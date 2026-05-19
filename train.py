"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘

This file is also the place where the W&B report experiments are launched
(see run_training_experiment and the EXPERIMENT_* helpers at the bottom).
"""

import math
import os
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import Transformer, make_src_mask, make_tgt_mask
from dataset import (
    Multi30kDataset, collate_batch,
    PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX,
)
from lr_scheduler import NoamScheduler


# ══════════════════════════════════════════════════════════════════════
# ❶  LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need" (§5.4, ε_ls = 0.1).

    WHY smooth the labels?
    ----------------------
    Plain cross-entropy pushes the probability of the correct word toward
    1.0 and every other word toward 0.0. That makes the model *over-confident*
    — bad for a translation task where several words are often acceptable.
    Label smoothing instead asks the model to put (1 - ε) probability on the
    correct word and spread the remaining ε uniformly over the other words.
    This acts as a regularizer: it slightly raises training perplexity but
    usually improves BLEU and calibration.

    Smoothed target distribution:
        y_smooth = (1 - eps) on the gold token,
                   eps / (vocab_size - 2) on every other NON-special token,
                   0 on <pad>.

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing      # mass kept on the gold token
        # KLDivLoss compares our log-probs against the smoothed target dist.
        # reduction='batchmean' averages the KL over the (non-pad) tokens.
        self.criterion  = nn.KLDivLoss(reduction="batchmean")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        # Convert raw logits to log-probabilities (KLDivLoss expects log-probs).
        log_probs = torch.log_softmax(logits, dim=-1)

        # Build the smoothed target distribution, same shape as log_probs.
        # We divide the smoothing mass over (vocab_size - 2): we exclude the
        # gold token itself and the <pad> token.
        smooth = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 2))
        # Put the big 'confidence' mass on the gold token of each row.
        smooth.scatter_(1, target.unsqueeze(1), self.confidence)
        # <pad> column must carry zero probability.
        smooth[:, self.pad_idx] = 0.0

        # Rows whose gold token is <pad> are pure padding -> zero them out
        # entirely so they contribute no loss.
        pad_rows = (target == self.pad_idx)
        if pad_rows.any():
            smooth.index_fill_(0, pad_rows.nonzero().squeeze(1), 0.0)

        return self.criterion(log_probs, smooth)


# ══════════════════════════════════════════════════════════════════════
# ❷  TRAINING / EVALUATION LOOP
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
    grad_log_callback=None,
) -> float:
    """
    Run one epoch of training or evaluation.

    The Transformer is trained with "teacher forcing": the decoder is fed the
    GOLD previous words and asked to predict the next word at every position
    in parallel. We therefore split each target sentence into:
        tgt_input  = tgt[:, :-1]   (what the decoder SEES   — drops <eos>)
        tgt_output = tgt[:, 1:]    (what it must PREDICT     — drops <sos>)

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.
        grad_log_callback : optional fn(step:int) called after each optimizer
                            step — used by the gradient-norm ablation.

    Returns:
        avg_loss : Average loss over the epoch (float).
    """
    model.train() if is_train else model.eval()

    total_loss   = 0.0
    total_tokens = 0

    # Disable gradient bookkeeping entirely during evaluation (faster, less RAM).
    grad_context = torch.enable_grad() if is_train else torch.no_grad()

    with grad_context:
        for src, tgt in data_iter:
            src = src.to(device)
            tgt = tgt.to(device)

            # Teacher forcing: split target into decoder-input / gold-output.
            tgt_input  = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            # Build the masks for this batch.
            src_mask = make_src_mask(src, PAD_IDX)
            tgt_mask = make_tgt_mask(tgt_input, PAD_IDX)

            # Forward pass -> logits [batch, tgt_len, vocab].
            logits = model(src, tgt_input, src_mask, tgt_mask)

            # Flatten to [batch*tgt_len, vocab] / [batch*tgt_len] for the loss.
            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1),
            )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping guards against the occasional exploding
                # gradient — standard practice for Transformers.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()        # Noam updates the LR every step
                if grad_log_callback is not None:
                    grad_log_callback(epoch_num)

            # Track loss weighted by number of real (non-pad) target tokens.
            n_tokens = (tgt_output != PAD_IDX).sum().item()
            total_loss   += loss.item() * n_tokens
            total_tokens += n_tokens

    return total_loss / max(total_tokens, 1)


# ══════════════════════════════════════════════════════════════════════
# ❸  GREEDY DECODING
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int = EOS_IDX,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    "Greedy" = at every step we pick the single highest-probability token
    (argmax) and feed it back in as input for the next step. We never look
    back or explore alternatives (that would be beam search).

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.
    """
    model.eval()
    src = src.to(device)
    src_mask = src_mask.to(device)

    # Encode the source ONCE — the encoder output never changes during decode.
    memory = model.encode(src, src_mask)

    # Seed the output with just the <sos> token.
    ys = torch.full((1, 1), start_symbol, dtype=torch.long, device=device)

    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys, PAD_IDX)
        logits   = model.decode(memory, src_mask, ys, tgt_mask)  # [1, cur, vocab]
        # Greedy pick on the LAST time-step.
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
        ys = torch.cat([ys, next_tok], dim=1)
        if next_tok.item() == end_symbol:
            break

    return ys


# ══════════════════════════════════════════════════════════════════════
# ❹  BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════════

def _ids_to_tokens(ids, itos):
    """Map a list of indices to words, dropping the 4 special tokens."""
    out = []
    for idx in ids:
        if idx in (SOS_IDX, PAD_IDX):
            continue
        if idx == EOS_IDX:
            break
        out.append(itos[idx] if idx < len(itos) else "<unk>")
    return out


def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    BLEU compares the model's output against the reference translation by
    counting overlapping n-grams (1- to 4-grams) and applying a brevity
    penalty for too-short outputs. Higher is better; we report it on a
    0–100 scale.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split, yielding
                          (src, tgt) token-index tensors.
        tgt_vocab       : Vocabulary object with an `itos` list (and/or
                          `lookup_token`) for the English side.
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).
    """
    model.eval()

    # Resolve the index->word mapping regardless of vocab object flavour.
    if hasattr(tgt_vocab, "itos"):
        itos = tgt_vocab.itos
    elif hasattr(tgt_vocab, "get_itos"):
        itos = tgt_vocab.get_itos()
    else:                                              # last-resort fallback
        itos = [tgt_vocab.lookup_token(i) for i in range(len(tgt_vocab))]

    hypotheses = []   # list of token-lists  (model outputs)
    references = []   # list of [token-list] (gold; nested for sacre/nltk API)

    with torch.no_grad():
        for src, tgt in test_dataloader:
            # Decode each sentence in the batch one at a time (batch=1 logic).
            for i in range(src.size(0)):
                src_i = src[i : i + 1].to(device)
                src_mask = make_src_mask(src_i, PAD_IDX)

                pred = greedy_decode(
                    model, src_i, src_mask,
                    max_len=max_len, start_symbol=SOS_IDX,
                    end_symbol=EOS_IDX, device=device,
                )
                hyp_tokens = _ids_to_tokens(pred.squeeze(0).tolist(), itos)
                ref_tokens = _ids_to_tokens(tgt[i].tolist(), itos)

                hypotheses.append(hyp_tokens)
                references.append([ref_tokens])   # one reference per sentence

    return _corpus_bleu(hypotheses, references)


def _corpus_bleu(hypotheses, references) -> float:
    """
    Corpus-level BLEU on the 0–100 scale.

    Prefers NLTK's corpus_bleu (with smoothing) and falls back to a small
    self-contained implementation so this never crashes if nltk is missing.
    """
    # ── Preferred path: NLTK ──────────────────────────────────────────
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        smooth = SmoothingFunction().method1
        score = corpus_bleu(references, hypotheses, smoothing_function=smooth)
        return score * 100.0
    except Exception:
        pass

    # ── Fallback: minimal BLEU-4 implementation ───────────────────────
    from collections import Counter

    def ngrams(tokens, n):
        return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))

    weights = [0.25, 0.25, 0.25, 0.25]
    precisions = []
    for n in range(1, 5):
        match, total = 0, 0
        for hyp, refs in zip(hypotheses, references):
            hyp_ng = ngrams(hyp, n)
            ref_ng = ngrams(refs[0], n)
            for ng, cnt in hyp_ng.items():
                match += min(cnt, ref_ng.get(ng, 0))
            total += max(sum(hyp_ng.values()), 1)
        # +1 smoothing to keep log() finite when an n-gram order has 0 matches.
        precisions.append((match + 1) / (total + 1))

    hyp_len = sum(len(h) for h in hypotheses)
    ref_len = sum(len(r[0]) for r in references)
    # Brevity penalty: punishes outputs shorter than the reference.
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))

    score = bp * math.exp(sum(w * math.log(p) for w, p in zip(weights, precisions)))
    return score * 100.0


# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    Args:
        model     : Transformer instance.
        optimizer : Optimizer instance.
        scheduler : NoamScheduler instance.
        epoch     : Current epoch number.
        path      : File path to save to.

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'
    """
    torch.save(
        {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            # model_config lets anyone rebuild Transformer(**model_config).
            "model_config":         getattr(model, "config", {}),
        },
        path,
    )
    print(f"[checkpoint] saved -> {path}  (epoch {epoch})")


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Args:
        path      : Path to checkpoint file saved by save_checkpoint.
        model     : Uninitialised Transformer with matching architecture.
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).
    """
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    return ckpt.get("epoch", 0)


# ══════════════════════════════════════════════════════════════════════
# ❻  EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

# Default hyper-parameters for the main training run. Tuned to fit the small
# Multi30k dataset on a single Kaggle GPU in a reasonable time.
DEFAULT_HPARAMS = dict(
    d_model      = 512,
    N            = 6,
    num_heads    = 8,
    d_ff         = 2048,
    dropout      = 0.1,
    batch_size   = 128,
    num_epochs   = 20,
    warmup_steps = 4000,
    smoothing    = 0.1,
    min_freq     = 2,
)


def build_dataloaders(batch_size: int, min_freq: int):
    """
    Build train / validation / test DataLoaders + return the dataset objects.

    The vocab is built from the TRAIN split and reused everywhere else, so
    the train/test isolation the assignment demands is guaranteed.
    """
    train_ds = Multi30kDataset(split="train", min_freq=min_freq)
    val_ds   = Multi30kDataset(
        split="validation",
        src_vocab=train_ds.src_vocab, tgt_vocab=train_ds.tgt_vocab,
        de_tokenizer=train_ds.de_tokenizer, en_tokenizer=train_ds.en_tokenizer,
    )
    test_ds  = Multi30kDataset(
        split="test",
        src_vocab=train_ds.src_vocab, tgt_vocab=train_ds.tgt_vocab,
        de_tokenizer=train_ds.de_tokenizer, en_tokenizer=train_ds.en_tokenizer,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_batch)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              collate_fn=collate_batch)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False,
                              collate_fn=collate_batch)
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def run_training_experiment(
    hparams: dict = None,
    use_wandb: bool = True,
    scheduler_type: str = "noam",      # 'noam' or 'fixed'
    use_scaling: bool = True,          # √dk scaling on/off  (Section 2.2)
    pos_encoding: str = "sinusoidal",  # 'sinusoidal' or 'learned' (Section 2.4)
    smoothing_override: float = None,  # override ε_ls           (Section 2.5)
    run_name: str = "main",
) -> dict:
    """
    Set up and run the full training experiment.

    This single function powers BOTH the main training run AND every
    W&B ablation (just call it with different flags). It returns a dict
    of final metrics so the Kaggle notebook can collect results.

    Steps:
        1. Init W&B
        2. Build dataset / vocabs / DataLoaders
        3. Instantiate Transformer (load_pretrained=False -> fresh weights)
        4. Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
        5. NoamScheduler  OR  a fixed LR
        6. LabelSmoothingLoss
        7. Epoch loop: train + validate + checkpoint
        8. Final BLEU on the test set
    """
    hp = dict(DEFAULT_HPARAMS)
    if hparams:
        hp.update(hparams)
    if smoothing_override is not None:
        hp["smoothing"] = smoothing_override

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[experiment '{run_name}'] device = {device}")

    # ── 1) W&B ────────────────────────────────────────────────────────
    wandb = None
    if use_wandb:
        try:
            import wandb as _wandb
            wandb = _wandb
            wandb.init(
                project="da6401-a3",
                name=run_name,
                config={**hp, "scheduler_type": scheduler_type,
                        "use_scaling": use_scaling, "pos_encoding": pos_encoding},
            )
        except Exception as e:
            print(f"[wandb] disabled ({e})")
            wandb = None

    # ── 2) Data ───────────────────────────────────────────────────────
    (train_ds, val_ds, test_ds,
     train_loader, val_loader, test_loader) = build_dataloaders(
        hp["batch_size"], hp["min_freq"]
    )
    src_vocab_size = len(train_ds.src_vocab)
    tgt_vocab_size = len(train_ds.tgt_vocab)
    print(f"[experiment] VOCAB SIZES -> src(de)={src_vocab_size}, "
          f"tgt(en)={tgt_vocab_size}   <-- put these in model.py defaults")

    # Save the vocab bundle so model.infer() can use it later.
    train_ds.export_vocab_bundle("vocab.pkl")

    # ── 3) Model ──────────────────────────────────────────────────────
    # load_pretrained=False -> a fresh random model (no Drive download).
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=hp["d_model"], N=hp["N"], num_heads=hp["num_heads"],
        d_ff=hp["d_ff"], dropout=hp["dropout"],
        pos_encoding=pos_encoding,
        load_pretrained=False,
    ).to(device)
    # Attach vocab so model.infer() works for spot-checks during training.
    model.attach_vocab(
        train_ds.src_vocab.stoi, train_ds.tgt_vocab.stoi,
        train_ds.tgt_vocab.itos, de_tokenizer=train_ds.de_tokenizer,
    )

    # Section 2.2 ablation: turn off the 1/√dk scaling if requested.
    if not use_scaling:
        _disable_attention_scaling()

    # ── 4) Optimizer ──────────────────────────────────────────────────
    # Adam with the exact betas/eps from the paper.
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9,
    )

    # ── 5) Scheduler ──────────────────────────────────────────────────
    if scheduler_type == "noam":
        scheduler = NoamScheduler(optimizer, d_model=hp["d_model"],
                                  warmup_steps=hp["warmup_steps"])
    else:
        # Fixed-LR baseline for Section 2.1: constant 1e-4, no warm-up.
        for g in optimizer.param_groups:
            g["lr"] = 1e-4
        scheduler = None

    # ── 6) Loss ───────────────────────────────────────────────────────
    loss_fn = LabelSmoothingLoss(tgt_vocab_size, PAD_IDX, smoothing=hp["smoothing"])

    # ── 7) Training loop ──────────────────────────────────────────────
    best_val = float("inf")
    for epoch in range(hp["num_epochs"]):
        train_loss = run_epoch(train_loader, model, loss_fn, optimizer,
                               scheduler, epoch, is_train=True, device=device)
        val_loss   = run_epoch(val_loader, model, loss_fn, None,
                               None, epoch, is_train=False, device=device)

        # Validation accuracy = next-token prediction accuracy on the val set.
        val_acc = compute_token_accuracy(model, val_loader, device)
        print(f"epoch {epoch:02d} | train {train_loss:.4f} | "
              f"val {val_loss:.4f} | val_acc {val_acc:.4f}")

        if wandb:
            wandb.log({"epoch": epoch, "train_loss": train_loss,
                       "val_loss": val_loss, "val_acc": val_acc})

        # Always keep the best-on-validation checkpoint.
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, "best_checkpoint.pth")
        save_checkpoint(model, optimizer, scheduler, epoch, "checkpoint.pth")

    # ── 8) Final test-set BLEU ────────────────────────────────────────
    # Reload the best checkpoint before the final score.
    load_checkpoint("best_checkpoint.pth", model)
    bleu = evaluate_bleu(model, test_loader, train_ds.tgt_vocab, device=device)
    print(f"[experiment '{run_name}'] TEST BLEU = {bleu:.2f}")
    if wandb:
        wandb.log({"test_bleu": bleu})
        wandb.finish()

    return {"run_name": run_name, "test_bleu": bleu,
            "best_val_loss": best_val,
            "src_vocab_size": src_vocab_size,
            "tgt_vocab_size": tgt_vocab_size}


# ══════════════════════════════════════════════════════════════════════
# ❼  SMALL HELPERS  (validation accuracy, ablation switches, etc.)
# ══════════════════════════════════════════════════════════════════════

def compute_token_accuracy(model, loader, device) -> float:
    """Next-token prediction accuracy on a loader (ignores <pad> targets)."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            logits = model(src, tgt_in,
                           make_src_mask(src, PAD_IDX),
                           make_tgt_mask(tgt_in, PAD_IDX))
            pred = logits.argmax(-1)
            mask = (tgt_out != PAD_IDX)
            correct += ((pred == tgt_out) & mask).sum().item()
            total   += mask.sum().item()
    return correct / max(total, 1)


def _disable_attention_scaling():
    """
    Section 2.2 ablation: monkey-patch scaled_dot_product_attention so it
    does NOT divide by sqrt(d_k). Used to demonstrate why the scaling
    matters (without it, large dot products saturate the softmax and the
    Q/K gradients vanish).
    """
    import model as _m

    def _unscaled(Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1))      # NO /sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask, float("-1e9"))
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, V), attn

    _m.scaled_dot_product_attention = _unscaled
    print("[ablation] √dk scaling DISABLED")


def main():
    """Plain (no-W&B) training entry point — handy for a local smoke test."""
    run_training_experiment(use_wandb=False, run_name="main")


if __name__ == "__main__":
    main()