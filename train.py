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



class LabelSmoothingLoss(nn.Module):

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing
        # KLDivLoss(reduction='batchmean') sums KL then divides by batch size.
        self.criterion  = nn.KLDivLoss(reduction="batchmean")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits, dim=-1)

        smooth = torch.full_like(log_probs, self.smoothing / (self.vocab_size - 2))

        smooth.scatter_(1, target.unsqueeze(1), self.confidence)
   
        smooth[:, self.pad_idx] = 0.0

        pad_rows = (target == self.pad_idx)
        if pad_rows.any():
            smooth.index_fill_(0, pad_rows.nonzero(as_tuple=False).squeeze(1), 0.0)

        return self.criterion(log_probs, smooth)




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
    model.train() if is_train else model.eval()

    total_loss   = 0.0
    total_tokens = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for src, tgt in data_iter:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input  = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_mask = make_src_mask(src, PAD_IDX)
            tgt_mask = make_tgt_mask(tgt_input, PAD_IDX)

            logits = model(src, tgt_input, src_mask, tgt_mask)  # [B, T, V]

            loss = loss_fn(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1),
            )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                if grad_log_callback is not None:
                    grad_log_callback(epoch_num)

            n_tokens = (tgt_output != PAD_IDX).sum().item()
            total_loss   += loss.item() * n_tokens
            total_tokens += n_tokens

    return total_loss / max(total_tokens, 1)



def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int = EOS_IDX,
    device: str = "cpu",
) -> torch.Tensor:
    model.eval()
    src      = src.to(device)
    src_mask = src_mask.to(device)

    # Encode once.
    memory = model.encode(src, src_mask)

    # Seed with <sos>.
    ys = torch.full((1, 1), start_symbol, dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_len - 1):
            tgt_mask = make_tgt_mask(ys, PAD_IDX)
            logits   = model.decode(memory, src_mask, ys, tgt_mask)
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_tok], dim=1)
            if next_tok.item() == end_symbol:
                break

    return ys



def _ids_to_tokens(ids, itos):
    words = []
    for idx in ids:
        if idx in (SOS_IDX, PAD_IDX):
            continue
        if idx == EOS_IDX:
            break
        words.append(itos[idx] if idx < len(itos) else "<unk>")
    return words


def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 150,
) -> float:
    model.eval()

    if hasattr(tgt_vocab, "itos"):
        itos = tgt_vocab.itos
    elif hasattr(tgt_vocab, "get_itos"):
        itos = tgt_vocab.get_itos()
    else:
        itos = [tgt_vocab.lookup_token(i) for i in range(len(tgt_vocab))]

    hypotheses = []
    references = []

    with torch.no_grad():
        for src, tgt in test_dataloader:
            for i in range(src.size(0)):
                src_i    = src[i:i+1].to(device)
                src_mask = make_src_mask(src_i, PAD_IDX)

                pred = greedy_decode(
                    model, src_i, src_mask,
                    max_len=max_len,
                    start_symbol=SOS_IDX,
                    end_symbol=EOS_IDX,
                    device=device,
                )
                hyp = _ids_to_tokens(pred.squeeze(0).tolist(), itos)
                ref = _ids_to_tokens(tgt[i].tolist(), itos)

                hypotheses.append(hyp)
                references.append([ref])

    return _corpus_bleu(hypotheses, references)


def _corpus_bleu(hypotheses, references) -> float:
   
    try:
        from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
        score = corpus_bleu(
            references, hypotheses,
            smoothing_function=SmoothingFunction().method1,
        )
        return score * 100.0
    except Exception:
        pass
    from collections import Counter

    def ngrams(tokens, n):
        return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

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
        precisions.append((match + 1) / (total + 1))   # +1 smoothing

    hyp_len = sum(len(h)    for h in hypotheses)
    ref_len = sum(len(r[0]) for r in references)
    bp      = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(hyp_len, 1))
    score   = bp * math.exp(sum(w * math.log(p) for w, p in zip(weights, precisions)))
    return score * 100.0



def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pth",
) -> None:
    torch.save(
        {
            "epoch":                epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
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
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and ckpt.get("optimizer_state_dict"):
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    return ckpt.get("epoch", 0)



def compute_token_accuracy(model: Transformer, loader, device: str) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
            logits = model(
                src, tgt_in,
                make_src_mask(src, PAD_IDX),
                make_tgt_mask(tgt_in, PAD_IDX),
            )
            pred = logits.argmax(-1)
            mask = tgt_out != PAD_IDX
            correct += ((pred == tgt_out) & mask).sum().item()
            total   += mask.sum().item()
    return correct / max(total, 1)


def _disable_attention_scaling():
    import model as _m

    def _unscaled(Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1))   # no /√dk
        if mask is not None:
            scores = scores.masked_fill(mask, float("-1e9"))
        attn = torch.softmax(scores, dim=-1)
        return torch.matmul(attn, V), attn

    _m.scaled_dot_product_attention = _unscaled
    print("[ablation] √dk scaling DISABLED")


DEFAULT_HPARAMS = dict(
    d_model      = 256,
    N            = 3,
    num_heads    = 8,
    d_ff         = 512,
    dropout      = 0.1,
    batch_size   = 64,      
    num_epochs   = 50,
    warmup_steps = 4000,
    smoothing    = 0.1,
    min_freq     = 2,
    patience     = 10,      
)


def build_dataloaders(batch_size: int, min_freq: int):
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

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate_batch
    )
    val_loader   = DataLoader(
        val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )
    test_loader  = DataLoader(
        test_ds,  batch_size=1,          shuffle=False, collate_fn=collate_batch
    )
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader


def run_training_experiment(
    hparams: dict            = None,
    use_wandb: bool          = True,
    scheduler_type: str      = "noam",       # 'noam' | 'fixed'
    use_scaling: bool        = True,         # √dk on/off  (§2.2 ablation)
    pos_encoding: str        = "sinusoidal", # 'sinusoidal' | 'learned' (§2.4)
    smoothing_override: float = None,        # override ε_ls  (§2.5)
    run_name: str            = "main",
) -> dict:
    hp = dict(DEFAULT_HPARAMS)
    if hparams:
        hp.update(hparams)
    if smoothing_override is not None:
        hp["smoothing"] = smoothing_override

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  Experiment : {run_name}")
    print(f"  Device     : {device}")
    print(f"  Hparams    : {hp}")
    print(f"{'='*60}\n")

    wandb = None
    if use_wandb:
        try:
            import wandb as _wandb
            wandb = _wandb
            wandb.init(
                project="da6401-a3",
                name=run_name,
                config={**hp,
                        "scheduler_type": scheduler_type,
                        "use_scaling":    use_scaling,
                        "pos_encoding":   pos_encoding},
            )
        except Exception as e:
            print(f"[wandb] disabled ({e})")
            wandb = None

    (train_ds, val_ds, test_ds,
     train_loader, val_loader, test_loader) = build_dataloaders(
        hp["batch_size"], hp["min_freq"]
    )
    src_vocab_size = len(train_ds.src_vocab)
    tgt_vocab_size = len(train_ds.tgt_vocab)
    print(f"VOCAB SIZES: src(de)={src_vocab_size},  tgt(en)={tgt_vocab_size}")
    print("  ^^ Copy these into DEFAULT_SRC_VOCAB_SIZE / DEFAULT_TGT_VOCAB_SIZE in model.py")

    # Save vocab bundle for infer()
    train_ds.export_vocab_bundle("vocab.pkl")

    model = Transformer(
        src_vocab_size = src_vocab_size,
        tgt_vocab_size = tgt_vocab_size,
        d_model        = hp["d_model"],
        N              = hp["N"],
        num_heads      = hp["num_heads"],
        d_ff           = hp["d_ff"],
        dropout        = hp["dropout"],
        pos_encoding   = pos_encoding,
        load_pretrained= False,
    ).to(device)

    model.attach_vocab(
        train_ds.src_vocab.stoi,
        train_ds.tgt_vocab.stoi,
        train_ds.tgt_vocab.itos,
        de_tokenizer=train_ds.de_tokenizer,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    
    if not use_scaling:
        _disable_attention_scaling()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )

    if scheduler_type == "noam":
        scheduler = NoamScheduler(
            optimizer, d_model=hp["d_model"], warmup_steps=hp["warmup_steps"]
        )
    else:
        for g in optimizer.param_groups:
            g["lr"] = 1e-4
        scheduler = None

    loss_fn = LabelSmoothingLoss(tgt_vocab_size, PAD_IDX, smoothing=hp["smoothing"])

    best_val   = float("inf")
    no_improve = 0

    for epoch in range(hp["num_epochs"]):
        train_loss = run_epoch(
            train_loader, model, loss_fn, optimizer, scheduler,
            epoch, is_train=True, device=device,
        )
        val_loss = run_epoch(
            val_loader, model, loss_fn, None, None,
            epoch, is_train=False, device=device,
        )
        val_acc = compute_token_accuracy(model, val_loader, device)

        print(f"epoch {epoch:03d} | "
              f"train {train_loss:.4f} | val {val_loss:.4f} | acc {val_acc:.4f}")

        if wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss":   val_loss,
                "val_acc":    val_acc,
                "lr": optimizer.param_groups[0]["lr"],
            })

        # Save best checkpoint
        if val_loss < best_val:
            best_val   = val_loss
            no_improve = 0
            save_checkpoint(model, optimizer, scheduler, epoch, "best_checkpoint.pth")
            print(f"  ↳ new best val loss {best_val:.4f} — checkpoint saved")
        else:
            no_improve += 1
            if no_improve >= hp["patience"]:
                print(f"Early stopping at epoch {epoch} "
                      f"(no improvement for {hp['patience']} epochs)")
                break

        save_checkpoint(model, optimizer, scheduler, epoch, "checkpoint.pth")

    print("\nLoading best checkpoint for final evaluation …")
    load_checkpoint("best_checkpoint.pth", model)
    bleu = evaluate_bleu(model, test_loader, train_ds.tgt_vocab, device=device)
    print(f"\n[{run_name}]  TEST BLEU = {bleu:.2f}")

    if wandb:
        wandb.log({"test_bleu": bleu})
        wandb.finish()

    return {
        "run_name":       run_name,
        "test_bleu":      bleu,
        "best_val_loss":  best_val,
        "src_vocab_size": src_vocab_size,
        "tgt_vocab_size": tgt_vocab_size,
    }


def main():
    run_training_experiment(use_wandb=False, run_name="main")


if __name__ == "__main__":
    main()
