"""
model.py — Transformer Architecture
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘

Design notes
------------
* Base architecture from Vaswani et al. (2017), but deliberately smaller
  (d_model=256, N=3) because Multi30k has only 29k training pairs — the
  full paper-size model (512/6) overfits severely on this dataset.
* Post-LayerNorm ("Add & Norm" AFTER the residual) exactly as in the paper.
* infer() uses beam search (beam_size=5) for better translation quality
  than greedy decoding.
* All vocab/tokenizer/weight loading happens inside __init__ as required.
  The autograder simply does:
      model = Transformer().to(device); model.eval()
      english = model.infer(german_sentence)
"""

import math
import copy
import os
import pickle
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import gdown
except Exception:
    gdown = None


# ══════════════════════════════════════════════════════════════════════
#  GLOBAL DEFAULTS
#  — Must match the architecture of the checkpoint on Google Drive.
#  — After training on Kaggle, update these to your actual vocab sizes.
# ══════════════════════════════════════════════════════════════════════

DEFAULT_SRC_VOCAB_SIZE = 7853   # German  (printed by train.py on Kaggle)
DEFAULT_TGT_VOCAB_SIZE = 5893   # English (printed by train.py on Kaggle)

DEFAULT_D_MODEL   = 256
DEFAULT_N         = 3
DEFAULT_NUM_HEADS = 8
DEFAULT_D_FF      = 512
DEFAULT_DROPOUT   = 0.1
DEFAULT_MAX_LEN   = 150

PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX = 1, 0, 2, 3

# ── Replace with your own Google Drive file IDs after uploading ──────
GDRIVE_WEIGHTS_ID = "1MKpa-kFlD1_SL4fM-6nj7hM8D_LAJYrK"   # checkpoint .pth
GDRIVE_VOCAB_ID   = "10gRX-7r3Ktc-bUaXveCiuQtpfdjrEfzW"     # vocab.pkl


# ══════════════════════════════════════════════════════════════════════
# ❶  SCALED DOT-PRODUCT ATTENTION
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q    : (..., seq_q, d_k)
        K    : (..., seq_k, d_k)
        V    : (..., seq_k, d_v)
        mask : BoolTensor broadcastable to (..., seq_q, seq_k).
               True  → masked out (set to -inf before softmax).
    Returns:
        output : (..., seq_q, d_v)
        attn_w : (..., seq_q, seq_k)   attention weights (sum to 1 over seq_k)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float("-1e9"))

    attn_w = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_w, V)
    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
# ❷  MASK HELPERS
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(src: torch.Tensor, pad_idx: int = PAD_IDX) -> torch.Tensor:
    """
    Padding mask for the encoder.
    Returns BoolTensor [batch, 1, 1, src_len]  (True = PAD, mask it out).
    """
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = PAD_IDX) -> torch.Tensor:
    """
    Combined padding + causal mask for the decoder.
    Returns BoolTensor [batch, 1, tgt_len, tgt_len]  (True = masked out).
    """
    batch_size, tgt_len = tgt.shape
    device = tgt.device

    # Positions whose KEY token is a pad.
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)        # [B,1,1,T]

    # Upper-triangular = future positions (causal mask).
    causal = torch.triu(
        torch.ones(tgt_len, tgt_len, device=device, dtype=torch.bool),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(1)                                   # [1,1,T,T]

    return pad_mask | causal                                      # [B,1,T,T]


# ══════════════════════════════════════════════════════════════════════
# ❸  MULTI-HEAD ATTENTION  (torch.nn.MultiheadAttention NOT used)
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (§3.2.2).
    Uses a single large projection then reshapes into h heads — identical
    to h independent projections but more efficient.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(p=dropout)
        self.attn_weights: Optional[torch.Tensor] = None   # for W&B visualisation

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, S, d_model] -> [B, h, S, d_k]"""
        B, S, _ = x.shape
        return x.view(B, S, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, h, S, d_k] -> [B, S, d_model]"""
        B, h, S, dk = x.shape
        return x.transpose(1, 2).contiguous().view(B, S, h * dk)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        Q = self._split_heads(self.w_q(query))   # [B, h, Sq, dk]
        K = self._split_heads(self.w_k(key))
        V = self._split_heads(self.w_v(value))

        attn_out, attn_w = scaled_dot_product_attention(Q, K, V, mask)
        attn_out = self.dropout(attn_out)
        self.attn_weights = attn_w.detach()      # cache for visualisation

        merged = self._merge_heads(attn_out)     # [B, Sq, d_model]
        return self.w_o(merged)


# ══════════════════════════════════════════════════════════════════════
# ❹  POSITIONAL ENCODING  (sinusoidal, registered as buffer)
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal PE (§3.5):
        PE(pos, 2i)   = sin( pos / 10000^(2i/d_model) )
        PE(pos, 2i+1) = cos( pos / 10000^(2i/d_model) )

    Registered as a buffer (not trainable; saved in state_dict; moves
    with .to(device)). The autograder checks it is a buffer, not a param.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, S, d_model]  ->  same shape, PE added."""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
# ❹b  LEARNED POSITIONAL ENCODING  (ablation §2.4 only)
# ══════════════════════════════════════════════════════════════════════

class LearnedPositionalEncoding(nn.Module):
    """Trainable nn.Embedding indexed by position. Drop-in for §2.4."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout    = nn.Dropout(p=dropout)
        self.pos_embed  = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        return self.dropout(x + self.pos_embed(positions))


# ══════════════════════════════════════════════════════════════════════
# ❺  FEED-FORWARD NETWORK  (position-wise, §3.3)
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """FFN(x) = max(0, x·W₁+b₁)·W₂+b₂"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
# ❻  SUBLAYER CONNECTION  ("Add & Norm", Post-LayerNorm)
# ══════════════════════════════════════════════════════════════════════

class SublayerConnection(nn.Module):
    """
    Post-LayerNorm residual: output = LayerNorm(x + Dropout(Sublayer(x)))

    Post-LN is the exact ordering in the original paper (§3.1). On Multi30k
    with the Noam warm-up schedule it trains stably — warm-up keeps early
    gradients small enough to avoid the instability Post-LN is otherwise
    known for at large LR.
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm    = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        return self.norm(x + self.dropout(sublayer(x)))


# ══════════════════════════════════════════════════════════════════════
# ❼  ENCODER LAYER
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_fwd  = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        x = self.sublayer1(x, lambda t: self.self_attn(t, t, t, src_mask))
        x = self.sublayer2(x, self.feed_fwd)
        return x


# ══════════════════════════════════════════════════════════════════════
# ❽  DECODER LAYER
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_fwd   = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer1  = SublayerConnection(d_model, dropout)
        self.sublayer2  = SublayerConnection(d_model, dropout)
        self.sublayer3  = SublayerConnection(d_model, dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = self.sublayer1(x, lambda t: self.self_attn(t, t, t, tgt_mask))
        x = self.sublayer2(x, lambda t: self.cross_attn(t, memory, memory, src_mask))
        x = self.sublayer3(x, self.feed_fwd)
        return x


# ══════════════════════════════════════════════════════════════════════
# ❾  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

def _clones(module: nn.Module, n: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class Encoder(nn.Module):
    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = _clones(layer, N)
        self.norm   = nn.LayerNorm(layer.sublayer1.norm.normalized_shape[0])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = _clones(layer, N)
        self.norm   = nn.LayerNorm(layer.sublayer1.norm.normalized_shape[0])

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
# ❿  TOKEN EMBEDDING  (scaled by √d_model, §3.4)
# ══════════════════════════════════════════════════════════════════════

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embed  = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.d_model = d_model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embed(tokens) * math.sqrt(self.d_model)


# ══════════════════════════════════════════════════════════════════════
# ⓫  FULL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer.

    Default args match the trained checkpoint on Google Drive.
    Autograder usage:
        model = Transformer().to(device)
        model.eval()
        english = model.infer(german_sentence)
    """

    def __init__(
        self,
        src_vocab_size: int   = DEFAULT_SRC_VOCAB_SIZE,
        tgt_vocab_size: int   = DEFAULT_TGT_VOCAB_SIZE,
        d_model:        int   = DEFAULT_D_MODEL,
        N:              int   = DEFAULT_N,
        num_heads:      int   = DEFAULT_NUM_HEADS,
        d_ff:           int   = DEFAULT_D_FF,
        dropout:        float = DEFAULT_DROPOUT,
        checkpoint_path: str  = "checkpoint.pth",
        vocab_path:      str  = "vocab.pkl",
        load_pretrained: bool = True,
        pos_encoding:   str   = "sinusoidal",
        max_len:        int   = DEFAULT_MAX_LEN,
    ) -> None:
        super().__init__()

        self.config = dict(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=d_model, N=N, num_heads=num_heads,
            d_ff=d_ff, dropout=dropout,
            pos_encoding=pos_encoding, max_len=max_len,
        )
        self.d_model = d_model
        self.max_len = max_len

        # ── Embeddings ────────────────────────────────────────────────
        self.src_embed = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_embed = TokenEmbedding(tgt_vocab_size, d_model)

        # ── Positional encoding ───────────────────────────────────────
        PE = PositionalEncoding if pos_encoding == "sinusoidal" else LearnedPositionalEncoding
        self.pos_encoder = PE(d_model, dropout, max_len=5000)
        self.pos_decoder = PE(d_model, dropout, max_len=5000)

        # ── Encoder / Decoder stacks ──────────────────────────────────
        self.encoder = Encoder(EncoderLayer(d_model, num_heads, d_ff, dropout), N)
        self.decoder = Decoder(DecoderLayer(d_model, num_heads, d_ff, dropout), N)

        # ── Output projection ─────────────────────────────────────────
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        # Weight tying (§3.4): share weights between tgt embedding and
        # the output projection. Reduces parameters and improves BLEU.
        self.generator.weight = self.tgt_embed.embed.weight

        # ── Parameter initialisation (Xavier uniform) ─────────────────
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # ── Vocab / tokenizer placeholders ────────────────────────────
        self.src_vocab     = None   # word -> idx  (German)
        self.tgt_vocab     = None   # word -> idx  (English)
        self.tgt_itos      = None   # idx  -> word (English)
        self._de_tokenizer = None

        # ── Load weights + vocab from Drive (autograder path) ─────────
        if load_pretrained:
            self._download_and_load(checkpoint_path, vocab_path)

    # ──────────────────────────────────────────────────────────────────
    #  WEIGHT / VOCAB LOADING
    # ──────────────────────────────────────────────────────────────────

    def _download_and_load(self, checkpoint_path: str, vocab_path: str) -> None:
        """Download checkpoint + vocab from Google Drive, then load both."""
        # 1) Weights
        try:
            if not os.path.exists(checkpoint_path):
                if gdown is None:
                    raise RuntimeError("gdown not installed")
                print(f"[Transformer] Downloading weights …")
                gdown.download(id=GDRIVE_WEIGHTS_ID, output=checkpoint_path, quiet=False)

            ckpt  = torch.load(checkpoint_path, map_location="cpu")
            state = ckpt.get("model_state_dict", ckpt)
            self.load_state_dict(state, strict=True)
            print(f"[Transformer] Loaded weights from {checkpoint_path}")
        except Exception as e:
            print(f"[Transformer] WARNING: could not load weights ({e}). "
                  f"Running with random weights.")

        # 2) Vocab + tokenizer
        self._load_vocab(vocab_path)

    def _load_vocab(self, vocab_path: str = "vocab.pkl") -> None:
        """Load pickled vocab bundle and initialise the German tokenizer."""
        try:
            if not os.path.exists(vocab_path):
                if gdown is None:
                    raise RuntimeError("gdown not installed")
                print(f"[Transformer] Downloading vocab …")
                gdown.download(id=GDRIVE_VOCAB_ID, output=vocab_path, quiet=False)

            with open(vocab_path, "rb") as f:
                bundle = pickle.load(f)

            self.src_vocab = bundle["src_vocab"]   # dict word->idx
            self.tgt_vocab = bundle["tgt_vocab"]   # dict word->idx
            self.tgt_itos  = bundle["tgt_itos"]    # list idx->word
            print(f"[Transformer] Loaded vocab from {vocab_path} "
                  f"(src={len(self.src_vocab)}, tgt={len(self.tgt_vocab)})")
        except Exception as e:
            print(f"[Transformer] WARNING: could not load vocab ({e}). "
                  f"infer() unavailable until attach_vocab() is called.")

        self._init_tokenizer()

    def _init_tokenizer(self) -> None:
        """Load spaCy German tokenizer once."""
        try:
            import spacy
            try:
                nlp = spacy.load("de_core_news_sm")
            except Exception:
                nlp = spacy.blank("de")
            self._de_tokenizer = lambda text: [t.text for t in nlp.tokenizer(text)]
        except Exception as e:
            print(f"[Transformer] WARNING: spaCy unavailable ({e}); "
                  f"falling back to whitespace tokenization.")
            self._de_tokenizer = lambda text: text.strip().split()

    def attach_vocab(self, src_vocab, tgt_vocab, tgt_itos, de_tokenizer=None) -> None:
        """Manually attach vocab (used during training when load_pretrained=False)."""
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.tgt_itos  = tgt_itos
        if de_tokenizer is not None:
            self._de_tokenizer = de_tokenizer
        elif self._de_tokenizer is None:
            self._init_tokenizer()

    # ──────────────────────────────────────────────────────────────────
    #  AUTOGRADER HOOKS
    # ──────────────────────────────────────────────────────────────────

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Run the encoder stack.
        src:      [B, src_len]
        src_mask: [B, 1, 1, src_len]
        Returns:  [B, src_len, d_model]
        """
        x = self.pos_encoder(self.src_embed(src))
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the decoder stack + project to vocabulary logits.
        Returns: [B, tgt_len, tgt_vocab_size]
        """
        x = self.pos_decoder(self.tgt_embed(tgt))
        dec_out = self.decoder(x, memory, src_mask, tgt_mask)
        return self.generator(dec_out)

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Full encoder-decoder forward pass. Returns logits [B, T, vocab]."""
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    # ──────────────────────────────────────────────────────────────────
    #  END-TO-END INFERENCE  with beam search
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def infer(self, src_sentence: str, beam_size: int = 5) -> str:
        """
        Translate a German sentence to English.

        Pipeline:
          raw German text
            -> spaCy tokenization (lowercased)
            -> word indices (+ <sos>/<eos>)
            -> encoder
            -> beam-search decoder (beam_size=5)
            -> English word list
            -> detokenized string

        Args:
            src_sentence : Raw German text.
            beam_size    : Number of beams (default 5).

        Returns:
            Translated English string.
        """
        if self.src_vocab is None or self.tgt_vocab is None:
            raise RuntimeError(
                "infer() needs vocab. Either build Transformer() with "
                "load_pretrained=True, or call attach_vocab() first."
            )

        self.eval()
        device = next(self.parameters()).device

        # ── 1) Tokenize ───────────────────────────────────────────────
        raw_tokens = self._de_tokenizer(src_sentence.lower().strip())

        # ── 2) Words -> indices ───────────────────────────────────────
        src_ids = (
            [SOS_IDX]
            + [self.src_vocab.get(tok, UNK_IDX) for tok in raw_tokens]
            + [EOS_IDX]
        )
        src      = torch.tensor([src_ids], dtype=torch.long, device=device)
        src_mask = make_src_mask(src, PAD_IDX)

        # ── 3) Encode once ────────────────────────────────────────────
        memory = self.encode(src, src_mask)

        # ── 4) Beam search ────────────────────────────────────────────
        # Each beam: (length-normalised score, token list)
        beams: List[Tuple[float, List[int]]] = [(0.0, [SOS_IDX])]
        completed: List[Tuple[float, List[int]]] = []

        for _ in range(self.max_len - 1):
            if not beams:
                break

            candidates: List[Tuple[float, List[int]]] = []

            for score, token_seq in beams:
                ys       = torch.tensor([token_seq], dtype=torch.long, device=device)
                tgt_mask = make_tgt_mask(ys, PAD_IDX)
                logits   = self.decode(memory, src_mask, ys, tgt_mask)   # [1,cur,V]
                log_probs = torch.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)

                topk_lp, topk_ids = log_probs.topk(beam_size)
                cur_len = len(token_seq)

                for lp, tok_id in zip(topk_lp.tolist(), topk_ids.tolist()):
                    new_seq   = token_seq + [tok_id]
                    # Length normalisation: divide cumulative score by new length
                    new_score = (score * (cur_len - 1) + lp) / cur_len
                    if tok_id == EOS_IDX:
                        completed.append((new_score, new_seq))
                    else:
                        candidates.append((new_score, new_seq))

            # Keep top-beam_size active beams
            candidates.sort(key=lambda x: x[0], reverse=True)
            beams = candidates[:beam_size]

            # Stop early if we have enough finished hypotheses
            if len(completed) >= beam_size * 2:
                break

        # Fall back to active beams if nothing finished
        if not completed:
            completed = beams

        # Pick the best completed hypothesis
        completed.sort(key=lambda x: x[0], reverse=True)
        best_tokens = completed[0][1]

        # ── 5) Indices -> English words ───────────────────────────────
        words = []
        for idx in best_tokens:
            if idx in (SOS_IDX, PAD_IDX):
                continue
            if idx == EOS_IDX:
                break
            words.append(
                self.tgt_itos[idx] if idx < len(self.tgt_itos) else "<unk>"
            )

        return " ".join(words).strip()


# ══════════════════════════════════════════════════════════════════════
#  QUICK SHAPE CHECK  (python model.py)
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    torch.manual_seed(0)
    m = Transformer(
        src_vocab_size=100, tgt_vocab_size=120,
        d_model=32, N=2, num_heads=4, d_ff=64, dropout=0.1,
        load_pretrained=False,
    )
    src = torch.randint(4, 100, (2, 9))
    tgt = torch.randint(4, 120, (2, 7))
    out = m(src, tgt, make_src_mask(src), make_tgt_mask(tgt))
    print("forward shape:", tuple(out.shape), "  (expect (2, 7, 120))")

    q = torch.randn(2, 4, 5, 8)
    _, w = scaled_dot_product_attention(q, q, q)
    print("attn weight sum:", w.sum(-1).mean().item(), "  (expect ~1.0)")