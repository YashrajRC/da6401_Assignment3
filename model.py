"""
model.py — Transformer Architecture
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  scaled_dot_product_attention(Q, K, V, mask) -> (out, weights)
  MultiHeadAttention.forward(q, k, v, mask)   -> Tensor
  PositionalEncoding.forward(x)               -> Tensor
  make_src_mask(src, pad_idx)                 -> BoolTensor
  make_tgt_mask(tgt, pad_idx)                 -> BoolTensor
  Transformer.encode(src, src_mask)           -> Tensor
  Transformer.decode(memory, src_m, tgt, tgt_m) -> Tensor

Usage by the autograder:
  model = Transformer().to(device); model.eval()
  english = model.infer(german_sentence)
"""

import math
import copy
import os
import pickle
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# gdown is imported safely so importing this module never fails without it.
try:
    import gdown
except Exception:
    gdown = None


# ── Default hyper-parameters (autograder builds Transformer() with no args) ──
DEFAULT_SRC_VOCAB_SIZE = 7853       # German vocab size  (from Kaggle run)
DEFAULT_TGT_VOCAB_SIZE = 5893       # English vocab size (from Kaggle run)

DEFAULT_D_MODEL   = 256      # was 512
DEFAULT_N         = 3        # was 6
DEFAULT_NUM_HEADS = 8        # unchanged
DEFAULT_D_FF      = 512      # was 2048
DEFAULT_DROPOUT   = 0.1 
DEFAULT_MAX_LEN   = 128

# Special-token indices (must match dataset.py).
PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX = 1, 0, 2, 3

# Google-Drive file-ids of the trained checkpoint and the vocab bundle.
GDRIVE_WEIGHTS_ID = "1JRHL0_yAVRFYQZRl0U8g4FlmSSp_j0YF"
GDRIVE_VOCAB_ID   = "1COs30j8_WmTnKxHFQZ63ItK8Lw-6V0UN"


# ══════════════════════════════════════════════════════════════════════
#  STANDALONE ATTENTION FUNCTION
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Attention(Q,K,V) = softmax(Q·Kᵀ/√dₖ)·V ; mask==True positions masked out."""
    d_k = Q.size(-1)                                                # depth per head
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # raw similarity
    if mask is not None:
        scores = scores.masked_fill(mask, float("-1e9"))            # masked -> -inf
    attn_w = torch.softmax(scores, dim=-1)                          # weights sum to 1
    output = torch.matmul(attn_w, V)                                # weighted value sum
    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
#  MASK HELPERS
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(src: torch.Tensor, pad_idx: int = PAD_IDX) -> torch.Tensor:
    """Encoder padding mask -> [batch, 1, 1, src_len], True where <pad>."""
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt: torch.Tensor, pad_idx: int = PAD_IDX) -> torch.Tensor:
    """Decoder pad + causal mask -> [batch, 1, tgt_len, tgt_len], True = masked."""
    batch_size, tgt_len = tgt.shape
    device = tgt.device
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)           # pad positions
    causal_mask = torch.triu(                                      # future positions
        torch.ones(tgt_len, tgt_len, device=device, dtype=torch.bool),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(1)
    return pad_mask | causal_mask                                  # masked if pad OR future


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention (§3.2.2). nn.MultiheadAttention is NOT used."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads          # depth per head

        self.w_q = nn.Linear(d_model, d_model)         # Q projection
        self.w_k = nn.Linear(d_model, d_model)         # K projection
        self.w_v = nn.Linear(d_model, d_model)         # V projection
        self.w_o = nn.Linear(d_model, d_model)         # output projection W_O

        self.dropout = nn.Dropout(p=dropout)
        self.attn_weights: Optional[torch.Tensor] = None   # cached for report viz

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[batch, seq, d_model] -> [batch, heads, seq, d_k]."""
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[batch, heads, seq, d_k] -> [batch, seq, d_model]."""
        batch_size, num_heads, seq_len, d_k = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, num_heads * d_k)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Project, split into heads, attend, merge, project out -> [batch, seq_q, d_model]."""
        Q = self._split_heads(self.w_q(query))         # [batch, heads, seq_q, d_k]
        K = self._split_heads(self.w_k(key))           # [batch, heads, seq_k, d_k]
        V = self._split_heads(self.w_v(value))         # [batch, heads, seq_k, d_k]

        attn_out, attn_w = scaled_dot_product_attention(Q, K, V, mask)
        attn_out = self.dropout(attn_out)
        self.attn_weights = attn_w.detach()            # store for visualization

        merged = self._merge_heads(attn_out)           # back to [batch, seq_q, d_model]
        return self.w_o(merged)


# ══════════════════════════════════════════════════════════════════════
#  POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (§3.5): PE(pos,2i)=sin, PE(pos,2i+1)=cos."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)                          # position table
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(                                       # 1/10000^(2i/d_model)
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)                # even dims -> sin
        pe[:, 1::2] = torch.cos(position * div_term)                # odd dims  -> cos
        pe = pe.unsqueeze(0)                                        # [1, max_len, d_model]
        self.register_buffer("pe", pe)                              # buffer, not a parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add the position signal to the token embeddings."""
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """Trainable positional embedding — used only for the Section 2.4 ablation."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pos_embed = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pos_embed(positions)
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """Position-wise FFN (§3.3): FFN(x) = max(0, x·W₁+b₁)·W₂+b₂."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)        # widen   d_model -> d_ff
        self.linear2 = nn.Linear(d_ff, d_model)        # project d_ff   -> d_model
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ReLU between two linears, dropout in between."""
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
#  SUBLAYER CONNECTION  ("Add & Norm", Post-LayerNorm)
# ══════════════════════════════════════════════════════════════════════

class SublayerConnection(nn.Module):
    """Post-LN residual block: output = LayerNorm(x + Sublayer(x)) — the paper's ordering."""

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)              # assignment requires nn.LayerNorm
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        """Add the residual first, then normalize (Post-LN)."""
        return self.norm(x + self.dropout(sublayer(x)))


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """One encoder block: Self-Attention -> Add&Norm -> FFN -> Add&Norm."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_fwd  = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """[batch, src_len, d_model] -> same shape."""
        x = self.sublayer1(x, lambda t: self.self_attn(t, t, t, src_mask))  # self-attention
        x = self.sublayer2(x, self.feed_fwd)                                # feed-forward
        return x


# ══════════════════════════════════════════════════════════════════════
#  DECODER LAYER
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """One decoder block: Masked Self-Attn -> Cross-Attn -> FFN, each + Add&Norm."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)   # masked self-attn
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)   # encoder-decoder attn
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
        """[batch, tgt_len, d_model] -> same shape."""
        x = self.sublayer1(x, lambda t: self.self_attn(t, t, t, tgt_mask))            # masked self-attn
        x = self.sublayer2(x, lambda t: self.cross_attn(t, memory, memory, src_mask)) # cross-attn
        x = self.sublayer3(x, self.feed_fwd)                                          # feed-forward
        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

def _clones(module: nn.Module, n: int) -> nn.ModuleList:
    """Return n independent deep copies of a layer."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class Encoder(nn.Module):
    """Stack of N EncoderLayers with a final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = _clones(layer, N)
        self.norm = nn.LayerNorm(layer.sublayer1.norm.normalized_shape[0])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N DecoderLayers with a final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        self.layers = _clones(layer, N)
        self.norm = nn.LayerNorm(layer.sublayer1.norm.normalized_shape[0])

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
#  TOKEN-EMBEDDING HELPER
# ══════════════════════════════════════════════════════════════════════

class TokenEmbedding(nn.Module):
    """Token embedding scaled by sqrt(d_model) (§3.4)."""

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.d_model = d_model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embed(tokens) * math.sqrt(self.d_model)


# ══════════════════════════════════════════════════════════════════════
#  FULL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """Full encoder-decoder Transformer for German -> English translation."""

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
        load_pretrained: bool = True,
        pos_encoding:   str   = "sinusoidal",
        max_len:        int   = DEFAULT_MAX_LEN,
    ) -> None:
        super().__init__()

        # Config kept so save_checkpoint can reconstruct the model later.
        self.config = dict(
            src_vocab_size=src_vocab_size, tgt_vocab_size=tgt_vocab_size,
            d_model=d_model, N=N, num_heads=num_heads, d_ff=d_ff,
            dropout=dropout, pos_encoding=pos_encoding, max_len=max_len,
        )
        self.d_model = d_model
        self.max_len = max_len

        # Embeddings.
        self.src_embed = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_embed = TokenEmbedding(tgt_vocab_size, d_model)

        # Positional encoding (sinusoidal by default; 'learned' for ablation).
        if pos_encoding == "learned":
            self.pos_encoder = LearnedPositionalEncoding(d_model, dropout, max_len=5000)
            self.pos_decoder = LearnedPositionalEncoding(d_model, dropout, max_len=5000)
        else:
            self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=5000)
            self.pos_decoder = PositionalEncoding(d_model, dropout, max_len=5000)

        # Encoder / decoder stacks.
        self.encoder = Encoder(EncoderLayer(d_model, num_heads, d_ff, dropout), N)
        self.decoder = Decoder(DecoderLayer(d_model, num_heads, d_ff, dropout), N)

        # Final projection to target-vocab logits.
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        # Xavier init on every >1-D weight.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Vocab / tokenizer placeholders (filled below or via attach_vocab).
        self.src_vocab = None        # German word -> idx
        self.tgt_vocab = None        # English word -> idx
        self.tgt_itos  = None        # English idx -> word
        self._de_tokenizer = None    # str -> list[str]

        # Load trained weights + vocab from Drive (all inside __init__).
        if load_pretrained:
            self._download_and_load(checkpoint_path)

    # ── Weight / vocab loading ────────────────────────────────────────

    def _download_and_load(self, checkpoint_path: str) -> None:
        """Download and load the trained checkpoint + vocab bundle from Drive."""
        try:
            if not os.path.exists(checkpoint_path):
                if gdown is None:
                    raise RuntimeError("gdown is not installed.")
                gdown.download(id=GDRIVE_WEIGHTS_ID, output=checkpoint_path, quiet=False)
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            self.load_state_dict(state, strict=True)
            print(f"[Transformer] Loaded weights from {checkpoint_path}")
        except Exception as e:
            print(f"[Transformer] WARNING: could not load weights ({e}).")
        self._load_vocab_from_drive()

    def _load_vocab_from_drive(self, vocab_path: str = "vocab.pkl") -> None:
        """Download and attach the pickled vocab bundle from Drive."""
        try:
            if not os.path.exists(vocab_path):
                if gdown is None:
                    raise RuntimeError("gdown is not installed.")
                gdown.download(id=GDRIVE_VOCAB_ID, output=vocab_path, quiet=False)
            with open(vocab_path, "rb") as f:
                bundle = pickle.load(f)
            self.src_vocab = bundle["src_vocab"]
            self.tgt_vocab = bundle["tgt_vocab"]
            self.tgt_itos  = bundle["tgt_itos"]
            print(f"[Transformer] Loaded vocab bundle from {vocab_path} "
                  f"(src={len(self.src_vocab)}, tgt={len(self.tgt_vocab)})")
        except Exception as e:
            print(f"[Transformer] WARNING: could not load vocab ({e}).")
        self._init_tokenizer()

    def _init_tokenizer(self) -> None:
        """Load the spaCy German tokenizer used by infer()."""
        try:
            import spacy
            try:
                nlp = spacy.load("de_core_news_sm")
            except Exception:
                nlp = spacy.blank("de")            # tokenizer-only fallback
            self._de_tokenizer = lambda text: [t.text for t in nlp.tokenizer(text)]
        except Exception as e:
            print(f"[Transformer] WARNING: spaCy unavailable ({e}); using whitespace split.")
            self._de_tokenizer = lambda text: text.strip().split()

    def attach_vocab(self, src_vocab, tgt_vocab, tgt_itos, de_tokenizer=None) -> None:
        """Manually attach vocab + tokenizer (used during training)."""
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.tgt_itos  = tgt_itos
        if de_tokenizer is not None:
            self._de_tokenizer = de_tokenizer
        elif self._de_tokenizer is None:
            self._init_tokenizer()

    # ── Autograder hooks ──────────────────────────────────────────────

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """Embed + positional encode + run the encoder stack -> memory."""
        x = self.pos_encoder(self.src_embed(src))
        return self.encoder(x, src_mask)

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Run the decoder stack and project to vocab logits."""
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
        """Full encoder-decoder forward pass -> logits [batch, tgt_len, tgt_vocab]."""
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    # ── End-to-end inference (German string -> English string) ────────

    @torch.no_grad()
    def infer(self, src_sentence: str) -> str:
        """German -> English greedy translation (fast + CPU-safe for the autograder)."""
        if self.src_vocab is None or self.tgt_vocab is None:
            raise RuntimeError("infer() needs vocab; build with load_pretrained=True.")

        self.eval()
        device = next(self.parameters()).device

        # 1) tokenize the German input.
        tokens = self._de_tokenizer(src_sentence.lower().strip())

        # 2) words -> source indices, wrapped with <sos>/<eos>.
        src_ids = [SOS_IDX] + [self.src_vocab.get(t, UNK_IDX) for t in tokens] + [EOS_IDX]
        src = torch.tensor([src_ids], dtype=torch.long, device=device)

        # 3) encode the source once.
        src_mask = make_src_mask(src, PAD_IDX)
        memory   = self.encode(src, src_mask)

        # 4) greedy decode — cap length to stay well under the 3s autograder limit.
        max_decode = min(50, src.size(1) + 10)        # short Multi30k sentences only
        ys = torch.tensor([[SOS_IDX]], dtype=torch.long, device=device)
        for _ in range(max_decode - 1):
            tgt_mask = make_tgt_mask(ys, PAD_IDX)
            logits   = self.decode(memory, src_mask, ys, tgt_mask)   # [1, cur, vocab]
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True) # greedy pick
            ys = torch.cat([ys, next_tok], dim=1)
            if next_tok.item() == EOS_IDX:            # stop as soon as <eos> is emitted
                break

        # 5) indices -> English words, drop special tokens.
        words = []
        for idx in ys.squeeze(0).tolist():
            if idx in (SOS_IDX, PAD_IDX):
                continue
            if idx == EOS_IDX:
                break
            words.append(self.tgt_itos[idx] if idx < len(self.tgt_itos) else "<unk>")

        # 6) detokenize into a clean sentence.
        return " ".join(words).strip()


# ══════════════════════════════════════════════════════════════════════
#  QUICK SHAPE SANITY CHECK  (run:  python model.py)
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
    print("forward output shape :", tuple(out.shape), "(expect (2, 7, 120))")
    q = torch.randn(2, 4, 5, 8)
    _, w = scaled_dot_product_attention(q, q, q)
    print("attn weights sum     :", w.sum(-1).mean().item(), "(expect ~1.0)")