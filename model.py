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
* This file implements the *base* architecture of Vaswani et al. (2017).
* We use **Post-LayerNorm** ("Add & Norm" applied AFTER the residual add),
  which is the exact ordering described in the original paper. See the
  SublayerConnection docstring for the justification asked for in the report.
* The Transformer class additionally exposes an `infer()` method that runs the
  full German -> English pipeline (tokenize -> encode -> greedy decode ->
  detokenize). All vocab / tokenizer / weight loading happens inside
  __init__ so that the autograder can simply do:

      model = Transformer().to(device)
      model.eval()
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

# gdown is only needed when we have to download the trained weights from Drive.
# We import it lazily-safely so that importing this module never fails even if
# gdown is missing in some minimal test environment.
try:
    import gdown
except Exception:  # pragma: no cover
    gdown = None


# ══════════════════════════════════════════════════════════════════════
#  GLOBAL DEFAULT HYPER-PARAMETERS
#  ---------------------------------------------------------------------
#  The autograder constructs the model with NO arguments: Transformer().
#  Therefore EVERY constructor argument must have a sensible default, and
#  the defaults below must match the architecture of the checkpoint we
#  trained on Kaggle and uploaded to Google Drive.
#
#  IMPORTANT: After you train on Kaggle you will know the exact vocab
#  sizes. Put those numbers here so that Transformer() (no-args) builds an
#  architecture that EXACTLY matches the saved weights.
# ══════════════════════════════════════════════════════════════════════

# These two values are produced by dataset.py when the vocab is built.
# Replace them with the numbers printed by the Kaggle notebook (Step "VOCAB
# SIZES"). If you forget, load_state_dict will raise a size-mismatch error.
DEFAULT_SRC_VOCAB_SIZE = 7853     # German vocab size  (UPDATE after Kaggle run)
DEFAULT_TGT_VOCAB_SIZE = 5893     # English vocab size (UPDATE after Kaggle run)

DEFAULT_D_MODEL   = 512
DEFAULT_N         = 6
DEFAULT_NUM_HEADS = 8
DEFAULT_D_FF      = 2048
DEFAULT_DROPOUT   = 0.1
DEFAULT_MAX_LEN   = 128            # max decode length used by infer()

# Special-token indices. These MUST be identical to what dataset.py assigns.
PAD_IDX, UNK_IDX, SOS_IDX, EOS_IDX = 1, 0, 2, 3

# Google-Drive file-id of the trained checkpoint (.pth). Fill this in AFTER
# you upload your Kaggle checkpoint to Drive and make it shareable.
# Example: if the share link is
#   https://drive.google.com/file/d/1AbCdEfGhIjK/view?usp=sharing
# then GDRIVE_WEIGHTS_ID = "1AbCdEfGhIjK"
GDRIVE_WEIGHTS_ID = "187c8NjOHUWPC3IkrcYgd6lFXfyGOuIWu"

# Google-Drive file-id of the pickled vocab object produced on Kaggle.
# infer() needs the vocab to map words<->indices, so we ship it too.
GDRIVE_VOCAB_ID = "14FCZPQoWu4LbNtygN9RXEvkA1FZ_vxRA"


# ══════════════════════════════════════════════════════════════════════
# ❶  STANDALONE ATTENTION FUNCTION
#    Exposed at module level so the autograder can import and test it
#    independently of MultiHeadAttention.
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

        Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q    : Query tensor,  shape (..., seq_q, d_k)
        K    : Key tensor,    shape (..., seq_k, d_k)
        V    : Value tensor,  shape (..., seq_k, d_v)
        mask : Optional Boolean mask, shape broadcastable to
               (..., seq_q, seq_k).
               Positions where mask is True are MASKED OUT
               (set to -inf before softmax).

    Returns:
        output : Attended output,   shape (..., seq_q, d_v)
        attn_w : Attention weights, shape (..., seq_q, seq_k)
    """
    # d_k is the depth of a single head. We need it for the 1/sqrt(d_k) scaling.
    d_k = Q.size(-1)

    # scores[i,j] = how much query i "matches" key j  (a raw similarity score)
    # Q  : (..., seq_q, d_k)
    # Kᵀ : (..., d_k, seq_k)   -> matmul gives (..., seq_q, seq_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # ── Masking ───────────────────────────────────────────────────────
    # Wherever mask == True we want the softmax weight to become 0.
    # softmax(-inf) = 0, so we replace those score entries with -inf
    # (we use a very large negative number to stay numerically safe).
    if mask is not None:
        scores = scores.masked_fill(mask, float("-1e9"))

    # softmax over the LAST axis (the key axis) -> each query row sums to 1.
    attn_w = torch.softmax(scores, dim=-1)

    # Weighted sum of the value vectors -> the attended output.
    output = torch.matmul(attn_w, V)

    return output, attn_w


# ══════════════════════════════════════════════════════════════════════
# ❷  MASK HELPERS
#    Exposed at module level so they can be tested independently and
#    reused inside Transformer.forward.
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = PAD_IDX,
) -> torch.Tensor:
    """
    Build a padding mask for the encoder (source sequence).

    Args:
        src     : Source token-index tensor, shape [batch, src_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, 1, src_len]
        True  → position is a PAD token (will be masked out)
        False → real token
    """
    # (src == pad_idx) is True exactly at the padded positions.
    # We add two singleton dims so the mask broadcasts over
    # (num_heads) and (seq_q) when used inside attention:
    #   [batch, src_len] -> [batch, 1, 1, src_len]
    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = PAD_IDX,
) -> torch.Tensor:
    """
    Build a combined padding + causal (look-ahead) mask for the decoder.

    Args:
        tgt     : Target token-index tensor, shape [batch, tgt_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, tgt_len, tgt_len]
        True → position is masked out (PAD or future token)
    """
    batch_size, tgt_len = tgt.shape
    device = tgt.device

    # ── Padding part ──────────────────────────────────────────────────
    # True where the *key* token is a pad. Shape -> [batch, 1, 1, tgt_len].
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)

    # ── Causal (look-ahead) part ──────────────────────────────────────
    # We must stop position i from attending to any position j > i.
    # torch.triu(..., diagonal=1) is an upper-triangular matrix of ones
    # ABOVE the main diagonal -> exactly the "future" positions.
    # Shape -> [1, 1, tgt_len, tgt_len] so it broadcasts over batch+heads.
    causal_mask = torch.triu(
        torch.ones(tgt_len, tgt_len, device=device, dtype=torch.bool),
        diagonal=1,
    ).unsqueeze(0).unsqueeze(1)

    # A position is masked if it is EITHER padding OR in the future.
    # Broadcasting: [batch,1,1,tgt_len] | [1,1,tgt_len,tgt_len]
    #            -> [batch,1,tgt_len,tgt_len]
    return pad_mask | causal_mask


# ══════════════════════════════════════════════════════════════════════
# ❸  MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need", §3.2.2.

        MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
        head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)

    Instead of literally building h separate small projections, we use ONE
    big linear layer of size d_model and then *reshape* it into h heads.
    This is the standard, efficient implementation and is mathematically
    identical to h independent projections.

    You are NOT allowed to use torch.nn.MultiheadAttention.

    Args:
        d_model   (int)  : Total model dimensionality. Must be divisible by num_heads.
        num_heads (int)  : Number of parallel attention heads h.
        dropout   (float): Dropout probability applied to attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads   # depth per head

        # One linear layer each for Q, K, V projections (W_Q, W_K, W_V).
        # Each maps d_model -> d_model; the d_model output is later split
        # into num_heads chunks of size d_k.
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        # Final output projection W_O after concatenating the heads.
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

        # Stored after every forward pass so the report code can pull out
        # attention heat-maps (Section 2.3 of the assignment).
        self.attn_weights: Optional[torch.Tensor] = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        [batch, seq, d_model]  ->  [batch, num_heads, seq, d_k]

        We view the last dim as (num_heads, d_k) and then move the head
        axis next to the batch axis so each head is processed in parallel.
        """
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 2)          # -> [batch, num_heads, seq, d_k]

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        [batch, num_heads, seq, d_k]  ->  [batch, seq, d_model]

        Inverse of _split_heads: glue the heads back into one vector.
        """
        batch_size, num_heads, seq_len, d_k = x.shape
        x = x.transpose(1, 2).contiguous()              # [batch, seq, heads, d_k]
        return x.view(batch_size, seq_len, num_heads * d_k)

    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : shape [batch, seq_q, d_model]
            key   : shape [batch, seq_k, d_model]
            value : shape [batch, seq_k, d_model]
            mask  : Optional BoolTensor broadcastable to
                    [batch, num_heads, seq_q, seq_k]
                    True → masked out (attend nowhere)

        Returns:
            output : shape [batch, seq_q, d_model]
        """
        # 1) Linear projections: produce Q, K, V in d_model space.
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # 2) Split each into num_heads parallel heads of depth d_k.
        Q = self._split_heads(Q)          # [batch, heads, seq_q, d_k]
        K = self._split_heads(K)          # [batch, heads, seq_k, d_k]
        V = self._split_heads(V)          # [batch, heads, seq_k, d_k]

        # 3) Scaled dot-product attention, applied to all heads at once.
        #    The mask has shape [batch,1,*,*] and broadcasts over heads.
        attn_out, attn_w = scaled_dot_product_attention(Q, K, V, mask)

        # Dropout on the attention output (regularization).
        attn_out = self.dropout(attn_out)

        # Cache the per-head weights for visualization in the W&B report.
        self.attn_weights = attn_w.detach()

        # 4) Merge heads back together and apply the output projection W_O.
        merged = self._merge_heads(attn_out)            # [batch, seq_q, d_model]
        output = self.w_o(merged)
        return output


# ══════════════════════════════════════════════════════════════════════
# ❹  POSITIONAL ENCODING
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.

        PE(pos, 2i)   = sin( pos / 10000^(2i/d_model) )
        PE(pos, 2i+1) = cos( pos / 10000^(2i/d_model) )

    Self-attention is permutation-invariant (it has no built-in notion of
    word order), so we *add* a fixed position-dependent signal to the token
    embeddings. The sinusoids of geometrically-spaced frequencies let the
    model attend by relative offsets and, in principle, extrapolate to
    sequences longer than those seen in training.

    Args:
        d_model  (int)  : Embedding dimensionality.
        dropout  (float): Dropout applied after adding encodings.
        max_len  (int)  : Maximum sequence length to pre-compute (default 5000).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # pe will hold the [max_len, d_model] table of position signals.
        pe = torch.zeros(max_len, d_model)

        # position : column vector [max_len, 1]  ->  pos = 0,1,2,...
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # div_term implements 1 / 10000^(2i/d_model) for i = 0,1,2,...
        # We compute it in log-space for numerical stability:
        #   10000^(2i/d_model) = exp( 2i/d_model * ln(10000) )
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )

        # Even dimensions get sin, odd dimensions get cos.
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dim -> [1, max_len, d_model] so it broadcasts.
        pe = pe.unsqueeze(0)

        # register_buffer: pe moves with .to(device) and is saved in the
        # state_dict, but is NOT a trainable parameter (no gradient).
        # The autograder explicitly checks that PE is a buffer.
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input embeddings, shape [batch, seq_len, d_model]

        Returns:
            Tensor of same shape [batch, seq_len, d_model]
            = x  +  PE[:, :seq_len, :]
        """
        seq_len = x.size(1)
        # Slice the pre-computed table to the current sequence length and add.
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
# ❹b  LEARNED POSITIONAL EMBEDDING  (for the report ablation, Section 2.4)
# ══════════════════════════════════════════════════════════════════════

class LearnedPositionalEncoding(nn.Module):
    """
    Drop-in replacement for PositionalEncoding that uses a *trainable*
    nn.Embedding indexed by position instead of fixed sinusoids.

    Used ONLY for the W&B ablation in Section 2.4. The autograder tests
    the sinusoidal PositionalEncoding above, not this class.
    """

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
# ❺  FEED-FORWARD NETWORK
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network, §3.3:

        FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

    "Position-wise" means the SAME two-layer MLP is applied independently
    to every position in the sequence. It widens the representation to
    d_ff, applies a ReLU non-linearity, then projects back to d_model.

    Args:
        d_model (int)  : Input / output dimensionality (e.g. 512).
        d_ff    (int)  : Inner-layer dimensionality (e.g. 2048).
        dropout (float): Dropout applied between the two linears.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)     # widen   d_model -> d_ff
        self.linear2 = nn.Linear(d_ff, d_model)     # project d_ff   -> d_model
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : shape [batch, seq_len, d_model]
        Returns:
              shape [batch, seq_len, d_model]
        """
        # ReLU between the two linear layers; dropout for regularization.
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ══════════════════════════════════════════════════════════════════════
# ❻  SUBLAYER CONNECTION  ("Add & Norm")
# ══════════════════════════════════════════════════════════════════════

class SublayerConnection(nn.Module):
    """
    Residual connection followed by Layer Normalization — the "Add & Norm"
    block from the paper.

    We use **POST-LayerNorm**:        output = LayerNorm( x + Sublayer(x) )

    Justification (asked for in the report, Section 1.2):
      * Post-LN is the *exact* ordering published in Vaswani et al. (2017),
        so it is the faithful reproduction of the base paper.
      * On a small dataset like Multi30k with a 6-layer model, Post-LN
        trains fine PROVIDED the Noam warm-up schedule is used — warm-up
        keeps the early-step gradients small enough to avoid the
        instability that Post-LN is otherwise known for.
      * (Pre-LN — LayerNorm before the sublayer — is more stable without
        warm-up but deviates from the paper; we mention it in the report
        as the alternative.)
    """

    def __init__(self, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)       # assignment requires nn.LayerNorm
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        """
        x        : input tensor
        sublayer : a callable (lambda) that runs the actual sublayer
                   (self-attention / cross-attention / FFN).
        """
        # Post-LN: add the residual first, then normalize.
        return self.norm(x + self.dropout(sublayer(x)))


# ══════════════════════════════════════════════════════════════════════
# ❼  ENCODER LAYER
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder sub-layer:
        x → [Self-Attention → Add & Norm] → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_fwd  = PositionwiseFeedForward(d_model, d_ff, dropout)

        # Two independent "Add & Norm" blocks: one after attention, one after FFN.
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            shape [batch, src_len, d_model]
        """
        # Sub-layer 1: multi-head SELF-attention (query=key=value=x).
        x = self.sublayer1(x, lambda t: self.self_attn(t, t, t, src_mask))
        # Sub-layer 2: position-wise feed-forward network.
        x = self.sublayer2(x, self.feed_fwd)
        return x


# ══════════════════════════════════════════════════════════════════════
# ❽  DECODER LAYER
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder sub-layer:
        x → [Masked Self-Attn → Add & Norm]
          → [Cross-Attn(memory) → Add & Norm]
          → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        # Masked self-attention over the (partially generated) target.
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        # Cross-attention: query=decoder state, key/value=encoder memory.
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_fwd   = PositionwiseFeedForward(d_model, d_ff, dropout)

        # Three "Add & Norm" blocks, one after each sub-layer.
        self.sublayer1 = SublayerConnection(d_model, dropout)
        self.sublayer2 = SublayerConnection(d_model, dropout)
        self.sublayer3 = SublayerConnection(d_model, dropout)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : Encoder output, shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            shape [batch, tgt_len, d_model]
        """
        # 1) Masked self-attention: the causal tgt_mask stops a position
        #    from peeking at future tokens it hasn't generated yet.
        x = self.sublayer1(x, lambda t: self.self_attn(t, t, t, tgt_mask))

        # 2) Cross-attention: queries come from the decoder (x), keys and
        #    values come from the encoder output (memory). src_mask hides
        #    padded source positions.
        x = self.sublayer2(x, lambda t: self.cross_attn(t, memory, memory, src_mask))

        # 3) Position-wise feed-forward network.
        x = self.sublayer3(x, self.feed_fwd)
        return x


# ══════════════════════════════════════════════════════════════════════
# ❾  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

def _clones(module: nn.Module, n: int) -> nn.ModuleList:
    """Produce n *independent* deep copies of a layer (own weights each)."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        self.layers = _clones(layer, N)
        # Final LayerNorm applied once at the end of the whole stack.
        self.norm = nn.LayerNorm(layer.sublayer1.norm.normalized_shape[0])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : shape [batch, src_len, d_model]
            mask : shape [batch, 1, 1, src_len]
        Returns:
            shape [batch, src_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

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
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            shape [batch, tgt_len, d_model]
        """
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ══════════════════════════════════════════════════════════════════════
# ❿  TOKEN-EMBEDDING HELPER
# ══════════════════════════════════════════════════════════════════════

class TokenEmbedding(nn.Module):
    """
    Standard token embedding scaled by sqrt(d_model).

    §3.4 of the paper multiplies the embedding output by sqrt(d_model) so
    that the embedding magnitudes are comparable to the positional-encoding
    magnitudes that get added on top.
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        self.d_model = d_model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embed(tokens) * math.sqrt(self.d_model)


# ══════════════════════════════════════════════════════════════════════
# ⓫  FULL TRANSFORMER
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Args:
        src_vocab_size (int)  : Source vocabulary size.
        tgt_vocab_size (int)  : Target vocabulary size.
        d_model        (int)  : Model dimensionality (default 512).
        N              (int)  : Number of encoder/decoder layers (default 6).
        num_heads      (int)  : Number of attention heads (default 8).
        d_ff           (int)  : FFN inner dimensionality (default 2048).
        dropout        (float): Dropout probability (default 0.1).
        checkpoint_path(str)  : If given, trained weights are downloaded
                                from Google Drive and loaded here.
        load_pretrained(bool) : If True (the DEFAULT), __init__ downloads the
                                checkpoint + vocab from Drive and loads them.
                                Set to False during training on Kaggle so a
                                fresh, randomly-initialised model is built.
        pos_encoding   (str)  : 'sinusoidal' (default) or 'learned' — only
                                used for the report ablation in Section 2.4.

    AUTOGRADER USAGE:
        model = Transformer().to(device)   # no args -> defaults used
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
        load_pretrained: bool = True,
        pos_encoding:   str   = "sinusoidal",
        max_len:        int   = DEFAULT_MAX_LEN,
    ) -> None:
        super().__init__()

        # Remember the config so save_checkpoint() can reconstruct us later.
        self.config = dict(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=d_model,
            N=N,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            pos_encoding=pos_encoding,
            max_len=max_len,
        )
        self.d_model = d_model
        self.max_len = max_len

        # ── Embeddings ────────────────────────────────────────────────
        self.src_embed = TokenEmbedding(src_vocab_size, d_model)
        self.tgt_embed = TokenEmbedding(tgt_vocab_size, d_model)

        # ── Positional encoding ───────────────────────────────────────
        if pos_encoding == "learned":
            self.pos_encoder = LearnedPositionalEncoding(d_model, dropout, max_len=5000)
            self.pos_decoder = LearnedPositionalEncoding(d_model, dropout, max_len=5000)
        else:
            self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=5000)
            self.pos_decoder = PositionalEncoding(d_model, dropout, max_len=5000)

        # ── Encoder / Decoder stacks ──────────────────────────────────
        self.encoder = Encoder(EncoderLayer(d_model, num_heads, d_ff, dropout), N)
        self.decoder = Decoder(DecoderLayer(d_model, num_heads, d_ff, dropout), N)

        # ── Final projection to target-vocab logits ───────────────────
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        # ── Parameter initialisation ──────────────────────────────────
        # Xavier/Glorot uniform on every >1-D weight, as commonly done for
        # Transformers — keeps activation variance stable across layers.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # ── Vocab / tokenizer placeholders ────────────────────────────
        # infer() needs these. They are populated either by
        # attach_vocab() (during training) or by _load_vocab_from_drive()
        # (at inference time, inside this __init__).
        self.src_vocab = None        # word -> idx  (German)
        self.tgt_vocab = None        # word -> idx  (English)
        self.tgt_itos  = None        # idx  -> word (English)
        self._de_tokenizer = None    # callable: str -> list[str]

        # ── Load trained weights + vocab from Google Drive ────────────
        # This is the key requirement: the autograder builds Transformer()
        # with no arguments, so EVERYTHING needed for inference must be
        # set up right here in __init__.
        if load_pretrained:
            self._download_and_load(checkpoint_path)

    # ──────────────────────────────────────────────────────────────────
    #  WEIGHT / VOCAB LOADING  (all happens inside __init__)
    # ──────────────────────────────────────────────────────────────────

    def _download_and_load(self, checkpoint_path: str) -> None:
        """
        Download the trained .pth checkpoint and the pickled vocab from
        Google Drive (via gdown) and load both into this instance.

        We never upload weights to Gradescope — gdown fetches them at
        construction time, exactly like Assignment 2.
        """
        # 1) Download + load the model weights.
        try:
            if not os.path.exists(checkpoint_path):
                if gdown is None:
                    raise RuntimeError("gdown is not installed.")
                gdown.download(id=GDRIVE_WEIGHTS_ID, output=checkpoint_path, quiet=False)

            ckpt = torch.load(checkpoint_path, map_location="cpu")
            # The checkpoint dict stores weights under 'model_state_dict'.
            state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            self.load_state_dict(state, strict=True)
            print(f"[Transformer] Loaded weights from {checkpoint_path}")
        except Exception as e:                                  # pragma: no cover
            print(f"[Transformer] WARNING: could not load weights ({e}). "
                  f"Model is running with random weights.")

        # 2) Download + load the vocab/tokenizer bundle.
        self._load_vocab_from_drive()

    def _load_vocab_from_drive(self, vocab_path: str = "vocab.pkl") -> None:
        """
        Download the pickled vocab bundle from Drive and attach it.

        The bundle is a dict produced on Kaggle with keys:
            'src_vocab', 'tgt_vocab', 'tgt_itos'
        (see dataset.py -> Multi30kDataset.export_vocab_bundle).
        """
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
        except Exception as e:                                  # pragma: no cover
            print(f"[Transformer] WARNING: could not load vocab ({e}). "
                  f"infer() will not work until attach_vocab() is called.")

        # Build the German tokenizer (spaCy). Done once, here.
        self._init_tokenizer()

    def _init_tokenizer(self) -> None:
        """Load the spaCy German tokenizer used by infer()."""
        try:
            import spacy
            try:
                nlp = spacy.load("de_core_news_sm")
            except Exception:
                # If the full model is unavailable, fall back to a blank
                # German pipeline — it still tokenizes correctly.
                nlp = spacy.blank("de")
            self._de_tokenizer = lambda text: [t.text for t in nlp.tokenizer(text)]
        except Exception as e:                                  # pragma: no cover
            print(f"[Transformer] WARNING: spaCy German tokenizer unavailable "
                  f"({e}); falling back to whitespace tokenization.")
            self._de_tokenizer = lambda text: text.strip().split()

    def attach_vocab(self, src_vocab, tgt_vocab, tgt_itos, de_tokenizer=None) -> None:
        """
        Manually attach vocab + tokenizer.

        Used during TRAINING on Kaggle (where load_pretrained=False), so
        that infer() can be smoke-tested before the checkpoint exists.
        """
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.tgt_itos  = tgt_itos
        if de_tokenizer is not None:
            self._de_tokenizer = de_tokenizer
        elif self._de_tokenizer is None:
            self._init_tokenizer()

    # ──────────────────────────────────────────────────────────────────
    #  AUTOGRADER HOOKS — keep these signatures exactly
    # ──────────────────────────────────────────────────────────────────

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Run the full encoder stack.

        Args:
            src      : Token indices, shape [batch, src_len]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            memory : Encoder output, shape [batch, src_len, d_model]
        """
        # embed -> add positional encoding -> N encoder layers.
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
        Run the full decoder stack and project to vocabulary logits.

        Args:
            memory   : Encoder output,  shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt      : Token indices,   shape [batch, tgt_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        x = self.pos_decoder(self.tgt_embed(tgt))
        dec_out = self.decoder(x, memory, src_mask, tgt_mask)
        # Project the decoder output to vocab-sized logits.
        return self.generator(dec_out)

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full encoder-decoder forward pass.

        Args:
            src      : shape [batch, src_len]
            tgt      : shape [batch, tgt_len]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask)

    # ──────────────────────────────────────────────────────────────────
    #  END-TO-END INFERENCE  (German string -> English string)
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def infer(self, src_sentence: str) -> str:
        """
        Translate a German sentence to English using greedy autoregressive
        decoding.

        Pipeline (all self-contained, as required):
            raw German text
              -> spaCy tokenization
              -> map words to source indices (+ <sos>/<eos>)
              -> encoder forward pass
              -> token-by-token greedy decoding through the decoder
              -> map output indices back to English words
              -> detokenize into a clean string

        Args:
            src_sentence : The raw German text.

        Returns:
            The fully translated English string, detokenized and clean.
        """
        # Guard: vocab must be available (it is, if load_pretrained=True).
        if self.src_vocab is None or self.tgt_vocab is None:
            raise RuntimeError(
                "infer() needs vocab. Build Transformer() with "
                "load_pretrained=True, or call attach_vocab(...)."
            )

        self.eval()
        device = next(self.parameters()).device

        # ── 1) Tokenize the German input ──────────────────────────────
        tokens = self._de_tokenizer(src_sentence.lower().strip())

        # ── 2) Words -> indices, wrapped with <sos> ... <eos> ─────────
        src_ids = (
            [SOS_IDX]
            + [self.src_vocab.get(tok, UNK_IDX) for tok in tokens]
            + [EOS_IDX]
        )
        src = torch.tensor([src_ids], dtype=torch.long, device=device)  # [1, src_len]

        # ── 3) Encode the source once ─────────────────────────────────
        src_mask = make_src_mask(src, PAD_IDX)
        memory   = self.encode(src, src_mask)

        # ── 4) Greedy autoregressive decoding ─────────────────────────
        # Start with just <sos>; append the best token each step.
        ys = torch.tensor([[SOS_IDX]], dtype=torch.long, device=device)
        for _ in range(self.max_len - 1):
            tgt_mask = make_tgt_mask(ys, PAD_IDX)
            logits   = self.decode(memory, src_mask, ys, tgt_mask)  # [1, cur, vocab]
            # Take the logits of the LAST position and pick the argmax token.
            next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True) # [1, 1]
            ys = torch.cat([ys, next_tok], dim=1)
            if next_tok.item() == EOS_IDX:
                break

        # ── 5) Indices -> English words, drop special tokens ─────────
        out_ids = ys.squeeze(0).tolist()
        words = []
        for idx in out_ids:
            if idx in (SOS_IDX, PAD_IDX):
                continue
            if idx == EOS_IDX:
                break
            words.append(self.tgt_itos[idx] if idx < len(self.tgt_itos) else "<unk>")

        # ── 6) Detokenize into a clean sentence ───────────────────────
        return " ".join(words).strip()


# ══════════════════════════════════════════════════════════════════════
#  QUICK SHAPE SANITY CHECK  (run:  python model.py)
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Build a TINY random model (load_pretrained=False -> no Drive download).
    torch.manual_seed(0)
    m = Transformer(
        src_vocab_size=100, tgt_vocab_size=120,
        d_model=32, N=2, num_heads=4, d_ff=64, dropout=0.1,
        load_pretrained=False,
    )
    src = torch.randint(4, 100, (2, 9))
    tgt = torch.randint(4, 120, (2, 7))
    s_mask = make_src_mask(src)
    t_mask = make_tgt_mask(tgt)
    out = m(src, tgt, s_mask, t_mask)
    print("forward output shape :", tuple(out.shape), "(expect (2, 7, 120))")

    q = torch.randn(2, 4, 5, 8)
    o, w = scaled_dot_product_attention(q, q, q)
    print("attn weights sum     :", w.sum(-1).mean().item(), "(expect ~1.0)")
