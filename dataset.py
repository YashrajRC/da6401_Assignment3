"""
dataset.py — Multi30k loading, tokenization, vocab building
DA6401 Assignment 3: "Attention Is All You Need"

What this file does
-------------------
The Transformer works on integer token-indices, not raw text. This file is
the bridge between raw German/English sentences and those integers:

    raw sentence text
        -> spaCy tokenization        (split into words)
        -> vocabulary lookup         (word -> integer index)
        -> tensor of indices         (what the model actually consumes)

It exposes:
  * Multi30kDataset  : loads one split, tokenizes, builds/uses a vocab.
  * Vocab            : a tiny word<->index mapping object.
  * collate_batch    : pads a list of variable-length samples into a
                       rectangular [batch, seq_len] tensor for a DataLoader.

Special tokens (indices MUST match model.py):
    <unk> = 0   unknown / out-of-vocabulary word
    <pad> = 1   padding to make all sequences in a batch equal length
    <sos> = 2   start-of-sentence marker
    <eos> = 3   end-of-sentence marker
"""

import pickle
from collections import Counter
from typing import List, Dict

import torch
from torch.utils.data import Dataset

# Special tokens — keep IDs identical to model.py (PAD_IDX, UNK_IDX, ...).
UNK, PAD, SOS, EOS = "<unk>", "<pad>", "<sos>", "<eos>"
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIALS = [UNK, PAD, SOS, EOS]   # order fixes the indices above


# ══════════════════════════════════════════════════════════════════════
#  VOCABULARY OBJECT
# ══════════════════════════════════════════════════════════════════════

class Vocab:
    """
    A minimal vocabulary: two dictionaries that convert between words and
    integer indices.

        stoi : "string to index"   word -> int
        itos : "index to string"   int  -> word   (a list, index == position)

    We deliberately keep this as a plain picklable object so the whole
    vocab can be saved to disk and shipped to the autograder via gdown.
    """

    def __init__(self, stoi: Dict[str, int], itos: List[str]) -> None:
        self.stoi = stoi
        self.itos = itos

    def __len__(self) -> int:
        return len(self.itos)

    def __getitem__(self, token: str) -> int:
        """word -> index, returning <unk> for any unknown word."""
        return self.stoi.get(token, UNK_IDX)

    # The assignment template mentions these accessors — provide both.
    def lookup_token(self, idx: int) -> str:
        """index -> word."""
        return self.itos[idx] if 0 <= idx < len(self.itos) else UNK

    def lookup_indices(self, tokens: List[str]) -> List[int]:
        """list of words -> list of indices."""
        return [self[t] for t in tokens]


def build_vocab_from_counter(counter: Counter, min_freq: int = 2) -> Vocab:
    """
    Turn a word-frequency Counter into a Vocab.

    Args:
        counter  : Counter mapping word -> number of occurrences.
        min_freq : Words appearing fewer than this many times are dropped
                   (they will be treated as <unk>). This keeps the vocab
                   small and prevents the model from memorizing rare noise.
    """
    # Specials always occupy indices 0..3.
    itos = list(SPECIALS)
    # Sort by frequency (desc) then alphabetically for reproducibility.
    for word, freq in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
        if freq >= min_freq and word not in SPECIALS:
            itos.append(word)
    stoi = {w: i for i, w in enumerate(itos)}
    return Vocab(stoi, itos)


# ══════════════════════════════════════════════════════════════════════
#  TOKENIZERS  (spaCy — required by the assignment)
# ══════════════════════════════════════════════════════════════════════

def get_tokenizers():
    """
    Return (german_tokenizer, english_tokenizer), each a function
    str -> list[str].

    We use spaCy as the assignment requires all pre-processing to be done
    with spaCy. If the full language models are not installed we fall back
    to spaCy's blank pipelines (still a proper rule-based tokenizer).
    """
    import spacy

    def _load(full_name: str, lang_code: str):
        try:
            return spacy.load(full_name)
        except Exception:
            # blank() gives a tokenizer-only pipeline — fine for our needs.
            return spacy.blank(lang_code)

    nlp_de = _load("de_core_news_sm", "de")
    nlp_en = _load("en_core_web_sm", "en")

    de_tok = lambda text: [t.text for t in nlp_de.tokenizer(text.lower().strip())]
    en_tok = lambda text: [t.text for t in nlp_en.tokenizer(text.lower().strip())]
    return de_tok, en_tok


# ══════════════════════════════════════════════════════════════════════
#  MULTI30K DATASET
# ══════════════════════════════════════════════════════════════════════

class Multi30kDataset(Dataset):
    """
    Loads ONE split (train / validation / test) of the Multi30k de->en
    dataset and converts every sentence pair into index tensors.

    Usage pattern (see train.py / the Kaggle notebook):

        train_ds = Multi30kDataset(split='train')          # builds vocab
        val_ds   = Multi30kDataset(split='validation',
                                   src_vocab=train_ds.src_vocab,
                                   tgt_vocab=train_ds.tgt_vocab)  # REUSES vocab

    IMPORTANT: the vocabulary is built ONLY from the training split and is
    then *reused* for validation and test. Building it from val/test would
    leak information about the held-out data — exactly the kind of data
    leakage the assignment forbids.
    """

    HF_DATASET = "bentrevett/multi30k"   # HuggingFace dataset id

    def __init__(
        self,
        split: str = "train",
        src_vocab: Vocab = None,
        tgt_vocab: Vocab = None,
        de_tokenizer=None,
        en_tokenizer=None,
        min_freq: int = 2,
    ) -> None:
        self.split = split

        # ── 1) Tokenizers ─────────────────────────────────────────────
        if de_tokenizer is None or en_tokenizer is None:
            de_tokenizer, en_tokenizer = get_tokenizers()
        self.de_tokenizer = de_tokenizer
        self.en_tokenizer = en_tokenizer

        # ── 2) Load raw text pairs from HuggingFace ───────────────────
        from datasets import load_dataset
        # HF uses 'validation' as the split name (not 'val').
        hf_split = {"val": "validation"}.get(split, split)
        raw = load_dataset(self.HF_DATASET, split=hf_split)

        # Each row is {'de': '...', 'en': '...'}. Tokenize once, up-front.
        self.src_tokens = [self.de_tokenizer(row["de"]) for row in raw]
        self.tgt_tokens = [self.en_tokenizer(row["en"]) for row in raw]

        # ── 3) Vocabulary ─────────────────────────────────────────────
        if src_vocab is None or tgt_vocab is None:
            # No vocab passed in -> this must be the TRAIN split: build it.
            assert split in ("train",), \
                "Vocab must be built from the train split, then reused."
            self.src_vocab = self._build_vocab(self.src_tokens, min_freq)
            self.tgt_vocab = self._build_vocab(self.tgt_tokens, min_freq)
        else:
            # Reuse the vocab from the train split (no leakage).
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab

        # ── 4) Convert tokens -> index lists (the 'process_data' step) ─
        self.src_data, self.tgt_data = self._process_data()

    # ------------------------------------------------------------------
    @staticmethod
    def _build_vocab(token_lists: List[List[str]], min_freq: int) -> Vocab:
        """Count word frequencies across all sentences and build a Vocab."""
        counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)
        return build_vocab_from_counter(counter, min_freq=min_freq)

    # Public alias matching the original template method name.
    def build_vocab(self) -> None:
        """(kept for template-compatibility — vocab is built in __init__)."""
        self.src_vocab = self._build_vocab(self.src_tokens, min_freq=2)
        self.tgt_vocab = self._build_vocab(self.tgt_tokens, min_freq=2)

    # ------------------------------------------------------------------
    def _process_data(self):
        """
        Convert every tokenized sentence into a list of integer indices,
        wrapped with <sos> at the start and <eos> at the end.

        <sos>/<eos> tell the decoder where a sentence begins and ends —
        the model learns to emit <eos> when the translation is complete.
        """
        src_data, tgt_data = [], []
        for s_tok, t_tok in zip(self.src_tokens, self.tgt_tokens):
            s_ids = [SOS_IDX] + self.src_vocab.lookup_indices(s_tok) + [EOS_IDX]
            t_ids = [SOS_IDX] + self.tgt_vocab.lookup_indices(t_tok) + [EOS_IDX]
            src_data.append(torch.tensor(s_ids, dtype=torch.long))
            tgt_data.append(torch.tensor(t_ids, dtype=torch.long))
        return src_data, tgt_data

    # Public alias matching the original template method name.
    def process_data(self):
        """(kept for template-compatibility)."""
        return self._process_data()

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.src_data)

    def __getitem__(self, idx: int):
        """Return one (src_indices, tgt_indices) pair."""
        return self.src_data[idx], self.tgt_data[idx]

    # ------------------------------------------------------------------
    def export_vocab_bundle(self, path: str = "vocab.pkl") -> None:
        """
        Pickle the vocab so model.Transformer.infer() can use it later.

        The bundle layout matches what Transformer._load_vocab_from_drive
        expects:
            'src_vocab' : dict  word -> idx   (German)
            'tgt_vocab' : dict  word -> idx   (English)
            'tgt_itos'  : list  idx  -> word  (English)
        """
        bundle = {
            "src_vocab": self.src_vocab.stoi,
            "tgt_vocab": self.tgt_vocab.stoi,
            "tgt_itos":  self.tgt_vocab.itos,
        }
        with open(path, "wb") as f:
            pickle.dump(bundle, f)
        print(f"[dataset] vocab bundle written to {path} "
              f"(src={len(self.src_vocab)}, tgt={len(self.tgt_vocab)})")


# ══════════════════════════════════════════════════════════════════════
#  COLLATE FUNCTION  (pads a batch of variable-length sequences)
# ══════════════════════════════════════════════════════════════════════

def collate_batch(batch):
    """
    Turn a list of (src, tgt) index-tensors of DIFFERENT lengths into two
    rectangular padded tensors that a Transformer can consume.

    A DataLoader passes this as `collate_fn`.

    Args:
        batch : list of (src_tensor, tgt_tensor) pairs.

    Returns:
        src_padded : LongTensor [batch, max_src_len]
        tgt_padded : LongTensor [batch, max_tgt_len]
        (padding positions are filled with PAD_IDX)
    """
    from torch.nn.utils.rnn import pad_sequence

    src_list, tgt_list = zip(*batch)
    # pad_sequence stacks along dim 0 and pads the rest with padding_value.
    src_padded = pad_sequence(list(src_list), batch_first=True, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(list(tgt_list), batch_first=True, padding_value=PAD_IDX)
    return src_padded, tgt_padded


# ══════════════════════════════════════════════════════════════════════
#  QUICK SELF-TEST  (run:  python dataset.py)
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train = Multi30kDataset(split="train")
    print("train pairs        :", len(train))
    print("German vocab size  :", len(train.src_vocab))
    print("English vocab size :", len(train.tgt_vocab))
    s, t = train[0]
    print("sample src indices :", s.tolist()[:12], "...")
    print("sample tgt indices :", t.tolist()[:12], "...")
