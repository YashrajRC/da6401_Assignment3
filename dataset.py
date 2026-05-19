"""
dataset.py — Multi30k loading, tokenization, vocab building
DA6401 Assignment 3: "Attention Is All You Need"

Special tokens (indices MUST match model.py):
    <unk> = 0   unknown / out-of-vocabulary word
    <pad> = 1   padding to make all sequences in a batch equal length
    <sos> = 2   start-of-sentence marker
    <eos> = 3   end-of-sentence marker
"""

import pickle
from collections import Counter
from typing import List, Dict, Tuple

import torch
from torch.utils.data import Dataset

# Special tokens — keep IDs identical to model.py
UNK, PAD, SOS, EOS = "<unk>", "<pad>", "<sos>", "<eos>"
UNK_IDX, PAD_IDX, SOS_IDX, EOS_IDX = 0, 1, 2, 3
SPECIALS = [UNK, PAD, SOS, EOS]


# ══════════════════════════════════════════════════════════════════════
#  VOCABULARY
# ══════════════════════════════════════════════════════════════════════

class Vocab:
    """
    Minimal vocabulary: stoi (word -> idx) and itos (idx -> word).
    Plain picklable object for shipping to autograder via gdown.
    """

    def __init__(self, stoi: Dict[str, int], itos: List[str]) -> None:
        self.stoi = stoi
        self.itos = itos

    def __len__(self) -> int:
        return len(self.itos)

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, UNK_IDX)

    def get(self, token: str, default: int = UNK_IDX) -> int:
        return self.stoi.get(token, default)

    def lookup_token(self, idx: int) -> str:
        return self.itos[idx] if 0 <= idx < len(self.itos) else UNK

    def lookup_indices(self, tokens: List[str]) -> List[int]:
        return [self[t] for t in tokens]


def build_vocab_from_counter(counter: Counter, min_freq: int = 2) -> Vocab:
    """Build Vocab from a frequency Counter; specials always at 0–3."""
    itos = list(SPECIALS)
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
    Return (de_tokenizer, en_tokenizer), each: str -> list[str].
    Falls back to blank spaCy pipelines if full models are unavailable.
    """
    import spacy

    def _load(full_name: str, lang_code: str):
        try:
            return spacy.load(full_name)
        except Exception:
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
    Loads one split of Multi30k (de->en) and converts to index tensors.

    Vocab is built ONLY from the train split and reused for val/test
    to avoid data leakage.
    """

    HF_DATASET = "bentrevett/multi30k"

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

        # ── Tokenizers ────────────────────────────────────────────────
        if de_tokenizer is None or en_tokenizer is None:
            de_tokenizer, en_tokenizer = get_tokenizers()
        self.de_tokenizer = de_tokenizer
        self.en_tokenizer = en_tokenizer

        # ── Raw text from HuggingFace ─────────────────────────────────
        from datasets import load_dataset
        hf_split = {"val": "validation"}.get(split, split)
        raw = load_dataset(self.HF_DATASET, split=hf_split)

        self.src_tokens = [self.de_tokenizer(row["de"]) for row in raw]
        self.tgt_tokens = [self.en_tokenizer(row["en"]) for row in raw]

        # ── Vocabulary ────────────────────────────────────────────────
        if src_vocab is None or tgt_vocab is None:
            assert split == "train", \
                "Vocab must be built from the train split and reused elsewhere."
            self.src_vocab = self._build_vocab(self.src_tokens, min_freq)
            self.tgt_vocab = self._build_vocab(self.tgt_tokens, min_freq)
        else:
            self.src_vocab = src_vocab
            self.tgt_vocab = tgt_vocab

        # ── Tokens -> index tensors ───────────────────────────────────
        self.src_data, self.tgt_data = self._process_data()

    @staticmethod
    def _build_vocab(token_lists: List[List[str]], min_freq: int) -> Vocab:
        counter = Counter()
        for tokens in token_lists:
            counter.update(tokens)
        return build_vocab_from_counter(counter, min_freq=min_freq)

    def build_vocab(self) -> None:
        """Template-compat alias."""
        self.src_vocab = self._build_vocab(self.src_tokens, min_freq=2)
        self.tgt_vocab = self._build_vocab(self.tgt_tokens, min_freq=2)

    def _process_data(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Tokenize -> indices, wrap with <sos>…<eos>."""
        src_data, tgt_data = [], []
        for s_tok, t_tok in zip(self.src_tokens, self.tgt_tokens):
            s_ids = [SOS_IDX] + self.src_vocab.lookup_indices(s_tok) + [EOS_IDX]
            t_ids = [SOS_IDX] + self.tgt_vocab.lookup_indices(t_tok) + [EOS_IDX]
            src_data.append(torch.tensor(s_ids, dtype=torch.long))
            tgt_data.append(torch.tensor(t_ids, dtype=torch.long))
        return src_data, tgt_data

    def process_data(self):
        """Template-compat alias."""
        return self._process_data()

    def __len__(self) -> int:
        return len(self.src_data)

    def __getitem__(self, idx: int):
        return self.src_data[idx], self.tgt_data[idx]

    def export_vocab_bundle(self, path: str = "vocab.pkl") -> None:
        """
        Pickle the vocab so Transformer.infer() can load it later.
        Bundle layout expected by Transformer._load_vocab_from_drive:
            src_vocab : dict  word->idx  (German)
            tgt_vocab : dict  word->idx  (English)
            tgt_itos  : list  idx->word  (English)
        """
        bundle = {
            "src_vocab": self.src_vocab.stoi,
            "tgt_vocab": self.tgt_vocab.stoi,
            "tgt_itos":  self.tgt_vocab.itos,
        }
        with open(path, "wb") as f:
            pickle.dump(bundle, f)
        print(f"[dataset] vocab bundle -> {path}  "
              f"(src={len(self.src_vocab)}, tgt={len(self.tgt_vocab)})")


# ══════════════════════════════════════════════════════════════════════
#  COLLATE (pads variable-length sequences into a rectangular batch)
# ══════════════════════════════════════════════════════════════════════

def collate_batch(batch):
    """
    Pad a list of (src, tgt) tensors to equal length.
    Returns:
        src_padded : LongTensor [batch, max_src_len]
        tgt_padded : LongTensor [batch, max_tgt_len]
    """
    from torch.nn.utils.rnn import pad_sequence
    src_list, tgt_list = zip(*batch)
    src_padded = pad_sequence(list(src_list), batch_first=True, padding_value=PAD_IDX)
    tgt_padded = pad_sequence(list(tgt_list), batch_first=True, padding_value=PAD_IDX)
    return src_padded, tgt_padded


# ══════════════════════════════════════════════════════════════════════
#  QUICK SELF-TEST
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    train = Multi30kDataset(split="train")
    print("train pairs        :", len(train))
    print("German vocab size  :", len(train.src_vocab))
    print("English vocab size :", len(train.tgt_vocab))
    s, t = train[0]
    print("sample src :", s.tolist()[:12], "...")
    print("sample tgt :", t.tolist()[:12], "...")