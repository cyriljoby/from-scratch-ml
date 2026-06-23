"""Data pipeline for Multi30k German→English seq2seq.

Pipeline order (each stage's output feeds the next):
    raw .de/.en files
      -> read_parallel   list of (src_str, tgt_str) pairs
      -> tokenize        list of (src_tokens, tgt_tokens)
      -> Vocab.build     stoi / itos
      -> Seq2SeqDataset  (src_ids, tgt_ids) per item
      -> collate_fn      padded (batch, seq_len) tensors
"""

from __future__ import annotations

from pathlib import Path
import re
import json
from sys import path


def read_parallel(src_path: Path, tgt_path: Path) -> list[tuple[str, str]]:
    """Read a line-aligned parallel corpus into (source, target) string pairs.

    The two files are parallel: line i of `src_path` is the translation of
    line i of `tgt_path`. Each returned tuple is one such (source, target) pair,
    with trailing newlines stripped. Tokenization happens later — this stage only
    reads raw text.

    Args:
        src_path: path to the source-language file (e.g. train.de).
        tgt_path: path to the target-language file (e.g. train.en).

    Returns:
        A list of (source_str, target_str) pairs, one per line, in file order.
    """
    # TODO(data):
    with open(src_path, "r", encoding="utf-8") as src_file, open(
        tgt_path, "r", encoding="utf-8"
    ) as tgt_file:
        pairs = []
        for src_line, tgt_line in zip(src_file, tgt_file, strict = True):
            pairs.append((src_line.strip(), tgt_line.strip()))
    return pairs

def tokenize_sentence(sentence: str) -> list[str]:
    # Lowercase the sentence
    sentence = sentence.lower()
    # Use regex to split on whitespace and punctuation
    tokens = re.findall(r"\w+|[^\w\s]", sentence, re.UNICODE)
    return tokens

def tokenize(pairs: list[tuple[str, str]]) -> list[tuple[list[str], list[str]]]:
    """ Tokenize a list of (source_str, target_str) pairs into (source_tokens, target_tokens)."""
    return [(tokenize_sentence(src_str), tokenize_sentence(tgt_str)) for src_str, tgt_str in pairs]

class Vocab:
    def __init__(self, tokens: list[list[str]], min_freq: int = 2) -> None:
        """Build a vocabulary from a list of tokenized sentences.

        Args:
            tokens: A list of tokenized sentences (list of list of strings).
            min_freq: Minimum frequency for a token to be included in the vocabulary.
        """
        # string to index mapping used for encoding tokens to integers
        self.stoi = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}  # start with special tokens
        # index-to-string mapping: the list position IS the token's id, so itos[id] -> token. Used to decode ids back to tokens.
        self.itos = ["<pad>", "<sos>", "<eos>", "<unk>"]  # start with special tokens
        self.build(tokens, min_freq)
    
    
    def build(self, tokens: list[list[str]], min_freq: int) -> None:
        """Build the vocabulary from tokenized sentences."""
        # Count token frequencies
        freq = {}
        for sentence in tokens:
            for token in sentence:
                freq[token] = freq.get(token, 0) + 1
        
        # Add tokens to the vocabulary if they meet the min_freq requirement
        for token, count in freq.items():
            if count >= min_freq:
                self.stoi[token] = len(self.itos)
                self.itos.append(token)
    def encode(self, tokens: list[str]) -> list[int]:
        """Convert a list of tokens to a list of integer IDs using the vocabulary."""
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in tokens]
    
    def decode(self, ids: list[int]) -> list[str]:
        """Convert a list of integer IDs back to a list of tokens using the vocabulary."""
        return [self.itos[id] for id in ids]
    
    def __len__(self) -> int:
        return len(self.itos)  # Return the size of the vocabulary

    def save(self, path: str | Path) -> None:
        """Write the vocab to JSON. Only itos is saved (stoi is its inverse)."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.itos, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        """Reconstruct a Vocab from a saved JSON file; rebuilds stoi from itos.

        Bypasses __init__ (which would re-count tokens) and fills state directly.
        """
        with open(path, "r", encoding="utf-8") as f:
            itos = json.load(f)
        vocab = cls.__new__(cls)  # blank instance, no __init__/build
        vocab.itos = itos
        vocab.stoi = {token: i for i, token in enumerate(itos)}
        return vocab
    