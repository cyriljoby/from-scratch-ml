"""Data pipeline for Multi30k German→English translation.

Copied from 02-seq2seq (same task, same pipeline) with the source-reversal
trick removed — the Sutskever reversal is unnecessary here because the
bidirectional encoder + attention lets the decoder look at any source position.

Pipeline order (each stage's output feeds the next):
    raw .de/.en files
    - read_parallel() files -> (src_str, tgt_str) pairs
    - tokenize()      strings -> (src_tokens, tgt_tokens) pairs
    - Vocab.build()   tokens -> stoi/itos mappings
    - Seq2SeqDataset  tokens + vocab -> (src_ids, tgt_ids) pairs
    - collate()       id lists -> rectangular src/tgt tensors
"""

from __future__ import annotations

from pathlib import Path
import re
import json
from torch import Tensor
import torch
from torch.utils.data import Dataset

from constants import PAD_ID, SOS_ID, EOS_ID, UNK_ID
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
        self.stoi = {"<pad>": PAD_ID, "<sos>": SOS_ID, "<eos>": EOS_ID, "<unk>": UNK_ID}  # start with special tokens
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

# Implemented as a subclass of torch.utils.data.Dataset, which is the base class for all PyTorch datasets.
# It requires implementing __len__ and __getitem__ methods. This allows PyTorch's DataLoader to efficiently load and batch data during training.
class Seq2SeqDataset(Dataset):
    def __init__(self, pairs: list[tuple[list[str], list[str]]], src_vocab: Vocab, tgt_vocab: Vocab) -> None:
        """Create a dataset of (source_ids, target_ids) pairs from tokenized sentences."""
        self.data = [
            (src_vocab.encode(src_tokens), tgt_vocab.encode(tgt_tokens))
            for src_tokens, tgt_tokens in pairs
        ]
    def __getitem__(self, i: int) -> tuple[list[int], list[int]]:
        return self.data[i]
    def __len__(self) -> int:
        return len(self.data)
    
def collate(batch: list[tuple[list[int], list[int]]]) -> tuple[Tensor, Tensor]:
    """ Returns two tensors, one for src and other for tgt, each padded so they are rectangular."""
    # Unzip the batch into separate lists
    src_batch, tgt_batch = zip(*batch)
    def shape(seqs: list[list[int]]) -> Tensor:
        out = []
        for sentence in seqs:
            out.append([SOS_ID] + sentence + [EOS_ID])  # Add <sos> and <eos> tokens
        # find the length of the longest sentence
        max_len = max(len(s) for s in out)
        return torch.tensor([s + [PAD_ID] * (max_len - len(s)) for s in out], dtype=torch.long)  # Pad every sentence up to that length to form a rectangular tensor
    src = shape(src_batch)
    tgt = shape(tgt_batch)
    return src, tgt