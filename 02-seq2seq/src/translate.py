"""Translate German sentences with a trained seq2seq model.

Usage (from 02-seq2seq/):
    python src/translate.py "ein hund läuft im park"   # one sentence
    python src/translate.py                             # interactive: type sentences, Ctrl-D to quit

Loads the not-reversed model (best_model.pth). Vocabs are rebuilt from the
training data (deterministic under the fixed seed), so no vocab file is needed.
Out-of-vocab German words become <unk>.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

from data import read_parallel, tokenize, tokenize_sentence, Vocab, collate
from model import Encoder, Decoder, Seq2Seq
from train import greedy_decode, ids_to_text, seed_everything

CKPT = "best_model.pth"   # forward (reverse=False) model, BLEU 17.21
REVERSE = False           # must match how CKPT was trained


def build():
    seed_everything()
    data = Path(__file__).resolve().parents[2] / "data"
    train_tokens = tokenize(read_parallel(data / "training" / "train.de", data / "training" / "train.en"))
    src_sents, tgt_sents = zip(*train_tokens)
    src_vocab, tgt_vocab = Vocab(src_sents), Vocab(tgt_sents)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = Seq2Seq(Encoder(len(src_vocab)), Decoder(len(tgt_vocab))).to(device)
    model.load_state_dict(torch.load(Path(__file__).resolve().parents[1] / CKPT, map_location=device))
    model.eval()
    return model, src_vocab, tgt_vocab, device


def translate(sentence: str, model, src_vocab, tgt_vocab, device) -> str:
    ids = src_vocab.encode(tokenize_sentence(sentence))
    src, _ = collate([(ids, ids)], reverse=REVERSE)  # dummy tgt; only src is used
    pred = greedy_decode(model, src, device)
    return ids_to_text(pred[0].tolist(), tgt_vocab)


def main() -> None:
    model, src_vocab, tgt_vocab, device = build()
    args = sys.argv[1:]
    if args:
        print(translate(" ".join(args), model, src_vocab, tgt_vocab, device))
    else:
        print("Type German sentences (Ctrl-D to quit):")
        for line in sys.stdin:
            line = line.strip()
            if line:
                print("  ->", translate(line, model, src_vocab, tgt_vocab, device))


if __name__ == "__main__":
    main()
