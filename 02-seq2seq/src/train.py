"""Training script for the Multi30k German->English seq2seq model.

Wires the data pipeline (data.py) to the model (model.py): build vocabs, make
DataLoaders, then run a train/validate loop with early stopping.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import sacrebleu
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from constants import PAD_ID, SOS_ID, EOS_ID
from data import read_parallel, tokenize, Vocab, Seq2SeqDataset, collate
from model import Encoder, Decoder, Seq2Seq


def seed_everything(seed: int = 0) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    total_loss = 0.0
    model.train()
    for src, tgt in loader:
        optimizer.zero_grad()
        src, tgt = src.to(device), tgt.to(device)
        outputs = model(src, tgt, teacher_forcing_ratio=1.0)
        predicted = outputs[:, 1:, :].reshape(-1, outputs.size(2))
        tgt = tgt[:, 1:].reshape(-1)
        loss = criterion(predicted, tgt)
        total_loss+=loss.item()
        loss.backward() # computes gradient
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) # clips gradient to avoid exploding gradient problem
        optimizer.step() # gradient descent
    return total_loss/len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> float:
    model.eval()
    total_loss = 0.0
    for src, target in loader:
        src, target = src.to(device), target.to(device)
        # teacher forcing on: per-token loss given correct history — a clean,
        # train-comparable generalization signal for early stopping. Real
        # free-running quality is measured separately by BLEU.
        outputs = model(src, target, teacher_forcing_ratio=1.0)
        predicted = outputs[:, 1:, :].reshape(-1, outputs.size(2))
        target = target[:, 1:].reshape(-1)
        loss = criterion(predicted, target)
        total_loss += loss.item()
    return total_loss / len(loader)


def main() -> None:
    seed_everything()
    data = Path(__file__).resolve().parents[2] / "data"
    train_pairs = read_parallel(data / "training" / "train.de", data / "training" / "train.en")
    val_pairs = read_parallel(data / "validation" / "val.de", data / "validation" / "val.en")
    train_tokens = tokenize(train_pairs)
    val_tokens = tokenize(val_pairs)
    print(f"loaded {len(train_tokens)} train / {len(val_tokens)} val pairs")
    src_sentences, tgt_sentences = zip(*train_tokens)
    src_vocab = Vocab(src_sentences)
    tgt_vocab = Vocab(tgt_sentences)
    print(f"vocab sizes — src(de): {len(src_vocab)}  tgt(en): {len(tgt_vocab)}")
    train_loader = DataLoader(
        Seq2SeqDataset(train_tokens, src_vocab, tgt_vocab),
        batch_size=32,
        shuffle=True,
        collate_fn=lambda b: collate(b, reverse = True),
    )
    val_loader = DataLoader(
        Seq2SeqDataset(val_tokens, src_vocab, tgt_vocab),
        batch_size=32,
        shuffle=False,
        collate_fn=lambda b: collate(b, reverse = True),
    )
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    encoder = Encoder(vocab_size=len(src_vocab), embed_dim=256, hidden_dim=256, num_layers=2, dropout=0.5)
    decoder = Decoder(vocab_size=len(tgt_vocab), embed_dim=256, hidden_dim=256, num_layers=2, dropout=0.5)
    model = Seq2Seq(encoder, decoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    MAX_EPOCHS = 20
    PATIENCE = 5
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params:,} params | device: {device} | {len(train_loader)} train batches/epoch")
    print(f"training up to {MAX_EPOCHS} epochs (patience {PATIENCE})...")
    for epoch in range(MAX_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = evaluate(model, val_loader, criterion, device)
        best = val_loss < best_val_loss
        print(f"epoch {epoch+1:2d} | train {train_loss:.4f} | val {val_loss:.4f}{'  *' if best else ''}")
        if best:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print("Early stopping triggered.")
                break

if __name__ == "__main__":
    main()
