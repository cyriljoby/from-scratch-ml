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


def train_one_epoch(model, loader, optimizer, criterion, device, teacher_forcing_ratio=1.0) -> float:
    total_loss = 0.0
    model.train()
    for src, tgt in loader:
        optimizer.zero_grad()
        src, tgt = src.to(device), tgt.to(device)
        outputs = model(src, tgt, teacher_forcing_ratio=teacher_forcing_ratio)
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


@torch.no_grad()
def greedy_decode(model, src, device, max_len: int = 50) -> torch.Tensor:
    """Free-running generation (no teacher forcing): feed <sos>, take the argmax
    token, feed it back, until <eos> or max_len. Returns (batch, gen_len) ids."""
    model.eval()
    batch = src.size(0)
    hidden, cell = model.encoder(src.to(device))
    input_token = torch.full((batch,), SOS_ID, dtype=torch.long, device=device)
    finished = torch.zeros(batch, dtype=torch.bool, device=device)
    generated = []
    for _ in range(max_len):
        logits, hidden, cell = model.decoder(input_token, hidden, cell)
        input_token = logits.argmax(1)
        generated.append(input_token)
        finished |= input_token == EOS_ID
        if finished.all():
            break
    return torch.stack(generated, dim=1)


def ids_to_text(ids, tgt_vocab) -> str:
    """Predicted ids -> detokenized string: drop specials, stop at first <eos>."""
    words = []
    for i in ids:
        if i == EOS_ID:
            break
        if i in (PAD_ID, SOS_ID):
            continue
        words.append(tgt_vocab.itos[i])
    return " ".join(words)


@torch.no_grad()
def compute_bleu(model, test_tokens, src_vocab, tgt_vocab, device, reverse: bool,
                 batch_size: int = 32, max_len: int = 50) -> float:
    """Corpus BLEU on the test set. Hypotheses are greedy-decoded; references are
    the true target tokens (NOT round-tripped through the vocab, so no <unk>).
    `reverse` must match how the model was trained."""
    model.eval()
    refs = [" ".join(tgt_tokens) for _, tgt_tokens in test_tokens]
    hyps = []
    for i in range(0, len(test_tokens), batch_size):
        chunk = test_tokens[i:i + batch_size]
        encoded = [(src_vocab.encode(s), tgt_vocab.encode(t)) for s, t in chunk]
        src, _ = collate(encoded, reverse=reverse)  # encode + pad + (maybe) reverse source
        pred = greedy_decode(model, src, device, max_len)
        hyps.extend(ids_to_text(row, tgt_vocab) for row in pred.tolist())
    # force=True: hyps/refs are already tokenized by our tokenizer, so we
    # suppress sacreBLEU's "detokenize?" warning (effect is <1 BLEU here).
    return sacrebleu.corpus_bleu(hyps, [refs], force=True).score


def main() -> None:
    # Experiment config
    REVERSE = True   # reverse the source sentence (paper's trick). A/B: True vs False
    TRAIN_TF = 0.0  # teacher-forcing ratio during training. ablate: 1.0 / 0.5 / 0.0
    # per-config checkpoint so runs never overwrite / BLEU each other's models
    CKPT = f"best_r{int(REVERSE)}_tf{TRAIN_TF}.pth"

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
        batch_size=64,
        shuffle=True,
        collate_fn=lambda b: collate(b, reverse=REVERSE),
    )
    val_loader = DataLoader(
        Seq2SeqDataset(val_tokens, src_vocab, tgt_vocab),
        batch_size=64,
        shuffle=False,
        collate_fn=lambda b: collate(b, reverse=REVERSE),
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
    MAX_EPOCHS = 40
    PATIENCE = 5
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params:,} params | device: {device} | {len(train_loader)} train batches/epoch")
    print(f"config: reverse={REVERSE}  train_tf={TRAIN_TF}")
    print(f"training up to {MAX_EPOCHS} epochs (patience {PATIENCE})...")
    for epoch in range(MAX_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, TRAIN_TF)
        val_loss = evaluate(model, val_loader, criterion, device)
        best = val_loss < best_val_loss
        print(f"epoch {epoch+1:2d} | train {train_loss:.4f} | val {val_loss:.4f}{'  *' if best else ''}")
        if best:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), CKPT)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print("Early stopping triggered.")
                break

    # test BLEU on the best checkpoint (greedy, free-running)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    test_pairs = read_parallel(data / "mmt16_task1_test" / "test.de", data / "mmt16_task1_test" / "test.en")
    test_tokens = tokenize(test_pairs)
    bleu = compute_bleu(model, test_tokens, src_vocab, tgt_vocab, device, reverse=REVERSE)
    print(f"best val loss {best_val_loss:.4f} | test BLEU {bleu:.2f}  (reverse={REVERSE}, train_tf={TRAIN_TF})")


if __name__ == "__main__":
    main()
