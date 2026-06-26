"""Overfit-one-batch test (CLAUDE.md gate).

A correct seq2seq model with teacher forcing should be able to *memorize* a
single fixed batch — drive the loss to ~0. If it can't, there's a bug (most
often in the logits/target alignment for the loss). This must pass before any
full training run.
"""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn

from constants import PAD_ID, SOS_ID, EOS_ID
from model import Encoder, Decoder, Seq2Seq


def seed_everything(seed: int = 0) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def make_batch(
    batch: int, seq_len: int, vocab: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """One fixed (src, tgt) batch. Body tokens avoid special ids; tgt is framed
    with <sos> at position 0 and <eos> at the end so it matches collate output."""
    src = torch.randint(4, vocab, (batch, seq_len))
    tgt = torch.randint(4, vocab, (batch, seq_len))
    tgt[:, 0] = SOS_ID
    tgt[:, -1] = EOS_ID
    return src, tgt


def test_overfit_one_batch() -> None:
    seed_everything(0)

    BATCH, SEQ_LEN, VOCAB = 8, 6, 30
    HIDDEN, EMBED, LAYERS = 64, 32, 2

    src, tgt = make_batch(BATCH, SEQ_LEN, VOCAB)

    encoder = Encoder(VOCAB, EMBED, HIDDEN, LAYERS, dropout=0.0)
    decoder = Decoder(VOCAB, EMBED, HIDDEN, LAYERS, dropout=0.0)
    model = Seq2Seq(encoder, decoder)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    loss = torch.tensor(float("inf"))
    for _ in range(600):
        optimizer.zero_grad() #clear gradients from previous step
        outputs = model(src, tgt, teacher_forcing_ratio=1.0)  # (BATCH, SEQ_LEN, VOCAB)
        predictions = outputs[:, 1:, :].reshape(-1, outputs.size())  # drop <sos> and flatten into (BATCH * (SEQ_LEN-1), VOCAB)
        targets = tgt[:, 1:].reshape(-1)  # drop <sos> and flatten into (BATCH * (SEQ_LEN-1),)
        loss = criterion(predictions, targets)  # compute the loss
        loss.backward() # compute gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step() # apply gradients to update model parameters

    assert loss.item() < 0.01, f"failed to overfit: final loss {loss.item():.4f}"
