"""Attention model: RNNsearch (Bahdanau et al., 2015).

Three changes from 02-seq2seq:
  1. Encoder is bidirectional and returns ALL hidden states (the annotations
     h_j), since each one will be scored against the decoder state at every step.
  2. A hand-written additive Attention module scores each annotation against
     the decoder state to form the context vector c_i = Σ_j α_ij h_j.
  3. The decoder consumes [embedded token; context] at every step.

Deviations from the paper:
LSTM instead of GRU: keeps the same cell as 02-seq2seq, so attention is the only changed variable in the comparison. At this scale differences between GRU and LSTM are relatively small.
Single layer: The paper itself is single layer, unlike Sutskever's seq2seq which was 4 layers.
Plain linear output instead of maxout: maxout was a popular 2013 technique, but the LSTM already has tanh and sigmoid gates that make the output nonlinear, so the additional nonlinearity offers negligible gain at this scale.

"""

from __future__ import annotations

import random

import torch
import torch.nn as nn
from torch import Tensor

from constants import PAD_ID


class Attention(nn.Module):
    """Additive (Bahdanau) attention: e_ij = v · tanh(W_s s_{i-1} + W_h h_j).

    Given the decoder's current state and all encoder annotations, produce a
    context vector (weighted sum of annotations) and the weights themselves
    (kept for alignment heatmaps).
    """

    def __init__(self, hidden_dim: int, annotation_dim: int, attn_dim: int) -> None:
        """
        Args:
            hidden_dim: decoder hidden state size (input to W_s).
            annotation_dim: encoder annotation size = 2 * encoder hidden (input to W_h).
            attn_dim: internal scoring dimension (output of W_s / W_h, input to v).
        """
        super().__init__()
        self.W_s = nn.Linear(hidden_dim, attn_dim, bias=False)
        # annotation_dim is 2 * hidden_dim because the encoder is bidirectional
        self.W_h = nn.Linear(annotation_dim, attn_dim, bias=False)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self, decoder_hidden: Tensor, annotations: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Score every source position against the decoder state.
        """
        #e_ij = v · tanh(W_s s_{i-1} + W_h h_j)
        # one decoder hidden state per batch, broadcast over src_len to score all annotations at once
        e = self.v(torch.tanh(self.W_s(decoder_hidden).unsqueeze(1) + self.W_h(annotations))).squeeze(-1)  # (batch, src_len, 1) -> (batch, src_len)
        e = e.masked_fill(~mask, float("-inf"))  # -inf at padded positions so softmax gives 0 weight there
        weights = torch.softmax(e, dim=1)
        context = (weights.unsqueeze(1) @ annotations).squeeze(1)
        return context, weights

class Encoder(nn.Module):
    """Bidirectional LSTM encoder producing one annotation per source token.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)
        self.fc_cell = nn.Linear(2*hidden_dim, hidden_dim)
        self.fc_hidden = nn.Linear(2*hidden_dim, hidden_dim)

    def forward(self, src: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Encode a batch of source sentences.

        Args:
            src: (batch, src_len) token ids.

        Returns:
            annotations: (batch, src_len, 2 * hidden_dim) — h_j for every
                source position, forward and backward halves concatenated.
            hidden: (1, batch, hidden_dim) — decoder's initial hidden state.
            cell:   (1, batch, hidden_dim) — decoder's initial cell state.
        """
        embedded = self.dropout(self.embedding(src)) # (batch, seq_len) -> (batch, seq_len, embed_dim)
        o, (h, c) = self.rnn(embedded) # (batch, seq_len, 2*hidden_dim), (num_layer * num_directions, batch, hidden_dim), (num_layer * num_directions, batch, hidden_dim)
        # combines forward, backward states into one continuous row
        h = torch.cat([h[0], h[1]], dim = 1) # (2, batch, hidden_dim) -> (batch, 2*hidden_dim)
        c = torch.cat([c[0], c[1]], dim = 1)
        # project the bidirectional summary down to decoder size, then add back the
        # (num_layers,) axis the decoder's LSTM expects on its state
        hidden = self.fc_hidden(h).unsqueeze(0) # (batch, 2*hidden_dim) -> (1, batch, hidden_dim)
        cell = self.fc_cell(c).unsqueeze(0)
        # o is the annotations: every position's hidden state, both directions — what attention scores against
        return o, hidden, cell

class Decoder(nn.Module):
    """One-step LSTM decoder that attends over the annotations each step.

    Differences from 02's decoder:
      - owns an Attention module;
      - the LSTM input is [embedded token; context] (embed_dim + annotation_dim);
      - the output layer sees [hidden; context] (hidden_dim + annotation_dim).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        annotation_dim: int = 512,
        attn_dim: int = 256,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_ID)
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(hidden_dim, annotation_dim, attn_dim)
        # input is [embedded token; context] so the LSTM sees both what came before
        # and what the source says is relevant right now
        self.rnn = nn.LSTM(embed_dim + annotation_dim, hidden_dim, num_layers=1, batch_first=True)
        # logits come from [new hidden state; context] rather than hidden alone
        self.fc_out = nn.Linear(hidden_dim + annotation_dim, vocab_size)

    def forward(
        self,
        input_token: Tensor,
        hidden: Tensor,
        cell: Tensor,
        annotations: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """One decoding step.

        Args:
            input_token: (batch,) previous target token ids.
            hidden: (1, batch, hidden_dim) decoder hidden state s_{i-1}.
            cell:   (1, batch, hidden_dim) decoder cell state.
            annotations: (batch, src_len, annotation_dim) encoder outputs.
            mask: (batch, src_len) bool, True at real source tokens.

        Returns:
            logits: (batch, vocab_size) scores for the next token.
            hidden: (1, batch, hidden_dim) updated state s_i.
            cell:   (1, batch, hidden_dim) updated cell.
            attn_weights: (batch, src_len) — α_ij for this step (for heatmaps).
        """
        # 1. attend BEFORE updating state: s_{i-1} asks "what's relevant now?"
        #    hidden is (1, batch, hidden_dim) but attention wants (batch, hidden_dim)
        context, attn_weights = self.attention(hidden.squeeze(0), annotations, mask) # (batch, annotation_dim), (batch, src_len)
        # 2. one LSTM step over [embedded token; context]
        embedded = self.dropout(self.embedding(input_token.unsqueeze(1))) # (batch,) -> (batch, 1) -> (batch, 1, embed_dim)
        lstm_input = torch.cat([embedded, context.unsqueeze(1)], dim=2) # (batch, 1, embed_dim + annotation_dim)
        o, (hidden, cell) = self.rnn(lstm_input, (hidden, cell)) # (batch, 1, hidden_dim), (1, batch, hidden_dim), (1, batch, hidden_dim)
        # 3. predict from [new hidden state; context]
        logits = self.fc_out(torch.cat([o.squeeze(1), context], dim=1)) # (batch, hidden_dim + annotation_dim) -> (batch, vocab_size)
        return logits, hidden, cell, attn_weights


class Seq2Seq(nn.Module):
    """Ties encoder + decoder; owns the teacher-forcing loop and the pad mask."""

    def __init__(self, encoder: Encoder, decoder: Decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(
        self, src: Tensor, tgt: Tensor, teacher_forcing_ratio: float = 1.0
    ) -> Tensor:
        """Same contract as 02: returns (batch, tgt_len, vocab_size) logits,
        row 0 left as zeros (position 0 is <sos>, never predicted).

        New vs 02: build mask = (src != PAD_ID) once, encode once, and pass
        (annotations, mask) into every decoder step.
        """
        batch, seq_len = tgt.size()
        outputs = torch.zeros(batch, seq_len, self.decoder.vocab_size, device=src.device)
        # encode once; annotations and mask are reused unchanged by every decoder step
        mask = src != PAD_ID # (batch, src_len) bool, False at <pad> so attention can't look there
        annotations, hidden, cell = self.encoder(src)
        input_token = tgt[:, 0]  # first token is always <sos>. column 0 all rows
        for time_step in range(1, seq_len):
            logits, hidden, cell, _ = self.decoder(input_token, hidden, cell, annotations, mask)
            outputs[:, time_step] = logits
            # next input: ground truth (teacher forcing) or the model's own top prediction
            teacher_force = random.random() < teacher_forcing_ratio
            input_token = tgt[:, time_step] if teacher_force else logits.argmax(1)
        return outputs
