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

        Args:
            decoder_hidden: (batch, hidden_dim) — s_{i-1}, the state BEFORE
                this decoding step.
            annotations: (batch, src_len, annotation_dim) — encoder's h_j.
            mask: (batch, src_len) bool — True at real tokens, False at <pad>.

        Returns:
            context: (batch, annotation_dim) — c_i = Σ_j α_ij h_j.
            weights: (batch, src_len) — α_ij; each row sums to 1, exactly 0
                at padded positions.
        """
        #e_ij = v · tanh(W_s s_{i-1} + W_h h_j)
        # one decoder hidden state per batch, broadcast over src_len to score all annotations at once
        e = self.v(torch.tanh(self.W_s(decoder_hidden).unsqueeze(1) + self.W_h(annotations))).squeeze(-1)  # (batch, src_len, 1) -> (batch, src_len)
        e = e.masked_fill(~mask, float("-inf"))  # -inf at padded positions so softmax gives 0 weight there
        weights = torch.softmax(e, dim=1) 
        context = (weights.unsqueeze(1) @ annotations).squeeze(1)
        return weights, context


class Encoder(nn.Module):
    """Bidirectional LSTM encoder producing one annotation per source token.

    Unlike 02, the decoder needs the ENTIRE output sequence (the annotations),
    plus an initial state of its own size: the BiLSTM's final states are
    (2, batch, hidden_dim) — one per direction — so they get concatenated and
    projected down to the decoder's (1, batch, hidden_dim).
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
        # TODO: embedding (padding_idx=PAD_ID), dropout, bidirectional LSTM
        #       (batch_first=True), and projections from the concatenated
        #       forward+backward final state (2*hidden_dim) down to hidden_dim
        #       for the decoder's initial hidden and cell.
        raise NotImplementedError

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
        # TODO: embed + dropout, run the BiLSTM, then build the decoder's
        #       initial (hidden, cell) from the final directional states.
        raise NotImplementedError


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
        # TODO: embedding, dropout, Attention, single-layer LSTM with input
        #       size embed_dim + annotation_dim (batch_first=True), and fc_out
        #       from hidden_dim + annotation_dim to vocab_size.
        raise NotImplementedError

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
        # TODO: 1. attend: context, weights = attention(hidden.squeeze(0), ...)
        #       2. embed token, concat with context, run one LSTM step
        #       3. logits from [new hidden; context]
        raise NotImplementedError


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
        # TODO: mirror 02's loop — encode, start from tgt[:, 0], step through
        #       time, choose teacher forcing per step via random.random().
        raise NotImplementedError
