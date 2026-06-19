"""Hand-written LSTM cell.

Reference: Hochreiter & Schmidhuber (1997), "Long Short-Term Memory".
Gate ordering and parameter layout follow torch.nn.LSTMCell so the two can be
compared directly (see tests/): weights are stacked in the order
input (i), forget (f), cell/candidate (g), output (o).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class LSTMCell(nn.Module):
    """A single LSTM step, computed from scratch.

    Given an input ``x`` and the previous hidden/cell state ``(h, c)``, returns
    the next ``(h', c')``. Matches ``torch.nn.LSTMCell`` numerically when given
    the same parameters.

    Parameters are laid out as in PyTorch:
        weight_ih: (4 * hidden_size, input_size)
        weight_hh: (4 * hidden_size, hidden_size)
        bias_ih:   (4 * hidden_size,)
        bias_hh:   (4 * hidden_size,)
    with the four chunks ordered [i, f, g, o].
    """

    def __init__(self, input_size: int, hidden_size: int, bias: bool = True) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        # instead of creating 4 different matrices for the 4 gates, we can create one big matrix 
        # and split it laterfor more efficient matrix multiplication.
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size)) # manipulates new input
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size)) #manipulates previous hidden state
        if bias:
            self.bias_ih = nn.Parameter(torch.empty(4 * hidden_size)) # one bias for each gate
            self.bias_hh = nn.Parameter(torch.empty(4 * hidden_size))
        else:
            self.register_parameter("bias_ih", None)
            self.register_parameter("bias_hh", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize parameters uniformly, as torch.nn.LSTMCell does."""
        stdv = 1.0 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0.0
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def forward(
        self, x: Tensor, state: tuple[Tensor, Tensor] | None = None
    ) -> tuple[Tensor, Tensor]:
        """Compute one LSTM step.

        Args:
            x: input tensor of shape (batch, input_size).
            state: optional (h, c), each of shape (batch, hidden_size).
                Defaults to zeros.

        Returns:
            (h_next, c_next), each of shape (batch, hidden_size).
        """
        if state is None:
            zeros = x.new_zeros(x.size(0), self.hidden_size)
            h_prev, c_prev = zeros, zeros
        else:
            h_prev, c_prev = state
        

        gates = x @ self.weight_ih.T + h_prev @ self.weight_hh.T
        if self.bias:
            gates = gates + self.bias_ih + self.bias_hh
        # This is an arbitrary convention: conceptually forget happens before input but we follow PyTorch conv
        input_gate, forget , candidate, output = gates.chunk(4, dim=1)
        input_gate = torch.sigmoid(input_gate)    # what percent of potential memory(candidate) to remember
        forget = torch.sigmoid(forget)  # how much of c_prev to forget
        output = torch.sigmoid(output)  # % potential short term memory to remember
        candidate = torch.tanh(candidate)   # new potential long term memory
        # compute the next cell state (Long Term Memory) and the next hidden state (Short Term Memory)
        c_next = candidate * input_gate + forget * c_prev
        potential_short_term = torch.tanh(c_next)
        h_next = potential_short_term * output
        return h_next, c_next
