"""Gradient-flow experiment: LSTM vs. vanilla RNN.

Demonstrates the core contribution of Hochreiter & Schmidhuber (1997): LSTM's additive cell state path 
preserves gradient magnitude across many time steps, unlike vanilla RNNs where gradients vanish (or explode),
making it possible for the optimizer to learn long-range dependencies

Idea:
    1. Unroll each cell over a sequence of length T.
    2. Compute a loss on the FINAL hidden state only.
    3. Backprop, and record the gradient norm at every timestep's hidden state.
    4. Plot both curves on a log y-axis. The RNN line collapses to ~0; the LSTM
       line stays alive much longer.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from src import LSTMCell

SEED = 0
SEQ_LEN = 100
HIDDEN_SIZE = 32
INPUT_SIZE = 16
ASSETS_DIR = Path(__file__).parent / "assets"


def seed_everything(seed: int = SEED) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def lstm_step(cell: LSTMCell, x_t: Tensor, state: tuple[Tensor, Tensor]):
    """One LSTM step. Returns (h_next, c_next)."""
    return cell(x_t, state)


def rnn_step(cell: nn.RNNCell, x_t: Tensor, state: tuple[Tensor, Tensor]):
    """One vanilla-RNN step, wrapped to share the (h, c) interface.

    The RNN has no cell state, so c is carried through untouched.
    """
    h_prev, c_prev = state
    h_next = cell(x_t, h_prev)
    return h_next, c_prev


def measure_gradient_flow(step_fn, cell, seq_len: int = SEQ_LEN) -> list[float]:
    """Unroll `cell` over a random sequence and return the per-timestep gradient norm.

    Args:
        step_fn: a function (cell, x_t, (h, c)) -> (h_next, c_next).
        cell: the recurrent cell to drive (LSTMCell or nn.RNNCell).
        seq_len: number of timesteps to unroll.

    Returns:
        A list of length `seq_len`: the L2 norm of the gradient of the final loss
        w.r.t. each timestep's hidden state, ordered from t=0 (oldest) to t=T-1.
    """
    batch = 1
    h = torch.zeros(batch, HIDDEN_SIZE)
    c = torch.zeros(batch, HIDDEN_SIZE)

    # A fixed random input sequence (same across cells thanks to seeding upstream).
    inputs = [torch.randn(batch, INPUT_SIZE) for _ in range(seq_len)]

    hidden_states: list[Tensor] = []

    # passing in a 100 inputs and unrolling the cell for 100 steps
    for input in inputs:
        # for each step we get the hidden state and cell state, which is passed to the next step
        h, c = step_fn(cell, input, (h, c))
        h.retain_grad()
        hidden_states.append(h)
    # backprop needs a scalar to begin from
    loss = h.pow(2).sum()
    # begin backprop, which fills in the gradient at each
    loss.backward()
    return [hs.grad.norm().item() for hs in hidden_states]


def plot(rnn_norms: list[float], lstm_norms: list[float], perfect_memory_lstm_norms: list[float]) -> Path:
    """Plot both gradient-flow curves and save to assets/gradient_flow.png."""
    import matplotlib.pyplot as plt

    ASSETS_DIR.mkdir(exist_ok=True)
    out_path = ASSETS_DIR / "gradient_flow.png"
    fig, ax = plt.subplots()
    ax.plot(rnn_norms, label="vanilla RNN")
    ax.plot(lstm_norms, label="LSTM")
    ax.plot(perfect_memory_lstm_norms, label="Perfect Memory LSTM (forget gate ~= 1)")
    ax.set_yscale("log")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Gradient Norm")
    ax.set_title("Gradient Flow: LSTM vs Vanilla RNN")
    ax.legend()
    plt.savefig(out_path, dpi=150)
    return out_path


def main() -> None:
    seed_everything()
    lstm = LSTMCell(INPUT_SIZE, HIDDEN_SIZE)
    perfect_memory_lstm = LSTMCell(INPUT_SIZE, HIDDEN_SIZE)
    perfect_memory_lstm.bias_hh[HIDDEN_SIZE:2*HIDDEN_SIZE].data.fill_(5.0)  # set forget gate bias to +5, so it never forgets
    
    rnn = nn.RNNCell(INPUT_SIZE, HIDDEN_SIZE)

    lstm_norms = measure_gradient_flow(lstm_step, lstm)
    perfect_memory_lstm_norms = measure_gradient_flow(lstm_step, perfect_memory_lstm)
    rnn_norms = measure_gradient_flow(rnn_step, rnn)

    out_path = plot(rnn_norms, lstm_norms, perfect_memory_lstm_norms)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
