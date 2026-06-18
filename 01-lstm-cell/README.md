# LSTM Cell — From Scratch

A from-scratch PyTorch implementation of the Long Short-Term Memory cell
(Hochreiter & Schmidhuber, 1997), built as a learning exercise before using
`nn.LSTM` in the implementations of other NNs that follow.

## Why

Understanding the gate mechanics, forget, input, output, and the additive
cell state update that solves the vanishing gradient problem, by implementing them.

## Stages of an LSTM

An LSTM uses a gating mechanism to regulate the flow of information, letting the
network maintain both **long-term memory** (the cell state) and **short-term
memory** (the hidden state). Each gate takes the previous short-term memory and
the current input, applies weights, and uses sigmoid/tanh activations to decide
what to keep, add, or expose.

### 1. Forget Gate

Decides **how much of the long-term memory to keep**. A sigmoid over the previous
hidden state and current input produces a value between 0 and 1 for each element
of the cell state (0 = forget completely, 1 = keep completely).

> **Output:** `Long-Term Memory = Long-Term Memory × % Long-Term to Remember`

### 2. Input Gate

Decides **what new information to add** to the long-term memory, via two parallel
blocks:

- **Right block — Potential Long-Term Memory:** combines short-term memory and
  input, then passes them through `tanh` to create the candidate (potential)
  long-term memory.
- **Left block — % Potential Memory to Remember:** a sigmoid that determines what
  percentage of that potential memory to actually add.

> **Output:** `New Long-Term Memory = (Potential Long-Term Memory × % Potential Memory to Remember) + Long-Term Memory`

### 3. Output Gate

Decides the **new short-term memory** (hidden state) passed to the next step,
again via two blocks:

- **Right block — Potential Short-Term Memory:** the new long-term memory is
  passed through `tanh` to scale values between −1 and 1.
- **Left block — % Potential Memory to Remember:** a sigmoid (same form as
  above) selecting which parts to expose.

> **Output:** `New Short-Term Memory = Potential Short-Term Memory × % Potential Memory to Remember`


## How It Works

The cell takes the current input and previous (hidden state, cell state), and
produces updated versions of both through three gates:

- **Forget gate:** sigmoid over [h_prev, x] — determines how much of the
  existing cell state to keep
- **Input gate:** sigmoid over [h_prev, x] scaled against a tanh candidate —
  determines what new information to add to the cell state
- **Output gate:** sigmoid over [h_prev, x] scaled against tanh of the updated
  cell state — determines what to expose as the new hidden state

The cell state itself is updated additively (old * forget + new * input),
which gives gradients a direct path to flow backward through many time steps
without vanishing.

## Verification

The implementation is tested against PyTorch's `nn.LSTMCell` — given identical
weights and inputs, outputs match within `atol=1e-5`.

```bash
pytest tests/test_lstm_cell.py
```

## What's Next

This cell is the primitive used inside the encoder and decoder LSTMs in
[02-seq2seq/](../02-seq2seq/), where two separate LSTM networks are chained
through a shared hidden state to map one variable-length sequence to another.