# Paper Implementations

From-scratch implementations of foundational ML papers, ordered to show the progression from recurrent models to attention-based architectures. This is a portfolio/learning project so I am
prioritizing clarity and correctness matter more than performance.

## Reading order

Folders are numbered for reading order. Each is added only when its implementation begins. This reading order
might change as I discover more topics I want to explore and implement:

1. `01-lstm-cell/` — hand-written LSTM cell
2. `02-seq2seq/` — encoder–decoder sequence-to-sequence
3. `03-attention/` — additive/dot-product attention
4. `04-transformer/` — full from-scratch transformer

## Layout of each implementation

```
NN-name/
├── README.md          ← paper summary, design choices, deviations, results
├── src/               ← implementation files
└── tests/             ← correctness checks
```

# Current Progress
- Implemented LSTM cell with gradient flow visualization

