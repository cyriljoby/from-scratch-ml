# Attention -  From Scratch

 
## Paper
 
- **Title:** Neural Machine Translation by Jointly Learning to Align and Translate
- **Authors:** Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio
- **Year:** 2015 — ICLR (arXiv 2014)
- **arXiv:** https://arxiv.org/abs/1409.0473

## Core Idea

The seq2seq model in [02-seq2seq](../02-seq2seq/) has a bottleneck: the encoder
crushes the entire source sentence into a single fixed-size vector, and the
decoder only receives that vector once, as its initial state. Everything the
decoder needs about the source has to survive, undiluted, through its own
recurrence. For short sentences this works; for long ones the early source
information washes out. Source reversal was a partial hack around this — it
shortened the distance for the *first* few tokens — but the fundamental problem
remains: one fixed vector can't hold a long sentence.

Attention removes the bottleneck. Instead of compressing the source into one
vector, the encoder keeps a hidden state for *every* source token, and at each
decoding step the decoder looks back at all of them. It scores how relevant each
source position is to what it's currently trying to generate, turns those scores
into weights (a soft **alignment**), and takes a weighted sum of the source
states — a **context vector** that is recomputed at every step. So as the decoder
produces each target word, it can focus on the source words that matter for that
word, rather than relying on one static summary.

Concretely, at decoder step `i` with previous state `s_{i-1}` and encoder states
`h_1..h_T`:

- **Score:** `e_ij = a(s_{i-1}, h_j)` — a small feedforward network scores the
  match between the current decoder state and each source state.
- **Align:** `α_ij = softmax_j(e_ij)` — normalize the scores into weights that
  sum to 1 across source positions.
- **Context:** `c_i = Σ_j α_ij · h_j` — the weighted sum of source states, i.e.
  a source summary tailored to this step.

The context vector is then fed into the decoder alongside the previous token to
produce the next output. The alignment weights `α` are learned end-to-end (no
alignment supervision) and, as a bonus, are interpretable — plotting them shows
which source words the model attended to for each target word.

Bahdanau et al. also use a **bidirectional encoder**, so each `h_j` summarizes
the whole sentence centered on word `j` (reading it both left-to-right and
right-to-left), giving the attention mechanism richer per-position representations
to align against.