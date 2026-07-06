# Seq2Seq -  From Scratch
 
A from-scratch PyTorch implementation of the encoder-decoder LSTM architecture
for sequence-to-sequence learning, following Sutskever et al. (2014).
 
## Paper
 
- **Title:** Sequence to Sequence Learning with Neural Networks
- **Authors:** Ilya Sutskever, Oriol Vinyals, Quoc V. Le
- **Year:** 2014 — NIPS
- **arXiv:** https://arxiv.org/abs/1409.3215
## Core Idea
 
DNNs cannot handle sequences because they require fixed-size inputs and outputs.
Sequences are ordered lists of variable length where order carries meaning, and
a sequence-to-sequence task is an input-output mapping between two such sequences
that may differ in length.

A possible approach to this could be the Vanilla RNN, as it can take multiple inputs.
A vanilla RNN computes a sequence of outputs by iterating the following equation:
- h_t = sigmoid(W_hx · x_t + W_hh · h_{t-1})
	- This is the hidden state update and just combines the current input and previous hidden state with the learned weights
- y_t = W_yh · h_t
	- projects the hidden state through another learned state to produce a prediction

But, the vanilla RNN is unsuitable for seq 2 seq due to 2 primary reasons:
- The vanishing/exploding gradient problem which makes optimization difficult
- And more importantly, it cannot handle input-output mappings of different lengths


This paper solves the problem with a simple architecture: one LSTM (the encoder)
reads the input sequence token by token and compresses it into a single
fixed-dimensional vector, then a second LSTM (the decoder) generates the output
sequence from that vector one token at a time. The decoder is essentially a
recurrent neural network language model conditioned on the encoded input. At each
step it receives the previous token as input, combines it with its own evolving
hidden state, and produces the next predicted token, continuing until it predicts
an end-of-sequence marker.
 
Therefore the seq2seq architecture addresses each of the issues caused by
the vanilla RNN: LSTMs with their additive cell-state update mitigate the 
gradient problem (see [01-lstm-cell](../01-lstm-cell/)), and
the encoder-decoder split decouples reading the input from producing the output,
allowing dynamic input-output lengths.
 
## Architecture
 
Three key design choices beyond the basic encoder-decoder structure:
 
1. **Two separate LSTMs.** Encoder and decoder don't share weights, allowing each
   to specialize without extra sequential computation since the encoder finishes
   before the decoder starts.
2. **Deep (stacked) layers.** Instead of processing through one LSTM, the input
   goes through multiple stacked LSTMs, each with its own weights. At each time
   step, layer 1 processes the input and produces a hidden state, which becomes
   the input to layer 2, and so on. Each layer carries its own h/c forward
   through time independently. c stays local to the layer as that layer's private
   long-term memory, while h gets passed both forward in time and up to the next
   layer as input. Deeper layers build more abstract representations.
3. **Source sentence reversal.** The words of the input sentence are reversed
   before being fed to the encoder. This shortens the distance between the first
   source tokens and the first target tokens, giving the optimizer stronger
   gradient signal for those early dependencies. The paper reports this single
   trick improved BLEU from 25.9 to 30.6 and dropped perplexity from 5.8 to 4.7.
 
### Training Details (paper's scale)
 
- Deep LSTMs with 4 layers, 1000 cells at each layer and 1000-dim word embeddings
- Input vocab of 160k and output vocab of 80k
- WMT'14 English to French, 12M sentence pairs
- Initialized parameters with uniform distribution between -0.08 and 0.08
- SGD without momentum, fixed learning rate of 0.7
- After first 5 epochs, halved the learning rate every half epoch, concluding at 7.5
- Gradient norm clipped at 5 to prevent gradient explosion
- Batched by similar sentence length to minimize padding and wasted computation
- Beam size 2: keep the top 2 candidates alive at each step, expand both, keep the
  best 2 overall, repeat until done, pick the best full sequence. Slightly slower
  but avoids committing to a bad path early.
    - Beam size 1 (greedy) also performed well.


## Planned Architecture and Training Details
From the above training details it is clear that the same scale is impossible as Google operated with an 8 GPU Machine over 10 days. Therefore my dataset and training specs are listed below with full justifications for each choice:

- **Dataset:** [Multi30k](https://github.com/multi30k/dataset) German→English with ~29k training pairs. Same task (translation), but can train in minutes on a single GPU.
- **Model:** 2 layers, 256 hidden units, 256-dim embeddings. Smaller dataset thus using a smaller model to prevent overfitting.
- **Vocabulary:** built from training data every word that appears at least twice, rare words replaced with `<unk>`. 
- **Optimizer:** Adam with lr=3e-4. Converges faster than SGD and doesn't require manual learning rate schedule.
- **Gradient clipping:** norm clipped at 5, same as the paper, as LSTM gradients can still explode.
- **Training:** ~20-30 epochs with early stopping on validation loss. Can train for more epochs as less costly given scale of dataset.
- **Decoding:** greedy (beam size 1). The paper found greedy performed surprisingly well, so beam search(size 2) is more of a stretch goal.
- **Teacher forcing:** ratio of 1.0 (always ground truth) to match the paper. Will compare against 0.5 and 0.0 as well to determine how much exposure to its own predictions during training improves or worsens the model's robustness.
- **Source reversal:** configurable flag to reproduce my own before/after BLEU comparison to validate one of the paper's key findings.
- **Batching:** sorted by similar sentence length to minimize padding waste as mentioned in the paper.
- **Evaluation:** BLEU score on test set.

## My Results

### Final training setup (as run)

- **Batch size:** 64 — more efficient for repeated experiments than 32, with
  minimal quality loss (val loss barely higher).
- **Epochs:** 40 — also tested 50, but the last 10 epochs changed validation loss
  negligibly, so 40 was significantly faster for repeated runs.
- **Learning rate:** kept at Adam 3e-4 even after bumping batch 32→64; a larger
  batch sometimes wants a higher LR, but Adam at 3e-4 stayed stable.
- **Batching:** random shuffle with padding, *not* length-sorted as planned —
  padding waste is small at batch 64, so I skipped the length-bucketing
  optimization (it affects speed, not model quality).
- **Device:** Apple MPS (Metal). Note LSTM kernels aren't bit-reproducible on MPS,
  so BLEU wobbles ~1 point between otherwise-identical runs (see Caveats).

### BLEU (test set, greedy decode, sacreBLEU)

Methodology:
- **Test set:** measured on `mmt16_task1_test` (1000 held-out sentences), never seen in training or validation.
- **Greedy decode:** take the top token each step, no beam search implemented as of now (beam would likely score higher).
- **sacreBLEU:** standard, reproducible corpus-level BLEU. References are the true target tokens (not round-tripped through the vocab, so no `<unk>` leaks into them).
- **Controlled comparison:** same seed, batch size, learning rate, and epoch budget across all rows of a table — only the stated variable changes, so any BLEU difference is attributable to that variable.

**Source reversal (train_tf = 1.0):**

| Source order | best val loss | test BLEU |
|---|---|---|
| Reversed     | 2.4944 | 15.81 |
| Not reversed | 2.4914 | 17.21 |

**Teacher-forcing ratio:**

| Train TF ratio | best val loss | test BLEU |
|---|---|---|
| 1.0 |  |  |
| 0.5 |  |  |
| 0.0 |  |  |

### Did source reversal help?

**No.** Reversal was marginally *worse* (15.81 vs 17.21 BLEU), not better — a
failure to reproduce the paper's gain (25.9 → 30.6 on WMT'14) at this scale.

I tested whether Multi30k's short sentences were masking a real benefit (reversal
should help most on long sentences). Length-stratified BLEU says no:

| src length | n | forward | reversed | Δ |
|---|---|---|---|---|
| 1–9   | 270 | 19.74 | 18.87 | −0.87 |
| 10–12 | 323 | 19.24 | 18.00 | −1.23 |
| 13–16 | 273 | 17.27 | 14.75 | −2.52 |
| 17–35 | 134 | 12.24 | 11.99 | −0.24 |
| all   | 1000 | 17.21 | 15.81 | −1.40 |

Negative in every bucket, no positive trend with length. Likely Multi30k never
reaches the long-sentence regime where reversal pays off (≤35 tokens vs WMT'14's
much longer sentences). Caveat: single seed + MPS noise (~1 BLEU), so trust the
direction, not the exact deltas.


### Did teacher-forcing ratio matter?
<!-- your read: 1.0 vs 0.5 vs 0.0 -->


### Comparison to the paper
My best is 17.21 BLEU vs the paper's 30.6 (single reversed model). But, a direct comparison is not possible due to the following differences primarily due to scale and architectural differences.

- Task: De→En vs their En→Fr
- Data: 29k pairs vs ~12M (~400× less)
- Model: 2×256 vs 4×1000
- Decoding: greedy single model vs beam search + 5-model ensemble

What's comparable is the *method*, with
same encoder-decoder LSTM, same recipe scaled down and the gap is mostly
explained by data and model size.


### Measurement lesson: teacher forcing in eval
<!-- what went wrong with TF=0 eval, why, how you fixed it -->


### Debugging: Using the paper as a guide

My first reversal runs showed source reversal *halving* BLEU (~5 reversed vs ~17
not-reversed), which directly contradicted the paper's results where reversal improved the model.I used the
paper's expected result as a guide to debug this clear bug.

The other clue was internal: reversed and non-reversed models reached *nearly
identical validation loss* (~2.49) but wildly different BLEU (5 vs 17). Since val
loss (teacher-forced) was fine, the model wasn't broken, which meant something specific to
*decoding/evaluation* of the reversed run was failing.

I isolated it step by step:

1. **Is reversal applied consistently?** Printed the tensors from both the
   training `DataLoader` and the eval path which had byte-identical reversed input. This mean it
   wasn't a train/eval reversal mismatch. 
2. **Decoded  a reversed model in memory, both ways.** Trained
   `reverse=True` and evaluated the *in-memory* model (no checkpoint file) with
   reversed vs forward decoding: **10.97 matched vs 4.88 mismatched.** A reversed
   model correctly prefers reversed decoding — so reversal was working all along.

**Root cause:** a single shared `best_model.pth` written by every run. Because I
ran the configs back-to-back, the reversed run's BLEU step loaded a *forward*
model left on disk and decoded it reversed The fix was per-config checkpoint filenames
(`best_r{reverse}_tf{tf}.pth`), so no run can ever evaluate another run's weights.

### Sample translations
```
REF:
HYP:
```



## What's Next
 
The encoder compresses the entire source sentence into a single fixed-size vector,
and the decoder only receives this vector once, as its initial state. For short
sentences this works well, but for longer ones the source information has to
survive through the decoder's own recurrence, diluting with each step. This
bottleneck was addressed here by reversing the source sentence, but it is more
fundamentally solved by attention in Bahdanau et al. (2015), "Neural Machine
Translation by Jointly Learning to Align and Translate", which lets the decoder
look directly at every encoder hidden state at each decoding step rather than
relying on one compressed vector. That is the next implementation in
[03-attention/](../03-attention/).