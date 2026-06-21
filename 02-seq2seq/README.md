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

## My Architecture and Training Details
Once I decide my architecture, dataset and scale I will update this section. From the above 
training details it is clear that the same scale is impossible as Google operated with an 8 GPU 
Machine over 10 days.

## My Results
Coming Soon


 
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