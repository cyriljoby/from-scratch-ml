"""Shared constants for the seq2seq pipeline.

Special-token ids are fixed here so data.py (which builds vocabs) and model.py
(which needs PAD_ID for the embedding's padding_idx) agree on a single source of
truth without importing each other.
"""

from __future__ import annotations

PAD_ID, SOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
