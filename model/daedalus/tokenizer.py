"""Byte-level tokenizer.

Maps text <-> a sequence of integers via UTF-8 bytes. The vocabulary is
exactly 256 (all possible byte values), so there is no "unknown token" failure
mode: any string in any language, including emoji and exotic whitespace, is
representable. The cost is longer sequences (one byte = one token). A byte-level
BPE (GPT-2 style, Radford et al. 2019) is the recommended upgrade for shorter
sequences while keeping the no-UNK guarantee.
"""
from __future__ import annotations
from typing import List, Sequence


class ByteTokenizer:
    """A reversible text <-> byte-id mapping with a fixed 256-token vocabulary."""

    vocab_size: int = 256

    def encode(self, text: str) -> List[int]:
        """str -> list of ints in [0, 255]."""
        return list(text.encode("utf-8"))

    def decode(self, ids: Sequence[int]) -> str:
        """list of ints -> str. `errors='replace'` guards against a partial
        multi-byte character when a model generates bytes one at a time."""
        return bytes(ids).decode("utf-8", errors="replace")
