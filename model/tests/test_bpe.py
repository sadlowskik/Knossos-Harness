"""Isolation tests for the byte-level BPE tokenizer.

The property that matters most is **exact roundtrip**: a tokenizer that loses
information corrupts every downstream measurement silently, and the loss curve
will look perfectly healthy while it happens.
"""
from __future__ import annotations
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from daedalus import BPETokenizer, SPLIT_PATTERN

CORPUS = [
    "def add(a, b):\n    return a + b\n",
    "def sub(a, b):\n    return a - b\n",
    "class Foo:\n    def __init__(self):\n        self.x = 1\n",
    "the quick brown fox jumps over the lazy dog. " * 20,
    "fn main() { println!(\"hello\"); }\n" * 10,
]


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.train(CORPUS, vocab_size=600)


# ------------------------------------------------------------------ roundtrip

@pytest.mark.parametrize("text", [
    "",
    "a",
    "def add(a, b):\n    return a + b\n",
    "  \t \n\n   deeply\n        indented\n",
    "naïve café — em-dash, ünïcödé",
    "日本語のテキスト",
    "emoji 🚀🔥 and a zero-width​space",
    "\x00\x01\x02 raw control bytes \x7f",
    "mixed 123456789 numbers and !@#$%^&*() punctuation",
])
def test_roundtrip_is_exact(tok, text):
    assert tok.decode(tok.encode(text)) == text


def test_roundtrip_on_unseen_bytes(tok):
    """Every byte value must survive, including ones absent from training."""
    text = bytes(range(1, 256)).decode("latin-1")
    assert tok.decode(tok.encode(text)) == text


# ------------------------------------------------------------ the split regex
# The pre-tokenizer is the one place where characters can vanish without any
# error being raised: `findall` simply returns fewer chunks. A dropped `_`
# corrupts every identifier in a code corpus and nothing reports it, so
# losslessness is asserted directly rather than inferred from roundtrips.

@pytest.mark.parametrize("text", [
    "snake_case __init__ self._x _leading trailing_",
    "".join(chr(c) for c in range(1, 0x300)),
    "def f(x): return x_1 + _y  # comment_here\n\n\tindented_\n",
    "a_b-c.d/e\\f|g~h`i^j",
    "🚀_🔥 中_文 café_naïve",
])
def test_split_pattern_never_drops_a_character(text):
    assert "".join(SPLIT_PATTERN.findall(text)) == text


def test_underscore_survives_specifically(tok):
    """Regression: `[^\\s\\w]+` excluded `_` because `\\w` includes it."""
    assert "".join(SPLIT_PATTERN.findall("_")) == "_"
    assert tok.decode(tok.encode("__init__")) == "__init__"


# ---------------------------------------------------------------------- vocab

def test_ids_are_in_range(tok):
    ids = tok.encode("".join(CORPUS))
    assert ids, "encoding a non-empty corpus produced no ids"
    assert all(0 <= i < tok.vocab_size for i in ids)


def test_vocab_size_accounting():
    """vocab_size is a *bound*, not a promise: a small corpus runs out of
    repeated pairs first. What must always hold is the accounting identity."""
    t = BPETokenizer.train(CORPUS, vocab_size=500, specials=("<|endoftext|>",))
    assert t.vocab_size == 256 + len(t.merges) + len(t.specials)
    assert t.vocab_size <= 500
    assert max(t.specials.values()) == t.vocab_size - 1


def test_byte_ids_are_identity(tok):
    for i in range(256):
        assert tok.token_bytes(i) == bytes([i])


# --------------------------------------------------------------------- merges

def test_merges_actually_compress(tok):
    text = "".join(CORPUS)
    assert len(tok.encode(text)) < len(text.encode("utf-8"))


def test_merge_never_crosses_a_chunk_boundary(tok):
    """Each token's bytes must lie inside a single pre-tokenizer chunk.

    If a merge spanned two chunks the tokenizer would learn tokens like
    " the def" -- frequent in one corpus, useless in the next.
    """
    for chunk in SPLIT_PATTERN.findall("def add(a, b): return a + b"):
        ids = tok.encode(chunk)
        assert b"".join(tok.token_bytes(i) for i in ids) == chunk.encode("utf-8")


def test_encode_is_concatenative_over_chunks(tok):
    """encode(a+b) == encode(a) + encode(b) when a, b split cleanly."""
    a, b = "def add", "(a, b)"
    assert tok.encode(a + b) == tok.encode(a) + tok.encode(b)


# ------------------------------------------------------------------ specials

def test_specials_are_not_reachable_from_raw_text(tok):
    """Raw text must never produce a special id -- that is what makes them
    trustworthy as document separators."""
    ids = tok.encode("<|endoftext|> is just text here")
    assert all(i not in tok.specials.values() for i in ids)


def test_encode_with_specials_matches_literally(tok):
    eot = tok.specials["<|endoftext|>"]
    ids = tok.encode_with_specials("a<|endoftext|>b")
    assert eot in ids
    assert ids == tok.encode("a") + [eot] + tok.encode("b")


# ------------------------------------------------------- determinism, storage

def test_training_is_deterministic():
    a = BPETokenizer.train(CORPUS, vocab_size=500).merges
    b = BPETokenizer.train(CORPUS, vocab_size=500).merges
    assert a == b


def test_save_load_roundtrip(tok, tmp_path):
    path = str(tmp_path / "tokenizer.json")
    tok.save(path)
    other = BPETokenizer.load(path)
    text = "".join(CORPUS)
    assert other.merges == tok.merges
    assert other.specials == tok.specials
    assert other.encode(text) == tok.encode(text)


def test_cache_does_not_change_results(tok):
    text = "def add(a, b):"
    first = tok.encode(text)
    second = tok.encode(text)          # second call is served from the cache
    assert first == second
    assert tok.decode(first) == text


def test_vocab_size_floor_is_enforced():
    with pytest.raises(ValueError):
        BPETokenizer.train(CORPUS, vocab_size=100)
