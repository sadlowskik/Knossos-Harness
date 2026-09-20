"""Isolation tests for the pretraining benchmarks.

Evaluation code is uniquely dangerous: a broken metric does not crash, it just
reports a number, and that number then decides whether weeks of GPU time were
worth it. So each function is checked against a case whose answer is known in
closed form rather than against itself.
"""
from __future__ import annotations
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn as nn

from daedalus import Daedalus, Ariadne, ByteTokenizer, BPETokenizer
from daedalus.benchmarks import (model_log_probs, token_byte_lengths,
                                 evaluate_bits_per_byte, score_continuations,
                                 multiple_choice, parse_rate, balanced_delimiters)

V, T = 64, 16


class Uniform(nn.Module):
    """Predicts a uniform distribution: every token costs exactly log2(V) bits."""

    def __init__(self, vocab=V):
        super().__init__()
        self.vocab = vocab
        self.block_size = T

    def forward(self, x, targets=None):
        return torch.zeros(*x.shape, self.vocab), None


class Oracle(nn.Module):
    """Puts (almost) all mass on a designated token id."""

    def __init__(self, target, vocab=V, strength=20.0):
        super().__init__()
        self.target, self.vocab, self.strength = target, vocab, strength
        self.block_size = T

    def forward(self, x, targets=None):
        logits = torch.zeros(*x.shape, self.vocab)
        logits[..., self.target] = self.strength
        return logits, None


# ------------------------------------------------------- model_log_probs

def test_log_probs_are_normalised():
    lp = model_log_probs(Uniform(), torch.zeros(2, T, dtype=torch.long))
    assert lp.shape == (2, T, V)
    assert torch.allclose(lp.exp().sum(-1), torch.ones(2, T), atol=1e-5)


def test_log_probs_normalised_for_a_real_model():
    model = Daedalus(vocab_size=V, n_embd=32, n_head=4, n_layer=2, block_size=T)
    lp = model_log_probs(model, torch.randint(0, V, (2, T)))
    assert torch.allclose(lp.exp().sum(-1), torch.ones(2, T), atol=1e-5)


def test_ariadne_uses_the_halting_mixture_not_the_last_step():
    """Ariadne returns (p, per-step logits); the predictive distribution is the
    p-weighted mixture. Scoring the final step instead would grade a model the
    halting head never committed to."""
    torch.manual_seed(0)
    model = Ariadne(vocab_size=V, n_embd=32, n_head=4, core_layers=1, max_loops=3,
                    block_size=T).eval()
    x = torch.randint(0, V, (2, T))
    with torch.no_grad():
        p, step_logits = model(x)
        got = model_log_probs(model, x)
        expected = (p.unsqueeze(-1) * torch.softmax(step_logits.float(), -1)).sum(0)
        last_step = torch.softmax(step_logits[-1].float(), -1)
    assert torch.allclose(got.exp(), expected, atol=1e-5)
    assert not torch.allclose(got.exp(), last_step, atol=1e-3), \
        "mixture must differ from the final step, or the test proves nothing"


# ------------------------------------------------------ token_byte_lengths

def test_byte_lengths_from_a_bpe_tokenizer():
    tok = BPETokenizer.train(["hello world hello world hello"], vocab_size=300)
    lengths = token_byte_lengths(tok, tok.vocab_size)
    assert all(lengths[i].item() == 1 for i in range(256))
    for i in range(256, 256 + len(tok.merges)):
        assert lengths[i].item() == len(tok.token_bytes(i))


def test_special_tokens_count_as_zero_bytes():
    tok = BPETokenizer.train(["hello world hello"], vocab_size=300)
    lengths = token_byte_lengths(tok, tok.vocab_size)
    for tid in tok.specials.values():
        assert lengths[tid].item() == 0


# ----------------------------------------------------- bits per byte

@pytest.fixture
def byte_ids():
    torch.manual_seed(0)
    return torch.randint(0, 256, (4000,), dtype=torch.long)


def test_uniform_model_scores_exactly_log2_vocab(byte_ids):
    """Closed form: a uniform predictor over V tokens costs log2(V) bits per
    token, and with a byte tokenizer one token is one byte."""
    r = evaluate_bits_per_byte(Uniform(256), byte_ids, ByteTokenizer(),
                               block_size=T, batch_size=4, vocab_size=256)
    assert r["bits_per_byte"] == pytest.approx(math.log2(256), abs=1e-4)
    assert r["bits_per_token"] == pytest.approx(math.log2(256), abs=1e-4)
    assert r["perplexity"] == pytest.approx(256, rel=1e-3)


def test_bits_per_byte_is_invariant_to_stride(byte_ids):
    a = evaluate_bits_per_byte(Uniform(256), byte_ids, ByteTokenizer(), block_size=T,
                               stride=T // 2, batch_size=4, vocab_size=256)
    b = evaluate_bits_per_byte(Uniform(256), byte_ids, ByteTokenizer(), block_size=T,
                               stride=T // 4, batch_size=4, vocab_size=256)
    assert a["bits_per_byte"] == pytest.approx(b["bits_per_byte"], abs=1e-4)


def test_every_token_is_scored_exactly_once(byte_ids):
    """Overlapping windows must not double-count -- that would silently scale
    the metric by the overlap factor."""
    n = 1000
    r = evaluate_bits_per_byte(Uniform(256), byte_ids[:n], ByteTokenizer(),
                               block_size=T, stride=T // 2, batch_size=4,
                               vocab_size=256)
    starts = list(range(0, n - T - 1, T // 2))
    expected = T + (len(starts) - 1) * (T // 2)
    assert r["tokens"] == expected
    assert r["bytes"] == expected          # byte tokenizer: 1 byte per token


def test_multi_byte_tokens_change_bits_per_byte_but_not_bits_per_token():
    """The whole point of bits-per-byte: it is invariant to tokenization."""
    tok = BPETokenizer.train(["the quick brown fox " * 200], vocab_size=400)
    ids = torch.tensor(tok.encode("the quick brown fox " * 200), dtype=torch.long)
    r = evaluate_bits_per_byte(Uniform(tok.vocab_size), ids, tok, block_size=T,
                               batch_size=4, vocab_size=tok.vocab_size)
    assert r["bits_per_token"] == pytest.approx(math.log2(tok.vocab_size), abs=1e-3)
    # tokens average >1 byte, so bits/byte must come out strictly lower
    assert r["bits_per_byte"] < r["bits_per_token"]
    assert r["bits_per_byte"] == pytest.approx(
        r["bits_per_token"] * r["tokens"] / r["bytes"], rel=1e-6)


def test_short_split_raises():
    with pytest.raises(ValueError):
        evaluate_bits_per_byte(Uniform(256), torch.zeros(5, dtype=torch.long),
                               ByteTokenizer(), block_size=T, vocab_size=256)


# ---------------------------------------------------- score_continuations

def test_batched_scoring_matches_one_at_a_time():
    """The padding test. Mixed-length choices in one batch must score exactly
    as they do alone, or every multiple-choice result is biased by batch
    composition."""
    torch.manual_seed(0)
    model = Daedalus(vocab_size=V, n_embd=32, n_head=4, n_layer=2, block_size=T).eval()
    pairs = [([1, 2, 3], [4]),
             ([1, 2, 3], [4, 5, 6, 7]),
             ([9], [8, 8]),
             ([5, 5, 5, 5, 5], [1, 2, 3, 4, 5])]
    batched = score_continuations(model, pairs, block_size=T, batch_size=4)
    alone = score_continuations(model, pairs, block_size=T, batch_size=1)
    for b, a in zip(batched, alone):
        assert b == pytest.approx(a, abs=1e-4)


def test_only_the_continuation_is_scored():
    """A uniform model gives every token -log(V); the total must therefore be
    exactly len(continuation) * -log(V), independent of context length."""
    pairs = [([1, 2, 3, 4, 5, 6], [7, 8]), ([1], [7, 8])]
    got = score_continuations(Uniform(), pairs, block_size=T, batch_size=2)
    assert got[0] == pytest.approx(-2 * math.log(V), abs=1e-4)
    assert got[0] == pytest.approx(got[1], abs=1e-4)


def test_scores_match_a_hand_computed_log_prob():
    model = Oracle(target=7)
    hit = score_continuations(model, [([1, 2], [7, 7])], block_size=T)[0]
    miss = score_continuations(model, [([1, 2], [3, 3])], block_size=T)[0]
    assert hit > miss
    p_hit = math.exp(20.0) / (math.exp(20.0) + V - 1)
    assert hit == pytest.approx(2 * math.log(p_hit), abs=1e-3)


def test_long_sequences_keep_the_continuation():
    """Truncation must drop context, never the tokens being scored."""
    long_ctx = [i % V for i in range(200)]         # 200 tokens, block_size is 16
    got = score_continuations(Uniform(), [(long_ctx, [7, 8, 9])], block_size=T)[0]
    assert got == pytest.approx(-3 * math.log(V), abs=1e-4)


# -------------------------------------------------------- multiple_choice

def test_multiple_choice_picks_the_favoured_choice():
    tok = ByteTokenizer()
    examples = [{"context": "ab", "choices": ["\x07\x07", "cd"], "label": 0}]
    r = multiple_choice(Oracle(target=7, vocab=256), tok, examples, block_size=T)
    assert r["acc"] == 1.0 and r["acc_norm"] == 1.0
    assert r["n"] == 1 and r["chance"] == pytest.approx(0.5)


def test_acc_norm_removes_the_short_choice_bias():
    """Raw totals favour short continuations: every extra token adds another
    negative log-probability. Constructed so the two disagree --

        "y"      1 byte  at log p_miss = -5.57   total -5.57, per byte -5.57
        "xxxxx"  5 bytes at log p_hit  = -3.57   total -17.8, per byte -3.57

    so the raw argmax picks the short wrong answer and the normalised one picks
    the long right answer. If acc_norm ever regresses to acc, this fails.
    """
    tok = ByteTokenizer()
    model = Oracle(target=ord("x"), vocab=256, strength=2.0)
    examples = [{"context": "ab", "choices": ["y", "xxxxx"], "label": 1}]
    r = multiple_choice(model, tok, examples, block_size=T)
    assert r["acc"] == 0.0, "raw score should have been fooled by the short choice"
    assert r["acc_norm"] == 1.0, "per-byte normalisation should have corrected it"


def test_chance_is_reported_for_mixed_choice_counts():
    tok = ByteTokenizer()
    examples = [{"context": "a", "choices": ["b", "c"], "label": 0},
                {"context": "a", "choices": ["b", "c", "d", "e"], "label": 0}]
    r = multiple_choice(Uniform(256), tok, examples, block_size=T)
    assert r["chance"] == pytest.approx((0.5 + 0.25) / 2)


# -------------------------------------------------------------- parse_rate

def test_python_parse_rate_uses_the_real_grammar():
    good = ["x = 1\n", "def f():\n    return 1\n"]
    bad = ["def f(:\n", "x = = 1", "if True\n  pass"]
    assert parse_rate(good)["parse_rate"] == 1.0
    assert parse_rate(bad)["parse_rate"] == 0.0
    assert parse_rate(good)["method"] == "ast"
    assert parse_rate(good + bad)["parse_rate"] == pytest.approx(2 / 5)


def test_parse_rate_on_empty_input():
    assert parse_rate([])["parse_rate"] == 0.0


def test_non_python_is_labelled_as_a_weak_proxy():
    r = parse_rate(["fn main() { let x = 1; }"], language="rust")
    assert r["parse_rate"] == 1.0
    assert "not a parser" in r["method"]


@pytest.mark.parametrize("text,ok", [
    ("fn main() { }", True),
    ("fn main() { ", False),
    ("let s = \"unclosed", False),
    ("let s = \"a ) brace in a string\";", True),
    ("// a ) comment\nfn f() {}", True),
    ("/* block ( comment */ fn f() {}", True),
    ("let c = '\\'';", True),
    ("f(g[h{}])", True),
    ("f(]", False),
])
def test_balanced_delimiters(text, ok):
    assert balanced_delimiters(text) is ok
