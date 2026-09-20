"""Tests for the GEC path: loss masking, aligned packing, corruption, scoring."""
from __future__ import annotations

import json
import os
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from daedalus.ariadne import ponder_loss                                   # noqa: E402
from daedalus.bpe import BPETokenizer                                      # noqa: E402
from daedalus.full import DaedalusFullAdaptive                             # noqa: E402
from daedalus.gec import (GEC_SEP, build_prompt, corrupt_sentence,         # noqa: E402
                          gec_score, parse_completion, split_sentences)
from data import PairCorpus, load_corpus                                   # noqa: E402


# --------------------------------------------------------------- ponder_loss

def _ponder_inputs(b, t, v, n_steps, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(n_steps, b, t, v, generator=g)
    raw = torch.rand(n_steps, b, t, generator=g)
    p = raw / raw.sum(0, keepdim=True)          # a halting distribution sums to 1
    targets = torch.randint(0, v, (b, t), generator=g)
    return p, logits, targets


def test_ponder_loss_unmasked_matches_plain_mean():
    """With nothing ignored the new normalisation must be the old `.mean()`."""
    p, logits, targets = _ponder_inputs(2, 6, 11, 3)
    _, l_rec, _ = ponder_loss(p, logits, targets)

    n_steps, b, t, v = logits.shape
    ce = torch.stack([
        torch.nn.functional.cross_entropy(
            logits[n].reshape(-1, v), targets.reshape(-1), reduction="none"
        ).reshape(b, t) for n in range(n_steps)], dim=0)
    assert torch.allclose(l_rec, (p * ce).sum(0).mean(), atol=1e-6)


def test_ponder_loss_ignores_masked_positions():
    """Masked targets must be dropped from the numerator AND the denominator.

    This is the regression that motivated the change: `reduction="none"` plus a
    plain `.mean()` divides by B*T, so ignored positions would scale the loss
    down by the supervised fraction instead of being excluded.
    """
    p, logits, targets = _ponder_inputs(1, 6, 11, 3, seed=1)
    masked = targets.clone()
    masked[0, 3:] = -100                          # supervise only the first 3

    _, l_rec_masked, l_kl_masked = ponder_loss(p, logits, masked)
    # The same computation restricted to the supervised slice.
    _, l_rec_slice, l_kl_slice = ponder_loss(
        p[:, :, :3], logits[:, :, :3], targets[:, :3])

    assert torch.allclose(l_rec_masked, l_rec_slice, atol=1e-6)
    assert torch.allclose(l_kl_masked, l_kl_slice, atol=1e-6)

    # And it must NOT equal the buggy version that divides by B*T.
    buggy = l_rec_slice * (3 / 6)
    assert not torch.allclose(l_rec_masked, buggy, atol=1e-4)


def test_ponder_loss_all_masked_is_finite():
    """A fully-masked batch must not produce NaN via a zero denominator."""
    p, logits, targets = _ponder_inputs(1, 4, 7, 2, seed=2)
    total, l_rec, l_kl = ponder_loss(p, logits, torch.full_like(targets, -100))
    assert torch.isfinite(total) and torch.isfinite(l_rec) and torch.isfinite(l_kl)
    assert float(l_rec) == pytest.approx(0.0)


# ---------------------------------------------------------------- PairCorpus

def _write_pair_corpus(tmp_path, block_size, windows, masks, vocab=32):
    d = tmp_path / "gec"
    d.mkdir(exist_ok=True)
    flat_t = np.asarray([x for w in windows for x in w], dtype=np.uint16)
    flat_m = np.asarray([x for m in masks for x in m], dtype=np.uint8)
    flat_t.tofile(str(d / "train_00000.bin"))
    flat_m.tofile(str(d / "train_00000.mask"))
    (d / "meta.json").write_text(json.dumps(
        {"format": "pairs", "block_size": block_size, "vocab_size": vocab,
         "dtype": "uint16"}), encoding="utf-8")
    return str(d)


def test_pair_corpus_masks_targets_with_correct_offset(tmp_path):
    """y[j] is token j+1, so it is supervised iff mask[j+1] is set."""
    block = 4
    win = [1, 2, 3, 4, 5]
    msk = [0, 0, 1, 1, 0]
    d = _write_pair_corpus(tmp_path, block, [win], [msk])

    corpus = PairCorpus(d, "train")
    x, y = corpus.batch(1, block, "cpu")
    assert x[0].tolist() == [1, 2, 3, 4]
    # mask[1:] = [0, 1, 1, 0] -> keep positions 1 and 2 only
    assert y[0].tolist() == [-100, 3, 4, -100]


def test_pair_corpus_rejects_block_size_mismatch(tmp_path):
    d = _write_pair_corpus(tmp_path, 4, [[1, 2, 3, 4, 5]], [[0, 0, 1, 1, 0]])
    corpus = PairCorpus(d, "train")
    with pytest.raises(ValueError, match="packed for --block-size"):
        corpus.batch(1, 8, "cpu")


def test_pair_corpus_rejects_ragged_shard(tmp_path):
    """A shard that is not a whole number of windows means a size mismatch."""
    d = _write_pair_corpus(tmp_path, 4, [[1, 2, 3, 4, 5, 6]], [[0, 0, 1, 1, 0, 1]])
    with pytest.raises(ValueError, match="not a multiple"):
        PairCorpus(d, "train")


def test_pair_corpus_requires_mask_file(tmp_path):
    d = _write_pair_corpus(tmp_path, 4, [[1, 2, 3, 4, 5]], [[0, 0, 1, 1, 0]])
    os.remove(os.path.join(d, "train_00000.mask"))
    with pytest.raises(FileNotFoundError, match="no matching"):
        PairCorpus(d, "train")


def test_mask_files_are_not_globbed_as_shards(tmp_path):
    """`train_*.bin` must never match a mask file, or it trains on garbage."""
    d = _write_pair_corpus(tmp_path, 4, [[1, 2, 3, 4, 5]] * 3, [[0, 0, 1, 1, 0]] * 3)
    corpus = PairCorpus(d, "train")
    assert len(corpus.shards) == 1
    assert corpus.total_windows == 3


def test_load_corpus_dispatches_on_format(tmp_path):
    d = _write_pair_corpus(tmp_path, 4, [[1, 2, 3, 4, 5]], [[0, 0, 1, 1, 0]])
    assert isinstance(load_corpus(d)["train"], PairCorpus)


def test_every_sampled_window_has_supervision(tmp_path):
    d = _write_pair_corpus(tmp_path, 4, [[1, 2, 3, 4, 5]] * 8, [[0, 0, 1, 1, 0]] * 8)
    corpus = PairCorpus(d, "train")
    x, y = corpus.batch(8, 4, "cpu")
    assert (y != -100).sum() > 0
    assert x.shape == (8, 4) and y.shape == (8, 4)


# ---------------------------------------------------------------- corruption

def test_corrupt_reports_zero_edits_when_unchanged():
    rng = random.Random(0)
    text = "the cat sat on the mat"
    out, n = corrupt_sentence(text, rng, max_edits=2)
    assert (out == text) == (n == 0)


def test_corrupt_actually_changes_text_usually():
    rng = random.Random(7)
    base = ("They said that their results were quite good and it is a "
            "principle that does not affect the outcome here")
    changed = sum(1 for _ in range(50)
                  if corrupt_sentence(base, rng, 2)[0] != base)
    assert changed > 40, "corruption operators almost never fire"


def test_corrupt_is_deterministic_for_a_seed():
    a = corrupt_sentence("They said that their results were quite good indeed",
                         random.Random(3), 2)
    b = corrupt_sentence("They said that their results were quite good indeed",
                         random.Random(3), 2)
    assert a == b


def test_short_sentences_are_left_alone():
    out, n = corrupt_sentence("too short", random.Random(0), 2)
    assert out == "too short" and n == 0


def test_split_sentences_normalises_whitespace():
    text = "This is  a   sentence that is long enough to keep.\nAnd  another one here too."
    got = list(split_sentences(text, min_chars=10, max_chars=300))
    assert got == ["This is a sentence that is long enough to keep.",
                   "And another one here too."]
    assert all("  " not in s for s in got)


# ------------------------------------------------------------------- scoring

def test_gec_score_perfect_model():
    corrupt = ["their happy", "it is fine"]
    clean = ["they're happy", "it is fine"]
    s = gec_score(corrupt, clean, clean)
    assert s["exact_match_errored"] == 1.0
    assert s["copy_rate"] == 0.0
    assert s["false_positive_rate"] == 0.0


def test_gec_score_catches_identity_copy():
    """The degenerate corrector echoes its input and must be visible as such."""
    corrupt = ["their happy", "it is fine"]
    clean = ["they're happy", "it is fine"]
    s = gec_score(corrupt, clean, corrupt)
    assert s["exact_match_errored"] == 0.0
    assert s["copy_rate"] == 1.0
    assert s["exact_match_clean"] == 1.0          # right for the wrong reason


def test_gec_score_penalises_damaging_clean_text():
    s = gec_score(["it is fine"], ["it is fine"], ["it are fine"])
    assert s["false_positive_rate"] == 1.0
    assert s["n_errored"] == 0


def test_gec_score_rejects_length_mismatch():
    with pytest.raises(ValueError):
        gec_score(["a"], ["a"], ["a", "b"])


def test_parse_completion_cuts_at_boundary():
    assert parse_completion("they're happy\nnext thing") == "they're happy"
    assert parse_completion("fixed<|endoftext|>junk") == "fixed"
    assert parse_completion("  padded  ") == "padded"


def test_build_prompt_ends_with_separator():
    assert build_prompt("their happy").endswith(GEC_SEP)


# ------------------------------------------------------------------- end to end

@pytest.mark.slow
def test_make_gec_end_to_end_and_one_train_step(tmp_path):
    """Build a real packed corpus, load it, and take a gradient step on it."""
    sentences = [
        "The committee said that their proposal was quite good and it is fine.",
        "She told him that there were many principles which do not affect this.",
        "It is clear that the results were better than we had expected today.",
        "They have accepted the advice and it is now part of the new process.",
    ]
    src = tmp_path / "src"
    src.mkdir()
    for i in range(40):
        (src / f"doc{i}.txt").write_text(" ".join(sentences), encoding="utf-8")

    tok = BPETokenizer.train([" ".join(sentences)] * 4, vocab_size=400)
    tok_path = tmp_path / "tokenizer.json"
    tok.save(str(tok_path))

    out = tmp_path / "gec"
    block = 64
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "make_gec.py"),
         "--local", str(src), "--ext", ".txt", "--out", str(out),
         "--tokenizer", str(tok_path), "--block-size", str(block),
         "--target-examples", "60", "--val-examples", "4", "--test-examples", "4",
         "--windows-per-shard", "5"],
        capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr

    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["format"] == "pairs" and meta["block_size"] == block
    assert meta["examples"]["train"] > 0
    assert meta["supervised_tokens"]["train"] > 0
    assert (out / "gec_eval.jsonl").exists()
    # The corpus dir must be self-contained: evaluate.py looks for the tokenizer
    # beside the data, and phase 1's lives in a different directory.
    assert (out / "tokenizer.json").exists()
    assert meta["tokenizer"] == "tokenizer.json"

    rows = [json.loads(l) for l in
            (out / "gec_eval.jsonl").read_text(encoding="utf-8").splitlines() if l]
    assert rows and any(r["corrupt"] != r["clean"] for r in rows), \
        "eval set contains no corrupted examples"
    assert all(set(r) == {"corrupt", "clean"} for r in rows)

    corpus = load_corpus(str(out))
    assert isinstance(corpus["train"], PairCorpus)
    x, y = corpus["train"].batch(2, block, "cpu")
    assert x.shape == (2, block) and y.shape == (2, block)
    assert (y == -100).any(), "nothing was masked -- the mask file is not wired up"
    assert (y != -100).any(), "everything was masked"
    # Inputs are real ids; only targets carry the sentinel.
    assert int(x.min()) >= 0 and int(x.max()) < meta["vocab_size"]

    model = DaedalusFullAdaptive(vocab_size=meta["vocab_size"], n_embd=32, n_head=4,
                                 block_size=block, core_layers=1, max_loops=2,
                                 n_experts=2, n_gist=4, n_stages=1)
    _, loss, _ = model(x, y)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and any(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


@pytest.mark.slow
def test_masked_loss_differs_from_unmasked(tmp_path):
    """Masking must actually change the objective, not silently no-op."""
    torch.manual_seed(0)
    block = 16
    win = list(range(1, block + 2))
    msk = [0] * 8 + [1] * (block + 1 - 8)
    d = _write_pair_corpus(tmp_path, block, [win], [msk], vocab=64)
    corpus = PairCorpus(d, "train")
    x, y = corpus.batch(1, block, "cpu")

    model = DaedalusFullAdaptive(vocab_size=64, n_embd=32, n_head=4,
                                 block_size=block, core_layers=1, max_loops=2,
                                 n_experts=2, n_gist=4, n_stages=1)
    _, masked, _ = model(x, y)
    _, full, _ = model(x, torch.where(y == -100, torch.zeros_like(y), y))
    assert not torch.allclose(masked, full)
