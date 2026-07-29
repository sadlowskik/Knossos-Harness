"""Isolation tests for the training machinery added for real-scale runs.

These are the pieces that fail *silently* when they are wrong: a learning-rate
schedule that never warms up, weight decay applied to LayerNorm gains, an
untied embedding quietly doubling the parameter count, or a data loader whose
targets are not actually shifted by one. None of those raise -- they just make
the run worse than it should be.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import torch

from daedalus import Daedalus, Labyrinth, DaedalusFull
from data import Corpus
from train import init_weights, tie_weights, make_optimizer, lr_at, token_embedding

V, C, T = 512, 32, 16


def _args(**kw):
    base = dict(lr=6e-4, min_lr=6e-5, warmup=100, steps=1000, decay_steps=1000)
    base.update(kw)
    return argparse.Namespace(**base)


# ------------------------------------------------------------- weight tying

@pytest.mark.parametrize("build", [
    lambda: Daedalus(vocab_size=V, n_embd=C, n_head=4, n_layer=2, block_size=T),
    lambda: Labyrinth(vocab_size=V, n_embd=C, n_head=4, core_layers=2, block_size=T),
    lambda: DaedalusFull(vocab_size=V, n_embd=C, n_head=4, block_size=T,
                         core_layers=1, n_stages=2, n_gist=4),
])
def test_tie_weights_shares_one_tensor(build):
    model = build()
    assert tie_weights(model) is True
    emb = token_embedding(model)
    assert emb.weight is model.lm_head.weight, "tying must share, not copy"

    # A gradient through the head must reach the embedding, since they are one
    # tensor -- this is what makes tying save parameters rather than just memory.
    x = torch.randint(0, V, (2, T))
    out = model(x)
    logits = out[0] if isinstance(out, tuple) else out
    logits.sum().backward()
    assert emb.weight.grad is not None


def test_tying_reduces_the_parameter_count():
    a = Daedalus(vocab_size=V, n_embd=C, n_head=4, n_layer=2, block_size=T)
    b = Daedalus(vocab_size=V, n_embd=C, n_head=4, n_layer=2, block_size=T)
    tie_weights(b)
    n_a = len({id(p): p for p in a.parameters()}) and sum(
        p.numel() for p in a.parameters())
    n_b = sum(p.numel() for p in b.parameters())
    assert n_b == n_a - V * C


# -------------------------------------------------------------------- init

def test_residual_projections_are_scaled_down():
    """The projections writing into the residual stream get std 0.02/sqrt(2N)."""
    model = Daedalus(vocab_size=V, n_embd=C, n_head=4, n_layer=8, block_size=T)
    init_weights(model, n_effective_layers=8)
    proj = model.blocks[0].attn.proj.weight.std().item()
    qkv = model.blocks[0].attn.qkv.weight.std().item()
    assert proj < qkv / 2, f"residual proj std {proj:.4f} not scaled vs qkv {qkv:.4f}"


def test_init_gives_the_expected_starting_loss():
    """Loss at init must be ~ln(vocab) -- the project's oldest sanity check.

    Targets are drawn independently of the inputs on purpose. With tied weights
    the logits are `h @ E^T`, and at init `h` still contains the token's own
    embedding, so scoring `targets == inputs` measures the tying shortcut rather
    than the initialisation (it lands ~0.4 nats low).
    """
    torch.manual_seed(0)
    model = Daedalus(vocab_size=V, n_embd=C, n_head=4, n_layer=2, block_size=T)
    init_weights(model, 2)
    tie_weights(model)
    x = torch.randint(0, V, (8, T))
    y = torch.randint(0, V, (8, T))
    with torch.no_grad():
        loss = model(x, y)[1].item()
    assert abs(loss - np.log(V)) < 0.25, f"init loss {loss:.3f} vs ln(V)={np.log(V):.3f}"


def test_biases_are_zeroed():
    model = Daedalus(vocab_size=V, n_embd=C, n_head=4, n_layer=2, block_size=T)
    init_weights(model, 2)
    assert torch.all(model.blocks[0].attn.proj.bias == 0)


# --------------------------------------------------------------- optimizer

def test_only_matrices_are_weight_decayed():
    model = Daedalus(vocab_size=V, n_embd=C, n_head=4, n_layer=2, block_size=T)
    opt = make_optimizer(model, lr=1e-3, weight_decay=0.1, betas=(0.9, 0.95))
    decay, no_decay = opt.param_groups[0], opt.param_groups[1]
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() < 2 for p in no_decay["params"])
    # every trainable parameter lands in exactly one group
    n_grouped = len(decay["params"]) + len(no_decay["params"])
    assert n_grouped == len([p for p in model.parameters() if p.requires_grad])


def test_layernorm_gains_are_not_decayed():
    model = Daedalus(vocab_size=V, n_embd=C, n_head=4, n_layer=2, block_size=T)
    opt = make_optimizer(model, lr=1e-3, weight_decay=0.1, betas=(0.9, 0.95))
    ln = model.ln_f.weight
    assert any(p is ln for p in opt.param_groups[1]["params"])


# ---------------------------------------------------------------- schedule

def test_warmup_ramps_from_near_zero():
    a = _args(warmup=100)
    assert lr_at(0, a) == pytest.approx(a.lr / 100)
    assert lr_at(49, a) == pytest.approx(a.lr * 0.5)
    assert lr_at(99, a) == pytest.approx(a.lr)


def test_cosine_decays_monotonically_to_min_lr():
    a = _args(warmup=100, steps=1000, decay_steps=1000)
    seq = [lr_at(s, a) for s in range(100, 1001, 50)]
    assert all(x >= y - 1e-12 for x, y in zip(seq, seq[1:])), "not monotone"
    assert lr_at(1000, a) == pytest.approx(a.min_lr)
    assert lr_at(5000, a) == pytest.approx(a.min_lr), "must stay flat past the floor"


def test_schedule_peaks_exactly_once():
    a = _args(warmup=100, steps=1000, decay_steps=1000)
    lrs = [lr_at(s, a) for s in range(1000)]
    assert lrs.index(max(lrs)) == 99


# -------------------------------------------------------------------- corpus

@pytest.fixture
def corpus_dir(tmp_path):
    d = tmp_path / "corpus"
    d.mkdir()
    # two shards of different lengths, so proportional sampling is exercised
    np.arange(0, 5000, dtype=np.uint16).tofile(str(d / "train_00000.bin"))
    np.arange(5000, 6000, dtype=np.uint16).tofile(str(d / "train_00001.bin"))
    np.arange(0, 2000, dtype=np.uint16).tofile(str(d / "val_00000.bin"))
    (d / "meta.json").write_text(json.dumps(
        {"vocab_size": 8192, "dtype": "uint16"}), encoding="utf-8")
    return str(d)


def test_corpus_reports_total_length_across_shards(corpus_dir):
    c = Corpus(corpus_dir, "train")
    assert len(c) == 6000
    assert c.vocab_size == 8192


def test_targets_are_inputs_shifted_by_one(corpus_dir):
    c = Corpus(corpus_dir, "train")
    x, y = c.batch(batch_size=8, block_size=16, device="cpu")
    assert x.shape == (8, 16) and y.shape == (8, 16)
    assert torch.equal(x[:, 1:], y[:, :-1]), "y must be x shifted by one"


def test_windows_are_contiguous_runs(corpus_dir):
    """The shards hold arange data, so a correct window is consecutive ints."""
    c = Corpus(corpus_dir, "train")
    x, _ = c.batch(batch_size=16, block_size=16, device="cpu")
    diffs = (x[:, 1:] - x[:, :-1]).unique()
    assert diffs.tolist() == [1], f"windows not contiguous: deltas {diffs.tolist()}"


def test_batches_are_reproducible_given_a_generator(corpus_dir):
    c = Corpus(corpus_dir, "train")
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    x1, _ = c.batch(4, 16, "cpu", g1)
    x2, _ = c.batch(4, 16, "cpu", g2)
    assert torch.equal(x1, x2), "fixed-seed eval batches must repeat exactly"


def test_missing_split_raises(corpus_dir):
    with pytest.raises(FileNotFoundError):
        Corpus(corpus_dir, "test")
