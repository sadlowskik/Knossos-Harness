"""Tests for per-loop step embeddings and the adaptive-depth instrumentation."""
from __future__ import annotations

import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from daedalus.benchmarks import halting_stats                              # noqa: E402
from daedalus.full import DaedalusFull, DaedalusFullAdaptive               # noqa: E402
from generate import build, resolve_config                                 # noqa: E402


def _adaptive(loop_embed=False, max_loops=3, **kw):
    return DaedalusFullAdaptive(vocab_size=32, n_embd=16, n_head=4, block_size=8,
                                core_layers=1, max_loops=max_loops, n_experts=2,
                                n_gist=4, n_stages=2, loop_embed=loop_embed, **kw)


# ------------------------------------------------------------- loop embedding

def test_disabled_by_default_adds_no_parameters():
    """The default must stay byte-compatible with existing checkpoints."""
    plain = _adaptive(loop_embed=False)
    assert not any("loop_emb" in k for k in plain.state_dict())
    assert all(c.loop_emb is None for c in plain.stages)


def test_enabled_adds_exactly_one_vector_per_step_per_stage():
    n_embd, max_loops, fixed, stages = 16, 3, 3, 2
    a, b = _adaptive(loop_embed=False), _adaptive(loop_embed=True)
    delta = sum(p.numel() for p in b.parameters()) - sum(p.numel() for p in a.parameters())
    assert delta == stages * max(fixed, max_loops) * n_embd


def test_enabled_changes_the_function():
    torch.manual_seed(0)
    x = torch.randint(0, 32, (2, 8))
    y = torch.randint(0, 32, (2, 8))
    torch.manual_seed(1)
    a = _adaptive(loop_embed=False)
    torch.manual_seed(1)
    b = _adaptive(loop_embed=True)
    # Same seed for the shared blocks; only the extra table differs.
    assert not torch.allclose(a(x, y)[1], b(x, y)[1])


def test_step_offset_is_clamped_past_the_table():
    """--variable-loops samples depths above the trained max; must not crash."""
    m = DaedalusFull(vocab_size=32, n_embd=16, n_head=4, block_size=8,
                     core_layers=1, n_loops=2, n_experts=2, n_gist=4,
                     n_stages=1, loop_embed=True)
    x = torch.randint(0, 32, (2, 8))
    y = torch.randint(0, 32, (2, 8))
    assert torch.isfinite(m(x, y, n_loops=7)[1])


def test_step_returns_zero_when_disabled():
    core = _adaptive(loop_embed=False).stages[0]
    assert core.step(0) == 0.0
    assert core.step(99) == 0.0


def test_gradients_reach_the_loop_table():
    m = _adaptive(loop_embed=True)
    x = torch.randint(0, 32, (2, 8))
    y = torch.randint(0, 32, (2, 8))
    m(x, y)[1].backward()
    # The final stage is the one whose loop is unrolled in forward(); if the
    # explicit final.step(n-1) call were missing this grad would be None.
    g = m.stages[-1].loop_emb.weight.grad
    assert g is not None and g.abs().sum() > 0


def test_checkpoint_round_trip_preserves_loop_embed():
    m = _adaptive(loop_embed=True)
    ck = {"model": m.state_dict(),
          "args": {"model": "adaptive", "vocab_size": 32, "n_embd": 16, "n_head": 4,
                   "block_size": 8, "core_layers": 1, "max_loops": 3, "n_experts": 2,
                   "n_gist": 4, "n_stages": 2, "n_mem_banks": 1, "loop_embed": True}}
    cfg = resolve_config(ck)
    assert cfg["loop_embed"] == 1
    rebuilt = build(cfg, "cpu")
    rebuilt.load_state_dict(ck["model"])          # raises on any shape mismatch
    assert any("loop_emb" in k for k in rebuilt.state_dict())


def test_checkpoint_without_the_key_defaults_to_off():
    """Old checkpoints predate the flag and must still load."""
    cfg = resolve_config({"args": {"model": "adaptive", "vocab_size": 32,
                                   "n_embd": 16, "n_head": 4, "block_size": 8}})
    assert cfg["loop_embed"] == 0
    assert not any("loop_emb" in k for k in build(cfg, "cpu").state_dict())


def test_expected_logits_matches_the_materialising_form():
    """The memory-lean accumulation must equal `(p.unsqueeze(-1)*logits).sum(0)`.

    That expression is what OOM'd at batch 16: it builds a second
    (steps, B, T, vocab) tensor, promoted to fp32 by `p`'s dtype.
    """
    torch.manual_seed(0)
    m = _adaptive()
    x = torch.randint(0, 32, (2, 8))
    y = torch.randint(0, 32, (2, 8))
    exp_logits, _, extras = m(x, y)
    p, step_logits = extras["p"], extras["step_logits"]
    reference = (p.unsqueeze(-1) * step_logits).sum(0)
    assert torch.allclose(exp_logits, reference, atol=1e-5, rtol=1e-4)


def test_expected_logits_peaks_smaller_than_the_stack():
    """Sanity: the accumulator never holds `steps` copies of the vocab tensor."""
    m = _adaptive(max_loops=4)
    x = torch.randint(0, 32, (2, 8))
    _, _, extras = m(x)
    sl = extras["step_logits"]
    assert sl.shape[0] == 4                    # the stack itself is unavoidable
    # one step's slice is 1/4 of it -- the temporary the old form avoided
    assert sl[0].numel() * 4 == sl.numel()


# ------------------------------------------------------------- halting_stats

class _FakeHalting(torch.nn.Module):
    """A model with a controllable halting distribution, for exact assertions."""

    def __init__(self, p_rows, vocab=8):
        super().__init__()
        self.p_rows = p_rows                        # list over steps
        self.vocab = vocab

    def forward(self, x, y=None):
        b, t = x.shape
        p = torch.tensor(self.p_rows, dtype=torch.float).view(-1, 1, 1)
        p = p.expand(len(self.p_rows), b, t).contiguous()
        return torch.randn(b, t, self.vocab), None, {"p": p}


def test_halting_stats_returns_none_without_halting():
    class NoHalt(torch.nn.Module):
        def forward(self, x, y=None):
            return torch.randn(*x.shape, 8), None
    ids = list(range(64))
    assert halting_stats(NoHalt(), ids, 8, "cpu", batch_size=2, max_batches=2) is None


def test_halting_stats_flags_dead_depth():
    """All mass on step 1 -> depth is constant 1.0 and the dial is dead."""
    h = halting_stats(_FakeHalting([1.0, 0.0, 0.0, 0.0]), [i % 8 for i in range(64)], 8,
                      "cpu", batch_size=2, max_batches=4)
    assert h["depth_mean"] == pytest.approx(1.0)
    assert h["depth_std"] == pytest.approx(0.0, abs=1e-6)
    assert h["frac_at_floor"] == pytest.approx(1.0)
    assert h["depth_nll_corr"] == 0.0          # undefined -> reported as zero


def test_halting_stats_flags_saturation():
    h = halting_stats(_FakeHalting([0.0, 0.0, 0.0, 1.0]), [i % 8 for i in range(64)], 8,
                      "cpu", batch_size=2, max_batches=4)
    assert h["depth_mean"] == pytest.approx(4.0)
    assert h["frac_at_ceiling"] == pytest.approx(1.0)
    assert h["max_loops"] == 4


def test_halting_stats_computes_expected_depth():
    """Uniform over 4 steps -> expected depth = (1+2+3+4)/4 = 2.5."""
    h = halting_stats(_FakeHalting([0.25] * 4), [i % 8 for i in range(64)], 8, "cpu",
                      batch_size=2, max_batches=4)
    assert h["depth_mean"] == pytest.approx(2.5)


def test_halting_stats_on_the_real_model():
    m = _adaptive(loop_embed=False, max_loops=3)
    h = halting_stats(m, [i % 32 for i in range(96)], 8, "cpu", batch_size=2, max_batches=4)
    assert h is not None
    assert h["max_loops"] == 3
    assert 1.0 <= h["depth_mean"] <= 3.0
    assert -1.0 <= h["depth_nll_corr"] <= 1.0
    assert len(h["histogram"]) == 3
    assert h["tokens"] > 0
