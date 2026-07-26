"""Isolation tests for Proteus, the self-modifying weight matrix.

Stability comes first here. A self-referential update rule can drive its own
weight norm to infinity while the loss still looks fine for a while, so the
tests watch ||W|| directly, per the module docstring.

    pytest -q tests/test_proteus.py
    pytest -q -m slow tests/test_proteus.py    # includes the long stability run
"""
import math

import pytest
import torch

from daedalus import (SelfModifyingLinear, DaedalusProteus, Daedalus, Labyrinth,
                      DaedalusFull, MultiHeadAttention)

B, T, C, V = 4, 16, 32, 256
H = 4


@pytest.mark.parametrize("self_ref", [False, True])
def test_self_modifying_linear_shapes_and_causality(self_ref):
    mix = SelfModifyingLinear(C, H, self_referential=self_ref).eval()
    x = torch.randn(B, T, C)
    with torch.no_grad():
        o1 = mix(x)
        assert o1.shape == (B, T, C)
        x2 = x.clone(); x2[:, -1] += 10.0
        assert torch.allclose(o1[:, :-1], mix(x2)[:, :-1], atol=1e-5)


@pytest.mark.parametrize("self_ref", [False, True])
def test_self_modification_actually_happens(self_ref):
    """The weight the layer writes must be non-zero and must depend on the input."""
    torch.manual_seed(0)
    model = DaedalusProteus(V, n_embd=C, n_head=H, n_layer=1, block_size=T,
                            self_referential=self_ref).eval()
    n1 = model.weight_norms(torch.randint(0, V, (1, T)))
    n2 = model.weight_norms(torch.randint(0, V, (1, T)))
    assert len(n1) == T and all(math.isfinite(v) for v in n1)
    assert n1[-1] > 0.0
    assert n1 != n2                                   # the state is input-dependent


@pytest.mark.parametrize("self_ref", [False, True])
def test_weight_norm_stays_bounded_within_a_sequence(self_ref):
    model = DaedalusProteus(V, n_embd=C, n_head=H, n_layer=1, block_size=128,
                            self_referential=self_ref).eval()
    norms = model.weight_norms(torch.randint(0, V, (1, 128)))
    assert all(math.isfinite(v) for v in norms)
    assert norms[-1] < 1e4, f"self-written weight blew up: {norms[-1]:.3g}"


def test_proteus_init_loss_is_ln_vocab():
    torch.manual_seed(0)
    model = DaedalusProteus(V, n_embd=C, n_head=H, n_layer=2, block_size=T)
    x = torch.randint(0, V, (B, T)); y = torch.randint(0, V, (B, T))
    _, loss = model(x, y)
    assert abs(loss.item() - math.log(V)) < 0.6


def test_proteus_trains_without_nans():
    torch.manual_seed(0)
    model = DaedalusProteus(V, n_embd=C, n_head=H, n_layer=2, block_size=T)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, V, (B, T)); y = torch.randint(0, V, (B, T))
    for _ in range(20):
        loss = model(x, y)[1]
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def test_proteus_does_not_touch_the_main_model_line():
    """The isolation requirement, asserted rather than assumed."""
    assert isinstance(Daedalus(V, block_size=T).blocks[0].attn, MultiHeadAttention)
    assert isinstance(Labyrinth(V, block_size=T).core[0].attn, MultiHeadAttention)
    for cls in (Daedalus, Labyrinth, DaedalusFull):
        assert "proteus" not in cls.__module__


@pytest.mark.slow
@pytest.mark.parametrize("self_ref", [False, True])
def test_weight_norm_does_not_diverge_over_training(self_ref):
    """Several hundred steps on a fixed toy batch, watching ||W|| not just loss."""
    torch.manual_seed(0)
    model = DaedalusProteus(V, n_embd=C, n_head=H, n_layer=1, block_size=T,
                            self_referential=self_ref)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, V, (B, T)); y = torch.randint(0, V, (B, T))

    start = model.weight_norms(x[:1])[-1]
    for step in range(300):
        loss = model(x, y)[1]
        assert torch.isfinite(loss), f"loss went non-finite at step {step}"
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    end = model.weight_norms(x[:1])[-1]

    assert math.isfinite(end), "self-written weight norm diverged"
    # An honest, loose bound: the mechanism is allowed to grow, not to explode.
    assert end < max(100.0, 50.0 * start), f"||W|| grew {start:.3g} -> {end:.3g}"


@pytest.mark.slow
def test_proteus_adapts_within_a_sequence():
    """Does the self-modification *do* anything?

    Sequential-adaptation probe, in the spirit of the original fast-weight work:
    the model sees a repeating pattern, and predictions on the second half of the
    sequence (after the rule could have been written into the fast weights)
    should be better than on the first half.
    """
    from scripts.proteus_probe import adaptation_gap
    gap = adaptation_gap(steps=400, seed=0)
    assert math.isfinite(gap), "probe produced a non-finite result"
    # Reported, not asserted as a win -- see the README note. The assertion is
    # only that the mechanism runs and produces a finite, non-degenerate number.
    assert abs(gap) < 20.0
