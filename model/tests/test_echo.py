"""Isolation tests for Echo, loop self-distillation.

Two layers of testing, deliberately separated:

  * mechanism (fast, deterministic) -- the distillation term is zero when the
    passes agree, positive when they disagree, and gradient flows to the student
    only. These run in pytest.
  * the empirical claim (slow) -- "training with Echo makes the shallow end of
    the loop sweep better". That is a training comparison, so it lives in
    scripts/echo_sweep.py and is exercised here only as a small marked test.

    pytest -q tests/test_echo.py
    pytest -q -m slow tests/test_echo.py      # includes the training comparison
"""
import math

import pytest
import torch

from daedalus import Labyrinth, Ariadne, echo_loss, echo_step, echo_from_steps

B, T, C, V = 4, 16, 32, 256


def test_distill_is_zero_when_the_passes_agree():
    logits = torch.randn(B, T, V)
    for kind in ("kl", "mse"):
        assert echo_loss(logits, logits.clone(), kind).item() < 1e-6


def test_distill_is_positive_when_they_disagree():
    student, teacher = torch.randn(B, T, V), torch.randn(B, T, V)
    for kind in ("kl", "mse"):
        assert echo_loss(student, teacher, kind).item() > 0.0


def test_teacher_gets_no_gradient():
    """The teacher must be a stop-gradient, or the deep pass learns to be shallow."""
    student = torch.randn(B, T, V, requires_grad=True)
    teacher = torch.randn(B, T, V, requires_grad=True)
    echo_loss(student, teacher, "kl").backward()
    assert student.grad is not None and student.grad.abs().sum() > 0
    assert teacher.grad is None                       # detached inside echo_loss


def test_echo_step_off_by_default_equals_plain_ce():
    torch.manual_seed(0)
    lab = Labyrinth(V, n_embd=C, n_head=4, core_layers=2, n_loops=4, block_size=T).eval()
    x = torch.randint(0, V, (B, T)); y = torch.randint(0, V, (B, T))
    with torch.no_grad():
        plain = lab(x, y)[1]
        assert torch.allclose(echo_step(lab, x, y, lab.n_loops, weight=0.0), plain, atol=1e-6)


def test_echo_step_adds_a_positive_term():
    torch.manual_seed(0)
    lab = Labyrinth(V, n_embd=C, n_head=4, core_layers=2, n_loops=4, block_size=T)
    x = torch.randint(0, V, (B, T)); y = torch.randint(0, V, (B, T))
    loss = echo_step(lab, x, y, lab.n_loops, weight=1.0, generator=torch.Generator().manual_seed(0))
    loss.backward()
    assert torch.isfinite(loss) and loss.item() > 0
    assert all(torch.isfinite(p.grad).all() for p in lab.parameters() if p.grad is not None)


def test_echo_reuses_ariadnes_per_step_logits():
    """Ariadne already materialises per-loop logits -- the teacher costs nothing."""
    ari = Ariadne(V, n_embd=C, n_head=4, core_layers=2, max_loops=4, block_size=T)
    _, step_logits = ari(torch.randint(0, V, (B, T)))
    assert step_logits.shape == (4, B, T, V)
    term = echo_from_steps(step_logits, k=1, weight=0.5)
    assert torch.isfinite(term) and term.item() > 0
    assert echo_from_steps(step_logits, k=1, weight=0.0).item() == 0.0


@pytest.mark.slow
def test_echo_improves_the_shallow_end_of_the_loop_sweep():
    """The actual claim: Echo lowers loss at loop counts below the training depth.

    Small synthetic corpus so it runs in seconds; the same comparison at real
    scale is scripts/echo_sweep.py. Compares loop-1 val loss for two runs that
    differ only in --echo-weight.
    """
    from scripts.echo_sweep import train_one, sweep, synthetic_corpus

    train, val = synthetic_corpus(seed=0)
    off = sweep(train_one(train, echo_weight=0.0, steps=300, seed=0), val)
    on = sweep(train_one(train, echo_weight=1.0, steps=300, seed=0), val)
    assert on[1] < off[1], f"Echo did not help at loop 1: {on[1]:.4f} vs {off[1]:.4f}"
