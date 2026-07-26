"""Isolation tests for the halting policy.

Named `test_ariadne_halting` to keep it distinct from any test of the
model-level PonderNet head -- these are different mechanisms that share a name
on purpose, and conflating them in the test suite would undo that.

The load-bearing claims:

  1. The loop always terminates. The ceiling is the only thing guaranteeing it.
  2. Success is checked before exhaustion, so verifying on the last permitted
     step reports DONE rather than BUDGET_EXHAUSTED.
  3. Pressure escalates rather than appearing once -- the β=0.01 collapse says a
     weak, constant penalty is the same as no penalty.

No torch, no network.

    pytest -q tests/test_ariadne_halting.py
"""
import pytest

from harness.ariadne import Ariadne, Halt, StepOutcome


def passed():
    return StepOutcome(tool_calls=1, files_changed=1, verdict_passed=True)


def worked():
    return StepOutcome(tool_calls=2, files_changed=1)


def nothing():
    return StepOutcome()


# ------------------------------------------------------------------ stopping

def test_verification_passing_ends_the_run():
    assert Ariadne().assess(1, passed()) is Halt.DONE


def test_productive_steps_continue():
    assert Ariadne().assess(3, worked()) is Halt.CONTINUE


def test_the_ceiling_forces_a_halt():
    """Without this the loop has no termination guarantee at all."""
    a = Ariadne(max_steps=5, target_steps=3)
    assert a.assess(5, worked()) is Halt.BUDGET_EXHAUSTED
    assert a.assess(99, worked()) is Halt.BUDGET_EXHAUSTED


def test_success_on_the_final_step_reports_done_not_exhausted():
    a = Ariadne(max_steps=5, target_steps=3)
    assert a.assess(5, passed()) is Halt.DONE


def test_repeated_noops_are_stuck():
    a = Ariadne()
    assert a.assess(4, nothing(), consecutive_noops=1) is Halt.CONTINUE
    assert a.assess(4, nothing(), consecutive_noops=2) is Halt.STUCK


def test_a_failed_verdict_does_not_end_the_run():
    """Failing verification is a reason to keep working, not to stop."""
    failed = StepOutcome(tool_calls=1, files_changed=1, verdict_passed=False)
    assert Ariadne().assess(2, failed) is Halt.CONTINUE


def test_only_continue_is_non_terminal():
    assert not Halt.CONTINUE.is_terminal
    for halt in (Halt.DONE, Halt.STUCK, Halt.BUDGET_EXHAUSTED):
        assert halt.is_terminal


# ------------------------------------------------------------------ pressure

def test_no_pressure_before_the_target():
    a = Ariadne(max_steps=12, target_steps=6)
    assert a.pressure(1) is None
    assert a.pressure(6) is None


def test_pressure_escalates_toward_the_ceiling():
    """A constant nudge is what β=0.01 was; it collapsed to max depth."""
    a = Ariadne(max_steps=12, target_steps=6)
    mid, late, last = a.pressure(7), a.pressure(10), a.pressure(12)

    assert mid is not None and "Prefer finishing" in mid
    assert late is not None and "Do not begin anything new" in late
    assert last is not None and "final step" in last
    assert mid != late != last, "pressure must change, not repeat"


def test_pressure_names_the_actual_step_numbers():
    a = Ariadne(max_steps=8, target_steps=4)
    text = a.pressure(5)
    assert "5 of 8" in text


# -------------------------------------------------------------------- config

def test_a_target_above_the_ceiling_is_clamped():
    """Otherwise pressure silently never fires."""
    a = Ariadne(max_steps=4, target_steps=99)
    assert a.target_steps == 4
    assert a.pressure(4) is None


def test_a_zero_ceiling_is_refused():
    with pytest.raises(ValueError):
        Ariadne(max_steps=0)


# --------------------------------------------------------------- noop signal

def test_noop_detection_is_about_evidence_not_opinion():
    assert nothing().is_noop
    assert not worked().is_noop
    # A tool call that changed nothing still counts as work -- a read is progress.
    assert not StepOutcome(tool_calls=1, files_changed=0).is_noop
