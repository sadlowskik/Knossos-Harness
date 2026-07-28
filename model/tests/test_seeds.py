"""Tests for the seed-aggregation used by every acceptance gate.

This module is small, but it is what every reported number now passes through,
and PLAN.md's rule 4 exists because a single-seed figure once overstated an
effect roughly twofold. So the properties that matter are:

  1. A lost run is *reported*, never silently dropped. An exclusion that
     correlates with the arm is a confound, not noise (PLAN.md §1.9).
  2. One arm blowing up does not take the rest of the sweep with it.
  3. A difference smaller than the seed-to-seed spread is described as not
     measured, rather than as a result.

    pytest -q tests/test_seeds.py
"""
import pytest

from scripts.seeds import Arm, delta, run_arm, table


# ----------------------------------------------------------------- aggregation

def test_mean_and_spread_over_seeds():
    arm = run_arm("x", lambda s: float(s), [0, 1, 2, 3, 4], progress=False)

    assert arm.n == 5
    assert arm.mean == pytest.approx(2.0)
    assert arm.stdev == pytest.approx(1.5811, abs=1e-4)


def test_a_single_sample_has_no_spread_rather_than_raising():
    """`statistics.stdev` needs two points; one seed must still report."""
    arm = run_arm("x", lambda s: 1.0, [0], progress=False)

    assert arm.n == 1
    assert arm.stdev == 0.0
    assert "n=1" in arm.summary()


def test_an_arm_with_no_successful_runs_says_so():
    arm = run_arm("x", lambda s: None, [0, 1], progress=False)

    assert arm.n == 0
    assert arm.summary() == "no successful runs"


# ------------------------------------------------------------------ exclusions

def test_a_failed_run_is_recorded_not_dropped():
    """§1.9's lesson: a lost sample that correlates with the arm invalidates
    the comparison rather than merely thinning it."""
    def explodes(seed):
        if seed == 1:
            raise RuntimeError("diverged")
        return 1.0

    arm = run_arm("x", explodes, [0, 1, 2], progress=False)

    assert arm.n == 2
    assert 1 in arm.excluded
    assert "diverged" in arm.excluded[1]


def test_an_exploding_arm_does_not_end_the_sweep():
    arm = run_arm("x", lambda s: 1 / 0, [0, 1], progress=False)

    assert arm.n == 0
    assert set(arm.excluded) == {0, 1}


def test_exclusions_appear_in_the_table():
    arm = run_arm("x", lambda s: None if s else 1.0, [0, 1], progress=False)

    rendered = table([arm])

    assert "EXCLUDED" in rendered
    assert "seed 1" in rendered


def test_the_table_shows_every_seed_not_just_the_mean():
    """A mean hiding one wild seed is the failure this module exists to prevent."""
    arm = run_arm("x", lambda s: float(s), [0, 1, 2], progress=False)

    rendered = table([arm], "losses")

    for value in ("0.0000", "1.0000", "2.0000"):
        assert value in rendered


# ---------------------------------------------------------------------- deltas

def test_a_difference_inside_the_spread_is_not_called_a_result():
    noisy_a = Arm("a", [0.0, 2.0])          # mean 1.0, stdev ~1.41
    noisy_b = Arm("b", [0.1, 2.1])          # mean 1.1, same spread

    assert "not measured" in delta(noisy_a, noisy_b)


def test_a_difference_beyond_the_spread_is_reported_as_one():
    tight_a = Arm("a", [1.00, 1.01, 0.99])
    tight_b = Arm("b", [5.00, 5.01, 4.99])

    message = delta(tight_a, tight_b)

    assert "larger than the spread" in message
    assert "+4.0" in message


def test_comparing_against_an_empty_arm_is_refused():
    assert "not enough successful runs" in delta(Arm("a", [1.0]), Arm("b", []))
