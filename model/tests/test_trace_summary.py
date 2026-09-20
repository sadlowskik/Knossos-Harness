"""Tests for aggregating traces across runs.

This script is what anyone reads months after the console scrolled, so a field
it derives wrongly becomes a published claim. It already reconstructs
`unreachable` for traces predating that field; it now reconstructs `steps` the
same way, and the two reconstructions must not interfere with each other or
with the recorded values they stand in for.

    pytest -q tests/test_trace_summary.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.trace_summary import RunSummary, read_trace, summarise  # noqa: E402


def _trace(path: Path, header: dict, events=()) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(header) + "\n")
        for event in events:
            fh.write(json.dumps(event) + "\n")
    return path


# ------------------------------------------------------------- reconstruction


def test_a_recorded_step_count_is_used_as_is(tmp_path):
    path = _trace(tmp_path / "c.jsonl",
                  {"case": "c", "passed": True, "unreachable": False,
                   "steps": 7},
                  [{"kind": "step"}] * 3)
    header = read_trace(path)
    # Three `step` events, but the field says seven and the field wins.
    assert header["steps"] == 7
    assert header["steps_derived"] is False


def test_an_old_trace_has_its_steps_counted_from_the_events(tmp_path):
    path = _trace(tmp_path / "c.jsonl",
                  {"case": "c", "passed": True, "unreachable": False},
                  [{"kind": "step"}, {"kind": "tool"}, {"kind": "step"}])
    header = read_trace(path)
    assert header["steps"] == 2
    assert header["steps_derived"] is True


def test_reconstructing_steps_does_not_overwrite_a_recorded_unreachable(tmp_path):
    """The two fields have independent vintages.

    A trace can record `unreachable` and predate `steps`. Scanning the events
    for the step count must not also recompute `unreachable` from those same
    events -- a case that recovered from a refusal and went on to pass is
    recorded reachable, and rederiving it would flip a real result.
    """
    path = _trace(tmp_path / "c.jsonl",
                  {"case": "c", "passed": True, "unreachable": False,
                   "api_errors": 2},
                  [{"kind": "text", "text": "Request failed: HTTP 429"},
                   {"kind": "step"}])
    header = read_trace(path)
    assert header["steps"] == 1          # reconstructed
    assert header["unreachable"] is False  # recorded, and left alone
    assert header["api_errors"] == 2


def test_a_corrupt_trace_is_skipped_rather_than_guessed_at(tmp_path):
    path = tmp_path / "c.jsonl"
    path.write_text("not json\n", encoding="utf-8")
    assert read_trace(path) is None


# ------------------------------------------------------------- the steps mean


def test_an_external_harness_run_reports_no_mean_rather_than_zero(tmp_path):
    """A header-only trace is what `--harness-cmd` writes.

    Zero steps across every case means the count was unavailable, not that the
    harness finished instantly. A zero would sort as the most efficient run in
    the table.
    """
    run = tmp_path / "external"
    for i in range(3):
        _trace(run / f"c{i}.jsonl",
               {"case": f"c{i}", "passed": True, "unreachable": False,
                "steps": 0})
    summary, = summarise(tmp_path)
    assert summary.mean_steps is None
    assert summary.steps_missing == 3


def test_only_solved_cases_reach_the_mean():
    run = RunSummary("r")
    run.add("a", passed=True, unreachable=False, halt="done", tamper=False,
            derived=False, steps=5)
    run.add("b", passed=False, unreachable=False, halt="budget_exhausted",
            tamper=False, derived=False, steps=20)
    assert run.mean_steps == 5.0


def test_an_unreachable_case_contributes_no_steps():
    """It measured nothing on every other axis; this one is no different."""
    run = RunSummary("r")
    run.add("a", passed=True, unreachable=False, halt="done", tamper=False,
            derived=False, steps=8)
    run.add("b", passed=False, unreachable=True, halt="stuck", tamper=False,
            derived=False, steps=0)
    assert run.mean_steps == 8.0
    assert run.steps_missing == 0


def test_a_run_marks_its_mean_as_derived_only_when_it_is(tmp_path):
    run = RunSummary("r")
    run.add("a", passed=True, unreachable=False, halt="done", tamper=False,
            derived=False, steps=5, steps_derived=False)
    assert run.steps_derived is False
    run.add("b", passed=True, unreachable=False, halt="done", tamper=False,
            derived=False, steps=9, steps_derived=True)
    assert run.steps_derived is True
