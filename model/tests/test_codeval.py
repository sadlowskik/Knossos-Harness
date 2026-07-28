"""Tests for the coding evaluation -- the grader, not the agent.

A benchmark is a measuring instrument, and an uncalibrated one produces numbers
that are worse than no numbers because they get quoted. The claims here are
about the instrument:

  1. A correct patch scores; an absent one does not.
  2. **Editing the tests buys nothing.** The fixture's tests are restored before
     grading, so the obvious cheat is not merely discouraged, it is inert.
  3. Breaking a passing test fails the case even when the bug is fixed.
  4. A crashing agent is a failed case, not a failed suite.

`scripts/coding_eval.py --agent oracle|lazy|vandal` is the end-to-end version
of the same argument, run against the real `Talos`.

    pytest -q tests/test_codeval.py
"""
import json
from pathlib import Path

import pytest

from knossos.codeval import (CODING_CASES, CaseResult, CodingCase, grade,
                             materialise, restore_tests, run_case, run_tests)


@pytest.fixture()
def case():
    return CodingCase(
        id="demo",
        prompt="make add() work",
        files={
            "pkg/__init__.py": "",
            "pkg/m.py": "def add(a, b):\n    return a - b\n",
            "tests/test_m.py": (
                "from pkg.m import add\n\n\n"
                "def test_adds():\n    assert add(1, 2) == 3\n\n\n"
                "def test_zero():\n    assert add(0, 0) == 0\n"),
        },
        fail_to_pass=["tests/test_m.py::test_adds"],
        pass_to_pass=["tests/test_m.py::test_zero"],
    )


class Agent:
    """Applies a fixed set of file writes, then reports whatever it is told."""

    def __init__(self, writes=(), succeeded=True, boom=False):
        self.writes = dict(writes)
        self._succeeded = succeeded
        self._boom = boom
        self.root = None

    def run(self, prompt):
        if self._boom:
            raise RuntimeError("the engine fell over")
        for rel, content in self.writes.items():
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        outcome = type("Outcome", (), {})()
        outcome.succeeded = self._succeeded
        outcome.halt = type("H", (), {"value": "done"})()
        outcome.steps_used = 1
        outcome.changed = list(self.writes)
        return outcome


def factory(agent):
    def make(root: Path):
        agent.root = root
        return agent
    return make


FIXED = "def add(a, b):\n    return a + b\n"


# ------------------------------------------------------------- the happy path

def test_a_correct_patch_passes(case, tmp_path):
    result = run_case(case, factory(Agent({"pkg/m.py": FIXED})), tmp_path / "w")

    assert result.passed
    assert result.fixed == 1 and result.kept == 1


def test_doing_nothing_fails(case, tmp_path):
    result = run_case(case, factory(Agent()), tmp_path / "w")

    assert not result.passed
    assert result.fixed == 0


# ------------------------------------------------------------- anti-gaming

def test_editing_the_test_buys_nothing(case, tmp_path):
    """The whole reason `restore_tests` exists."""
    vandal = Agent({"tests/test_m.py": "def test_adds():\n    assert True\n"})

    result = run_case(case, factory(vandal), tmp_path / "w")

    assert not result.passed
    assert result.tamper, "the attempt is recorded even though it failed"


def test_deleting_the_test_buys_nothing(case, tmp_path):
    class Deleter(Agent):
        def run(self, prompt):
            (self.root / "tests" / "test_m.py").unlink()
            return super().run(prompt)

    result = run_case(case, factory(Deleter()), tmp_path / "w")

    assert not result.passed
    assert result.tamper


def test_the_restored_tests_are_the_fixtures_own(case, tmp_path):
    root = tmp_path / "w"
    materialise(case, root)
    (root / "tests" / "test_m.py").write_text("junk", encoding="utf-8")

    tampered = restore_tests(case, root)

    assert tampered
    assert (root / "tests" / "test_m.py").read_text(encoding="utf-8") == \
        case.files["tests/test_m.py"]


def test_restoring_an_untouched_suite_reports_no_tamper(case, tmp_path):
    root = tmp_path / "w"
    materialise(case, root)

    assert restore_tests(case, root) is False


# --------------------------------------------------------- breaking things

def test_fixing_the_bug_but_breaking_another_test_fails(case, tmp_path):
    """A change that fixes the bug and breaks the suite is not partial success."""
    sneaky = Agent({"pkg/m.py": "def add(a, b):\n"
                                "    if a == 1 and b == 2:\n        return 3\n"
                                "    return None\n"})

    result = run_case(case, factory(sneaky), tmp_path / "w")

    assert result.fixed == 1, "the named test does pass"
    assert result.kept == 0, "but the other one no longer does"
    assert not result.passed


def test_a_provider_refusal_is_counted_separately_from_a_bad_answer(case, tmp_path):
    """A starved run and an incapable model both score zero.

    Measured against a free tier capped at 8 000 tokens per minute: every case
    returned 0, and without this the run would have read as a capability result
    for two frontier models rather than as an exhausted quota.
    """
    class Refused:
        def run(self, prompt, on_event=None):
            if on_event:
                for _ in range(3):
                    on_event(type("E", (), {
                        "kind": "text", "step": 1,
                        "text": "Request failed: HTTP 429 — rate limit reached.",
                        "call": None, "result": None, "verdict": None})())
            outcome = type("Outcome", (), {})()
            outcome.succeeded = False
            outcome.halt = type("H", (), {"value": "stuck"})()
            outcome.steps_used = 3
            outcome.changed = []
            return outcome

    result = run_case(case, lambda root: Refused(), tmp_path / "w")

    assert not result.passed
    assert result.api_errors == 3


def test_the_suite_stops_once_the_provider_stops_answering(tmp_path):
    """Every further case would pay full retry cost to produce the same zero.

    Measured at 480s per case against an exhausted free tier -- 82.7 minutes to
    learn nothing that the second case had not already established.
    """
    from knossos.codeval import CODING_CASES, run_suite

    attempted = []

    class Refused:
        def __init__(self, case_id):
            attempted.append(case_id)

        def run(self, prompt, on_event=None):
            if on_event:
                on_event(type("E", (), {
                    "kind": "text", "step": 1,
                    "text": "Request failed: HTTP 429 — rate limit reached.",
                    "call": None, "result": None, "verdict": None})())
            outcome = type("Outcome", (), {})()
            outcome.succeeded = False
            outcome.halt = type("H", (), {"value": "stuck"})()
            outcome.steps_used = 1
            outcome.changed = []
            return outcome

    results = run_suite(lambda root: Refused(root.name), tmp_path,
                        cases=CODING_CASES[:5])

    assert len(results) == 2, "it kept running against a provider that was refusing"
    assert len(attempted) == 2


def test_a_normal_failure_records_no_api_errors(case, tmp_path):
    result = run_case(case, factory(Agent()), tmp_path / "w")

    assert not result.passed
    assert result.api_errors == 0


def test_a_crashing_agent_is_a_failed_case_not_a_failed_suite(case, tmp_path):
    result = run_case(case, factory(Agent(boom=True)), tmp_path / "w")

    assert not result.passed
    assert "the engine fell over" in result.error


# ------------------------------------------------------------ honesty tracking

def test_honest_records_whether_the_harness_was_right(case, tmp_path):
    liar = run_case(case, factory(Agent(succeeded=True)), tmp_path / "a")
    truthful = run_case(case, factory(Agent({"pkg/m.py": FIXED}, succeeded=True)),
                        tmp_path / "b")

    assert not liar.honest, "claimed success having changed nothing"
    assert truthful.honest


# ---------------------------------------------------------------- the fixtures

def test_every_case_starts_out_failing(tmp_path):
    """A `fail_to_pass` that already passes measures nothing.

    Run against the untouched fixture, every named test must be red -- otherwise
    the case would score for an agent that did nothing at all.
    """
    for case in CODING_CASES:
        root = tmp_path / case.id
        materialise(case, root)
        results = run_tests(root, case.fail_to_pass)
        assert not any(results.values()), \
            f"{case.id}: {[n for n, ok in results.items() if ok]} already passed"


def test_every_pass_to_pass_starts_out_green(tmp_path):
    """Otherwise the case is unwinnable and the score is a lie in the other
    direction."""
    for case in CODING_CASES:
        root = tmp_path / case.id
        materialise(case, root)
        results = run_tests(root, case.pass_to_pass)
        assert all(results.values()), \
            f"{case.id}: {[n for n, ok in results.items() if not ok]} already failed"


def test_case_ids_are_unique():
    ids = [c.id for c in CODING_CASES]
    assert len(ids) == len(set(ids))


def test_every_case_names_at_least_one_thing_to_fix():
    for case in CODING_CASES:
        assert case.fail_to_pass, f"{case.id} cannot be passed or failed"


# ------------------------------------------------- provider outage vs. failure
#
# A case the provider refused measured nothing. It nonetheless leaves exactly
# the fingerprint of an incapable model -- `passed=False`, and `halt="stuck"`,
# because an agent handed an error string in place of a reply makes no tool
# calls and trips the no-op detector. These tests pin the distinction, in the
# durable artefact as well as in memory: the console warning that says "this is
# your quota, not the model" scrolls away, and the trace file is what anyone
# aggregates months later.


def test_a_refused_case_is_unreachable_not_a_failure():
    result = CaseResult(case_id="x", kind="bug", passed=False, api_errors=3)
    assert result.unreachable


def test_a_case_that_recovered_from_a_refusal_still_counts():
    """One rate limit, a retry, then a solve. That measured the model."""
    result = CaseResult(case_id="x", kind="bug", passed=True, api_errors=1)
    assert not result.unreachable


def test_an_ordinary_failure_is_not_excused_as_unreachable():
    result = CaseResult(case_id="x", kind="bug", passed=False, api_errors=0)
    assert not result.unreachable


def test_the_trace_header_records_what_the_console_warning_says(tmp_path, case):
    """The console is ephemeral; this file is the evidence."""
    from knossos.codeval import _write_trace
    path = tmp_path / "t.jsonl"
    result = CaseResult(case_id="x", kind="bug", passed=False, api_errors=2)
    _write_trace(path, case, result, [])
    header = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert header["api_errors"] == 2
    assert header["unreachable"] is True


def test_unreachable_cases_leave_the_denominator():
    """0/2 measured, not 0/4 -- the two refused cases are not evidence."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.coding_eval import report

    results = [
        CaseResult(case_id="a", kind="bug", passed=True, harness_said_done=True),
        CaseResult(case_id="b", kind="bug", passed=False),
        CaseResult(case_id="c", kind="bug", passed=False, api_errors=4),
        CaseResult(case_id="d", kind="bug", passed=False, api_errors=4),
    ]
    tiers = {c.case_id: "core" for c in results}
    assert report(results, tiers) == 1


def test_honesty_is_not_inflated_by_cases_nobody_attempted():
    """An unreachable case is trivially `honest` -- true, and meaningless."""
    refused = CaseResult(case_id="c", kind="bug", passed=False, api_errors=4)
    assert refused.honest          # it did not claim a task it never ran
    assert refused.unreachable     # ...which is why it must not be counted


# --------------------------------------------------------- external suites
#
# The built-in cases are the calibration set and they are saturated: every model
# that can drive the loop scores 5/5 on the core tier. A saturated benchmark
# measures a floor, so comparing harness changes on anything harder means adding
# cases -- and that must not require editing this package.
#
# Everything that makes the grader trustworthy applies unchanged, because a
# loaded case is the same `CodingCase` the built-in set produces.


def _write_suite(tmp_path, cases):
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    return path


def _minimal(case_id="ext-1"):
    return {"id": case_id, "prompt": "fix add()",
            "files": {"pkg/m.py": "def add(a, b):\n    return a - b\n",
                      "tests/test_m.py": "from pkg.m import add\n\n"
                                         "def test_add():\n    assert add(1, 2) == 3\n"},
            "fail_to_pass": ["tests/test_m.py::test_add"]}


def test_an_external_suite_loads_as_ordinary_cases(tmp_path):
    from knossos.codeval import load_cases
    cases = load_cases(_write_suite(tmp_path, [_minimal()]))
    assert len(cases) == 1
    assert isinstance(cases[0], CodingCase)
    assert cases[0].fail_to_pass == ["tests/test_m.py::test_add"]
    # and the anti-gaming property survives the trip
    assert cases[0].test_files == ["tests/test_m.py"]


def test_a_loaded_case_is_gradeable_end_to_end(tmp_path):
    """The point of reusing `CodingCase`: nothing about grading changes."""
    from knossos.codeval import load_cases, materialise, run_tests
    case = load_cases(_write_suite(tmp_path, [_minimal()]))[0]
    root = tmp_path / "work"
    materialise(case, root)
    assert run_tests(root, case.fail_to_pass) == {
        "tests/test_m.py::test_add": False}, "must fail before the fix"


def test_a_suite_may_be_wrapped_in_an_object(tmp_path):
    from knossos.codeval import load_cases
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"cases": [_minimal()]}), encoding="utf-8")
    assert len(load_cases(path)) == 1


def test_a_case_with_nothing_to_fix_is_rejected(tmp_path):
    """It cannot be passed or failed, so it is a free point in the denominator."""
    from knossos.codeval import load_cases
    broken = _minimal()
    broken["fail_to_pass"] = []
    with pytest.raises(ValueError, match="fail_to_pass"):
        load_cases(_write_suite(tmp_path, [broken]))


def test_duplicate_ids_are_rejected(tmp_path):
    """Two cases with one id silently become one row in every report."""
    from knossos.codeval import load_cases
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(_write_suite(tmp_path, [_minimal(), _minimal()]))


def test_a_malformed_suite_raises_rather_than_dropping_cases(tmp_path):
    """A rate over a denominator nobody chose is worse than a crash."""
    from knossos.codeval import load_cases
    for bad in ([], [{"id": "x"}], ["not an object"],
                [{**_minimal(), "files": ["not", "a", "map"]}]):
        with pytest.raises(ValueError):
            load_cases(_write_suite(tmp_path, bad))


def test_a_run_that_never_made_a_request_is_unreachable():
    """A missing key raises before the first turn, so `api_errors` stays zero.

    Found by running the suite with no `ANTHROPIC_API_KEY`: every case reported
    a clean capability zero for a model that was never called.
    """
    result = CaseResult(case_id="x", kind="bug", passed=False, steps_used=0,
                        error="RuntimeError: ANTHROPIC_API_KEY is not set.")
    assert result.unreachable


def test_a_crash_after_real_turns_is_still_a_failure():
    """It measured the turns it took; only a run that never started measured nothing."""
    result = CaseResult(case_id="x", kind="bug", passed=False, steps_used=4,
                        error="RuntimeError: something broke mid-run")
    assert not result.unreachable


def test_a_clean_zero_step_failure_is_not_excused():
    """No error means the agent simply declined to act — that is a real result."""
    result = CaseResult(case_id="x", kind="bug", passed=False, steps_used=0)
    assert not result.unreachable
