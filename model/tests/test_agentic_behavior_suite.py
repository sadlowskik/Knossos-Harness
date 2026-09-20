"""Calibration for the non-bugfix agentic behavior suite."""

from pathlib import Path

from knossos.codeval import grade, load_cases, materialise, run_tests


SUITE = Path(__file__).parents[1] / "fixtures" / "agentic_behavior_suite.json"
REPO = Path(__file__).parents[2]
EXPERIMENT = REPO / "knossos-rs" / "cases" / "ox-alpha-experiment.json"


def _cases():
    return load_cases(SUITE)


def test_suite_covers_the_missing_behavior_archetypes():
    cases = _cases()
    assert len(cases) == 6
    assert {case.expected_action for case in cases} == {
        "edit", "edit_preserve", "no_op", "clarify"
    }
    assert {case.kind for case in cases} >= {
        "refactor", "no_op", "clarification", "config_change",
        "dependency_repair", "multi_step_feature",
    }


def test_fixture_starting_states_are_truthful(tmp_path):
    for case in _cases():
        root = tmp_path / case.id
        materialise(case, root)
        assert all(run_tests(root, case.pass_to_pass).values()), case.id
        assert not any(run_tests(root, case.fail_to_pass).values()), case.id
        assert all(not (root / path).exists() for path in case.held_out), case.id


def test_no_op_and_clarification_can_pass_without_a_diff(tmp_path):
    by_action = {case.expected_action: case for case in _cases()}

    no_op = by_action["no_op"]
    no_op_root = tmp_path / "no-op"
    materialise(no_op, no_op_root)
    assert grade(no_op, no_op_root, False)["passed"]

    clarify = by_action["clarify"]
    clarify_root = tmp_path / "clarify"
    materialise(clarify, clarify_root)
    assert grade(
        clarify, clarify_root, False,
        "Which retry dimension and workload should define ‘better’?",
    )["passed"]


def test_action_contracts_reject_a_plausible_but_wrong_trajectory(tmp_path):
    by_action = {case.expected_action: case for case in _cases()}

    refactor = by_action["edit_preserve"]
    refactor_root = tmp_path / "refactor"
    materialise(refactor, refactor_root)
    verdict = grade(refactor, refactor_root, False)
    assert not verdict["passed"]
    assert not verdict["action_passed"]

    no_op = by_action["no_op"]
    no_op_root = tmp_path / "invented-diff"
    materialise(no_op, no_op_root)
    (no_op_root / "pkg" / "seq.py").write_text(
        "def stable_unique(items):\n    return sorted(set(items))\n",
        encoding="utf-8",
    )
    verdict = grade(no_op, no_op_root, False)
    assert not verdict["passed"]
    assert not verdict["action_passed"]


def test_ox_experiment_is_capped_reproducible_and_points_to_real_suites():
    manifest = __import__("json").loads(EXPERIMENT.read_text(encoding="utf-8"))
    assert manifest["schema"] == "knossos-experiment/v1"
    assert manifest["controls"]["concurrency"] == 1
    assert manifest["controls"]["max_steps"] <= 8
    assert manifest["controls"]["max_output_tokens"] <= 2048
    assert manifest["controls"]["collect_exchanges"] is True
    assert {arm["id"] for arm in manifest["arms"]} == {
        "knossos-full", "knossos-no-context", "knossos-no-memory",
        "knossos-no-compaction",
    }
    assert manifest["controls"]["max_requests"] > 0
    assert manifest["controls"]["max_total_tokens"] > 0
    assert manifest["controls"]["checkpoint"]
    assert set(manifest["failure_cause_groups"]) == {
        "model", "harness", "infrastructure", "ambiguous",
    }
    assert set(manifest["behavior_coverage"]) == {
        "no_op", "clarification", "boundary", "recovery",
    }
    for key in ("canary_cases", "full_cases", "behavior_cases"):
        assert (REPO / manifest["suite"][key]).is_file(), key
