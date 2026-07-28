"""Tests for grading a third-party coding agent.

`run_case` reads the agent through `getattr`, and `CaseResult` stores plain
integers rather than an engine type -- deliberately, so a foreign harness can be
graded by the same instrument. These pin what that adapter observes, because a
harness-vs-harness number is only worth quoting if the adapter cannot be fooled
by a tool that merely claims to have worked.

    pytest -q tests/test_external_harness.py
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.coding_eval import (ExternalHarness, _snapshot,  # noqa: E402
                                 external_agent)

PY = sys.executable


@pytest.fixture()
def root(tmp_path):
    (tmp_path / "m.py").write_text("value = 1\n", encoding="utf-8")
    return tmp_path


def _run(root, code, timeout=60):
    return ExternalHarness([PY, "-c", code, "{prompt}"], root, timeout).run("task")


# ------------------------------------------------------------ what it measures

def test_changes_are_measured_not_taken_on_trust(root):
    """A harness cannot inflate its diff by claiming one — the tree is hashed."""
    outcome = _run(root, "open('new.py','w').write('x')")
    assert outcome.changed == ["new.py"]


def test_an_edit_to_an_existing_file_counts(root):
    outcome = _run(root, "open('m.py','w').write('value = 2')")
    assert outcome.changed == ["m.py"]


def test_a_harness_that_does_nothing_changes_nothing(root):
    """Exit 0 with no work is the false-pass case the calibration relies on."""
    outcome = _run(root, "pass")
    assert outcome.changed == []
    assert outcome.succeeded, "it claimed success — that is the point"


def test_a_nonzero_exit_is_not_a_claim_of_success(root):
    outcome = _run(root, "import sys; sys.exit(1)")
    assert not outcome.succeeded
    assert outcome.halt.value == "stuck"


def test_running_the_tests_is_not_counted_as_work(root):
    """A harness that runs pytest leaves `.pytest_cache`; that is not a fix."""
    outcome = _run(root, "import os; os.makedirs('__pycache__', exist_ok=True);"
                         " open('__pycache__/x.pyc','w').write('junk')")
    assert outcome.changed == []


def test_a_timeout_keeps_whatever_was_written(root):
    """The foreign equivalent of BUDGET_EXHAUSTED: no claim, but graded anyway."""
    outcome = _run(root, "import time; open('partial.py','w').write('x');"
                         " time.sleep(30)", timeout=3)
    assert not outcome.succeeded
    assert outcome.halt.value == "budget_exhausted"
    assert outcome.changed == ["partial.py"]


def test_a_turn_count_is_read_when_the_harness_prints_one(root):
    outcome = _run(root, "print('{\"num_turns\": 7}')")
    assert outcome.steps_used == 7


def test_steps_stay_zero_rather_than_being_guessed(root):
    """Blank is honest; a fabricated step count would be quoted as real."""
    assert _run(root, "print('done')").steps_used == 0


def test_a_missing_binary_is_an_error_not_a_silent_zero(root):
    harness = ExternalHarness(["definitely-not-a-real-binary-xyz", "{prompt}"],
                              root, 10)
    with pytest.raises(RuntimeError, match="could not run"):
        harness.run("task")


# ------------------------------------------------------------- the CLI surface

def test_the_prompt_reaches_the_command(root):
    """`{prompt}` is substituted per-token, so it survives spaces intact."""
    made = external_agent(f'"{PY}" -c '
                          '"import sys;open(\'got.txt\',\'w\').write(sys.argv[1])"'
                          ' {prompt}', 60)
    made(root).run("fix the bug in m.py")
    assert (root / "got.txt").read_text(encoding="utf-8") == "fix the bug in m.py"


def test_a_template_without_a_prompt_placeholder_is_rejected():
    with pytest.raises(SystemExit, match="prompt"):
        external_agent("claude -p", 60)


def test_an_empty_template_is_rejected():
    with pytest.raises(SystemExit):
        external_agent("   ", 60)


def test_the_command_is_never_handed_to_a_shell(root):
    """Same rule as the tool layer: tokenised, never interpreted."""
    made = external_agent(f'"{PY}" -c "print(1)" {{prompt}}', 60)
    outcome = made(root).run("x && touch pwned.txt")
    assert not (root / "pwned.txt").exists()
    assert outcome.changed == []


def test_snapshot_skips_directories_that_are_not_the_work(tmp_path):
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("ref: x", encoding="utf-8")
    (tmp_path / "keep.py").write_text("x = 1", encoding="utf-8")
    assert list(_snapshot(tmp_path)) == ["keep.py"]
