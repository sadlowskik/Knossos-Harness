"""Tests for grading a third-party coding agent.

`run_case` reads the agent through `getattr`, and `CaseResult` stores plain
integers rather than an engine type -- deliberately, so a foreign harness can be
graded by the same instrument. These pin what that adapter observes, because a
harness-vs-harness number is only worth quoting if the adapter cannot be fooled
by a tool that merely claims to have worked.

    pytest -q tests/test_external_harness.py
"""
import os
import sys
import time
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


# ------------------------------------------------------------- the control arm
#
# Every coding-eval number before this was model x harness with no way to
# separate them. `knossos.eval` has had a control from the start and the README
# calls it "the load-bearing part"; this is the same discipline applied to the
# weaker of the two instruments.


def test_the_baseline_gets_one_step_and_no_verifier(root):
    """The delta it measures is exactly iteration plus verification."""
    from knossos.talos import accept_everything
    from scripts.coding_eval import baseline_agent

    agent = baseline_agent(engine=None)(root)
    assert agent.ariadne.max_steps == 1
    assert agent.verify is accept_everything


def test_the_baseline_never_claims_to_be_verified(root):
    """Its `honest` is a tautology and must be labelled as one."""
    from scripts.coding_eval import baseline_agent
    assert baseline_agent(engine=None)(root).claim_source == "unverified"


def test_a_foreign_harness_claim_is_recorded_as_an_exit_code(root):
    """Not a self-assessment: most CLIs exit 0 unless they crash."""
    outcome = _run(root, "print('done')")
    assert outcome.claim_source == "exit_code"
    assert outcome.budget_kind == "wall_clock"


def test_the_two_arms_do_not_share_a_budget_unit(root):
    """A steps ceiling and a wall clock are different constraints.

    Comparing them without saying so is how a harness that was killed at 300
    seconds is read as having given up rather than as having been stopped.
    """
    from knossos.talos import Outcome
    assert Outcome.budget_kind == "steps"
    assert _run(root, "print('x')").budget_kind == "wall_clock"


# -------------------------------------------------- the per-case engine reset
#
# `_fresh_case` reaches `restore_limits` through `getattr(engine, ...)`, which
# is the right shape -- `AgentFactory` is "anything with `.run(prompt)`" and a
# scripted agent has no engine. But a duck-typed call cannot tell "this agent
# has no engine" from "the method was renamed and nothing calls it any more".
# The second silently restores the bug it was written to fix: every shrink path
# is one-way, so the suite goes back to measuring a descending staircase of
# allowances, and the only symptom is a score that quietly depends on case
# order. These tests exist so that failure is loud instead.


class _SpyEngine:
    """Records the reset without needing a provider. Talos never calls it here."""

    name = "spy"

    def __init__(self) -> None:
        self.restores = 0

    def restore_limits(self) -> bool:
        self.restores += 1
        return True


def test_the_engine_still_offers_the_reset_the_eval_reaches_for():
    """A contract test, because the caller's `getattr` cannot raise.

    If `restore_limits` is renamed, `_fresh_case` degrades to a no-op and every
    test above still passes -- they use a stub or `engine=None`. This is the one
    assertion that ties the eval's expectation to the real engine class.
    """
    from knossos.engine import OpenAICompatEngine

    assert callable(getattr(OpenAICompatEngine, "restore_limits", None)), \
        "coding_eval._fresh_case calls this by name; renaming it silently " \
        "reinstates the monotonically-degrading allowance bug"


def test_each_case_starts_from_a_restored_allowance(root):
    """`make_agent(root)` is the case boundary, so the reset belongs there."""
    from scripts.coding_eval import live_agent

    engine = _SpyEngine()
    make = live_agent(engine, max_steps=4, target_steps=2)
    make(root)
    make(root)

    assert engine.restores == 2, "once per case, not once per suite"


def test_the_control_arm_is_reset_too(root):
    """Otherwise the delta between the arms measures throttling history."""
    from scripts.coding_eval import baseline_agent

    engine = _SpyEngine()
    baseline_agent(engine)(root)

    assert engine.restores == 1


def test_an_agent_without_an_engine_is_left_alone(root):
    """The duck-typing has to keep working: scripted agents have no engine."""
    from scripts.coding_eval import _fresh_case

    _fresh_case(None)              # must not raise
    _fresh_case(object())


# --------------------------------------------------------- the snapshot cache
#
# `_snapshot` hashed every file in the tree, twice per case. Fine for the
# built-in fixtures and O(repo bytes x 2 x cases) the moment `--cases` points at
# a real repository, which is what `--cases` is for. The cache keys on
# (path, size, mtime_ns) so an unchanged file is not re-read -- but the value is
# still the hash, so the risk it introduces is a *stale* one, and that is what
# these test.
#
# An unchanged mtime is only evidence of an unchanged file if the clock behind
# it can separate two writes. It cannot, so the entry carries the time its bytes
# were read and is trusted only once `_SETTLED_NS` has passed. These pin both
# ends of that: the staleness it rules out, and the saving it must still make.
# Every one of them forces the timestamps with `os.utime` rather than waiting on
# a real clock -- a test that only reproduces the bug on a loaded machine is a
# test that reports the scheduler.


def test_a_rewritten_file_is_not_served_from_the_cache(tmp_path):
    """The failure mode the cache could introduce, and the only one that matters.

    A stale hash would report a file the agent rewrote as untouched, which is
    the external arm silently under-reporting the work it graded: a same-size
    edit scored as no work at all.
    """
    from scripts.coding_eval import _snapshot

    target = tmp_path / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")
    stamp = target.stat().st_mtime_ns
    before = _snapshot(tmp_path)

    # Same length, so `st_size` cannot tell the two apart; same mtime, so
    # neither can the timestamp. Reading the bytes is the only thing that can.
    target.write_text("x = 2\n", encoding="utf-8")
    os.utime(target, ns=(stamp, stamp))
    assert target.stat().st_mtime_ns == stamp, "the collision must be real"

    assert _snapshot(tmp_path)["a.py"] != before["a.py"]


def test_a_file_written_moments_ago_is_re_read(tmp_path):
    """The rule that makes the one above hold, stated directly.

    A file whose mtime is too recent to trust is the whole ambiguity, so the
    cache has to decline it. Asserted separately because the test above would
    still pass if `_snapshot` re-read *everything* -- this is what says the
    settle window is where the re-reading comes from.
    """
    import scripts.coding_eval as ce

    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    ce._snapshot(tmp_path)

    reads = []
    real = ce.hashlib.sha1
    try:
        ce.hashlib.sha1 = lambda data: (reads.append(data), real(data))[1]
        ce._snapshot(tmp_path)
    finally:
        ce.hashlib.sha1 = real

    assert reads, "a file inside the settle window must not be trusted"


def test_an_untouched_file_is_not_read_twice(tmp_path):
    """The saving itself, asserted rather than assumed.

    Aged past the settle window first, because that is the only state in which
    the saving is available at all. A file written moments ago is exactly the
    one whose mtime cannot rule out a rewrite, so asserting a cache hit without
    ageing it would be asserting the staleness bug rather than the saving.
    """
    import scripts.coding_eval as ce

    target = tmp_path / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")
    settled = target.stat().st_mtime_ns - 10 * ce._SETTLED_NS
    os.utime(target, ns=(settled, settled))
    ce._snapshot(tmp_path)

    reads = []
    real = ce.hashlib.sha1
    try:
        ce.hashlib.sha1 = lambda data: (reads.append(data), real(data))[1]
        second = ce._snapshot(tmp_path)
    finally:
        ce.hashlib.sha1 = real

    assert reads == [], "an unchanged file must not be hashed again"
    assert "a.py" in second, "...but it must still appear in the snapshot"


def test_the_snapshot_still_reports_content_not_timestamps(tmp_path):
    """Rewriting identical bytes moves the mtime but must not read as a change.

    The cache keys on mtime, so a touched-but-identical file misses the cache
    and is re-hashed -- and must then produce the same digest it did before.
    """
    from scripts.coding_eval import _snapshot

    target = tmp_path / "a.py"
    target.write_text("x = 1\n", encoding="utf-8")
    before = _snapshot(tmp_path)

    time.sleep(0.01)
    target.write_text("x = 1\n", encoding="utf-8")

    assert _snapshot(tmp_path)["a.py"] == before["a.py"]


# ------------------------------------------------- independence of the repeats
#
# `--repeat` exists to give a score its `n`. That only holds if the runs are
# independent, and the thing that makes them independent is one line choosing a
# directory. It is worth testing directly because sharing a directory does not
# fail -- it produces a full set of plausible numbers that are no longer samples
# of the same starting state.


def test_each_repeat_gets_its_own_fixture_directory(tmp_path):
    from scripts.coding_eval import run_dir

    dirs = [run_dir(tmp_path, i, 3) for i in range(3)]
    assert len(set(dirs)) == 3, "repeats must not share a tree"
    assert all(d.parent == tmp_path for d in dirs)


def test_a_single_run_keeps_the_bare_directory(tmp_path):
    """Nothing changes for the common case, so no trace path moves."""
    from scripts.coding_eval import run_dir

    assert run_dir(tmp_path, 0, 1) == tmp_path


def test_an_absent_trace_directory_stays_absent(tmp_path):
    """The trace dir is optional; `--repeat` must not conjure one."""
    from scripts.coding_eval import run_dir

    assert run_dir(None, 0, 5) is None


def test_materialise_leaves_files_the_agent_created(tmp_path):
    """The hazard the per-run directory defends against.

    Pinned as its own test because it is the reason, not the mechanism. If this
    ever stops being true -- if `materialise` grows a clean step -- then sharing
    a directory becomes safe and the extra nesting could go. Until then, anyone
    tempted to reuse one tree across repeats should fail this and see why.
    """
    from knossos.codeval import CodingCase, materialise

    case = CodingCase(id="c", prompt="p", files={"src/a.py": "x = 1\n"},
                      fail_to_pass=["tests/test_a.py::test_a"])
    materialise(case, tmp_path)
    stray = tmp_path / "src" / "agent_made_this.py"
    stray.write_text("leftover\n", encoding="utf-8")

    materialise(case, tmp_path)

    assert stray.exists(), \
        "materialise rewrites the case's own files and cannot know about " \
        "others -- which is exactly why repeats need separate directories"


# ------------------------------------------------------------------ the spread


def test_the_spread_reports_a_range_not_just_a_mean(capsys):
    from knossos.codeval import CaseResult
    from scripts.coding_eval import report_spread

    runs = [
        [CaseResult(case_id="a", kind="b", passed=True, harness_said_done=True,
                    steps_used=4)],
        [CaseResult(case_id="a", kind="b", passed=False, steps_used=9)],
        [CaseResult(case_id="a", kind="b", passed=True, harness_said_done=True,
                    steps_used=8)],
    ]
    report_spread(runs)
    out = capsys.readouterr().out
    assert "across 3 run(s)" in out
    assert "range 0-1" in out, out          # solved varied between runs
    assert "range 4.0-8.0" in out, out      # and so did the step count


def test_two_runs_are_flagged_as_too_few_to_quote(capsys):
    from knossos.codeval import CaseResult
    from scripts.coding_eval import report_spread

    report_spread([[CaseResult(case_id="a", kind="b", passed=True)]] * 2)
    assert "Three or more" in capsys.readouterr().out
