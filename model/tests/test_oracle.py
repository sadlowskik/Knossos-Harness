"""Isolation tests for the verification ladder.

The claims, in order of how much they matter:

  1. **Fail fast.** The first failing tier returns immediately; nothing below it
     runs. A syntax error must not be buried under the consequences of itself.
  2. **Tier 4 is gated.** Model judgement is unreachable unless every
     deterministic tier passed -- and a dry run can never satisfy that, because
     most of the ladder never ran.
  3. A missing tool is **skipped, not failed**. Absent ruff is not evidence of
     broken code.

Real subprocesses run here, but only `python -m`, and the fixtures are tiny.

    pytest -q tests/test_oracle.py
"""
import sys
from pathlib import Path

import pytest

from knossos.oracle import (PYTHON_TIERS, Oracle, Tier, TierResult,
                            _diagnostics)
from knossos.workspace import Workspace

#: The running interpreter, not the bare name. `python` is frequently not on
#: PATH -- on this machine it is not -- and a tier whose program cannot be found
#: is skipped, which would make every test below vacuously pass.
PY = sys.executable


@pytest.fixture()
def ws(tmp_path):
    (tmp_path / "mod.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    return Workspace(tmp_path)


def changed(ws, *names):
    return [ws.resolve(n) for n in names]


# ------------------------------------------------------- an empty change set

def test_an_empty_change_set_is_not_a_pass(ws):
    """Every tier is satisfied vacuously by nothing.

    Reported as a clean bill of health, this is what let a run that did nothing
    at all end with `stopReason: end_turn`.
    """
    verdict = Oracle(ws.root)(ws, [])

    assert verdict.nothing_to_verify
    assert "nothing to verify" in verdict.summary


def test_an_empty_change_set_cannot_reach_model_judgement(ws):
    """There is nothing for a judge to look at."""
    verdict = Oracle(ws.root)(ws, [])

    assert not verdict.deterministic_passed


def test_a_real_change_set_is_not_marked_empty(ws):
    verdict = Oracle(ws.root, tiers=[Tier(3, "ok", [PY, "-c", "pass"])])(
        ws, changed(ws, "mod.py"))

    assert not verdict.nothing_to_verify
    assert verdict.passed


def test_an_empty_change_set_runs_no_subprocess(ws):
    """A tier that would fail must not be reached -- nor run at all."""
    oracle = Oracle(ws.root, tiers=[Tier(3, "boom", [PY, "-c", "import sys; sys.exit(1)"])])

    verdict = oracle(ws, [])

    assert verdict.nothing_to_verify
    assert [t.label for t in verdict.tiers] == ["syntax"]


# ------------------------------------------------- per-diagnostic baselining

def emitting(number: int, label: str, lines: str, scopes=(".py",)) -> Tier:
    """A scopable tier that prints fixed diagnostics and fails."""
    return Tier(number, label,
                [PY, "-c", f"import sys; print({lines!r}); sys.exit(1)"],
                scopes=scopes)


def test_diagnostics_are_keyed_without_line_numbers():
    """Editing a file shifts everything below it. Keyed on position, every
    pre-existing complaint would look new the moment the agent touched the line
    above it -- which is exactly the case this forgives."""
    before = _diagnostics("mod.py:3:1: F401 `os` imported but unused")
    after = _diagnostics("mod.py:41:1: F401 `os` imported but unused")

    assert before == after
    assert sum(before.values()) == 1


def test_both_ladder_formats_parse():
    ruff = _diagnostics("src/a.py:1:1: F401 [*] `os` imported but unused")
    mypy = _diagnostics(
        'src/a.py:3: error: Incompatible return value type  [return-value]')

    assert sum(ruff.values()) == 1
    assert sum(mypy.values()) == 1
    # Summaries and headers are not diagnostics.
    assert not _diagnostics("Found 3 errors in 1 file (checked 9 source files)")


def test_ruffs_block_format_is_not_mis_parsed():
    """Modern ruff does not default to the concise form.

    It prints the location on a `-->` line inside a block. A path pattern that
    permits spaces matches `--> dirty.py` there and takes the *column* as the
    message, which keys every complaint in a file by its column number -- so two
    unrelated violations sharing a column become one key, and a newly added one
    is forgiven as pre-existing. That parsed cleanly and was completely wrong,
    which is worse than not parsing.

    The tier pins `--output-format=concise` so this shape should not arrive; the
    parser refuses it anyway, because failing to parse costs a forgiveness and
    mis-parsing costs a verdict.
    """
    block = ("F401 [*] `os` imported but unused\n"
             " --> dirty.py:1:8\n"
             "  |\n"
             "1 | import os\n"
             "  |        ^^\n"
             "  |\n"
             "help: Remove unused import: `os`\n")

    assert not _diagnostics(block), "the arrow format must yield nothing, not junk"


def test_mypy_notes_are_not_counted_as_separate_complaints():
    """A `note:` elaborates the error above it. One complaint, two lines."""
    text = ('a.py:4: error: Skipping analyzing "x": missing stubs  [import-untyped]\n'
            "a.py:4: note: See https://mypy.readthedocs.io/en/stable/running_mypy.html\n")

    assert sum(_diagnostics(text).values()) == 1


def test_the_concise_flag_is_on_the_shipped_ruff_tier():
    """Without it the parser sees the block format and forgives nothing, which
    is the dirty-file bug quietly returning."""
    ruff = next(t for t in PYTHON_TIERS if t.label == "ruff")
    assert "--output-format=concise" in ruff.argv


def test_a_whole_tree_baseline_passes_an_explicit_target(tmp_path):
    """ruff with no target walks the tree; mypy exits 2 with a usage error.

    Left as "no arguments", the mypy baseline came back empty on every run and
    mypy was never forgiven anything -- a half-working fix that reported nothing
    wrong.
    """
    seen = {}

    class Recording(Oracle):
        def _module_available(self, module):
            return True

    oracle = Recording(tmp_path, tiers=[])
    tier = Tier(2, "mypy", [sys.executable, "-m", "mypy"], scopes=(".py",))

    import subprocess as sp
    original = sp.run

    def capture(argv, **kw):
        seen["argv"] = argv
        return original([sys.executable, "-c", "pass"], **kw)

    sp.run = capture
    try:
        oracle._tier_output(tier, whole_tree=True)
    finally:
        sp.run = original

    assert seen["argv"][-1] == "."


def test_a_pre_existing_violation_in_an_edited_file_is_forgiven(ws):
    """Scoping fixed the unrelated file; it could not fix the dirty one.

    A scoped tier reports on the file, not on the diff, so editing one line of a
    module that already carried a violation failed the run for that violation --
    and told the engine to repair code it had never touched.
    """
    tier = emitting(1, "ruff", "mod.py:1:1: F401 `os` imported but unused")
    oracle = Oracle(ws.root, tiers=[tier])
    oracle.prepare(ws)                        # the violation is already there

    verdict = oracle(ws, changed(ws, "mod.py"))

    assert verdict.passed
    assert "already present before this change" in verdict.tiers[1].detail


def test_a_violation_the_change_added_is_not_forgiven(ws):
    """The other direction, and the one that matters for trusting the verdict."""
    oracle = Oracle(ws.root, tiers=[
        emitting(1, "ruff", "mod.py:1:1: F401 `os` imported but unused")])
    oracle.prepare(ws)
    # The tier now reports a second, different complaint.
    oracle.tiers = [emitting(1, "ruff",
                             "mod.py:1:1: F401 `os` imported but unused\n"
                             "mod.py:9:1: E711 comparison to None")]

    verdict = oracle(ws, changed(ws, "mod.py"))

    assert not verdict.passed
    assert verdict.failure.label == "ruff"


def test_a_second_instance_of_an_existing_violation_is_not_forgiven(ws):
    """Forgiveness is capped at the count that was already there.

    Otherwise one pre-existing complaint would license any number of copies of
    itself, which is a real way to make a file worse and be told it is fine.
    """
    line = "mod.py:1:1: F401 `os` imported but unused"
    oracle = Oracle(ws.root, tiers=[emitting(1, "ruff", line)])
    oracle.prepare(ws)
    oracle.tiers = [emitting(1, "ruff", f"{line}\n{line}")]

    verdict = oracle(ws, changed(ws, "mod.py"))

    assert not verdict.passed


def test_a_crash_is_never_forgiven_as_a_known_diagnostic(ws):
    """A tier that fails without emitting diagnostics failed for a reason no
    baseline can speak to."""
    oracle = Oracle(ws.root, tiers=[
        emitting(1, "ruff", "mod.py:1:1: F401 `os` imported but unused")])
    oracle.prepare(ws)
    oracle.tiers = [Tier(1, "ruff",
                         [PY, "-c", "import sys; sys.exit(2)"], scopes=(".py",))]

    verdict = oracle(ws, changed(ws, "mod.py"))

    assert not verdict.passed


def test_without_a_baseline_nothing_is_forgiven(ws):
    """`prepare` never ran, so the tier is held to the normal standard."""
    oracle = Oracle(ws.root, tiers=[
        emitting(1, "ruff", "mod.py:1:1: F401 `os` imported but unused")])

    verdict = oracle(ws, changed(ws, "mod.py"))

    assert not verdict.passed


def test_a_new_files_diagnostics_are_never_forgiven(ws):
    """It did not exist at baseline, so nothing about it can be pre-existing."""
    oracle = Oracle(ws.root, tiers=[
        emitting(1, "ruff", "mod.py:1:1: F401 `os` imported but unused")])
    oracle.prepare(ws)
    ws.write("fresh.py", "import os\n")
    oracle.tiers = [emitting(1, "ruff",
                             "fresh.py:1:1: F401 `os` imported but unused")]

    verdict = oracle(ws, changed(ws, "fresh.py"))

    assert not verdict.passed


def test_real_ruff_forgives_a_file_that_was_already_dirty(tmp_path):
    """End to end, with the actual linter rather than a scripted tier.

    This is not hypothetical for this repository: there is no ruff config, so
    defaults apply, and every module here violates them -- `argus.py` alone has
    57. Before this, touching any file failed the ruff tier on complaints the
    agent had not written, and the engine spent its budget being told to fix
    them.
    """
    # The shipped tier, so the pinned output format is part of what is tested.
    ruff = next(t for t in PYTHON_TIERS if t.label == "ruff")
    if Oracle(tmp_path, tiers=[ruff])._tier_output(ruff, whole_tree=True) == "":
        pytest.skip("ruff is not installed")

    dirty = tmp_path / "dirty.py"
    dirty.write_text("import os\n\n\ndef add(a, b):\n    return a + b\n",
                     encoding="utf-8")
    ws = Workspace(tmp_path)
    oracle = Oracle(tmp_path, tiers=[ruff])
    oracle.prepare(ws)

    # The agent edits the file, leaving the pre-existing unused import alone.
    ws.write("dirty.py",
             "import os\n\n\ndef add(a, b):\n    return a + b\n\n\n"
             "def sub(a, b):\n    return a - b\n")
    verdict = oracle(ws, changed(ws, "dirty.py"))

    assert verdict.passed, verdict.detail
    assert "already present before this change" in verdict.tiers[1].detail


def test_real_ruff_still_catches_what_the_change_added(tmp_path):
    """The half that has to keep working, or the fix is just a way to pass."""
    # The shipped tier, so the pinned output format is part of what is tested.
    ruff = next(t for t in PYTHON_TIERS if t.label == "ruff")
    if Oracle(tmp_path, tiers=[ruff])._tier_output(ruff, whole_tree=True) == "":
        pytest.skip("ruff is not installed")

    dirty = tmp_path / "dirty.py"
    dirty.write_text("import os\n\n\ndef add(a, b):\n    return a + b\n",
                     encoding="utf-8")
    ws = Workspace(tmp_path)
    oracle = Oracle(tmp_path, tiers=[ruff])
    oracle.prepare(ws)

    # Same pre-existing import, plus a new unused one the agent did write.
    ws.write("dirty.py",
             "import os\nimport sys\n\n\ndef add(a, b):\n    return a + b\n")
    verdict = oracle(ws, changed(ws, "dirty.py"))

    assert not verdict.passed
    assert verdict.failure.label == "ruff"


# ------------------------------------------------------------------- tier 0

def test_a_syntax_error_fails_at_tier_zero_and_stops_there(ws):
    ws.write("broken.py", "def nope(:\n")
    oracle = Oracle(ws.root, tiers=[Tier(3, "pytest", [PY, "-c", "pass"])])

    verdict = oracle(ws, changed(ws, "broken.py"))

    assert not verdict.passed
    assert verdict.failure.tier == 0
    assert len(verdict.tiers) == 1, "nothing below the failing tier should run"
    assert "broken.py" in verdict.detail


def test_valid_syntax_lets_the_ladder_continue(ws):
    oracle = Oracle(ws.root, tiers=[])
    verdict = oracle(ws, changed(ws, "mod.py"))
    assert verdict.passed
    assert verdict.tiers[0].label == "syntax"
    assert "1 file(s) parse cleanly" in verdict.tiers[0].detail


def test_non_python_files_are_not_syntax_checked(ws):
    ws.write("notes.md", "# not python at all (:\n")
    verdict = Oracle(ws.root, tiers=[])(ws, changed(ws, "notes.md"))
    assert verdict.passed


def test_tier_zero_reads_staged_content_in_a_dry_run(tmp_path):
    """The whole reason a preview can say anything: it checks what *would* be."""
    (tmp_path / "a.py").write_text("valid = 1\n", encoding="utf-8")
    dry = Workspace(tmp_path, dry_run=True)
    dry.write("a.py", "def broken(:\n")            # staged, not on disk

    verdict = Oracle(tmp_path, tiers=[])(dry, [dry.resolve("a.py")])

    assert not verdict.passed, "must check the staged version, not the disk one"
    assert verdict.failure.tier == 0


# ------------------------------------------------------------------ ordering

def test_the_first_failing_tier_stops_the_ladder(ws):
    oracle = Oracle(ws.root, tiers=[
        Tier(1, "always-fails", [PY, "-c", "raise SystemExit(1)"]),
        Tier(2, "never-runs", [PY, "-c", "pass"]),
    ])

    verdict = oracle(ws, changed(ws, "mod.py"))

    assert not verdict.passed
    assert verdict.failure.label == "always-fails"
    assert [t.label for t in verdict.tiers] == ["syntax", "always-fails"]


def test_a_passing_ladder_runs_every_tier(ws):
    oracle = Oracle(ws.root, tiers=[
        Tier(1, "one", [PY, "-c", "pass"]),
        Tier(2, "two", [PY, "-c", "pass"]),
    ])
    verdict = oracle(ws, changed(ws, "mod.py"))
    assert verdict.passed
    assert [t.label for t in verdict.tiers] == ["syntax", "one", "two"]


def test_a_warn_only_tier_does_not_block(ws):
    """Linter advice should surface without failing the verdict."""
    oracle = Oracle(ws.root, tiers=[
        Tier(1, "advisory", [PY, "-c", "raise SystemExit(1)"], fail_on_nonzero=False),
    ])
    verdict = oracle(ws, changed(ws, "mod.py"))
    assert verdict.passed
    assert verdict.tiers[1].passed


# ------------------------------------------------------------ missing tools

def test_a_missing_module_is_skipped_not_failed(ws):
    """Absent tooling is not evidence of broken code."""
    oracle = Oracle(ws.root, tiers=[
        Tier(1, "ghost", [PY, "-m", "definitely_not_installed_xyz"]),
    ])
    verdict = oracle(ws, changed(ws, "mod.py"))

    assert verdict.passed
    skipped = verdict.tiers[1]
    assert skipped.skipped and skipped.passed
    assert "not installed" in skipped.detail


def test_skipped_tiers_are_excluded_from_the_summary(ws):
    oracle = Oracle(ws.root, tiers=[
        Tier(1, "ghost", [PY, "-m", "definitely_not_installed_xyz"]),
        Tier(2, "real", [PY, "-c", "pass"]),
    ])
    verdict = oracle(ws, changed(ws, "mod.py"))
    assert "ghost" not in verdict.summary
    assert "real" in verdict.summary


# ------------------------------------------------------------------- tier 4

def test_judgement_is_unreachable_when_a_deterministic_tier_fails(ws):
    called = []

    def judge(workspace, files):
        called.append(True)
        return TierResult(4, "constitution", passed=True)

    oracle = Oracle(ws.root, judge=judge, tiers=[
        Tier(1, "fails", [PY, "-c", "raise SystemExit(1)"]),
    ])
    oracle(ws, changed(ws, "mod.py"))

    assert called == [], "tier 4 must not run over failing code"


def test_judgement_runs_once_everything_deterministic_passes(ws):
    def judge(workspace, files):
        return TierResult(4, "constitution", passed=True, detail="looks right")

    oracle = Oracle(ws.root, judge=judge, tiers=[Tier(1, "ok", [PY, "-c", "pass"])])
    verdict = oracle(ws, changed(ws, "mod.py"))

    assert verdict.passed
    assert verdict.tiers[-1].tier == 4


def test_a_failing_judgement_fails_the_verdict(ws):
    def judge(workspace, files):
        return TierResult(4, "constitution", passed=False, detail="does not do what was asked")

    oracle = Oracle(ws.root, judge=judge, tiers=[Tier(1, "ok", [PY, "-c", "pass"])])
    verdict = oracle(ws, changed(ws, "mod.py"))

    assert not verdict.passed
    assert "constitution" in verdict.summary
    assert "does not do what was asked" in verdict.detail


# ------------------------------------------------------------------ dry runs

def test_a_dry_run_stops_at_tier_zero_and_says_so(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    dry = Workspace(tmp_path, dry_run=True)
    dry.write("a.py", "y = 2\n")

    oracle = Oracle(tmp_path, tiers=[Tier(1, "pytest", [PY, "-c", "pass"])])
    verdict = oracle(dry, [dry.resolve("a.py")])

    assert verdict.passed
    assert verdict.preview
    assert "syntax only" in verdict.summary
    assert all(t.skipped for t in verdict.tiers if t.tier > 0)


def test_a_passing_dry_run_still_blocks_judgement(tmp_path):
    """A preview must never be mistaken for verification, even when it passes."""
    called = []

    def judge(workspace, files):
        called.append(True)
        return TierResult(4, "constitution", passed=True)

    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    dry = Workspace(tmp_path, dry_run=True)
    dry.write("a.py", "y = 2\n")

    verdict = Oracle(tmp_path, judge=judge, tiers=[])(dry, [dry.resolve("a.py")])

    assert verdict.passed
    assert not verdict.deterministic_passed
    assert called == [], "tier 4 must be unreachable from a preview"


# --------------------------------------------- the verdict is about the change

#: Fails if it is handed no path arguments, passes if it is handed any. Stands
#: in for "a linter that finds a pre-existing problem elsewhere in the tree".
DIRTY_ELSEWHERE = [
    PY, "-c",
    "import sys; sys.exit(0 if len(sys.argv) > 1 else 1)",
]


def test_a_scopable_tier_is_pointed_at_the_changed_files(ws):
    """`ruff check a.py` must not be able to fail because of `b.py`.

    Unscoped, this tier fails; scoped to the one changed file it is handed a
    path and passes. That difference is the whole fix.
    """
    tier = Tier(1, "lint", DIRTY_ELSEWHERE, scopes=(".py",))

    verdict = Oracle(ws.root, tiers=[tier])(ws, changed(ws, "mod.py"))

    assert verdict.passed, verdict.detail


def test_a_scopable_tier_is_skipped_when_nothing_of_its_kind_changed(ws):
    """Editing a README must not run the Python linter over the whole tree."""
    (ws.root / "notes.md").write_text("hello\n", encoding="utf-8")
    tier = Tier(1, "lint", DIRTY_ELSEWHERE, scopes=(".py",))

    verdict = Oracle(ws.root, tiers=[tier])(ws, changed(ws, "notes.md"))

    assert verdict.passed
    lint = [t for t in verdict.tiers if t.label == "lint"][0]
    assert lint.skipped and "no .py file changed" in lint.detail


def test_paths_outside_the_root_are_not_passed_to_a_tier(tmp_path):
    """Otherwise an absolute outside path silently widens the tier's scope."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "mod.py").write_text("x = 1\n", encoding="utf-8")
    outside = tmp_path / "elsewhere.py"
    outside.write_text("y = 2\n", encoding="utf-8")

    tier = Tier(1, "lint", DIRTY_ELSEWHERE, scopes=(".py",))

    assert tier.targets([outside], root.resolve()) == []
    assert tier.targets([root / "mod.py", outside], root.resolve()) == ["mod.py"]


def test_a_tier_already_failing_before_the_change_is_not_a_pass(ws):
    """Unattributable is not the same as verified.

    This asserted `verdict.passed` until the hard-tier eval showed what that
    licenses. On a task whose whole content is "fix this failing test", the
    failing test *is* the baseline failure -- so forgiving it reported success
    over a bug that was still there. The Oracle cannot tell the two cases apart,
    because it never sees the task, so it must not claim either.
    """
    always_red = Tier(3, "pytest", [PY, "-c", "import sys; sys.exit(1)"])
    oracle = Oracle(ws.root, tiers=[always_red])
    oracle.prepare(ws)

    verdict = oracle(ws, changed(ws, "mod.py"))

    assert not verdict.passed
    tier = [t for t in verdict.tiers if t.label == "pytest"][0]
    assert tier.forgiven, "must be marked unattributable, not as the agent's fault"
    assert "already failing before this change" in tier.detail


def test_a_pre_existing_failure_is_not_reported_as_the_agents_fault(ws):
    """Blame matters even though both outcomes block completion.

    An engine told it broke something it did not touch will go and repair code
    it never wrote, which is exactly how the step budget disappears.
    """
    always_red = Tier(3, "pytest", [PY, "-c", "import sys; sys.exit(1)"])
    oracle = Oracle(ws.root, tiers=[always_red])
    oracle.prepare(ws)

    verdict = oracle(ws, changed(ws, "mod.py"))

    assert "could not verify" in verdict.summary
    assert "FAILED at" not in verdict.summary
    assert "do not try to repair code you did not touch" in verdict.detail


def test_a_failure_the_change_caused_is_still_blamed_on_the_change(ws):
    """Forgiveness must not blunt the signal the ladder exists to produce."""
    marker = ws.root / "broken"
    script = "import sys, pathlib; sys.exit(1 if pathlib.Path('broken').exists() else 0)"
    oracle = Oracle(ws.root, tiers=[Tier(3, "pytest", [PY, "-c", script])])
    oracle.prepare(ws)                      # green: the marker does not exist

    marker.write_text("", encoding="utf-8")
    verdict = oracle(ws, changed(ws, "mod.py"))

    assert not verdict.passed
    assert "FAILED at pytest" in verdict.summary


def test_a_tier_the_change_broke_still_blocks(ws):
    """The baseline forgives what was already red -- and nothing else.

    Green at the baseline, red now, is exactly the signal the ladder exists to
    produce.
    """
    marker = ws.root / "broken"
    script = "import sys, pathlib; sys.exit(1 if pathlib.Path('broken').exists() else 0)"
    tier = Tier(3, "pytest", [PY, "-c", script])
    oracle = Oracle(ws.root, tiers=[tier])
    oracle.prepare(ws)          # green: the marker does not exist yet

    marker.write_text("", encoding="utf-8")
    verdict = oracle(ws, changed(ws, "mod.py"))

    assert not verdict.passed
    assert "FAILED at pytest" in verdict.summary


def test_the_baseline_is_taken_once(ws):
    """It runs a whole test suite, so a second call must not pay for it again."""
    counter = ws.root / "runs"
    script = ("import pathlib; p = pathlib.Path('runs'); "
              "p.write_text(p.read_text() + 'x' if p.exists() else 'x')")
    oracle = Oracle(ws.root, tiers=[Tier(3, "pytest", [PY, "-c", script])])

    oracle.prepare(ws)
    oracle.prepare(ws)
    oracle.prepare(ws)

    assert counter.read_text(encoding="utf-8") == "x"


def test_the_baseline_can_be_declined(ws):
    """A caller that does not want to pay for it at session start."""
    oracle = Oracle(ws.root, tiers=[Tier(3, "pytest", [PY, "-c", "sys.exit(1)"])],
                    baseline=False)

    oracle.prepare(ws)

    assert oracle._baseline is None


def test_a_scopable_tier_is_never_baselined(ws):
    """Scoping already makes it attributable; a baseline would be dead cost."""
    oracle = Oracle(ws.root, tiers=[Tier(1, "lint", DIRTY_ELSEWHERE, scopes=(".py",))])

    oracle.prepare(ws)

    assert oracle._baseline == {}


# --------------------------------------------------------- test-suite integrity

def _suite(root, body="def test_one():\n    assert True\n"):
    (root / "tests").mkdir(exist_ok=True)
    (root / "tests" / "test_thing.py").write_text(body, encoding="utf-8")


def test_deleting_a_test_does_not_buy_a_pass(tmp_path):
    """Found by the coding eval, not by review.

    A scripted agent that replaced a test file with
    `def test_placeholder(): assert True` was told on every case that it had
    completed and verified the task. Every tier passed honestly -- the suite it
    ran no longer contained anything that could fail.
    """
    (tmp_path / "mod.py").write_text("x = 1\n", encoding="utf-8")
    _suite(tmp_path, "def test_a():\n    assert True\n\n\ndef test_b():\n    assert True\n")
    ws = Workspace(tmp_path)
    oracle = Oracle(tmp_path, tiers=[])
    oracle.prepare(ws)

    _suite(tmp_path, "def test_a():\n    assert True\n")      # test_b removed

    verdict = oracle(ws, changed(ws, "mod.py"))

    assert not verdict.passed
    assert "test functions were removed" in verdict.detail.lower()
    assert "test_thing.py" in verdict.detail


def test_a_suite_that_grows_is_fine(tmp_path):
    """The check is one-directional on purpose."""
    (tmp_path / "mod.py").write_text("x = 1\n", encoding="utf-8")
    _suite(tmp_path, "def test_a():\n    assert True\n")
    ws = Workspace(tmp_path)
    oracle = Oracle(tmp_path, tiers=[])
    oracle.prepare(ws)

    _suite(tmp_path, "def test_a():\n    assert True\n\n\ndef test_b():\n    assert True\n")

    assert oracle(ws, changed(ws, "mod.py")).passed


def test_an_untouched_suite_is_fine(tmp_path):
    (tmp_path / "mod.py").write_text("x = 1\n", encoding="utf-8")
    _suite(tmp_path)
    ws = Workspace(tmp_path)
    oracle = Oracle(tmp_path, tiers=[])
    oracle.prepare(ws)

    assert oracle(ws, changed(ws, "mod.py")).passed


def test_a_test_file_that_cannot_import_is_still_counted(tmp_path):
    """`--collect-only` cannot see these; the textual count can.

    This is the gap that let one case through: on a feature task the test file
    references a function that does not exist yet, so it collects *zero* tests
    at baseline and can be replaced by a single trivial test as apparent growth.
    """
    (tmp_path / "mod.py").write_text("x = 1\n", encoding="utf-8")
    _suite(tmp_path, "from mod import missing\n\n\n"
                     "def test_a():\n    assert True\n\n\n"
                     "def test_b():\n    assert True\n")
    ws = Workspace(tmp_path)
    oracle = Oracle(tmp_path, tiers=[])
    oracle.prepare(ws)

    _suite(tmp_path, "def test_placeholder():\n    assert True\n")

    verdict = oracle(ws, changed(ws, "mod.py"))

    assert not verdict.passed, "a 2-test file became a 1-test file"


def test_a_module_level_skip_in_a_touched_test_file_is_caught(tmp_path):
    """What the collect count adds over the textual one.

    The `def test_` lines are all still there, so counting them sees nothing
    wrong -- but the module no longer collects, so the tests do not run. Only
    reachable when a test file was actually touched, which is exactly when the
    expensive check is allowed to run.
    """
    (tmp_path / "mod.py").write_text("x = 1\n", encoding="utf-8")
    _suite(tmp_path, "def test_a():\n    assert True\n\n\n"
                     "def test_b():\n    assert True\n")
    ws = Workspace(tmp_path)
    # An unscopable tier, because the collect count is only taken when the
    # ladder has something pytest-shaped in it to take one for.
    oracle = Oracle(tmp_path, tiers=[Tier(3, "suite", [PY, "-c", "pass"])])
    oracle.prepare(ws)

    _suite(tmp_path, 'import pytest\n\npytest.skip("nope", allow_module_level=True)\n\n\n'
                     "def test_a():\n    assert True\n\n\n"
                     "def test_b():\n    assert True\n")

    verdict = oracle(ws, changed(ws, "tests/test_thing.py"))

    assert not verdict.passed
    assert "disappeared" in verdict.detail


def test_the_expensive_check_is_skipped_when_no_test_was_touched(tmp_path):
    """It costs a pytest subprocess -- 3.4s on this repo -- per verification.

    A change that never went near a test cannot shrink the suite without also
    failing the pytest tier, so paying for it every time is waste.
    """
    (tmp_path / "mod.py").write_text("x = 1\n", encoding="utf-8")
    _suite(tmp_path)
    ws = Workspace(tmp_path)
    oracle = Oracle(tmp_path, tiers=[Tier(3, "suite", [PY, "-c", "pass"])])
    oracle.prepare(ws)

    calls = []
    oracle._collect_count = lambda: calls.append(1)      # type: ignore[method-assign]

    oracle(ws, changed(ws, "mod.py"))
    assert calls == [], "spawned pytest for a change that touched no test"

    oracle(ws, changed(ws, "tests/test_thing.py"))
    assert calls, "a touched test file must still be checked"


def test_integrity_is_skipped_without_a_baseline(tmp_path):
    """A check that could not run must not invent a result."""
    (tmp_path / "mod.py").write_text("x = 1\n", encoding="utf-8")
    _suite(tmp_path)
    ws = Workspace(tmp_path)
    oracle = Oracle(tmp_path, tiers=[], baseline=False)

    (tmp_path / "tests" / "test_thing.py").unlink()

    assert oracle(ws, changed(ws, "mod.py")).passed


# ------------------------------------------------ the workspace is not on the path

def test_a_tier_does_not_execute_a_module_shadowed_by_the_workspace(tmp_path):
    """`python -m X` with cwd=workspace runs the workspace's `X.py`.

    The verifier is harness-initiated: it never passes through `parse_calls`,
    the `run_command` allowlist, or the permission gate. So a repository that
    ships a file named after a checking tool used to get code execution just by
    being verified -- no misbehaviour from the model required.

    The marker file is the assertion: if the shadow ran, it wrote it.
    """
    pytest.importorskip("ruff", reason="the shadowed module must really be installed")

    marker = tmp_path / "SHADOW_RAN"
    # Named after a tool the ladder actually invokes. A made-up name proves
    # nothing: `_module_available` runs `find_spec` in the parent, so a module
    # the *harness* cannot import is skipped before the child ever starts. The
    # attack needs a module that genuinely exists -- which is precisely the case
    # that guard waves through.
    (tmp_path / "ruff.py").write_text(
        f"import pathlib; pathlib.Path(r{str(marker)!r}).write_text('x')\n",
        encoding="utf-8")
    (tmp_path / "mod.py").write_text("x = 1\n", encoding="utf-8")
    ws = Workspace(tmp_path)

    tier = [t for t in PYTHON_TIERS if t.label == "ruff"][0]
    Oracle(tmp_path, tiers=[tier])(ws, [ws.resolve("mod.py")])

    assert not marker.exists(), "the workspace's module was executed by the verifier"


def test_the_pytest_tier_keeps_the_workspace_importable(tmp_path):
    """`-P` is deliberately off for pytest, and this is why.

    The standard layout -- package at the root, tests in `tests/` -- imports the
    package through the cwd entry that `-P` removes. Turning it on everywhere
    would make every such repository a collection error, which is a worse bug
    than the one it fixes.
    """
    (tmp_path / "mypkg").mkdir()
    (tmp_path / "mypkg" / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_v.py").write_text(
        "from mypkg import VALUE\n\n\ndef test_v():\n    assert VALUE == 1\n",
        encoding="utf-8")
    (tmp_path / "mod.py").write_text("x = 1\n", encoding="utf-8")
    ws = Workspace(tmp_path)

    pytest_tier = [t for t in PYTHON_TIERS if t.label == "pytest"][0]
    verdict = Oracle(tmp_path, tiers=[pytest_tier], baseline=False)(
        ws, [ws.resolve("mod.py")])

    assert verdict.passed, verdict.detail


def test_only_path_taking_tiers_get_the_flag():
    """The split is the point: it is not safe to turn on everywhere."""
    by_label = {t.label: t for t in PYTHON_TIERS}

    assert by_label["ruff"].safe_path and by_label["mypy"].safe_path
    assert not by_label["pytest"].safe_path


# ----------------------------------------------------------- talos protocol

def test_oracle_satisfies_the_verifier_protocol(ws):
    """It is passed straight to Talos, so the shape has to match."""
    from knossos.talos import Talos

    oracle = Oracle(ws.root, tiers=[])

    class Engine:
        name = "scripted"

        def generate(self, prompt, context, cancelled):
            yield "Nothing to do."

    talos = Talos(Engine(), ws, verifier=oracle)
    outcome = talos.run("check the protocol")

    assert outcome.verdict is not None
    assert outcome.verdict.passed


# ------------------------------------------------------------ language ladders
#
# The Oracle could only ever verify Python: `PYTHON_TIERS` was the hardcoded
# default, so pointing the harness at a Go or TypeScript repository produced a
# ladder that checked nothing and a verdict that meant nothing. The Rust harness
# has had this seam since it shipped; this is the Python side catching up.


def _detect_names(root):
    from knossos.oracle import detect
    return [adapter.name for adapter in detect(root)]


def test_a_cargo_project_gets_the_rust_ladder(tmp_path):
    from knossos.oracle import tiers_for
    (tmp_path / "Cargo.toml").write_text("[package]\nname='x'\n")
    assert _detect_names(tmp_path) == ["rust"]
    assert [t.label for t in tiers_for(tmp_path)] == [
        "cargo check", "clippy", "cargo test"]


def test_a_go_module_gets_the_go_ladder(tmp_path):
    from knossos.oracle import tiers_for
    (tmp_path / "go.mod").write_text("module x\n")
    assert [t.label for t in tiers_for(tmp_path)] == [
        "go build", "go vet", "go test"]


def test_a_node_project_gets_the_node_ladder(tmp_path):
    from knossos.oracle import tiers_for
    (tmp_path / "package.json").write_text("{}")
    assert [t.label for t in tiers_for(tmp_path)] == ["tsc", "eslint", "npm test"]


def test_a_polyglot_repository_gets_every_applicable_ladder(tmp_path):
    """Checking only one language reports green on a change that broke the other."""
    from knossos.oracle import tiers_for
    (tmp_path / "Cargo.toml").write_text("[package]\nname='x'\n")
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    assert sorted(_detect_names(tmp_path)) == ["python", "rust"]
    labels = [t.label for t in tiers_for(tmp_path)]
    assert any("ruff" in l for l in labels)
    assert any("cargo test" in l for l in labels)


def test_ladders_interleave_cheapest_first_across_languages(tmp_path):
    """A run that will fail to compile must not first sit in another test suite."""
    from knossos.oracle import tiers_for
    (tmp_path / "Cargo.toml").write_text("[package]\nname='x'\n")
    (tmp_path / "go.mod").write_text("module x\n")
    labels = [t.label for t in tiers_for(tmp_path)]
    build = max(labels.index("rust: cargo check"), labels.index("go: go build"))
    tests = min(labels.index("rust: cargo test"), labels.index("go: go test"))
    assert build < tests


def test_tier_numbers_stay_unique_and_dense(tmp_path):
    """The baseline is keyed on the number; a collision forgives the wrong tier."""
    from knossos.oracle import tiers_for
    (tmp_path / "Cargo.toml").write_text("[package]\nname='x'\n")
    (tmp_path / "go.mod").write_text("module x\n")
    (tmp_path / "package.json").write_text("{}")
    numbers = [t.number for t in tiers_for(tmp_path)]
    assert numbers == list(range(1, len(numbers) + 1))


def test_labels_stay_distinct_when_several_languages_have_a_test_tier(tmp_path):
    from knossos.oracle import tiers_for
    (tmp_path / "Cargo.toml").write_text("[package]\nname='x'\n")
    (tmp_path / "go.mod").write_text("module x\n")
    labels = [t.label for t in tiers_for(tmp_path)]
    assert len(labels) == len(set(labels))


def test_an_unmarked_tree_still_gets_the_python_ladder(tmp_path):
    """Verifying nothing would make every task pass vacuously."""
    from knossos.oracle import tiers_for
    assert [t.label for t in tiers_for(tmp_path)] == [
        t.label for t in PYTHON_TIERS]


def test_the_oracle_picks_its_ladder_from_the_tree(tmp_path):
    (tmp_path / "go.mod").write_text("module x\n")
    assert [t.label for t in Oracle(tmp_path).tiers] == [
        "go build", "go vet", "go test"]


def test_explicit_tiers_still_win(tmp_path):
    """Detection must not override a caller that said what it wanted."""
    (tmp_path / "Cargo.toml").write_text("[package]\nname='x'\n")
    mine = [Tier(1, "mine", ["true"])]
    assert [t.label for t in Oracle(tmp_path, tiers=mine).tiers] == ["mine"]


def test_the_node_ladder_never_reaches_for_npx(tmp_path):
    """`npx` downloads and executes a package named in a config file.

    That turns a verification step into arbitrary code execution sourced from
    the very repository being verified. Absent tooling is skipped instead.
    """
    from knossos.oracle import NODE_TIERS
    programs = [t.argv[0] for t in NODE_TIERS]
    assert not any("npx" in p for p in programs)
    assert any("node_modules" in p for p in programs)
