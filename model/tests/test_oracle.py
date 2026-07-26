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

from harness.oracle import Oracle, Tier, TierResult
from harness.workspace import Workspace

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


# ----------------------------------------------------------- talos protocol

def test_oracle_satisfies_the_verifier_protocol(ws):
    """It is passed straight to Talos, so the shape has to match."""
    from harness.talos import Talos

    oracle = Oracle(ws.root, tiers=[])

    class Engine:
        name = "scripted"

        def generate(self, prompt, context, cancelled):
            yield "Nothing to do."

    talos = Talos(Engine(), ws, verifier=oracle)
    outcome = talos.run("check the protocol")

    assert outcome.verdict is not None
    assert outcome.verdict.passed
