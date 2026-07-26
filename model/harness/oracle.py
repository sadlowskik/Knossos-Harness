"""Oracle: tiered verification.

Two rules, and both are load-bearing.

**Fail fast.** The first failing tier returns immediately. There is no point
running the type checker on a file that does not parse, and the syntax error is
the one the engine needs to see. Running everything and reporting a pile buries
the actionable failure among consequences of it.

**Model judgement is the last tier, and unreachable until every deterministic
tier has passed.** "It parses, it lints, the tests are green -- but is it what
was asked for?" is the only question a model adds that a test runner cannot
answer. Asking it about code that does not compile spends tokens re-deriving
what the interpreter already said for free.

# Dry runs

In `dry_run` mode nothing is on disk, so pytest and mypy would inspect the *old*
code and return a confident verdict about the wrong source. Rather than run a
check whose answer is about something else, the ladder stops at tier 0 -- which
reads through the workspace and therefore sees staged content -- and the verdict
is marked `preview`. `Verdict.passed` stays honest, and `deterministic_passed`
returns False so tier 4 can never be reached from a preview.

# Missing tools

A tier whose program is not installed is **skipped, not failed**. Absent ruff is
not evidence of broken code, and treating it as such would make the verdict
depend on the developer's machine rather than the change.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from .talos import Verdict
from .workspace import Workspace

__all__ = ["TierResult", "OracleVerdict", "Tier", "Oracle", "PYTHON_TIERS"]

#: How long any single verification command may run.
TIER_TIMEOUT = 300
#: Cap on a tier's captured output.
MAX_DETAIL = 12_000


@dataclass
class TierResult:
    tier: int
    label: str
    passed: bool
    detail: str = ""
    #: True when the tier could not run (tool absent, or nothing on disk).
    skipped: bool = False


@dataclass
class OracleVerdict(Verdict):
    tiers: List[TierResult] = field(default_factory=list)
    #: Only tier 0 could run, because nothing was written to disk.
    preview: bool = False

    @property
    def failure(self) -> Optional[TierResult]:
        return next((t for t in self.tiers if not t.passed and not t.skipped), None)

    @property
    def deterministic_passed(self) -> bool:
        """Whether model judgement is permitted.

        A preview can never satisfy this: most of the ladder never ran.
        """
        return self.passed and not self.preview


@dataclass
class Tier:
    """One rung. Lower numbers run first."""

    number: int
    label: str
    argv: Sequence[str]
    #: Treat a non-zero exit as failure. Linters that warn by default set this
    #: False so advice is surfaced without blocking.
    fail_on_nonzero: bool = True


#: The deterministic ladder for a Python project, cheapest first.
#:
#: `python -m` rather than bare names so a virtualenv is honoured and PATH
#: cannot redirect the check.
PYTHON_TIERS: List[Tier] = [
    Tier(1, "ruff", [sys.executable, "-m", "ruff", "check", "."]),
    Tier(2, "mypy", [sys.executable, "-m", "mypy", "."], fail_on_nonzero=False),
    Tier(3, "pytest", [sys.executable, "-m", "pytest", "-q"]),
]


class Oracle:
    """A `talos.Verifier`. Call it with a workspace and the files that changed."""

    def __init__(self, root: str | Path,
                 tiers: Optional[Sequence[Tier]] = None,
                 judge: Optional[Callable[[Workspace, Sequence[Path]], TierResult]] = None,
                 timeout: int = TIER_TIMEOUT) -> None:
        self.root = Path(root).resolve()
        self.tiers = list(tiers) if tiers is not None else list(PYTHON_TIERS)
        self.judge = judge
        self.timeout = timeout

    # ------------------------------------------------------------------ api

    def __call__(self, ws: Workspace, changed: Sequence[Path]) -> OracleVerdict:
        results: List[TierResult] = [self._tier0(ws, changed)]

        if not results[0].passed:
            return self._verdict(results, preview=ws.dry_run)

        if ws.dry_run:
            # Nothing on disk: the rest of the ladder would judge the old code.
            for tier in self.tiers:
                results.append(TierResult(
                    tier.number, tier.label, passed=True, skipped=True,
                    detail="skipped: nothing written to disk yet"))
            return self._verdict(results, preview=True)

        for tier in self.tiers:
            result = self._run_tier(tier)
            results.append(result)
            if not result.passed and not result.skipped:
                return self._verdict(results, preview=False)

        verdict = self._verdict(results, preview=False)

        # Tier 4 is reachable only now.
        if self.judge is not None and verdict.deterministic_passed:
            judged = self.judge(ws, changed)
            verdict.tiers.append(judged)
            if not judged.passed:
                verdict.passed = False
                verdict.summary = f"FAILED at {judged.label} (tier {judged.tier})"
                verdict.detail = judged.detail
        return verdict

    # ---------------------------------------------------------------- tiers

    def _tier0(self, ws: Workspace, changed: Sequence[Path]) -> TierResult:
        """Syntax, in-process.

        Reads through the workspace, so in a dry run this checks the *staged*
        content rather than the stale file on disk -- which is the whole reason
        a preview can say anything useful at all.
        """
        broken: List[str] = []
        checked = 0

        for path in changed:
            if path.suffix != ".py":
                continue
            try:
                source = ws.read(path)
            except OSError:
                continue
            checked += 1
            try:
                compile(source, str(path), "exec")
            except SyntaxError as exc:
                broken.append(f"{ws.display(path)}:{exc.lineno}: {exc.msg}")

        if broken:
            return TierResult(0, "syntax", passed=False,
                              detail="\n".join(broken))
        return TierResult(0, "syntax", passed=True,
                          detail=f"{checked} file(s) parse cleanly")

    def _run_tier(self, tier: Tier) -> TierResult:
        program = tier.argv[0]
        module = tier.argv[2] if len(tier.argv) > 2 and tier.argv[1] == "-m" else None

        if shutil.which(program) is None and not Path(program).exists():
            return TierResult(tier.number, tier.label, passed=True, skipped=True,
                              detail=f"skipped: {program} not found")
        if module and not self._module_available(module):
            return TierResult(tier.number, tier.label, passed=True, skipped=True,
                              detail=f"skipped: {module} is not installed")

        try:
            proc = subprocess.run(tier.argv, cwd=self.root, capture_output=True,
                                  text=True, timeout=self.timeout,
                                  stdin=subprocess.DEVNULL, shell=False)
        except OSError as exc:
            return TierResult(tier.number, tier.label, passed=True, skipped=True,
                              detail=f"skipped: could not run ({exc})")
        except subprocess.TimeoutExpired:
            return TierResult(tier.number, tier.label, passed=False,
                              detail=f"{tier.label} timed out after {self.timeout}s")

        body = "\n".join(p for p in (proc.stdout, proc.stderr) if p.strip())
        passed = proc.returncode == 0 or not tier.fail_on_nonzero
        return TierResult(tier.number, tier.label, passed=passed,
                          detail=_cap(body) if body.strip() else "(no output)")

    def _module_available(self, module: str) -> bool:
        import importlib.util
        try:
            return importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            return False

    # -------------------------------------------------------------- verdict

    def _verdict(self, results: List[TierResult], preview: bool) -> OracleVerdict:
        failure = next((t for t in results if not t.passed and not t.skipped), None)

        if failure is not None:
            return OracleVerdict(passed=False,
                                 summary=f"FAILED at {failure.label} (tier {failure.tier})",
                                 detail=failure.detail, tiers=results, preview=preview)

        ran = [t.label for t in results if not t.skipped]
        if preview:
            summary = ("syntax only -- nothing is on disk, so the linter, type "
                       "checker and tests could not run")
        else:
            summary = f"passed {len(ran)} tier(s): {', '.join(ran)}"
        return OracleVerdict(passed=True, summary=summary, tiers=results, preview=preview)


def _cap(text: str) -> str:
    if len(text) <= MAX_DETAIL:
        return text
    return text[:MAX_DETAIL] + "\n\n[truncated]"
