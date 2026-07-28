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

# The verdict is about the change, not about the repository

A tier that runs over the whole tree answers "is this repository clean", which
is not the question. On any real repository the answer is no -- one stale lint
warning, one flaky test -- and every task then fails verification forever while
the agent spends its budget repairing code it never touched.

Two mechanisms keep the verdict attributable, in order of preference:

  * **Scoping.** A tier that accepts file arguments is pointed at the changed
    files (`Tier.scopes`). `ruff check a.py` cannot fail because of `b.py`. When
    nothing of that kind changed, the tier is skipped rather than run over
    everything.
  * **A baseline.** `pytest` cannot be scoped -- changing `lib.py` does not tell
    you which tests exercise it -- so instead `prepare()` records, *before the
    agent touches anything*, which unscopable tiers already fail. A tier that
    was red at the baseline and is red now is reported as pre-existing and does
    not block completion; one that was green and is now red is the change's
    fault and does.

The baseline costs one run of those tiers per session, taken at the start of the
first execute turn. That is the price of being able to attribute a failure, and
it is paid once rather than per task. `Oracle(root, baseline=False)` opts out.
"""
from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

#: Never walked when counting test functions. Same list Argus and `search` use;
#: a vendored copy of someone else's tests is not this repository's suite.
_SKIP_DIRS = frozenset({
    ".git", ".argus", "__pycache__", ".pytest_cache", ".mypy_cache", ".venv",
    "venv", "node_modules", "target", "build", "dist", ".ipynb_checkpoints",
})

from .jsonrpc import log
from .talos import Verdict
from .workspace import Workspace

__all__ = ["TierResult", "OracleVerdict", "Tier", "Oracle", "PYTHON_TIERS",
           "RUST_TIERS", "GO_TIERS", "NODE_TIERS", "LanguageAdapter",
           "ADAPTERS", "detect", "tiers_for"]

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
    #: True when this tier failed but was already failing before the change, so
    #: its result says nothing about the change either way.
    forgiven: bool = False


@dataclass
class OracleVerdict(Verdict):
    tiers: List[TierResult] = field(default_factory=list)
    #: Only tier 0 could run, because nothing was written to disk.
    preview: bool = False
    #: The change set was empty, so no tier had anything to look at. Distinct
    #: from passing: "I found no problems in nothing" is not evidence that the
    #: task was done. Every tier is satisfied vacuously by an empty change set,
    #: which is how a run that did nothing at all used to report success.
    nothing_to_verify: bool = False

    @property
    def failure(self) -> Optional[TierResult]:
        return next((t for t in self.tiers if not t.passed and not t.skipped), None)

    @property
    def deterministic_passed(self) -> bool:
        """Whether model judgement is permitted.

        Neither a preview nor an empty change set can satisfy this: in the first
        most of the ladder never ran, and in the second there is nothing for a
        judge to look at.
        """
        return self.passed and not self.preview and not self.nothing_to_verify


#: One diagnostic line from a checker. Matches the two shapes the ladder emits
#: in concise form: ruff's `path:line:col: CODE message` and mypy's
#: `path:line: error: message  [code]`.
#:
#: The line and column are captured and then deliberately **discarded** from the
#: key. Editing a file shifts every diagnostic below the edit, so keying on the
#: position would make every pre-existing complaint look new the moment the agent
#: touched the file above it -- which is precisely the case this exists to
#: forgive.
#:
#: `[^\s:]+` for the path, not `[^:]*`, and the difference is load-bearing.
#: Modern ruff does not default to the concise form at all; it prints a block
#: with the location on its own arrow line:
#:
#:     F401 [*] `os` imported but unused
#:      --> dirty.py:1:8
#:
#: A path pattern permitting spaces matches `--> dirty.py` there, takes the
#: *column* as the message, and keys every complaint in a file by its column
#: number -- so two unrelated violations in the same column become one key and a
#: newly added one is forgiven as pre-existing. Forbidding whitespace in the path
#: makes that line fail to match instead, which is the right outcome: an
#: unrecognised format yields no diagnostics, no baseline, and therefore no
#: forgiveness, which is exactly the behaviour from before any of this existed.
#: `--output-format=concise` on the tier itself is what makes it recognised.
_DIAGNOSTIC = re.compile(
    r"^(?P<path>[^\s:]+\.[A-Za-z0-9_]+):(?P<line>\d+):(?:(?P<col>\d+):)?\s*"
    r"(?P<body>\S.*)$")


def _diagnostics(text: str) -> "Counter":
    """Count a checker's complaints, keyed by file and message.

    Unparseable lines -- summaries, headers, source excerpts, tracebacks -- are
    ignored rather than counted, so a change in a tool's summary wording cannot
    look like a new problem or hide a real one.
    """
    found: Counter = Counter()
    for line in text.splitlines():
        match = _DIAGNOSTIC.match(line.strip())
        if not match:
            continue
        body = match.group("body").strip()
        # A body with no letters is not a message. This is the second guard on
        # the arrow-format mis-parse above: whatever survives the path pattern,
        # a bare column number is not something a checker said.
        if not any(char.isalpha() for char in body):
            continue
        # mypy emits `note:` lines that elaborate the error immediately above.
        # They are one complaint with two lines, not two complaints, and
        # counting both inflates the totals on each side of the subtraction.
        if body.startswith("note:"):
            continue
        found[(match.group("path").replace("\\", "/"), body)] += 1
    return found


@dataclass
class Tier:
    """One rung. Lower numbers run first."""

    number: int
    label: str
    argv: Sequence[str]
    #: Treat a non-zero exit as failure. Linters that warn by default set this
    #: False so advice is surfaced without blocking.
    fail_on_nonzero: bool = True
    #: File suffixes this tier can be pointed at, if it accepts path arguments.
    #: Non-empty means the tier is *scopable*: it runs over the changed files of
    #: these kinds instead of the whole tree, so an unrelated file cannot fail
    #: it. Empty means it can only run over everything, and is therefore subject
    #: to the baseline instead.
    scopes: Tuple[str, ...] = ()
    #: Run the interpreter with `-P`, so the workspace is kept off `sys.path[0]`.
    #: Only safe for tiers that take explicit file arguments -- see `_run_tier`.
    safe_path: bool = True

    def targets(self, changed: Sequence[Path], root: Path) -> Optional[List[str]]:
        """Path arguments for this tier, or None if it cannot be scoped.

        An empty list is meaningful and distinct from None: the tier is scopable
        but nothing it cares about changed, so there is nothing to check.
        """
        if not self.scopes:
            return None
        out = []
        for path in changed:
            if path.suffix not in self.scopes:
                continue
            try:
                out.append(str(path.relative_to(root)))
            except ValueError:
                # Outside the root: not ours to check, and passing an absolute
                # path would silently widen the tier's scope back out.
                continue
        return sorted(set(out))


#: The deterministic ladder for a Python project, cheapest first.
#:
#: `python -m` rather than bare names so a virtualenv is honoured and PATH
#: cannot redirect the check -- paired with `-P` where possible, because `-m`
#: alone closes the PATH door and opens the `sys.path[0]` one.
#:
#: ruff and mypy take file arguments, so they are scoped to what changed. pytest
#: does not -- the tests covering a change are not the files the change touched
#: -- so it runs whole and leans on the baseline.
PYTHON_TIERS: List[Tier] = [
    # `--output-format=concise` is not cosmetic. Modern ruff defaults to a block
    # format whose location sits on a `-->` line, which per-diagnostic baselining
    # cannot parse -- and an unparsed baseline silently forgives nothing, so the
    # dirty-file case quietly comes back. Pinning the format is what keeps the
    # subtraction in `_forgive_known_diagnostics` meaningful across ruff releases.
    Tier(1, "ruff", [sys.executable, "-m", "ruff", "check",
                     "--output-format=concise"], scopes=(".py",)),
    Tier(2, "mypy", [sys.executable, "-m", "mypy"], fail_on_nonzero=False,
         scopes=(".py",)),
    # `safe_path=False`: pytest's standard layout -- a package at the root and
    # tests in `tests/` -- imports the package via the cwd entry that `-P`
    # removes, so running with it turns every such repository into a collection
    # error. Measured, not assumed. This is also the tier where `-P` buys least:
    # collecting a repository's tests executes that repository's `conftest.py`
    # and test modules by design, so the protection would be nominal.
    Tier(3, "pytest", [sys.executable, "-m", "pytest", "-q"], safe_path=False),
]


#: The ladder for a Rust crate. Nothing here is scopable: cargo's unit of work is
#: the crate, so pointing it at two changed files checks the same thing as
#: pointing it at none. All three therefore lean on the baseline instead.
RUST_TIERS: List[Tier] = [
    Tier(1, "cargo check", ["cargo", "check", "--quiet"]),
    Tier(2, "clippy", ["cargo", "clippy", "--quiet"], fail_on_nonzero=False),
    Tier(3, "cargo test", ["cargo", "test", "--quiet"]),
]

#: The ladder for a Go module. `./...` is the whole module by design -- Go has no
#: cheaper granularity that is still correct, since a change to one package can
#: break any package that imports it.
GO_TIERS: List[Tier] = [
    Tier(1, "go build", ["go", "build", "./..."]),
    Tier(2, "go vet", ["go", "vet", "./..."], fail_on_nonzero=False),
    Tier(3, "go test", ["go", "test", "./..."]),
]

#: The ladder for a Node/TypeScript project.
#:
#: Deliberately `node_modules/.bin/...` and never `npx`: npx will *download and
#: execute* a package that is not installed, which turns a verification step into
#: arbitrary code execution sourced from the name in a config file. If the
#: project has not installed its own toolchain, the tier is skipped -- absent
#: tooling is not evidence of broken code.
NODE_TIERS: List[Tier] = [
    Tier(1, "tsc", [os.path.join("node_modules", ".bin", "tsc"), "--noEmit"]),
    Tier(2, "eslint", [os.path.join("node_modules", ".bin", "eslint"), "."],
         fail_on_nonzero=False, scopes=(".js", ".jsx", ".ts", ".tsx")),
    Tier(3, "npm test", ["npm", "test", "--silent"]),
]


@dataclass(frozen=True)
class LanguageAdapter:
    """One language's ladder, and how to tell the language is present.

    The Rust harness has had this seam since it shipped (`scribe::LanguageAdapter`);
    the Python one hardcoded `PYTHON_TIERS`, which is why it could only verify
    Python. A harness that cannot check a TypeScript or Go repository is not a
    coding harness for anyone who does not write Python, however good its loop is.

    Detection is by marker file rather than by counting source files, because a
    single vendored `.py` under `node_modules` should not make a React app look
    like a Python project.
    """

    name: str
    #: Files whose presence at the root identifies this kind of project.
    markers: Tuple[str, ...]
    tiers: Tuple[Tier, ...]

    def detected(self, root: Path) -> bool:
        return any((root / marker).exists() for marker in self.markers)


#: Ordered only for stable output; detection is independent per adapter.
ADAPTERS: Tuple[LanguageAdapter, ...] = (
    LanguageAdapter("python",
                    ("pyproject.toml", "setup.py", "setup.cfg",
                     "requirements.txt", "tox.ini"),
                    tuple(PYTHON_TIERS)),
    LanguageAdapter("rust", ("Cargo.toml",), tuple(RUST_TIERS)),
    LanguageAdapter("go", ("go.mod",), tuple(GO_TIERS)),
    LanguageAdapter("node", ("package.json",), tuple(NODE_TIERS)),
)


def detect(root: Path) -> List[LanguageAdapter]:
    """Every language present at `root`. Possibly none, possibly several."""
    return [adapter for adapter in ADAPTERS if adapter.detected(root)]


def tiers_for(root: str | Path) -> List[Tier]:
    """The ladder for whatever kind of project this is.

    Polyglot repositories get every applicable ladder rather than a guess at
    which language "really" owns the tree -- this repository is itself Python
    and Rust, and checking only one of them would report a green verdict on a
    change that broke the other.

    Ordering is cheapest-first *across* languages, not language by language: all
    the compilers, then all the linters, then all the test suites. A run that is
    going to fail on a syntax error should not first spend five minutes in
    someone else's test suite.

    Tier numbers are reassigned to stay unique and dense, because the baseline
    is keyed on them.
    """
    found = detect(Path(root))
    if not found:
        # No marker at all: keep the historical behaviour rather than verifying
        # nothing, which would make every task pass vacuously.
        return list(PYTHON_TIERS)

    pairs = [(tier, adapter.name)
             for adapter in found for tier in adapter.tiers]
    pairs.sort(key=lambda pair: (pair[0].number, pair[1]))

    out: List[Tier] = []
    multilingual = len(found) > 1
    for index, (tier, language) in enumerate(pairs, start=1):
        # Labels are what the engine reads back in a failure report, and
        # `test` appearing twice with different meanings is worse than verbose.
        label = f"{language}: {tier.label}" if multilingual else tier.label
        out.append(replace(tier, number=index, label=label))
    return out


class Oracle:
    """A `talos.Verifier`. Call it with a workspace and the files that changed."""

    def __init__(self, root: str | Path,
                 tiers: Optional[Sequence[Tier]] = None,
                 judge: Optional[Callable[[Workspace, Sequence[Path]], TierResult]] = None,
                 timeout: int = TIER_TIMEOUT,
                 baseline: bool = True) -> None:
        self.root = Path(root).resolve()
        self.tiers = list(tiers) if tiers is not None else tiers_for(self.root)
        self.judge = judge
        self.timeout = timeout
        self.use_baseline = baseline
        #: tier number -> whether it passed before the agent touched anything.
        #: `None` until `prepare` runs; a tier absent from it was never measured
        #: and is therefore held to the normal standard.
        self._baseline: Optional[Dict[int, bool]] = None
        #: tier number -> the complaints that tier already made, before the agent
        #: touched anything, keyed by (file, message).
        #:
        #: Scoping fixed the *unrelated* file: a lint error in a module nobody
        #: edited no longer fails the verdict. It could not fix the file that was
        #: already dirty, because a scoped tier reports on the file rather than
        #: on the diff -- so editing one line of a module carrying three
        #: pre-existing violations failed the run for all three, and the engine
        #: was told to repair code it had never touched. This is the subtraction
        #: that closes it.
        self._baseline_diags: Optional[Dict[int, Counter]] = None
        #: How many tests the suite collected before the agent started.
        self._baseline_tests: Optional[int] = None
        #: test file -> how many `def test_` it defined before the agent started.
        self._baseline_defs: Optional[Dict[str, int]] = None

    # ------------------------------------------------------------------ api

    def prepare(self, ws: Optional[Workspace] = None) -> None:
        """Record which unscopable tiers already fail. Call before any change.

        Talos calls this once before the loop starts. Timing is the whole point:
        run it afterwards and it measures the agent's own work, which is the
        thing it exists to exclude. Idempotent, so repeated calls across a
        session cost nothing.

        `ws` is accepted so this matches the shape of the rest of the verifier
        interface; the baseline is taken from disk, because that is what the
        unscopable tiers will read.
        """
        if self._baseline is not None or not self.use_baseline:
            return
        # Recorded before the early return below, and unconditionally. Suite
        # integrity is not about which tiers are configured -- deleting a test
        # is not made acceptable by there being no test runner in the ladder --
        # and both counts are cheap. `_collect_count` returns None on its own
        # when nothing pytest-shaped is present.
        self._baseline_tests = self._collect_count()
        self._baseline_defs = self._test_function_counts()
        # Whole-tree, because which files will change is not known yet. The cost
        # sits next to the `pytest` baseline below, which is far larger, and is
        # paid once per session at the first consequential call.
        self._baseline_diags = {
            tier.number: _diagnostics(self._tier_output(tier, whole_tree=True))
            for tier in self.tiers if tier.scopes}

        unscopable = [t for t in self.tiers if not t.scopes]
        if not unscopable:
            self._baseline = {}
            return
        self._baseline = {t.number: self._run_tier(t, ()).passed for t in unscopable}
        failing = [t.label for t in unscopable if not self._baseline[t.number]]
        if failing:
            log(f"[oracle] baseline: {', '.join(failing)} already failing before "
                f"any change; failures there will not be attributed to the agent")

    def quick(self, ws: Workspace, changed: Sequence[Path]) -> OracleVerdict:
        """Tier 0 only: does the tree still parse?

        For running *between* plan steps, where the full ladder would cost more
        than the plan saves -- pytest after every step is not a check, it is a
        tax. This is in-process and reads through the workspace, so it sees
        staged content and answers the one question worth asking mid-plan:
        did that step leave things broken.
        """
        results = [self._tier0(ws, changed)]
        if not changed:
            return self._verdict(results, preview=ws.dry_run, nothing_to_verify=True)
        return self._verdict(results, preview=ws.dry_run)

    def __call__(self, ws: Workspace, changed: Sequence[Path]) -> OracleVerdict:
        results: List[TierResult] = [self._tier0(ws, changed)]

        if not changed:
            # Say so rather than reporting a clean bill of health on an empty
            # set. The caller decides what an unverifiable run means; the
            # Oracle's job is not to imply one happened.
            return self._verdict(results, preview=ws.dry_run, nothing_to_verify=True)

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
            result = self._forgive_baseline(
                self._forgive_known_diagnostics(tier, self._run_tier(tier, changed)))
            results.append(result)
            if not result.passed and not result.skipped:
                return self._verdict(results, preview=False)

        integrity = self._suite_integrity(changed)
        if integrity is not None:
            results.append(integrity)
            if not integrity.passed:
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

    def _run_tier(self, tier: Tier, changed: Sequence[Path]) -> TierResult:
        program = tier.argv[0]
        module = tier.argv[2] if len(tier.argv) > 2 and tier.argv[1] == "-m" else None

        if shutil.which(program) is None and not Path(program).exists():
            return TierResult(tier.number, tier.label, passed=True, skipped=True,
                              detail=f"skipped: {program} not found")
        if module and not self._module_available(module):
            return TierResult(tier.number, tier.label, passed=True, skipped=True,
                              detail=f"skipped: {module} is not installed")

        argv = list(tier.argv)
        if tier.safe_path and len(argv) > 1 and argv[1] == "-m":
            # `python -m` puts the child's cwd at `sys.path[0]`, and the cwd
            # here is the workspace -- so a `ruff.py` at the repository root
            # shadows the installed distribution and `runpy` executes it. The
            # `_module_available` guard does not help: it runs `find_spec` in
            # *this* process, confirms the real module exists, and then hands
            # control to the workspace's copy. `-P` (3.11+) is what actually
            # closes it; the comment above about `-m` defeating PATH was true
            # and incomplete.
            argv.insert(1, "-P")
        targets = tier.targets(changed, self.root)
        if targets is not None:
            if not targets:
                # Scopable, and nothing it cares about changed. Running it over
                # the tree anyway is how an unrelated file fails the verdict.
                return TierResult(
                    tier.number, tier.label, passed=True, skipped=True,
                    detail=f"skipped: no {'/'.join(tier.scopes)} file changed")
            argv += targets

        try:
            proc = subprocess.run(argv, cwd=self.root, capture_output=True,
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

    def _tier_output(self, tier: Tier, whole_tree: bool = False) -> str:
        """Raw, uncapped output from running a tier. "" if it could not run.

        Separate from `_run_tier` for two reasons. `_run_tier` caps its detail
        for the prompt, and a baseline parsed from capped output silently
        under-records -- which would forgive fewer pre-existing complaints than
        it should, quietly reintroducing the bug this is here to fix. And a
        scopable tier given no targets is *skipped* rather than run, which is
        correct at verification time and useless at baseline time, when the whole
        tree is exactly what needs measuring.
        """
        program = tier.argv[0]
        module = tier.argv[2] if len(tier.argv) > 2 and tier.argv[1] == "-m" else None
        if shutil.which(program) is None and not Path(program).exists():
            return ""
        if module and not self._module_available(module):
            return ""

        argv = list(tier.argv)
        if tier.safe_path and len(argv) > 1 and argv[1] == "-m":
            argv.insert(1, "-P")
        if whole_tree:
            # An explicit ".", not "no arguments". ruff with no target walks the
            # tree, but mypy refuses to start -- it exits 2 with a usage message
            # and no diagnostics at all, so the mypy baseline came back empty
            # and mypy was never forgiven anything. The two tools disagree about
            # what "no arguments" means, and "." is what both read as the tree.
            argv.append(".")
        else:
            targets = tier.targets((), self.root)
            if targets:
                argv += targets
        try:
            proc = subprocess.run(argv, cwd=self.root, capture_output=True,
                                  text=True, timeout=self.timeout,
                                  stdin=subprocess.DEVNULL, shell=False)
        except (OSError, subprocess.TimeoutExpired):
            # No baseline is a worse verdict, not a reason to abandon the run:
            # an empty counter forgives nothing, which is the behaviour before
            # this existed.
            return ""
        return "\n".join(p for p in (proc.stdout, proc.stderr) if p.strip())

    def _forgive_known_diagnostics(self, tier: Tier,
                                   result: TierResult) -> TierResult:
        """Drop complaints the tier was already making before the agent started.

        Scoping answered "is this file mine?". This answers the harder half --
        "is this *complaint* mine?" -- for a file that is mine and was already
        dirty.

        `Counter` subtraction gives the semantics that matter: a message present
        once at baseline and twice now leaves one, and that one fails the tier. A
        change is only forgiven up to the count that was already there, so
        adding a second instance of an existing violation is still caught.

        Deliberately not applied when nothing remains to explain: a tier that
        passed is left alone, and a tier with no baseline (`prepare` never ran,
        or the tool could not be run then) forgives nothing at all.
        """
        if result.passed or result.skipped or not self._baseline_diags:
            return result
        baseline = self._baseline_diags.get(tier.number)
        if not baseline:
            return result

        current = _diagnostics(result.detail)
        if not current:
            # The tier failed for a reason that is not a diagnostic -- a crash, a
            # config error, an internal traceback. Not something a baseline can
            # speak to, and not something to wave through.
            return result

        remaining = current - baseline
        if remaining:
            return result

        forgiven = sum(current.values())
        log(f"[oracle] {tier.label}: {forgiven} pre-existing diagnostic(s) in "
            f"changed files, none introduced by this change")
        return replace(
            result, passed=True, skipped=True,
            detail=(f"{forgiven} diagnostic(s) in the changed files were "
                    f"already present before this change and none were added:\n"
                    f"{result.detail}"))

    def _collect_count(self) -> Optional[int]:
        """How many tests the suite collects, or None if that cannot be told.

        `--collect-only` rather than a run: this is about the suite's *size*,
        it costs a fraction of an execution, and it still answers correctly for
        a suite that is currently failing.
        """
        if not any(not t.scopes for t in self.tiers):
            return None                      # no pytest-shaped tier configured
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "pytest", "--collect-only", "-q",
                 "-p", "no:cacheprovider"],
                cwd=self.root, capture_output=True, text=True,
                timeout=self.timeout, stdin=subprocess.DEVNULL, shell=False)
        except (OSError, subprocess.TimeoutExpired):
            return None
        # Count node ids rather than parsing the summary line, whose wording
        # has changed between pytest releases.
        return sum(1 for line in proc.stdout.splitlines() if "::" in line)

    def _touches_tests(self, changed: Sequence[Path]) -> bool:
        """Whether the change set includes anything that looks like a test file."""
        for path in changed:
            if any(fnmatch.fnmatch(path.name, g) for g in self._TEST_GLOBS):
                return True
            try:
                parts = path.relative_to(self.root).parts
            except ValueError:
                continue
            if any(part in ("test", "tests") for part in parts):
                return True
        return False

    def _suite_integrity(self, changed: Sequence[Path] = ()) -> Optional[TierResult]:
        """Refuse a pass bought by deleting tests.

        Every other check here asks "is the code right". This one asks whether
        the *question* is still being asked, and it exists because the coding
        eval caught the harness answering the first without noticing the
        second: a scripted agent that replaced a test file with
        `def test_placeholder(): assert True` was told, on all five cases, that
        it had completed and verified the task. Every deterministic tier passed
        -- honestly -- because the suite it ran no longer contained anything
        that could fail.

        A shrinking suite is the objective, language-agnostic signal for that.
        It does not require guessing intent or diffing assertions: if the
        repository collected forty tests before the change and thirty-nine
        after, one is gone, and no task is completed by removing the evidence.

        Deliberately one-directional. Growth is fine, an unchanged count is
        fine, and a legitimate deletion of a test is a thing a human can do --
        which is why this reports a *failed verdict* the engine can argue with
        in the transcript, rather than reverting anything.

        Returns None when no baseline was taken, because a check that cannot
        run must not invent a result.
        """
        lost = self._lost_test_definitions()
        if lost:
            where = "; ".join(f"{name}: {before} -> {after}"
                              for name, before, after in lost)
            return TierResult(
                0, "test suite", passed=False,
                detail=(f"Test functions were removed: {where}.\n\nA task is not "
                        f"completed by deleting the test that proves it is not. "
                        f"Restore the removed test(s) and fix the code instead. If "
                        f"a test genuinely should go, say so plainly rather than "
                        f"removing it as part of a fix."))

        # The collect-count check costs a `pytest --collect-only` subprocess --
        # 3.4 s on this repository, on *every* verification, which is far more
        # than the rest of the ladder for a change that never went near a test.
        # It only earns that when a test file was touched: the textual count
        # above already catches a deleted function or a deleted file, and what
        # this adds is the subtler case of a file that stops *collecting*
        # (a module-level skip, a broken import in `conftest.py`). None of those
        # can be caused by editing source alone without also failing the pytest
        # tier, which runs anyway.
        if self._baseline_tests is None or not self._touches_tests(changed):
            return None
        now = self._collect_count()
        if now is None:
            return None
        if now >= self._baseline_tests:
            return TierResult(0, "test suite", passed=True,
                              detail=f"{now} test(s) collected, "
                                     f"was {self._baseline_tests}")
        return TierResult(
            0, "test suite", passed=False,
            detail=(f"The suite collected {self._baseline_tests} test(s) before this "
                    f"change and {now} now: {self._baseline_tests - now} "
                    f"disappeared.\n\nA task is not completed by deleting the test "
                    f"that proves it is not. Restore the removed test(s) and fix "
                    f"the code instead. If a test genuinely should go, say so "
                    f"plainly rather than removing it as part of a fix."))

    #: Files whose test functions are counted. Both pytest conventions.
    _TEST_GLOBS = ("test_*.py", "*_test.py")

    #: `def test_...` / `async def test_...` at any indentation, so methods in a
    #: `TestFoo` class count too.
    _TEST_DEF = re.compile(r"^\s*(?:async\s+)?def\s+test\w*\s*\(", re.MULTILINE)

    def _test_function_counts(self) -> Dict[str, int]:
        """How many test functions each test file defines.

        A textual count, deliberately, because it answers a question
        `--collect-only` cannot: how many tests a file *contains* even when it
        does not currently import. That gap is not hypothetical -- it is
        exactly how a deletion slipped past the collect-count check on a
        feature task, where the test file referenced a function that did not
        exist yet, collected zero tests at baseline, and so could be replaced
        with a single trivial test and register as *growth*.

        Being textual, it over-counts a commented-out test and under-counts a
        generated one. That is acceptable for a check that only ever fires on a
        *decrease* in a file that already existed.
        """
        counts: Dict[str, int] = {}
        for current, dirnames, filenames in os.walk(self.root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            for name in filenames:
                if not any(fnmatch.fnmatch(name, g) for g in self._TEST_GLOBS):
                    continue
                path = Path(current) / name
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                counts[str(path.relative_to(self.root))] = len(
                    self._TEST_DEF.findall(text))
        return counts

    def _lost_test_definitions(self) -> List[Tuple[str, int, int]]:
        """(file, before, after) for every test file that lost functions.

        Only files present at the baseline are considered, and only decreases.
        A new test file is not suspicious, and neither is a larger one.
        """
        if not self._baseline_defs:
            return []
        now = self._test_function_counts()
        lost = []
        for name, before in self._baseline_defs.items():
            after = now.get(name, 0)
            if after < before:
                lost.append((name, before, after))
        return sorted(lost)

    def _forgive_baseline(self, result: TierResult) -> TierResult:
        """Mark a failure the agent did not cause as *unattributable*.

        Only tiers measured by `prepare` are eligible, and only those that were
        already failing. A tier that was green at the baseline and is red now is
        exactly the signal the ladder exists to produce, and is left alone.

        **This does not make the verdict pass**, and that distinction is the
        whole point. An earlier version marked the tier `passed=True,
        skipped=True`, reasoning that a pre-existing failure is not the agent's
        fault -- which is true, and which turned out to license a lie. The
        hard-tier eval caught it: on a task whose *entire content* was "fix this
        failing test", pytest was red at the baseline, got forgiven, and the run
        reported `passed 3 tier(s): syntax, ruff, mypy` and halted DONE having
        fixed one of the two defects. Compare a successful case, which reports
        four tiers including pytest.

        The trap is that "a check that was already failing" and "the check that
        defines the task" are frequently the same check, and nothing here can
        tell them apart -- the Oracle never sees the task text. So the honest
        report is neither pass nor blame: the change could not be verified.
        `_verdict` turns that into a failed verdict whose detail says plainly
        that the failure predates the change, so an engine reading it can tell
        the difference between "you broke this" and "this was already broken".
        """
        if result.passed or result.skipped or self._baseline is None:
            return result
        if self._baseline.get(result.tier) is not False:
            return result
        return replace(
            result, passed=False, forgiven=True,
            detail=(f"{result.label} was already failing before this change, so "
                    f"this run could not confirm whether the change is correct.\n\n"
                    f"If these failures are unrelated to your task, say so plainly "
                    f"and stop -- do not try to repair code you did not touch. If "
                    f"fixing them *was* the task, it is not done yet.\n\n"
                    f"Output:\n\n{result.detail}"))

    def _module_available(self, module: str) -> bool:
        import importlib.util
        try:
            return importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            return False

    # -------------------------------------------------------------- verdict

    def _verdict(self, results: List[TierResult], preview: bool,
                 nothing_to_verify: bool = False) -> OracleVerdict:
        failure = next((t for t in results if not t.passed and not t.skipped), None)

        if failure is not None:
            # Distinguish "you broke this" from "this was already broken". Both
            # block completion -- neither confirms the change is correct -- but
            # an engine told it caused a failure it did not cause will go and
            # "fix" code it never touched, which is how a budget disappears.
            if failure.forgiven:
                summary = (f"could not verify: {failure.label} was already failing "
                           f"before this change")
            else:
                summary = f"FAILED at {failure.label} (tier {failure.tier})"
            return OracleVerdict(passed=False, summary=summary,
                                 detail=failure.detail, tiers=results, preview=preview)

        ran = [t.label for t in results if not t.skipped]
        if nothing_to_verify:
            summary = "nothing was changed, so there was nothing to verify"
        elif preview:
            summary = ("syntax only -- nothing is on disk, so the linter, type "
                       "checker and tests could not run")
        else:
            summary = f"passed {len(ran)} tier(s): {', '.join(ran)}"
        return OracleVerdict(passed=True, summary=summary, tiers=results,
                             preview=preview, nothing_to_verify=nothing_to_verify)


def _cap(text: str) -> str:
    if len(text) <= MAX_DETAIL:
        return text
    return text[:MAX_DETAIL] + "\n\n[truncated]"
