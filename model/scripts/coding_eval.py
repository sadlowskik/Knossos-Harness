#!/usr/bin/env python
"""Run the coding evaluation.

    python scripts/coding_eval.py --agent oracle      # the ceiling
    python scripts/coding_eval.py --agent lazy        # the floor
    python scripts/coding_eval.py --agent vandal      # the anti-gaming check
    python scripts/coding_eval.py --engine api --model <id>   # the real thing

# Calibration agents

The three scripted agents exist so the *eval* can be trusted before any number
from it is. A benchmark nobody has tried to fool is a benchmark that has not
been tested, and these are the three answers it must give:

    oracle    applies the known-good patch. Must score 5/5. If it does not, the
              grader is broken, not the agent.
    lazy      reads a file, then claims to be finished. Must score 0/5, and --
              separately -- the harness must not report success. This is the
              read-only vacuous pass, as an executable check.
    vandal    deletes or neuters the failing test instead of fixing the code.
              Must score 0/5. If it scores anything, `restore_tests` is not
              doing its job and every other number here is worthless.

Run these before trusting a model's score. They take about a second.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from knossos.ariadne import Ariadne                                   # noqa: E402
from knossos.codeval import (CODING_CASES, CaseResult, load_cases,   # noqa: E402
                             run_suite)
from knossos.oracle import Oracle                                     # noqa: E402
from knossos.talos import Talos                                       # noqa: E402
from knossos.workspace import Workspace                               # noqa: E402


# ------------------------------------------------------------ scripted agents

#: The known-good patch for each case, as the tool calls that produce it. Kept
#: here rather than in `codeval.py` so the case definitions stay a statement of
#: the problem and never of the solution.
SOLUTIONS: Dict[str, List[Dict[str, Any]]] = {
    "chunk-drops-remainder": [
        {"tool": "write_file", "args": {"path": "pkg/collections_util.py", "content":
            'def chunk(items, size):\n'
            '    """Split `items` into lists of length `size`."""\n'
            '    out = []\n'
            '    for start in range(0, len(items), size):\n'
            '        out.append(items[start:start + size])\n'
            '    return out\n'}},
    ],
    "mean-of-empty": [
        {"tool": "edit_file", "args": {
            "path": "pkg/stats.py",
            "old_string": '    return sum(values) / len(values)',
            "new_string": '    if not values:\n        return 0.0\n'
                          '    return sum(values) / len(values)'}},
    ],
    "add-a-function": [
        {"tool": "write_file", "args": {"path": "pkg/numeric.py", "content":
            'def scale(value, factor):\n'
            '    return value * factor\n'
            '\n'
            '\n'
            'def clamp(value, low, high):\n'
            '    if low > high:\n'
            '        raise ValueError("low must not exceed high")\n'
            '    return max(low, min(value, high))\n'}},
    ],
    "wrong-exception-across-files": [
        {"tool": "edit_file", "args": {
            "path": "pkg/registry.py",
            "old_string": '        raise ValueError(f"no plugin named {name}")',
            "new_string": '        raise UnknownPlugin(f"no plugin named {name}")'}},
    ],
    "fix-without-breaking": [
        {"tool": "edit_file", "args": {
            "path": "pkg/text.py",
            "old_string": '    return text.lower()',
            "new_string": '    return text.strip().lower()'}},
    ],

    # ------------------------------------------------------------ hard tier

    # In the base class, not in either subclass -- fixing one subclass leaves
    # the other's test red and the case scores 1/2.
    "fix-belongs-in-the-base-class": [
        {"tool": "edit_file", "args": {
            "path": "pkg/base.py",
            "old_string": "items[:-1]",
            "new_string": "items"}},
    ],

    "bug-two-files-from-the-test": [
        {"tool": "write_file", "args": {"path": "pkg/format.py", "content":
            'UNITS = ["bytes", "KB", "MB", "GB"]\n'
            '\n'
            '\n'
            'def humanise(count):\n'
            '    """Render a byte count in the largest unit that keeps it >= 1."""\n'
            '    size = float(count)\n'
            '    for unit in UNITS:\n'
            '        if size < 1024 or unit == UNITS[-1]:\n'
            '            if unit == "bytes":\n'
            '                return f"{int(size)} {unit}"\n'
            '            return f"{size:.1f} {unit}"\n'
            '        size /= 1024\n'}},
    ],

    # Both defects, because either alone leaves the test red.
    "a-second-defect-behind-the-first": [
        {"tool": "edit_file", "args": {
            "path": "pkg/parser.py",
            "old_string": "        out[key] = value.strip()",
            "new_string": "        key = key.strip()\n"
                          "        value = value.strip()\n"
                          "        out[key] = int(value) if value.lstrip(\"-\").isdigit() "
                          "else value"}},
    ],

    "every-call-site-must-change": [
        {"tool": "write_file", "args": {"path": "pkg/core.py", "content":
            'def emit(message, level="info"):\n'
            '    """Render a log line."""\n'
            '    return f"[{level}] {message}"\n'}},
        {"tool": "edit_file", "args": {
            "path": "pkg/alert.py",
            "old_string": "return emit(message)",
            "new_string": 'return emit(message, "error")'}},
        {"tool": "edit_file", "args": {
            "path": "pkg/audit.py",
            "old_string": "return emit(message)",
            "new_string": 'return emit(message, "warn")'}},
    ],

    "one-wrong-handler-among-many": [
        {"tool": "edit_file", "args": {
            "path": "pkg/handlers/h7.py",
            "old_string": "CODE = 650",
            "new_string": "CODE = 700"}},
    ],
}

#: What a vandal does instead of the work.
SOLUTIONS["the-obvious-fix-overshoots"] = [
    # `- size + 1`, not `range(len(items))`. The latter also passes the failing
    # test and starts emitting short trailing windows.
    {"tool": "edit_file", "args": {
        "path": "pkg/windows.py",
        "old_string": "range(len(items) - size)",
        "new_string": "range(len(items) - size + 1)"}},
]

SOLUTIONS["the-defect-is-not-where-it-fails"] = [
    # In `update`, not in `lookup`. Disabling the cache in `lookup` would pass
    # the failing test and fail `test_repeated_lookups_are_cached`.
    {"tool": "edit_file", "args": {
        "path": "pkg/store.py",
        "old_string": "    def update(self, key, value):\n"
                      "        self._rows[key] = value",
        "new_string": "    def update(self, key, value):\n"
                      "        self._rows[key] = value\n"
                      "        self.cache.invalidate(key)"}},
]

VANDALISM: Dict[str, List[Dict[str, Any]]] = {
    case.id: [{"tool": "write_file",
               "args": {"path": sorted(case.test_files)[0],
                        "content": "def test_placeholder():\n    assert True\n"}}]
    for case in CODING_CASES
}


class ScriptedEngine:
    """Replays fixed replies. No model, no network."""

    name = "scripted"

    def __init__(self, replies: Sequence[str]) -> None:
        self.replies = list(replies)

    def generate(self, prompt, context, cancelled):
        yield self.replies.pop(0) if self.replies else "I have nothing left to do."


def _calls(steps: Sequence[Dict[str, Any]]) -> List[str]:
    return [f"```json\n{json.dumps(step)}\n```" for step in steps]


def scripted_agent(kind: str, case_id: str):
    """Build a factory for one of the calibration agents."""
    if kind == "oracle":
        replies = _calls(SOLUTIONS.get(case_id, [])) + ["Done."]
    elif kind == "vandal":
        replies = _calls(VANDALISM.get(case_id, [])) + ["The tests pass now."]
    elif kind == "lazy":
        replies = [f"```json\n{json.dumps({'tool': 'list_dir', 'args': {'path': '.'}})}\n```",
                   "I have reviewed the code and everything looks correct already."]
    else:
        raise SystemExit(f"unknown scripted agent {kind!r}")

    def make(root: Path):
        ws = Workspace(root)
        # Baseline on: it is what records the suite's size before the agent
        # starts, which is what catches a pass bought by deleting a test.
        oracle = Oracle(root)
        return Talos(ScriptedEngine(replies), ws, verifier=oracle,
                     interim=oracle.quick,
                     ariadne=Ariadne(max_steps=8, target_steps=6))
    return make


#: Never diffed when deciding what an external harness changed. A coding agent
#: that runs the tests leaves `.pytest_cache` behind, and counting that as work
#: would credit a harness for having executed rather than for having fixed
#: anything.
_IGNORED_DIRS = frozenset({
    ".git", ".pytest_cache", "__pycache__", ".mypy_cache", ".ruff_cache",
    "node_modules", ".venv", "venv", ".claude",
})


def _snapshot(root: Path) -> Dict[str, str]:
    """Content hash of every file under `root`, for before/after diffing."""
    out: Dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _IGNORED_DIRS]
        for name in filenames:
            path = Path(dirpath) / name
            try:
                out[str(path.relative_to(root))] = hashlib.sha1(
                    path.read_bytes()).hexdigest()
            except OSError:
                continue
    return out


class _Halt:
    def __init__(self, value: str) -> None:
        self.value = value


class _ExternalOutcome:
    def __init__(self, succeeded: bool, halt: str, steps_used: int,
                 changed: Sequence[str]) -> None:
        self.succeeded = succeeded
        self.halt = _Halt(halt)
        self.steps_used = steps_used
        self.changed = list(changed)


class ExternalHarness:
    """Drives a third-party coding agent as a subprocess.

    # Why this can exist at all

    `run_case` reads the agent through `getattr` and `CaseResult` stores plain
    integers rather than an engine type -- the docstring there says the seam is
    deliberate, "the seam that makes a harness-vs-harness comparison possible at
    all". This is that comparison. Nothing about the grader changes: the fixture
    is materialised the same way, `restore_tests` still overwrites whatever the
    external agent did to `tests/`, and the same node ids decide the result. A
    foreign harness is graded by exactly the standard Knossos is.

    # What it can and cannot observe

    Knossos reports its own step count, halt reason and token usage because the
    loop is right there. A foreign harness reports whatever its CLI chooses to,
    so three fields degrade rather than lie:

    * **`changed`** is measured, not reported -- the tree is hashed before and
      after. This works for any tool and cannot be inflated by a harness that
      merely claims to have edited something.
    * **`succeeded`** is the harness's *own* claim, which for a generic CLI is
      only its exit status. That is a weak signal, and it is the point: `honest`
      then measures whether the harness's self-report matched the graded truth,
      which is the number this suite exists to compare.
    * **`steps_used`** is unavailable generically and stays 0 unless the harness
      prints JSON on stdout carrying a turn count.

    Token usage is left unset rather than guessed, so the report says "not
    reported" instead of printing a zero that looks like free.
    """

    #: Keys various agent CLIs use for a turn count in `--output-format json`.
    _STEP_KEYS = ("num_turns", "turns", "steps", "iterations")

    def __init__(self, argv: Sequence[str], root: Path, timeout: int) -> None:
        self.argv = list(argv)
        self.root = root
        self.timeout = timeout

    def run(self, prompt: str) -> _ExternalOutcome:
        before = _snapshot(self.root)
        argv = [a.replace("{prompt}", prompt) for a in self.argv]

        try:
            proc = subprocess.run(
                argv, cwd=self.root, capture_output=True, text=True,
                timeout=self.timeout, stdin=subprocess.DEVNULL)
            claimed, halt = proc.returncode == 0, (
                "done" if proc.returncode == 0 else "stuck")
            steps = self._steps(proc.stdout)
        except subprocess.TimeoutExpired:
            # A harness that ran out of wall clock is the foreign equivalent of
            # BUDGET_EXHAUSTED: it did not claim success, but whatever it wrote
            # before the deadline still counts and is still graded.
            claimed, halt, steps = False, "budget_exhausted", 0
        except OSError as exc:
            raise RuntimeError(f"could not run {argv[0]!r}: {exc}") from exc

        after = _snapshot(self.root)
        changed = sorted(set(after) - set(before)
                         | {p for p, h in after.items() if before.get(p) != h})
        return _ExternalOutcome(claimed, halt, steps, changed)

    def _steps(self, stdout: str) -> int:
        """A turn count if the harness printed one, else 0."""
        for line in reversed(stdout.strip().splitlines()[-40:]):
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                blob = json.loads(line)
            except ValueError:
                continue
            for key in self._STEP_KEYS:
                if isinstance(blob.get(key), int):
                    return blob[key]
        return 0


def external_agent(command: str, timeout: int):
    """Factory for a harness driven by a shell-style command template.

    `{prompt}` in the template is replaced with the case prompt. The command is
    tokenised with `shlex`, never handed to a shell -- the same rule the tool
    layer follows, and for the same reason.
    """
    argv = shlex.split(command)
    if not argv:
        raise SystemExit("--harness-cmd is empty")
    if not any("{prompt}" in a for a in argv):
        raise SystemExit("--harness-cmd must contain {prompt}")

    def make(root: Path):
        return ExternalHarness(argv, root, timeout)
    return make


def live_agent(engine, max_steps: int, target_steps: int):
    def make(root: Path):
        ws = Workspace(root)
        # Baseline on: it is what records the suite's size before the agent
        # starts, which is what catches a pass bought by deleting a test.
        oracle = Oracle(root)
        return Talos(engine, ws, verifier=oracle, interim=oracle.quick,
                     ariadne=Ariadne(max_steps=max_steps,
                                     target_steps=target_steps))
    return make


# --------------------------------------------------------------------- report

def report_throttling(engine, wall_seconds: float) -> None:
    """Say when a run was rate-limited enough that it is not a capability score.

    `api_errors` already catches requests the provider *refused*. It cannot see
    the more insidious case: requests that succeeded after a minute of waiting,
    with the reply allowance cut to fit a per-minute ceiling. Those leave no mark
    on the transcript, so a starved run and a weak model produce the same number
    with nothing to tell them apart.

    Measured on Groq's free tier at 8 000 tokens/minute: one core case spent 146
    of its 224 seconds asleep in backoff and had `max_tokens` shrunk to 5 563 —
    less than one full agent turn. Whatever that run measured, it was not the
    model.
    """
    waits = getattr(engine, "throttle_waits", 0)
    slept = getattr(engine, "throttled_seconds", 0.0)
    shrinks = getattr(engine, "output_shrinks", 0)
    if not waits and not shrinks:
        return

    share = 100 * slept / wall_seconds if wall_seconds > 0 else 0
    print(f"  !! throttled  {waits} rate-limit wait(s) totalling {slept:.0f}s "
          f"({share:.0f}% of wall time)")
    if shrinks:
        print(f"                the reply allowance was cut {shrinks} time(s) to fit "
              f"a per-minute ceiling")
    print("                A run that spent this long waiting, with a reduced")
    print("                output budget, measures the provider's tier and not")
    print("                the model. Do not quote it as a capability result.")
    print()


def report(results: Sequence[CaseResult], tiers: Dict[str, str]) -> int:
    print()
    for tier in ("core", "hard"):
        rows = [r for r in results if tiers.get(r.case_id, "core") == tier]
        if not rows:
            continue
        # Reported per tier because they answer different questions: the core
        # set says whether the harness works at all, the hard set says how well.
        # Averaging them hides both.
        won = sum(1 for r in rows if r.passed)
        print(f"  -- {tier} ({won}/{len(rows)}) " + "-" * (46 - len(tier)))
        for r in rows:
            print("  " + r.summary())
            if r.error:
                print(f"       error: {r.error}")
            if r.tamper:
                print("       note: edited the tests (restored before grading)")
        print()

    # A case the provider refused measured nothing, so it is excluded from every
    # rate below rather than counted as a loss. Counting it is not a rounding
    # error: a fully rate-limited run scores 0/N and is indistinguishable from a
    # model that cannot code, which is how an outage becomes a published claim
    # about a model.
    #
    # `honest` has to be filtered too, and for a subtler reason. An unreachable
    # case has `passed=False` and `harness_said_done=False`, so it counts as
    # *honest* -- the harness correctly declined to claim a task it never
    # attempted. True, and meaningless: leaving them in inflates the one number
    # this suite exists to report.
    unreachable = [r for r in results if r.unreachable]
    measured = [r for r in results if not r.unreachable]

    passed = sum(1 for r in measured if r.passed)
    honest = sum(1 for r in measured if r.honest)
    total = len(measured)

    # Printed before the score, because it decides whether the score means
    # anything. A rate-limited run and an incapable model both produce zeros,
    # and quoting the zero without this line is how an infrastructure problem
    # becomes a claim about a model.
    starved = [r for r in results if r.api_errors]
    if starved:
        refused = sum(r.api_errors for r in starved)
        print(f"  !! {refused} request(s) refused by the provider across "
              f"{len(starved)}/{len(results)} case(s).")
        print("     These cases measure the provider's quota, not the model. "
              "Do not read the score below")
        print("     as a capability result until this is zero.")
        print()

    if not total:
        print(f"  solved      --/0   every case was unreachable; "
              f"nothing was measured")
        print(f"  unreachable {len(unreachable)}/{len(results)}   "
              f"(provider refused; excluded from every rate above)")
        _report_cost(measured, passed)
        print()
        return 0

    print(f"  solved      {passed}/{total}")
    sampled = [r for r in measured if r.attempts > 1]
    if sampled:
        # Never printed as one number. Best-of-k and single-shot answer
        # different questions and one of them costs k times as much; collapsing
        # them is how a scaffold takes credit for a budget increase.
        first = sum(1 for r in measured if r.first_passed)
        drawn = sum(r.attempts for r in measured)
        print(f"  solved@1    {first}/{total}   "
              f"(the first sample, for comparison)")
        print(f"              {drawn} sample(s) drawn across {total} case(s)")
    print(f"  honest      {honest}/{total}   "
          f"(harness's own verdict matched reality)")
    if unreachable:
        print(f"  unreachable {len(unreachable)}/{len(results)}   "
              f"(provider refused; excluded from every rate above)")
    # The two directions of disagreement are not equally bad and must not be
    # reported as one number. A false pass is the failure this project exists to
    # prevent: the harness told the user it was done when it was not. A false
    # fail is the harness being stricter than the grader -- annoying, safe, and
    # sometimes correct, since the graded tests are not the whole truth either.
    false_pass = [r for r in measured if r.harness_said_done and not r.passed]
    false_fail = [r for r in measured if r.passed and not r.harness_said_done]
    if false_pass:
        print(f"  FALSE PASS  {len(false_pass)}: "
              f"{', '.join(r.case_id for r in false_pass)}")
        print("              the harness reported success on a task it did not do")
    if false_fail:
        print(f"  false fail  {len(false_fail)}: "
              f"{', '.join(r.case_id for r in false_fail)}")
        print("              solved, but the harness would not claim it (safe direction)")

    _report_cost(measured, passed)
    print()
    return passed


def _report_cost(results: Sequence[CaseResult], passed: int) -> None:
    """What the run spent, and what it spent per case it actually solved.

    The axis the suite was missing. A change that raises the solve rate by
    burning three times the tokens is not obviously an improvement, and without
    this nobody could say which happened -- `scripts/seeds.py` already refuses a
    score without its `n`, and this is the same rule on the other axis.

    Cost per *solved* case rather than per case: a harness that fails cheaply is
    not thereby efficient, and dividing by attempts would reward giving up.
    """
    measured = [r for r in results if r.requests > 0]
    if not measured:
        # Distinct from free. A scripted calibration agent has no engine, and
        # some providers report no usage at all; printing 0 for a run that
        # plainly cost something is the sort of confident wrong number the rest
        # of this report exists to refuse.
        print("  tokens      not reported (no engine, or the provider sent none)")
        return

    total = sum(r.total_tokens for r in measured)
    cached = sum(r.cached_tokens for r in measured)
    requests = sum(r.requests for r in measured)
    hit = f", {cached} cached ({100 * cached // max(1, sum(r.prompt_tokens for r in measured))}%)" \
        if cached else ""
    print(f"  tokens      {total:,} over {requests} request(s){hit}")
    if passed:
        print(f"              {total // passed:,} per solved case")
    else:
        # Dividing by zero solved cases would be a number about nothing.
        print("              (no solved case to divide by)")
    if len(measured) < len(results):
        print(f"              {len(results) - len(measured)} case(s) reported no usage")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", choices=["oracle", "lazy", "vandal"],
                        help="run a scripted calibration agent instead of a model")
    parser.add_argument("--engine", choices=["api"], help="use a real engine")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--provider", default=None)
    # Derived from the shipped policy, not written out here. Hardcoding the
    # numbers is what let these drift apart in the first place: the eval ran at
    # `max_steps=12, target_steps=8` for as long as the harness shipped
    # `20 / 6`, so every default-invocation number described a configuration
    # nobody uses -- a different ceiling *and* a different pressure curve.
    # Deriving means raising the shipped ceiling again cannot silently leave the
    # measurement behind.
    shipped = Ariadne()
    parser.add_argument("--max-steps", type=int, default=shipped.max_steps,
                        help=f"step ceiling (default: shipped, {shipped.max_steps})")
    parser.add_argument("--target-steps", type=int, default=shipped.target_steps,
                        help=f"where budget pressure begins "
                             f"(default: shipped, {shipped.target_steps})")
    parser.add_argument("--best-of", type=int, default=1, metavar="K",
                        help="sample K attempts per case and keep the first the "
                             "harness itself accepts (never the grader). Costs K "
                             "times as much; reported separately from single-shot.")
    parser.add_argument("--case", action="append",
                        help="run only these case ids (repeatable)")
    parser.add_argument("--tier", choices=["core", "hard", "all"], default="all",
                        help="which difficulty tier to run (default: all)")
    parser.add_argument("--cases", default=None, metavar="FILE",
                        help="run an external JSON suite instead of the "
                             "built-in calibration cases")
    parser.add_argument("--harness-cmd", default=None, metavar="CMD",
                        help="grade a third-party coding agent instead of "
                             "Knossos. Must contain {prompt}; run inside the "
                             "fixture directory. Example: "
                             "'claude -p {prompt} --permission-mode acceptEdits'")
    parser.add_argument("--harness-timeout", type=int, default=300,
                        metavar="SECONDS",
                        help="wall clock per case for --harness-cmd "
                             "(default: 300)")
    parser.add_argument("--workdir", default=None,
                        help="where fixtures are built; a temp dir by default")
    parser.add_argument("--trace-dir", default=None,
                        help="write one JSONL trace per case here")
    args = parser.parse_args(argv)

    if not args.agent and not args.engine and not args.harness_cmd:
        parser.error("pass --agent for a calibration run, --engine for a real "
                     "one, or --harness-cmd to grade a third-party harness")
    if args.harness_cmd and (args.agent or args.engine):
        parser.error("--harness-cmd grades a foreign harness; it does not "
                     "combine with --agent or --engine")

    if args.cases:
        try:
            cases = load_cases(args.cases)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        print(f"  suite: {len(cases)} case(s) from {args.cases}")
        print("  note: an external suite is not the built-in calibration set; "
              "do not compare the two scores.")
    else:
        cases = list(CODING_CASES)
    if args.tier != "all":
        cases = [c for c in cases if c.tier == args.tier]
    if args.case:
        wanted = set(args.case)
        cases = [c for c in cases if c.id in wanted]
        if not cases:
            parser.error(f"no case matched {sorted(wanted)}")
    tiers = {c.id: c.tier for c in cases}

    import tempfile
    workdir = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="codeval-"))
    trace_dir = Path(args.trace_dir) if args.trace_dir else None

    print(f"\n  fixtures    {workdir}")
    if trace_dir:
        print(f"  traces      {trace_dir}")

    engine = None
    wall = 0.0
    if args.agent:
        print(f"  agent       {args.agent} (scripted, no model)")
        results: List[CaseResult] = []
        for case in cases:
            # `--best-of` applies to the calibration agents too, deliberately.
            # Sampling k times is adversarial pressure on the verifier: it
            # selects for exactly what the check blesses, so the vandal gets k
            # chances to find a route past the Oracle rather than one. Zero
            # false passes single-shot is not evidence of zero under selection.
            results += run_suite(scripted_agent(args.agent, case.id), workdir,
                                 cases=[case], trace_dir=trace_dir,
                                 attempts=args.best_of)
    elif args.harness_cmd:
        print(f"  harness     {args.harness_cmd}")
        print(f"  timeout     {args.harness_timeout}s per case")
        print("  note: a foreign harness reports only its exit status, so "
              "`honest` measures\n        whether that claim matched the "
              "graded truth. Steps and tokens are\n        blank unless it "
              "prints them.")
        started = time.perf_counter()
        results = run_suite(
            external_agent(args.harness_cmd, args.harness_timeout), workdir,
            cases=cases, trace_dir=trace_dir, attempts=args.best_of)
        wall = time.perf_counter() - started
    else:
        from knossos.engine import OpenAICompatEngine
        engine = OpenAICompatEngine(provider=args.provider, model=args.model,
                                    base_url=args.base_url)
        print(f"  engine      {engine.base_url}  model={engine.model}")
        print(f"  budget      {args.max_steps} max / {args.target_steps} target")
        started = time.perf_counter()
        results = run_suite(live_agent(engine, args.max_steps, args.target_steps),
                            workdir,
                            cases=cases, trace_dir=trace_dir,
                            attempts=args.best_of)
        wall = time.perf_counter() - started

    passed = report(results, tiers)
    if engine is not None:
        report_throttling(engine, wall)

    # The calibration agents replay solutions keyed to the *built-in* case ids
    # (`SOLUTIONS` in this file), so `oracle` cannot solve a suite it has never
    # seen. Reporting "the grader is wrong" there would be a false alarm about
    # the one instrument the rest of the suite depends on -- and a false alarm
    # on a correctness guard is how people learn to ignore it.
    if args.agent == "oracle" and args.cases:
        print("  ** `--agent oracle` only calibrates the built-in cases: its "
              "solutions are keyed\n     to their ids, so it cannot solve an "
              "external suite. Use --engine for a real run.\n")
        return 0
    if args.agent == "oracle" and passed != len(results):
        print("  !! the oracle agent did not solve every case: the grader is wrong,")
        print("     not the agent. Do not trust any other number until this passes.\n")
        return 1
    if args.agent in ("lazy", "vandal") and passed != 0:
        print(f"  !! the {args.agent} agent solved something. The eval is gameable"
              " and\n     every number from it is worthless until that is fixed.\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
