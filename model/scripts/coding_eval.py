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
from knossos.talos import Talos, accept_everything                    # noqa: E402
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


#: (path, size, mtime_ns) -> (content hash, when those bytes were read).
#:
#: This is a prefilter, not a substitute: the value returned is still the hash,
#: so "changed" still means *the bytes differ* rather than "the timestamp moved".
#: What it removes is re-reading a file to learn something already known.
#: `_snapshot` ran twice per case over the whole tree, which is nothing for the
#: built-in fixtures and O(repo bytes x 2 x cases) the moment `--cases` points
#: at a real repository -- which is the entire reason `--cases` exists.
#:
#: The read time is the half that makes the key safe to believe. An unchanged
#: mtime only means "not rewritten" if the clock that stamped it can resolve the
#: gap between two writes, and it cannot: Windows advances the file-write clock
#: in ~15.6ms ticks, so a same-size rewrite inside one tick lands on the
#: identical mtime_ns, matches this key, and is served a hash of the *previous*
#: bytes. That is the external arm quietly under-reporting the work it graded --
#: an agent's same-size edit scored as no change at all. So an entry is believed
#: only once its file has gone quiet for `_SETTLED_NS`; see `_snapshot`.
#:
#: Process-lifetime and unbounded, which is the right size here: one entry per
#: file the eval has looked at, in a process that exists to walk that tree.
_HASHES: Dict[tuple, tuple] = {}


#: How far an mtime must sit below the read that hashed it before that hash can
#: be reused. It has to exceed the stamping clock's granularity, and comparing
#: the two timestamps directly does not: `time.time_ns` is sub-microsecond on
#: Windows while `st_mtime_ns` is quantised down to the ~15.6ms tick, so a read
#: lands *above* the mtime of a write it raced and the entry would look safe.
#: Two seconds clears every granularity that turns up in practice -- 15.6ms on
#: NTFS, 2s on FAT and some network mounts -- so past it no later write can
#: still quantise onto the mtime already recorded here.
#:
#: The cost is that a file touched in the last two seconds is re-read rather
#: than trusted, which is the pre-cache behaviour for the one window where the
#: timestamp genuinely cannot distinguish the two. It costs nothing in the
#: workload this exists for: `--cases` walks a tree that was materialised once
#: and then sat still for however long the foreign harness ran.
_SETTLED_NS = 2_000_000_000


def _snapshot(root: Path) -> Dict[str, str]:
    """Content hash of every file under `root`, for before/after diffing."""
    out: Dict[str, str] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in _IGNORED_DIRS]
        for name in filenames:
            path = Path(dirpath) / name
            try:
                info = path.stat()
                key = (str(path), info.st_size, info.st_mtime_ns)
                cached = _HASHES.get(key)
                if cached is not None and (
                        info.st_mtime_ns + _SETTLED_NS <= cached[1]):
                    digest = cached[0]
                else:
                    # Sampled before the read, never after: a write landing
                    # mid-read must leave an mtime this entry refuses, and a
                    # timestamp taken afterwards would sit above it and pass.
                    read_at = time.time_ns()
                    digest = hashlib.sha1(path.read_bytes()).hexdigest()
                    _HASHES[key] = (digest, read_at)
                out[str(path.relative_to(root))] = digest
            except OSError:
                continue
    return out


class _Halt:
    def __init__(self, value: str) -> None:
        self.value = value


class _ExternalOutcome:
    #: An exit status, not a self-assessment. Recorded so `report` can refuse to
    #: print this run's `honest` as though it were Oracle's verdict.
    claim_source = "exit_code"
    #: A foreign harness is stopped by `--harness-timeout`, not by a step
    #: ceiling. Naming it keeps a wall-clock arm from being compared against a
    #: steps arm as if the same constraint bound both.
    budget_kind = "wall_clock"

    def __init__(self, succeeded: bool, halt: str, steps_used: int,
                 changed: Sequence[str], tools_used: int = 0) -> None:
        self.succeeded = succeeded
        self.halt = _Halt(halt)
        self.steps_used = steps_used
        self.tools_used = tools_used
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


def run_dir(base: Optional[Path], index: int, total: int) -> Optional[Path]:
    """Where run `index` of `total` puts its files. A separate dir per repeat.

    Extracted from `main` so the property can be tested, because it is the one
    that makes `--repeat` mean anything and it fails silently when wrong.
    `materialise` rewrites each case's own files, but it has no way to know
    about a file the *agent* created -- nothing enumerates those. Share one
    directory across repeats and run 2 starts from whatever run 1 left behind:
    still a number, still plausible, no longer an independent sample of the same
    thing. Since the whole purpose of `--repeat` is to give a score its `n`,
    that failure would quietly void the flag while appearing to work.

    `None` passes through for the trace directory, which is optional, and a
    single run keeps the bare path so nothing changes for the common case.
    """
    if base is None or total <= 1:
        return base
    return base / f"run{index + 1}"


def _fresh_case(engine) -> None:
    """Undo the previous case's transient throttling, if any.

    `make_agent(root)` is called once per case, so this is the case boundary.
    Without it a 413 in case 1 held `max_tokens` down for every case after it --
    every shrink path in the engine is one-way -- and the suite measured a
    descending staircase of allowances rather than a model. Duck-typed because
    `AgentFactory` is "anything with `.run(prompt)`" and a scripted agent has no
    engine to restore.
    """
    restore = getattr(engine, "restore_limits", None)
    if callable(restore):
        restore()


def live_agent(engine, max_steps: int, target_steps: int):
    def make(root: Path):
        _fresh_case(engine)
        ws = Workspace(root)
        # Baseline on: it is what records the suite's size before the agent
        # starts, which is what catches a pass bought by deleting a test.
        oracle = Oracle(root)
        return Talos(engine, ws, verifier=oracle, interim=oracle.quick,
                     ariadne=Ariadne(max_steps=max_steps,
                                     target_steps=target_steps))
    return make


def baseline_agent(engine):
    """The control arm: the same model, one step, nothing checking it.

    # Why this has to exist

    Every number this script produced before now was *model x harness* with no
    way to separate the two. A 12/12 could mean the harness works or it could
    mean the model is strong enough that a single API call would also score
    12/12, and nothing here could tell those apart. `knossos.eval` has had a
    control arm from the start -- it runs every case `raw` and `harness` and the
    README calls the control "the load-bearing part", correctly noting that a
    lift on it would invalidate the treatment number too. The coding eval was
    the weaker instrument of the two and this closes that gap.

    # What it does and does not hold constant

    Held constant: the engine, the tool vocabulary, the workspace jail, the
    prompt, the fixtures and the grader. Removed: **the loop and the verifier**,
    and only those. `max_steps=1` means one turn with no chance to read a
    result and correct, and `accept_everything` means the run ends on the
    engine's own say-so.

    So the delta this measures is precisely *what iteration and verification
    buy on top of the same model*. It is deliberately **not** a claim about
    "the harness versus no harness at all" -- retrieval, the path jail and the
    tool layer are all still present, and a bare `curl` to the same provider
    would score lower than this for reasons that have nothing to do with Talos.
    Quoting it as the latter would be the same overclaim the control exists to
    prevent.

    # Reading the result

    Expect the baseline to post a high `FALSE PASS` count. That is not a bug in
    the control -- `accept_everything` agrees with the engine by construction,
    so the run reports success whenever the model stops calling tools. It is the
    measurement: it is what "completion decided by the engine" scores, which is
    the claim the whole project exists to refuse. `claim_source` is recorded as
    `unverified` so the number can never be printed beside a verified arm's
    `honest` as though they were the same quantity.
    """
    def make(root: Path):
        # The control arm has to start each case from the same allowance the
        # treatment does, or the comparison measures throttling history.
        _fresh_case(engine)
        ws = Workspace(root)
        # No Oracle, and no `interim`. A verifier here would be the treatment.
        return Talos(engine, ws, verifier=accept_everything,
                     ariadne=Ariadne(max_steps=1, target_steps=1))
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
    # Labelled by where the claim came from, because the two sources are not the
    # same measurement. Oracle's verdict is a deterministic self-assessment; a
    # foreign CLI's exit status is process health. Most agent CLIs exit 0 unless
    # they crash, so for an external arm `harness_said_done` is almost always
    # True, `honest` collapses into `passed`, and a `false fail` becomes
    # unscoreable. Printing both under one heading invites exactly the
    # cross-harness comparison the number cannot support.
    by_exit = [r for r in measured if r.claim_source == "exit_code"]
    if by_exit and len(by_exit) == len(measured):
        print(f"  honest*     {honest}/{total}   "
              f"(* exit status, not a verdict -- see below)")
    else:
        print(f"  honest      {honest}/{total}   "
              f"(harness's own verdict matched reality)")
    if by_exit:
        print(f"              {len(by_exit)}/{total} case(s) claimed completion "
              f"by exit code alone. A CLI that")
        print("              exits 0 unless it crashes cannot score a false "
              "fail, so this is")
        print("              not comparable with a verifier-backed `honest`.")

    # A case the provider throttled or shrunk did reach the model, so it is not
    # `unreachable` -- but it did not run the configuration that was asked for.
    # The console already warns; this puts it next to the score it qualifies.
    degraded = [r for r in measured if r.degraded]
    if degraded:
        print(f"  degraded    {len(degraded)}/{total}   "
              f"(ran with a reduced allowance or after backoff)")

    # The failure `tamper` cannot see, and it must be louder than a pass.
    overfit = [r for r in measured if r.overfit]
    if overfit:
        print(f"  OVERFIT     {len(overfit)}: "
              f"{', '.join(r.case_id for r in overfit)}")
        print("              passed every visible test and failed a held-out "
              "one: the")
        print("              assertions were solved, not the task")
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

    _report_steps(measured)
    _report_cost(measured, passed)
    print()
    return passed


def _report_steps(results: Sequence[CaseResult]) -> None:
    """How many turns the harness spent on the cases it actually solved.

    The axis that still has signal once `solved` saturates, and the number was
    already being collected -- `CaseResult.steps_used` is populated for every
    run and `summary()` prints it per case. Only the aggregate was missing, so
    two runs that differ by a factor of two looked identical in the report that
    gets quoted.

    Measured on the built-in suite: Gemini 3.1 Pro and Gemini 3.1 Flash-Lite
    both score 12/12, which is the saturation this tier cannot see past, while
    the same two runs sit at 6.9 and 14.2 mean steps. On the easiest case in the
    set the gap is 3 steps against 20-and-the-ceiling.

    Solved cases only, for the reason `_report_cost` gives about tokens: a
    harness that fails cheaply is not thereby efficient. A case that exhausted
    its budget reports the ceiling rather than a cost, so averaging it in
    rewards giving up early and penalises persistence that went on to work.

    `steps_used == 0` is dropped rather than averaged in. A foreign harness
    whose CLI prints no turn count reports zero (`ExternalHarness._steps`), and
    a zero folded into a mean reads as a harness that solved the task
    instantly -- the same confidently wrong number the token line refuses to
    print. Excluded cases are counted on their own line instead.
    """
    solved = [r for r in results if r.passed]
    counted = sorted(r.steps_used for r in solved if r.steps_used > 0)
    if not counted:
        print("  steps       not reported (the harness printed no turn count)")
        return

    mean = sum(counted) / len(counted)
    print(f"  steps       {mean:.1f} mean per solved case   "
          f"(range {counted[0]}-{counted[-1]}, n={len(counted)})")
    missing = len(solved) - len(counted)
    if missing:
        print(f"              {missing} solved case(s) reported no turn count "
              f"and are excluded")

    # Printed next to steps rather than instead of it, because neither is
    # sufficient alone. A step is not a fixed unit of work: this loop batches
    # adjacent parallel-safe calls into one turn, so a model that emits its
    # reads together spends fewer steps for identical work and would rank as
    # more capable on steps alone. Tools-per-step is what makes that visible --
    # if two runs differ on steps and agree on tools, the difference is batching
    # habit, not capability.
    with_tools = [r for r in solved if r.tools_used > 0]
    tool_steps = sum(r.steps_used for r in with_tools)
    if with_tools:
        total = sum(r.tools_used for r in with_tools)
        # A harness can report tool calls without reporting turns, so the ratio
        # is guarded rather than assumed -- the mean is still worth printing.
        ratio = (f" ({total / tool_steps:.1f} per step)" if tool_steps else "")
        print(f"  tools       {total / len(with_tools):.1f} mean per solved case  "
              f"{ratio}")


def report_spread(runs: Sequence[Sequence[CaseResult]]) -> None:
    """What repeating the whole suite showed, and how far the runs disagreed.

    `scripts/seeds.py` says a score without its `n` is not a result, and until
    now neither eval imported it: every number this suite has ever produced was
    single-shot. That is the weaker claim in two directions. A run that scores
    12/12 once might score 10/12 typically, and the README already documents the
    opposite error -- at n=1 the retrieval eval understated an effect by half.

    Steps need this more than `solved` does, not less. A pass is one bit with
    bounded variance; a step count is unbounded, and the biggest per-case gap
    observed so far (3 steps against 20) is exactly the shape of a difference
    that could be a single unlucky sample.

    Range rather than a standard deviation: at the `n` anyone will actually pay
    for -- three, maybe five -- a standard deviation is a statistic about too
    few points to mean much, while "best and worst run" is exactly what a reader
    wants to know and cannot be over-read.
    """
    def measured(batch):
        return [r for r in batch if not r.unreachable]

    print()
    print(f"  == across {len(runs)} run(s) " + "=" * 40)
    for label, pick in (("solved", lambda b: sum(1 for r in b if r.passed)),
                        ("honest", lambda b: sum(1 for r in b if r.honest))):
        counts = [pick(measured(b)) for b in runs]
        denom = [len(measured(b)) for b in runs]
        if not counts or not any(denom):
            continue
        spread = ("" if min(counts) == max(counts)
                  else f"   range {min(counts)}-{max(counts)}")
        print(f"  {label:<11} {sum(counts) / len(counts):.1f} mean "
              f"of {max(denom)}{spread}")

    per_run = []
    for batch in runs:
        counted = [r.steps_used for r in batch if r.passed and r.steps_used > 0]
        if counted:
            per_run.append(sum(counted) / len(counted))
    if per_run:
        spread = ("" if len(per_run) < 2 or min(per_run) == max(per_run)
                  else f"   range {min(per_run):.1f}-{max(per_run):.1f}")
        print(f"  {'steps':<11} {sum(per_run) / len(per_run):.1f} mean "
              f"per solved case{spread}")
    if len(runs) < 3:
        # Said plainly, because two runs that agree look far more convincing
        # than they are and this is the number people will quote.
        print("  note        n=2 shows whether the runs differ, not by how much. "
              "Three or more")
        print("              before quoting a mean as though it had a spread.")
    print()


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
    parser.add_argument("--baseline", action="store_true",
                        help="control arm: the same engine with one step and no "
                             "verifier, isolating what the loop and Oracle buy. "
                             "Requires --engine.")
    parser.add_argument("--repeat", type=int, default=1, metavar="N",
                        help="run the whole suite N times and report the spread. "
                             "Different from --best-of, which keeps the best "
                             "sample and inflates the score; this reports every "
                             "run and the range across them.")
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
    if args.baseline and not args.engine:
        parser.error("--baseline is a control for a real run: pass --engine")
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")
    if args.repeat > 1 and args.agent:
        # A scripted agent replays a fixed script, so repeating it samples the
        # same run N times. The spread would be zero by construction and would
        # read as an unusually tight result rather than a deterministic one.
        parser.error("--repeat measures sampling variance; the scripted agents "
                     "are deterministic")

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
    runs: List[List[CaseResult]] = []

    def repeated(factory) -> List[CaseResult]:
        """Run the suite `--repeat` times, reporting each, and keep them all.

        Each repeat gets its own fixture directory. Sharing one would let a file
        the agent *created* in run 1 survive into run 2 -- `materialise` rewrites
        the case's own files but has no way to know about anything else, so the
        second run would start from a tree the first one left behind and the
        repeats would not be independent samples of the same thing.
        """
        for index in range(args.repeat):
            tag = f"run{index + 1}"
            here = run_dir(workdir, index, args.repeat)
            where = run_dir(trace_dir, index, args.repeat)
            if args.repeat > 1:
                print(f"\n  == {tag} of {args.repeat} "
                      + "=" * 40)
            batch = run_suite(factory, here, cases=cases, trace_dir=where,
                              attempts=args.best_of)
            runs.append(batch)
            if args.repeat > 1:
                report(batch, tiers)
        return runs[-1]

    if args.agent:
        print(f"  agent       {args.agent} (scripted, no model)")
        results = []
        for case in cases:
            # `--best-of` applies to the calibration agents too, deliberately.
            # Sampling k times is adversarial pressure on the verifier: it
            # selects for exactly what the check blesses, so the vandal gets k
            # chances to find a route past the Oracle rather than one. Zero
            # false passes single-shot is not evidence of zero under selection.
            results += run_suite(scripted_agent(args.agent, case.id), workdir,
                                 cases=[case], trace_dir=trace_dir,
                                 attempts=args.best_of)
        # Not routed through `repeated`: these agents replay a fixed script, so
        # every run is identical by construction and a spread over them would be
        # a confidence interval of exactly zero -- which reads as a strong result
        # rather than as a deterministic one.
        runs.append(results)
    elif args.harness_cmd:
        print(f"  harness     {args.harness_cmd}")
        print(f"  timeout     {args.harness_timeout}s per case")
        print("  note: a foreign harness reports only its exit status, so "
              "`honest` measures\n        whether that claim matched the "
              "graded truth. Steps and tokens are\n        blank unless it "
              "prints them.")
        started = time.perf_counter()
        results = repeated(external_agent(args.harness_cmd, args.harness_timeout))
        wall = time.perf_counter() - started
    else:
        from knossos.engine import OpenAICompatEngine
        engine = OpenAICompatEngine(provider=args.provider, model=args.model,
                                    base_url=args.base_url)
        print(f"  engine      {engine.base_url}  model={engine.model}")
        if args.baseline:
            print("  arm         BASELINE control -- 1 step, no verifier")
            print("              Isolates what the loop and Oracle buy on this "
                  "model. Expect")
            print("              false passes: nothing is checking the claim. "
                  "Not a harness score.")
            factory = baseline_agent(engine)
        else:
            print(f"  budget      {args.max_steps} max / {args.target_steps} target")
            factory = live_agent(engine, args.max_steps, args.target_steps)
        started = time.perf_counter()
        results = repeated(factory)
        wall = time.perf_counter() - started

    if args.repeat > 1:
        report_spread(runs)
        passed = sum(1 for r in results if r.passed)
    else:
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
