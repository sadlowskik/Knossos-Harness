"""Aggregate coding-eval traces across runs, without lying about outages.

# Why this exists

`coding_eval.py` prints a report at the end of a run and the report is careful:
it says, loudly, when the provider refused requests and the score is therefore
not a capability result. Then the terminal scrolls, and what survives is
`traces/<run>/<case>.jsonl`.

A trace header written before this script existed carries `passed` and `halt`
and nothing about whether the model was ever reached. Both fields are populated
identically by an incapable model and by an exhausted quota: `passed: false`,
and `halt: "stuck"`, because an agent handed `Request failed: HTTP 429` in place
of a reply makes no tool calls and trips the no-op detector on the second step.

So anyone aggregating a directory of old traces -- which is the entire reason to
keep them -- reads a rate limit as a score. That is not hypothetical; it is what
happened the first time these traces were summarised, and the conclusion drawn
was that a frontier model had scored zero on a task a 4B local model solved.

# What it does about it

New traces record `unreachable` directly (`codeval._write_trace`). For traces
written before that, this script reconstructs it the only way available: by
looking for the engine's failure prefix in the run's own `text` events, which is
exactly what `run_case` does live. Reconstructed rows are marked as such,
because a derived field and a recorded one do not deserve the same confidence.

Cases that never reached the model are excluded from the rate and counted on
their own line. A run with nothing left after that exclusion prints `--`, not
`0%`.

    python -m scripts.trace_summary
    python -m scripts.trace_summary --dir model/traces --verbose
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib
import sys
from typing import Dict, List, Optional, Tuple

#: Must match `codeval._API_FAILURE`. Imported rather than copied when the
#: package is importable; the literal is the fallback so this script still works
#: against a checkout whose `knossos` is not on the path.
try:
    from knossos.codeval import _API_FAILURE
except Exception:                                 # noqa: BLE001 - best effort
    _API_FAILURE = "Request failed:"


class RunSummary:
    """One trace directory, read honestly."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.passed = 0
        self.measured = 0
        self.unreachable = 0
        self.tamper = 0
        self.reconstructed = 0
        self.halts: collections.Counter = collections.Counter()
        self.cases: List[Tuple[str, bool, bool, str]] = []
        #: Turn counts of the cases this run solved. Solved-only and non-zero
        #: only, matching `coding_eval._report_steps` -- see the reasoning
        #: there. Kept as a list rather than a running sum so the range can be
        #: shown; the spread is where the signal was.
        self.steps: List[int] = []
        #: Solved cases whose turn count was unavailable, reported separately
        #: rather than folded in as zeros.
        self.steps_missing = 0
        self.steps_derived = False
        #: Cases that reached the model but not in the configuration asked for.
        #: The category that had no home: `unreachable` covers a refused
        #: request, and nothing covered a request that succeeded after two
        #: minutes of backoff with the reply allowance halved.
        self.degraded = 0
        #: Cases whose completion claim was a process exit code rather than a
        #: verdict, so this run's `halts` cannot be read as self-assessment.
        self.by_exit = 0
        #: Cases that passed everything visible and failed a held-out test.
        self.overfit = 0

    @property
    def total(self) -> int:
        return self.measured + self.unreachable

    @property
    def rate(self) -> Optional[float]:
        """None when nothing was measured -- distinct from zero."""
        return 100.0 * self.passed / self.measured if self.measured else None

    @property
    def mean_steps(self) -> Optional[float]:
        """None when no solved case reported a turn count -- not zero."""
        return sum(self.steps) / len(self.steps) if self.steps else None

    def add(self, case: str, passed: bool, unreachable: bool, halt: str,
            tamper: bool, derived: bool, steps: int = 0,
            steps_derived: bool = False, degraded: bool = False,
            claim_source: str = "", overfit: bool = False) -> None:
        self.cases.append((case, passed, unreachable, halt))
        self.tamper += bool(tamper)
        self.reconstructed += bool(derived)
        self.overfit += bool(overfit)
        if claim_source == "exit_code":
            self.by_exit += 1
        if unreachable:
            self.unreachable += 1
            return
        # Counted only among measured cases: a case the provider refused was
        # not degraded, it was absent, and reporting it as both would double
        # count the same outage under two headings.
        self.degraded += bool(degraded)
        self.measured += 1
        self.passed += bool(passed)
        self.halts[halt or "?"] += 1
        if passed:
            if steps > 0:
                self.steps.append(steps)
                self.steps_derived |= bool(steps_derived)
            else:
                self.steps_missing += 1


def read_trace(path: pathlib.Path) -> Optional[Dict]:
    """Header plus a reconstructed `unreachable`, or None if unreadable.

    A truncated or corrupt trace is skipped rather than guessed at. A partial
    run is not a zero.
    """
    try:
        with path.open(encoding="utf-8") as fh:
            first = fh.readline()
            if not first.strip():
                return None
            header = json.loads(first)
            if "passed" not in header:
                return None

            derived = "unreachable" not in header
            needs_steps = "steps" not in header
            if not (derived or needs_steps):
                return {**header, "derived": False, "steps_derived": False}

            # Old trace: recover the facts from the events, the same way
            # `run_case` counts them live. One pass serves both fields.
            api_errors = 0
            steps = 0
            for line in fh:
                try:
                    event = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if event.get("kind") == "step":
                    steps += 1
                if (event.get("kind") == "text"
                        and _API_FAILURE in (event.get("text") or "")):
                    api_errors += 1
    except (OSError, ValueError, TypeError):
        return None

    # Guarded, because a trace can predate one field and not the other: an
    # unconditional write here would overwrite a recorded `unreachable` with a
    # value derived from the same events it was already computed from.
    if derived:
        header["api_errors"] = api_errors
        header["unreachable"] = bool(api_errors) and not header.get("passed")
    if needs_steps:
        # Zero here means "no `step` events", which for an external-harness
        # trace means the count is unavailable rather than genuinely zero.
        # `RunSummary.add` drops zeros instead of averaging them.
        header["steps"] = steps
    header["derived"] = derived
    header["steps_derived"] = needs_steps
    return header


def summarise(directory: pathlib.Path) -> List[RunSummary]:
    runs = []
    for child in sorted(directory.iterdir()):
        if not child.is_dir():
            continue
        run = RunSummary(child.name)
        for trace in sorted(child.glob("*.jsonl")):
            header = read_trace(trace)
            if header is None:
                continue
            run.add(case=header.get("case", trace.stem),
                    passed=bool(header.get("passed")),
                    unreachable=bool(header.get("unreachable")),
                    halt=header.get("halt", ""),
                    tamper=bool(header.get("tamper")),
                    derived=bool(header.get("derived")),
                    steps=int(header.get("steps") or 0),
                    steps_derived=bool(header.get("steps_derived")),
                    degraded=bool(header.get("degraded")),
                    claim_source=str(header.get("claim_source") or ""),
                    overfit=bool(header.get("overfit")))
        if run.total:
            runs.append(run)
    return runs


def report(runs: List[RunSummary], verbose: bool = False) -> None:
    # Sorted so unmeasured runs sink rather than sorting as zeroes among real
    # results -- the whole point is that they are not the bottom of a ranking,
    # they are absent from it.
    runs.sort(key=lambda r: (r.rate is None, -(r.rate or 0.0), r.name))

    print(f"{'run':38} {'solved':>9} {'rate':>7} {'steps':>8} "
          f"{'unreach':>8}  halts")
    print("-" * 104)
    for run in runs:
        rate = "     --" if run.rate is None else f"{run.rate:6.1f}%"
        solved = ("      --" if run.rate is None
                  else f"{run.passed:3}/{run.measured:<4}")
        marker = "*" if run.reconstructed else " "
        # `--` rather than 0.0 when nothing reported a count: a run whose
        # harness prints no turn count has not been measured on this axis, and
        # a zero would sort as the most efficient run in the table.
        # ASCII marker on purpose: this script prints to a Windows console,
        # where the default code page is cp1252 and a `†` lands as a literal
        # `?`. A legend nobody can read is worse than a plainer one.
        steps = ("      --" if run.mean_steps is None
                 else f"{run.mean_steps:7.1f}{'+' if run.steps_derived else ' '}")
        print(f"{run.name:38}{marker}{solved:>8} {rate} {steps:>8} "
              f"{run.unreachable:4}/{run.total:<3}  {dict(run.halts)}")
        if verbose:
            for case, passed, unreachable, halt in run.cases:
                state = ("unreachable" if unreachable
                         else ("PASS" if passed else "fail"))
                print(f"    {state:>12}  {case:<38} {halt}")

    measured_runs = [r for r in runs if r.rate is not None]
    dead = [r for r in runs if r.rate is None]
    print("-" * 104)
    print(f"{len(runs)} run(s); {len(measured_runs)} measured something, "
          f"{len(dead)} reached no model at all.")
    if any(r.reconstructed for r in runs):
        print("* `unreachable` reconstructed from the event stream: this trace "
              "predates the recorded field.")
    if any(r.steps_derived for r in runs):
        print("+ steps reconstructed by counting `step` events, for traces "
              "predating the recorded field.")
    missing = sum(r.steps_missing for r in runs)
    if missing:
        # Named rather than silently dropped: this is exactly the population an
        # external-harness comparison lands in, and a reader who does not know
        # the column is empty for them will read `--` as a result.
        print(f"{missing} solved case(s) reported no turn count and are excluded "
              f"from `steps`.")
        print("  An external harness reports one only if its CLI prints a turn "
              "count on stdout")
        print("  (`ExternalHarness._STEP_KEYS`); Claude Code needs "
              "`--output-format json`.")
    if dead:
        print("Runs that measured nothing (excluded from every rate above):")
        print("  " + ", ".join(r.name for r in dead))
    degraded = [r for r in runs if r.degraded]
    if degraded:
        # The whole reason this script exists, applied to the field it did not
        # cover: the console said so at run time and the console is gone.
        print("Ran degraded (reduced reply allowance, or after rate-limit "
              "backoff):")
        for run in degraded:
            print(f"  {run.name}: {run.degraded}/{run.measured} case(s). "
                  f"Not a clean capability result.")
    by_exit = [r for r in runs if r.by_exit]
    if by_exit:
        print("Completion claimed by exit code, not by a verifier "
              "(`honest` is not comparable):")
        print("  " + ", ".join(f"{r.name} ({r.by_exit})" for r in by_exit))
    overfit = [r for r in runs if r.overfit]
    if overfit:
        print("PASSED THE VISIBLE TESTS AND FAILED A HELD-OUT ONE:")
        print("  " + ", ".join(f"{r.name} ({r.overfit})" for r in overfit))
    tampered = sum(r.tamper for r in runs)
    print(f"Test-file tampering attempts across all runs: {tampered}"
          + ("  (restored before grading; bought nothing)" if tampered else ""))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarise coding-eval traces, excluding provider outages.")
    parser.add_argument("--dir", default="model/traces", type=pathlib.Path,
                        help="directory of per-run trace directories")
    parser.add_argument("--verbose", action="store_true",
                        help="list every case, not just the per-run totals")
    args = parser.parse_args(argv)

    if not args.dir.is_dir():
        print(f"no such directory: {args.dir}", file=sys.stderr)
        return 2
    runs = summarise(args.dir)
    if not runs:
        print(f"no readable traces under {args.dir}", file=sys.stderr)
        return 1
    report(runs, verbose=args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
