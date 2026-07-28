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

    @property
    def total(self) -> int:
        return self.measured + self.unreachable

    @property
    def rate(self) -> Optional[float]:
        """None when nothing was measured -- distinct from zero."""
        return 100.0 * self.passed / self.measured if self.measured else None

    def add(self, case: str, passed: bool, unreachable: bool, halt: str,
            tamper: bool, derived: bool) -> None:
        self.cases.append((case, passed, unreachable, halt))
        self.tamper += bool(tamper)
        self.reconstructed += bool(derived)
        if unreachable:
            self.unreachable += 1
            return
        self.measured += 1
        self.passed += bool(passed)
        self.halts[halt or "?"] += 1


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
            if not derived:
                return {**header, "derived": False}

            # Old trace: recover the fact from the events, the same way
            # `run_case` counts it live.
            api_errors = 0
            for line in fh:
                try:
                    event = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if (event.get("kind") == "text"
                        and _API_FAILURE in (event.get("text") or "")):
                    api_errors += 1
    except (OSError, ValueError, TypeError):
        return None

    header["api_errors"] = api_errors
    header["unreachable"] = bool(api_errors) and not header.get("passed")
    header["derived"] = True
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
                    derived=bool(header.get("derived")))
        if run.total:
            runs.append(run)
    return runs


def report(runs: List[RunSummary], verbose: bool = False) -> None:
    # Sorted so unmeasured runs sink rather than sorting as zeroes among real
    # results -- the whole point is that they are not the bottom of a ranking,
    # they are absent from it.
    runs.sort(key=lambda r: (r.rate is None, -(r.rate or 0.0), r.name))

    print(f"{'run':38} {'solved':>9} {'rate':>7} {'unreach':>8}  halts")
    print("-" * 96)
    for run in runs:
        rate = "     --" if run.rate is None else f"{run.rate:6.1f}%"
        solved = ("      --" if run.rate is None
                  else f"{run.passed:3}/{run.measured:<4}")
        marker = "*" if run.reconstructed else " "
        print(f"{run.name:38}{marker}{solved:>8} {rate} "
              f"{run.unreachable:4}/{run.total:<3}  {dict(run.halts)}")
        if verbose:
            for case, passed, unreachable, halt in run.cases:
                state = ("unreachable" if unreachable
                         else ("PASS" if passed else "fail"))
                print(f"    {state:>12}  {case:<38} {halt}")

    measured_runs = [r for r in runs if r.rate is not None]
    dead = [r for r in runs if r.rate is None]
    print("-" * 96)
    print(f"{len(runs)} run(s); {len(measured_runs)} measured something, "
          f"{len(dead)} reached no model at all.")
    if any(r.reconstructed for r in runs):
        print("* `unreachable` reconstructed from the event stream: this trace "
              "predates the recorded field.")
    if dead:
        print("Runs that measured nothing (excluded from every rate above):")
        print("  " + ", ".join(r.name for r in dead))
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
