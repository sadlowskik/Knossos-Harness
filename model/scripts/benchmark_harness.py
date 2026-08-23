#!/usr/bin/env python
"""Measure Knossos overhead without contacting a model.

The benchmark launches the release binary against a local fixture and records
wall time, peak resident memory, output size, and binary size.  It intentionally
uses `--engine none index`: retrieval/index construction is real, while provider
latency and model memory are excluded.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import pathlib
import statistics
import subprocess
import sys
import time
from typing import Any, Optional, Sequence


def rss_bytes(pid: int) -> Optional[int]:
    if sys.platform == "win32":
        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        process = ctypes.windll.kernel32.OpenProcess(0x0400 | 0x0010, False, pid)
        if not process:
            return None
        try:
            counters = Counters()
            counters.cb = ctypes.sizeof(counters)
            if not ctypes.windll.psapi.GetProcessMemoryInfo(
                process, ctypes.byref(counters), counters.cb
            ):
                return None
            return int(counters.WorkingSetSize)
        finally:
            ctypes.windll.kernel32.CloseHandle(process)

    status = pathlib.Path(f"/proc/{pid}/status")
    if status.exists():
        for line in status.read_text(errors="replace").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return None


def run_once(command: Sequence[str], timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    process = subprocess.Popen(
        list(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    peak = 0
    deadline = started + timeout
    while process.poll() is None:
        sample = rss_bytes(process.pid)
        if sample is not None:
            peak = max(peak, sample)
        if time.perf_counter() >= deadline:
            process.kill()
            stdout, stderr = process.communicate()
            raise TimeoutError(
                f"benchmark command exceeded {timeout}s: {' '.join(command)}\n{stderr}"
            )
        time.sleep(0.005)
    stdout, stderr = process.communicate()
    elapsed = time.perf_counter() - started
    if process.returncode:
        raise RuntimeError(
            f"benchmark command exited {process.returncode}: {' '.join(command)}\n{stderr}"
        )
    return {
        "wall_ms": round(elapsed * 1000, 3),
        "peak_rss_bytes": peak or None,
        "stdout_bytes": len(stdout.encode()),
        "stderr_bytes": len(stderr.encode()),
    }


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile needs at least one value")
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.5))]


def analyze_trace(path: pathlib.Path) -> dict[str, Any]:
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            events.append(value)
    exchanges = [
        event for event in events
        if event.get("event") in {"exchange", "exchange_delta"}
    ]
    context = [
        int(event.get("response", {}).get("usage", {}).get("input_tokens", 0) or 0)
        for event in exchanges
    ]
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "events": len(events),
        "exchanges": len(exchanges),
        "first_context_tokens": context[0] if context else None,
        "max_context_tokens": max(context) if context else None,
        "context_growth_tokens": (context[-1] - context[0]) if context else None,
    }


def main() -> int:
    repo = pathlib.Path(__file__).resolve().parents[2]
    suffix = ".exe" if sys.platform == "win32" else ""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        type=pathlib.Path,
        default=repo / "knossos-rs" / "target" / "release" / f"daedalus{suffix}",
    )
    parser.add_argument(
        "--workspace",
        type=pathlib.Path,
        default=repo / "knossos-rs" / "tests" / "fixtures" / "passing",
    )
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="unmeasured launches used to separate OS cold-start effects",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--trace", type=pathlib.Path,
        help="collected trace used to measure trace size and prompt-context growth",
    )
    parser.add_argument("--out", type=pathlib.Path)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    if args.warmup < 0:
        parser.error("--warmup cannot be negative")
    if not args.binary.is_file():
        parser.error(f"binary does not exist: {args.binary} (run cargo build --release)")

    command = [
        str(args.binary),
        "--engine",
        "none",
        "--workspace",
        str(args.workspace.resolve()),
        "index",
    ]
    warmup_runs = [run_once(command, args.timeout) for _ in range(args.warmup)]
    runs = [run_once(command, args.timeout) for _ in range(args.iterations)]
    verify_command = [
        str(args.binary), "--engine", "none", "--workspace",
        str(args.workspace.resolve()), "verify",
    ]
    verify_runs = [run_once(verify_command, args.timeout) for _ in range(args.iterations)]
    wall = [run["wall_ms"] for run in runs]
    rss = [run["peak_rss_bytes"] for run in runs if run["peak_rss_bytes"] is not None]
    report = {
        "schema": "knossos-harness-benchmark/v1",
        "command": command,
        "binary_bytes": args.binary.stat().st_size,
        "warmup_runs": warmup_runs,
        "iterations": args.iterations,
        "wall_ms": {
            "min": min(wall),
            "median": round(statistics.median(wall), 3),
            "p95": percentile(wall, 0.95),
            "max": max(wall),
        },
        "peak_rss_bytes": {
            "max": max(rss) if rss else None,
            "samples": len(rss),
        },
        "runs": runs,
        "verification": {
            "command": verify_command,
            "median_ms": round(statistics.median(
                run["wall_ms"] for run in verify_runs
            ), 3),
            "peak_rss_bytes": max(
                (run["peak_rss_bytes"] for run in verify_runs
                 if run["peak_rss_bytes"] is not None), default=None
            ),
            "runs": verify_runs,
        },
        "trace": analyze_trace(args.trace) if args.trace else None,
        "scope": "harness startup, indexing, verification, and trace/context overhead; excludes model inference",
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
