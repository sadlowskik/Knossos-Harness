"""Run the four Knossos harness arms against identical recorded responses."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ARMS = {
    "knossos-full": [],
    "knossos-no-context": ["--no-context"],
    "knossos-no-memory": ["--no-memory"],
    "knossos-no-compaction": ["--no-compaction"],
}


def commands(binary: Path, cases: Path, recordings: Path, out: Path,
             case_ids: list[str]) -> list[list[str]]:
    result = []
    for case_id in case_ids:
        recording = recordings / f"{case_id}.jsonl"
        if not recording.is_file():
            raise FileNotFoundError(f"missing recording for {case_id}: {recording}")
        for arm, flags in ARMS.items():
            trace = out / arm / f"{case_id}.jsonl"
            result.append([
                str(binary), "--engine", "none", "eval", "--cases", str(cases),
                "--case-id", case_id, "--experiment-arm", arm,
                "--replay-responses", str(recording), "--trace", str(trace),
                "--collect-exchanges", "--max-concurrency", "1", *flags,
            ])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--recordings", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--case-id", action="append", required=True, dest="case_ids")
    parser.add_argument("--print-only", action="store_true")
    args = parser.parse_args()

    invocations = commands(
        args.binary.resolve(), args.cases.resolve(), args.recordings.resolve(),
        args.out.resolve(), args.case_ids,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "knossos-recorded-ablation/v1",
        "arms": list(ARMS),
        "cases": args.case_ids,
        "recordings": str(args.recordings.resolve()),
        "commands": invocations,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    if args.print_only:
        print(json.dumps(manifest, indent=2))
        return 0

    failed = False
    for command in invocations:
        Path(command[command.index("--trace") + 1]).parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(command, check=False)
        failed |= completed.returncode != 0
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
