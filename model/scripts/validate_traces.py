"""Fail-closed validation for raw Knossos v2 training traces."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from curate_traces import contains_secret


SCHEMA = "knossos-trace/v2"
LEGACY_SCHEMAS = {"daedalus-trace/v2"}
EVENTS = {
    "experiment_metadata", "task_started", "plan_produced", "step_started",
    "agent_message", "thought", "context_considered", "exchange",
    "exchange_delta", "tool_call", "oracle_verdict", "context_compacted",
    "interjected", "halt", "hypothesis", "redirected", "task_finished",
    "evaluation_finished",
}


def validate(path: Path) -> list[str]:
    errors: list[str] = []
    events = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append(f"line {number}: invalid JSON: {error.msg}")
            continue
        if not isinstance(event, dict):
            errors.append(f"line {number}: event must be an object")
            continue
        events.append(event)
        for field in ("at", "schema_version", "run_id", "event"):
            if field not in event:
                errors.append(f"line {number}: missing {field}")
        if event.get("schema_version") not in {SCHEMA, *LEGACY_SCHEMAS}:
            errors.append(f"line {number}: unsupported schema {event.get('schema_version')!r}")
        if event.get("event") not in EVENTS:
            errors.append(f"line {number}: unknown event {event.get('event')!r}")
        if event.get("event") in {"exchange", "exchange_delta"} and "response" not in event:
            errors.append(f"line {number}: exchange has no response")
    if contains_secret(events):
        errors.append("secret detected: quarantine the complete trace")
    if not any(event.get("event") == "evaluation_finished" for event in events):
        errors.append("missing terminal evaluation_finished label")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    files = sorted(
        path for item in args.paths
        for path in ([item] if item.is_file() else item.rglob("*.jsonl"))
    )
    seen: dict[str, Path] = {}
    failed = False
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:
            print(f"ERROR {path}: byte-identical duplicate of {seen[digest]}")
            failed = True
            continue
        seen[digest] = path
        for error in validate(path):
            print(f"ERROR {path}: {error}")
            failed = True
    print(json.dumps({"schema": "knossos-trace-validation/v1", "files": len(files),
                      "unique": len(seen), "valid": not failed}))
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
