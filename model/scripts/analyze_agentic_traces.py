"""Summarize labelled Knossos v2 experiment traces by arm and case.

The report keeps provider/infrastructure failures separate from behavioral
failures so an outage cannot lower a model's apparent solve rate or enter DPO.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
from collections import defaultdict
from typing import Any


def cause_group(reason: str) -> str:
    if reason == "success":
        return "success"
    if reason in {"test_failure", "action_violation", "boundary_violation", "stuck"}:
        return "model"
    if reason == "harness":
        return "harness"
    if reason in {"provider", "infrastructure"}:
        return "infrastructure"
    return "ambiguous"


def read_events(path: pathlib.Path) -> list[dict[str, Any]]:
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def timestamp(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def summarize(path: pathlib.Path) -> dict[str, Any] | None:
    events = read_events(path)
    metadata = next(
        (event for event in events if event.get("event") == "experiment_metadata"),
        None,
    )
    result = next(
        (event for event in reversed(events) if event.get("event") == "evaluation_finished"),
        None,
    )
    if metadata is None or result is None:
        return None

    usage = [
        event.get("response", {}).get("usage", {})
        for event in events
        if event.get("event") == "exchange_delta"
        and isinstance(event.get("response"), dict)
    ]
    starts = [event for event in events if event.get("event") == "step_started"]
    tools = [event for event in events if event.get("event") == "tool_call"]
    considered = [event for event in events if event.get("event") == "context_considered"]
    halts = [event for event in events if event.get("event") == "halt"]
    dated = [event for event in events if isinstance(event.get("at"), str)]
    duration = 0.0
    if len(dated) >= 2:
        duration = (timestamp(dated[-1]["at"]) - timestamp(dated[0]["at"])).total_seconds()

    provider_ok = result.get("provider_status") == "ok"
    infrastructure_ok = result.get("infrastructure_status") == "ok"
    completion_consistent = bool(
        not halts
        or halts[-1].get("reason") == "done"
        or metadata.get("expected_action") == "clarify"
    )
    passed = bool(
        result.get("grader_pass")
        and result.get("verifier_pass")
        and completion_consistent
    )
    failure_class = result.get("halt_reason", "unknown")
    if not completion_consistent and failure_class == "done":
        failure_class = halts[-1].get("reason", "incomplete")
    elif not provider_ok:
        failure_class = "provider"
    elif not infrastructure_ok:
        failure_class = "infrastructure"
    elif passed:
        failure_class = "success"

    return {
        "trace": path.name,
        "arm": metadata.get("arm", "unassigned"),
        "case_id": metadata.get("case_id", "unknown"),
        "provider": metadata.get("provider"),
        "model": metadata.get("model"),
        "started_at": metadata.get("started_at", ""),
        "passed": passed and provider_ok and infrastructure_ok,
        "failure_class": failure_class,
        "cause_group": cause_group(failure_class),
        "provider_ok": provider_ok,
        "infrastructure_ok": infrastructure_ok,
        "steps": max((int(event.get("index", 0)) for event in starts), default=0),
        "tool_calls": len(tools),
        "requests": sum(1 for item in usage if item),
        "input_tokens": sum(int(item.get("input_tokens", 0) or 0) for item in usage),
        "output_tokens": sum(int(item.get("output_tokens", 0) or 0) for item in usage),
        "duration_seconds": round(duration, 3),
        "context_injected": any(bool(event.get("injected")) for event in considered),
    }


def report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["arm"]].append(row)

    arms: dict[str, Any] = {}
    for arm, cases in sorted(grouped.items()):
        successful = sum(1 for case in cases if case["passed"])
        behavioral = sum(
            1
            for case in cases
            if not case["passed"] and case["provider_ok"] and case["infrastructure_ok"]
        )
        arms[arm] = {
            "cases": len(cases),
            "passed": successful,
            "behavioral_failures": behavioral,
            "provider_failures": sum(1 for case in cases if not case["provider_ok"]),
            "infrastructure_failures": sum(
                1 for case in cases if case["provider_ok"] and not case["infrastructure_ok"]
            ),
            "cause_groups": dict(sorted(
                (group, sum(1 for case in cases if case["cause_group"] == group))
                for group in {case["cause_group"] for case in cases}
            )),
            "pass_rate": round(successful / len(cases), 4),
            "steps": sum(case["steps"] for case in cases),
            "tool_calls": sum(case["tool_calls"] for case in cases),
            "requests": sum(case["requests"] for case in cases),
            "input_tokens": sum(case["input_tokens"] for case in cases),
            "output_tokens": sum(case["output_tokens"] for case in cases),
            "duration_seconds": round(sum(case["duration_seconds"] for case in cases), 3),
            "case_results": sorted(cases, key=lambda case: case["case_id"]),
        }
    return {"schema": "daedalus-experiment-analysis/v1", "traces": len(rows), "arms": arms}


def latest_attempts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["arm"], row["case_id"])
        previous = selected.get(key)
        marker = (row.get("started_at", ""), row["trace"])
        if previous is None or marker > (
            previous.get("started_at", ""),
            previous["trace"],
        ):
            selected[key] = row
    return list(selected.values())


def markdown(data: dict[str, Any]) -> str:
    lines = [
        "# Agentic experiment summary",
        "",
        "| Arm | Pass | Behavioral | Provider | Infrastructure | Steps | Tool calls | Input tok | Output tok | Seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, arm in data["arms"].items():
        lines.append(
            f"| {name} | {arm['passed']}/{arm['cases']} | {arm['behavioral_failures']} | "
            f"{arm['provider_failures']} | {arm['infrastructure_failures']} | {arm['steps']} | "
            f"{arm['tool_calls']} | {arm['input_tokens']} | {arm['output_tokens']} | "
            f"{arm['duration_seconds']:.1f} |"
        )
    lines.extend(["", "## Cases", ""])
    for name, arm in data["arms"].items():
        lines.append(f"### {name}")
        lines.append("")
        for case in arm["case_results"]:
            state = "PASS" if case["passed"] else f"FAIL ({case['failure_class']})"
            lines.append(
                f"- `{case['case_id']}`: {state}; {case['steps']} steps, "
                f"{case['tool_calls']} tools, {case['input_tokens']}+{case['output_tokens']} tokens, "
                f"{case['duration_seconds']:.1f}s."
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", required=True, type=pathlib.Path)
    parser.add_argument("--pattern", default="*.jsonl")
    parser.add_argument("--out", type=pathlib.Path)
    parser.add_argument("--markdown", type=pathlib.Path)
    parser.add_argument(
        "--latest-attempt",
        action="store_true",
        help="keep only the newest trace for each arm/case pair",
    )
    args = parser.parse_args()

    paths = [args.traces] if args.traces.is_file() else sorted(args.traces.glob(args.pattern))
    rows = [row for path in paths if (row := summarize(path)) is not None]
    if args.latest_attempt:
        rows = latest_attempts(rows)
    data = report(rows)
    rendered = json.dumps(data, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(data), encoding="utf-8")


if __name__ == "__main__":
    main()
