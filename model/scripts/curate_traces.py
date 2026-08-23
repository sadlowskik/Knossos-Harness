"""Curate Knossos traces into deterministic, leakage-resistant SFT/DPO splits.

Only fully labelled successful evaluations enter SFT by default. DPO pairs are
formed only within the same case/task key: a successful final turn is chosen and
a failed final turn is rejected. Splits are assigned by task key, so variants of
one benchmark can never leak across train/validation/test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from trace_to_sft import read_trace, records, succeeded

SECRET_PATTERNS = [
    re.compile(r"\bsk-or-v1-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"(?i)authorization\s*:\s*bearer\s+\S+"),
    re.compile(r"(?i)\b(?:api[_-]?key|token|password)\b\s*[=:]\s*['\"]?\S{12,}"),
]


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def contains_secret(events: Iterable[Dict[str, Any]]) -> bool:
    text = canonical(list(events))
    return any(pattern.search(text) for pattern in SECRET_PATTERNS)


def evaluation(events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    labels = [event for event in events if event.get("event") == "evaluation_finished"]
    return labels[-1] if labels else None


def failure_class(events: List[Dict[str, Any]]) -> str:
    """Stable reason a trajectory is or is not eligible for preference data."""
    label = evaluation(events)
    if label is None:
        return "unlabelled"
    if succeeded(events):
        return "success"
    if label.get("provider_status") != "ok":
        return "provider"
    if label.get("infrastructure_status") == "harness_error":
        return "harness"
    if label.get("infrastructure_status") != "ok":
        return "infrastructure"
    if label.get("tamper") is True:
        return "boundary_violation"
    reason = label.get("halt_reason")
    if reason in {"test_failure", "action_violation"}:
        return str(reason)
    if label.get("verifier_pass") is False:
        return "verifier_failure"
    return "grader_failure"


def cause_group(reason: str) -> str:
    """Collapse detailed labels into the four audit ownership buckets."""
    if reason == "success":
        return "success"
    if reason in {"test_failure", "action_violation", "boundary_violation"}:
        return "model"
    if reason == "harness":
        return "harness"
    if reason in {"provider", "infrastructure"}:
        return "infrastructure"
    return "ambiguous"


def engine_name(events: List[Dict[str, Any]]) -> str:
    starts = [event for event in events if event.get("event") == "task_started"]
    return str(starts[-1].get("engine") or "unknown") if starts else "unknown"


def experiment_metadata(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    records = [event for event in events if event.get("event") == "experiment_metadata"]
    if not records:
        return {}
    return {key: value for key, value in records[-1].items() if key != "event"}


def task_key(events: List[Dict[str, Any]], path: pathlib.Path) -> str:
    label = evaluation(events) or {}
    if label.get("case_id"):
        return f"case:{label['case_id']}"
    starts = [event for event in events if event.get("event") == "task_started"]
    if starts and starts[-1].get("task"):
        return f"task:{digest(starts[-1]['task'])[:20]}"
    return f"trace:{path.stem}"


def split_for(key: str, salt: str) -> str:
    bucket = int(hashlib.sha256(f"{salt}\0{key}".encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 90 else "validation" if bucket < 95 else "test"


def valid_chat(record: Dict[str, Any]) -> bool:
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return False
    target = messages[-1]
    if target.get("role") != "assistant":
        return False
    outstanding: set[str] = set()
    for message in messages:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                call_id = call.get("id")
                if not call_id or call_id in outstanding:
                    return False
                outstanding.add(call_id)
        elif message.get("role") == "tool":
            call_id = message.get("tool_call_id")
            if call_id not in outstanding:
                return False
            outstanding.remove(call_id)
    return True


@dataclass
class Trajectory:
    path: pathlib.Path
    key: str
    split: str
    success: bool
    labelled: bool
    recovered: bool
    failure_class: str
    cause_group: str
    engine: str
    experiment: Dict[str, Any]
    records: List[Dict[str, Any]]


def load_trajectory(path: pathlib.Path, salt: str, allow_legacy: bool) -> tuple[Optional[Trajectory], str]:
    events = read_trace(path)
    if not events:
        return None, "empty_or_invalid"
    if contains_secret(events):
        return None, "secret_quarantine"
    label = evaluation(events)
    if label is None and not allow_legacy:
        return None, "unlabelled"
    recs = [record for record in records(events) if valid_chat(record)]
    if not recs:
        return None, "no_valid_records"
    key = task_key(events, path)
    success = succeeded(events)
    recovered = success and any(
        event.get("event") == "oracle_verdict" and event.get("passed") is False
        for event in events
    )
    reason = failure_class(events)
    return Trajectory(path, key, split_for(key, salt), success, label is not None,
                      recovered, reason, cause_group(reason), engine_name(events),
                      experiment_metadata(events), recs), "ok"


def write_jsonl(path: pathlib.Path, rows: Iterable[Dict[str, Any]]) -> int:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")
    return len(rows)


def source_paths(traces: pathlib.Path, out_dir: pathlib.Path) -> List[pathlib.Path]:
    """Raw traces only; never recursively ingest a previous curation output."""
    if traces.is_file():
        return [traces]
    output = out_dir.resolve()
    return [
        path
        for path in sorted(traces.rglob("*.jsonl"))
        if not path.resolve().is_relative_to(output)
    ]


def curate(paths: List[pathlib.Path], out_dir: pathlib.Path, salt: str,
           allow_legacy: bool = False) -> Dict[str, Any]:
    stats: Counter[str] = Counter()
    trajectories: List[Trajectory] = []
    for path in sorted(paths, key=lambda item: item.as_posix()):
        trajectory, reason = load_trajectory(path, salt, allow_legacy)
        stats[reason] += 1
        if trajectory:
            trajectories.append(trajectory)
            stats[f"failure_class:{trajectory.failure_class}"] += 1
            stats[f"cause_group:{trajectory.cause_group}"] += 1

    seen: set[str] = set()
    sft: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for trajectory in trajectories:
        if not trajectory.success:
            continue
        for record in trajectory.records:
            record_id = digest(record["messages"])
            if record_id in seen:
                stats["duplicate_record"] += 1
                continue
            seen.add(record_id)
            enriched = dict(record)
            enriched["id"] = record_id
            enriched["task_key"] = trajectory.key
            enriched["recovery"] = trajectory.recovered
            enriched["cause_group"] = trajectory.cause_group
            enriched["engine"] = trajectory.engine
            enriched["experiment"] = trajectory.experiment
            enriched["source_trace"] = trajectory.path.name
            sft[trajectory.split].append(enriched)

    by_key: Dict[str, Dict[bool, List[Trajectory]]] = defaultdict(lambda: defaultdict(list))
    eligible_rejections = {"test_failure", "action_violation", "verifier_failure"}
    for trajectory in trajectories:
        if trajectory.success or trajectory.failure_class in eligible_rejections:
            by_key[trajectory.key][trajectory.success].append(trajectory)
        elif not trajectory.success:
            stats[f"dpo_rejected:{trajectory.failure_class}"] += 1
    dpo: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for key in sorted(by_key):
        chosen = sorted(by_key[key][True], key=lambda item: item.path.as_posix())
        rejected = sorted(by_key[key][False], key=lambda item: item.path.as_posix())
        if not chosen or not rejected:
            continue
        good, bad = chosen[0].records[-1], rejected[0].records[-1]
        # Pair only when both turns share an identical prompt. Otherwise the
        # preference could be about different context rather than behaviour.
        prompt = good["messages"][:-1]
        if canonical(prompt) != canonical(bad["messages"][:-1]):
            stats["dpo_prompt_mismatch"] += 1
            continue
        row = {
            "id": digest([key, prompt, good["messages"][-1], bad["messages"][-1]]),
            "task_key": key,
            "engine": good.get("engine", chosen[0].engine),
            "experiment": chosen[0].experiment,
            "chosen_source": chosen[0].path.name,
            "rejected_source": rejected[0].path.name,
            "rejected_failure_class": rejected[0].failure_class,
            "rejected_cause_group": rejected[0].cause_group,
            "prompt": prompt,
            "chosen": good["messages"][-1],
            "rejected": bad["messages"][-1],
        }
        dpo[chosen[0].split].append(row)

    counts = {"sft": {}, "dpo": {}}
    for split in ("train", "validation", "test"):
        counts["sft"][split] = write_jsonl(out_dir / f"sft-{split}.jsonl", sft[split])
        counts["dpo"][split] = write_jsonl(out_dir / f"dpo-{split}.jsonl", dpo[split])
    manifest = {
        "schema": "daedalus-corpus/v1",
        "split_salt": salt,
        "allow_legacy": allow_legacy,
        "input_traces": len(paths),
        "counts": counts,
        "stats": dict(sorted(stats.items())),
        "rules": {
            "sft": "labelled grader+verifier pass; provider/infrastructure ok; no tamper",
            "dpo": "same task and byte-identical prompt; successful chosen vs behavioral failure rejected; provider/infrastructure/tamper failures excluded",
            "secrets": "matching traces quarantined, never redacted into training data",
            "split_unit": "case id or hashed task, never individual turn",
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", required=True, type=pathlib.Path)
    parser.add_argument("--out-dir", required=True, type=pathlib.Path)
    parser.add_argument("--split-salt", default="daedalus-v1")
    parser.add_argument("--allow-legacy", action="store_true")
    args = parser.parse_args()
    paths = source_paths(args.traces, args.out_dir)
    manifest = curate(paths, args.out_dir, args.split_salt, args.allow_legacy)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if paths else 1


if __name__ == "__main__":
    raise SystemExit(main())
