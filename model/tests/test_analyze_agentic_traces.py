import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_agentic_traces import latest_attempts, report, summarize


def test_summarize_keeps_infrastructure_separate_from_behavior(tmp_path):
    trace = tmp_path / "one.jsonl"
    events = [
        {
            "at": "2026-01-01T00:00:00+00:00",
            "event": "experiment_metadata",
            "arm": "full",
            "case_id": "case-a",
            "provider": "openrouter",
            "model": "model",
        },
        {"at": "2026-01-01T00:00:01+00:00", "event": "step_started", "index": 1},
        {
            "at": "2026-01-01T00:00:02+00:00",
            "event": "exchange_delta",
            "response": {"usage": {"input_tokens": 10, "output_tokens": 4}},
        },
        {"at": "2026-01-01T00:00:03+00:00", "event": "tool_call"},
        {
            "at": "2026-01-01T00:00:04+00:00",
            "event": "evaluation_finished",
            "grader_pass": False,
            "verifier_pass": False,
            "halt_reason": "infrastructure",
            "provider_status": "ok",
            "infrastructure_status": "unverifiable",
        },
    ]
    trace.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")

    row = summarize(trace)
    assert row is not None
    assert row["failure_class"] == "infrastructure"
    assert row["input_tokens"] == 10
    assert row["output_tokens"] == 4
    assert row["duration_seconds"] == 4
    data = report([row])
    assert data["arms"]["full"]["infrastructure_failures"] == 1
    assert data["arms"]["full"]["behavioral_failures"] == 0


def test_summarize_rejects_a_green_label_after_a_stuck_halt(tmp_path):
    trace = tmp_path / "stuck.jsonl"
    events = [
        {"at": "2026-01-01T00:00:00+00:00", "event": "experiment_metadata",
         "arm": "full", "case_id": "case-b"},
        {"at": "2026-01-01T00:00:01+00:00", "event": "halt", "reason": "stuck"},
        {"at": "2026-01-01T00:00:02+00:00", "event": "evaluation_finished",
         "grader_pass": True, "verifier_pass": True, "halt_reason": "done",
         "provider_status": "ok", "infrastructure_status": "ok"},
    ]
    trace.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")

    row = summarize(trace)
    assert row is not None
    assert not row["passed"]
    assert row["failure_class"] == "stuck"


def test_latest_attempt_replaces_an_earlier_retry():
    rows = [
        {"arm": "full", "case_id": "a", "started_at": "2026-01-01", "trace": "v1"},
        {"arm": "full", "case_id": "a", "started_at": "2026-01-02", "trace": "v2"},
        {"arm": "base", "case_id": "a", "started_at": "2026-01-01", "trace": "base"},
    ]
    chosen = latest_attempts(rows)
    assert {row["trace"] for row in chosen} == {"v2", "base"}
