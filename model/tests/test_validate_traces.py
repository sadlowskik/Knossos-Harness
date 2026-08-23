import json
import pathlib
import sys

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from validate_traces import validate  # noqa: E402


def _event(kind, **extra):
    return {
        "at": "2026-01-01T00:00:00+00:00",
        "schema_version": "daedalus-trace/v2",
        "run_id": "run",
        "event": kind,
        **extra,
    }


def test_complete_v2_trace_validates(tmp_path):
    path = tmp_path / "trace.jsonl"
    events = [
        _event("task_started"),
        _event("exchange_delta", response={}),
        _event("evaluation_finished"),
    ]
    path.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
    assert validate(path) == []


def test_schema_and_secret_fail_closed(tmp_path):
    path = tmp_path / "trace.jsonl"
    events = [
        _event("task_started", schema_version="v1", task="sk-or-v1-" + "a" * 40),
        _event("evaluation_finished"),
    ]
    path.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
    errors = validate(path)
    assert any("unsupported schema" in error for error in errors)
    assert any("secret detected" in error for error in errors)
