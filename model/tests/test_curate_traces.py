import json
import pathlib
import sys

SCRIPTS = pathlib.Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from curate_traces import (  # noqa: E402
    cause_group,
    contains_secret,
    curate,
    failure_class,
    source_paths,
    split_for,
    valid_chat,
)


def test_source_paths_excludes_the_output_subtree(tmp_path):
    raw = tmp_path / "raw.jsonl"
    raw.write_text("{}\n", encoding="utf-8")
    out = tmp_path / "curated"
    out.mkdir()
    (out / "train.jsonl").write_text("{}\n", encoding="utf-8")

    assert source_paths(tmp_path, out) == [raw]


def test_split_is_stable_and_task_level():
    assert split_for("case:same", "salt") == split_for("case:same", "salt")


def test_openrouter_keys_are_quarantined():
    events = [{"event": "task_started", "task": "key sk-or-v1-" + "a" * 40}]
    assert contains_secret(events)


def test_tool_result_must_follow_a_known_call():
    bad = {"messages": [
        {"role": "system", "content": "s"},
        {"role": "tool", "tool_call_id": "missing", "content": "x"},
        {"role": "assistant", "content": "done"},
    ]}
    assert not valid_chat(bad)


def test_plain_chat_with_assistant_target_is_valid():
    record = {"messages": [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "answer"},
    ]}
    assert valid_chat(record)


def _events(case_id, passed, *, provider="ok", infrastructure="ok",
            tamper=False, reason="test_failure"):
    return [
        {"event": "task_started", "task": "fix it", "engine": "openrouter:ox"},
        {
            "event": "exchange",
            "schema_version": "knossos-trace/v2",
            "request": {
                "system": "be correct",
                "messages": [{
                    "role": "user",
                    "content": [{"kind": "text", "text": "fix it"}],
                }],
            },
            "response": {
                "content": [{"kind": "text", "text": "done" if passed else "gave up"}],
            },
        },
        {
            "event": "evaluation_finished",
            "case_id": case_id,
            "grader_pass": passed,
            "verifier_pass": passed,
            "halt_reason": "done" if passed else reason,
            "provider_status": provider,
            "infrastructure_status": infrastructure,
            "tamper": tamper,
        },
    ]


def _write(path, events):
    path.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")


def test_failure_class_separates_behavior_from_provider_and_infrastructure():
    assert failure_class(_events("c", True)) == "success"
    assert failure_class(_events("c", False)) == "test_failure"
    assert failure_class(_events("c", False, provider="error")) == "provider"
    assert failure_class(_events("c", False, infrastructure="timeout")) == "infrastructure"
    assert failure_class(_events("c", False, infrastructure="harness_error")) == "harness"
    assert failure_class(_events("c", False, tamper=True)) == "boundary_violation"
    assert cause_group("test_failure") == "model"
    assert cause_group("harness") == "harness"
    assert cause_group("provider") == "infrastructure"
    assert cause_group("verifier_failure") == "ambiguous"


def test_dpo_never_uses_provider_or_infrastructure_failures(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    _write(traces / "chosen.jsonl", _events("same", True))
    _write(traces / "behavior.jsonl", _events("same", False))
    _write(traces / "provider.jsonl", _events("same", False, provider="error"))
    _write(traces / "infra.jsonl", _events("same", False, infrastructure="timeout"))

    manifest = curate(sorted(traces.glob("*.jsonl")), tmp_path / "out", "test")
    dpo_rows = []
    for output in (tmp_path / "out").glob("dpo-*.jsonl"):
        dpo_rows.extend(json.loads(line) for line in output.read_text().splitlines())

    assert len(dpo_rows) == 1
    assert dpo_rows[0]["rejected_source"] == "behavior.jsonl"
    assert dpo_rows[0]["rejected_failure_class"] == "test_failure"
    assert dpo_rows[0]["rejected_cause_group"] == "model"
    assert manifest["stats"]["dpo_rejected:provider"] == 1
    assert manifest["stats"]["dpo_rejected:infrastructure"] == 1
