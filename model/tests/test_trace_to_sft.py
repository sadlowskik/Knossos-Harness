"""Traces in, training records out.

The claims worth testing are about *fidelity*, not plumbing: a converter that
silently reshapes a conversation produces a corpus that trains something other
than what ran, and nothing downstream will tell you.

  1. The target is the model's own reply, and it is last.
  2. Tool calls survive with `arguments` as a JSON string, which is what the
     format requires and what a naive dict would get wrong.
  3. Tool results become `tool` messages carrying the id they answer.
  4. A trajectory that did not reach Done is dropped by default.

No torch, no network.

    pytest -q tests/test_trace_to_sft.py
"""
import json
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

from scripts.trace_to_sft import records, succeeded, to_chat  # noqa: E402


def exchange(step, messages, response, system="Be correct."):
    return {"event": "exchange", "step": step,
            "request": {"system": system, "messages": messages},
            "response": {"content": response}}


def text(s):
    return {"kind": "text", "text": s}


def tool_use(id_, name, inp):
    return {"kind": "tool_use", "id": id_, "name": name, "input": inp}


def tool_result(id_, content, is_error=False):
    return {"kind": "tool_result", "id": id_, "content": content,
            "is_error": is_error}


def test_the_reply_is_the_last_message_and_is_the_target():
    events = [exchange(1, [{"role": "user", "content": [text("fix it")]}],
                       [text("on it")])]
    (rec,) = list(records(events))

    assert rec["messages"][0]["role"] == "system"
    assert rec["messages"][-1] == {"role": "assistant", "content": "on it"}


def test_a_tool_call_survives_with_arguments_as_a_string():
    # OpenAI wants `arguments` as JSON text, not an object. A dict here parses
    # fine and trains the model to emit the wrong type.
    events = [exchange(1, [{"role": "user", "content": [text("go")]}],
                       [tool_use("t1", "write_file", {"path": "a.rs"})])]
    (rec,) = list(records(events))

    call = rec["messages"][-1]["tool_calls"][0]
    assert call["id"] == "t1"
    assert call["function"]["name"] == "write_file"
    assert isinstance(call["function"]["arguments"], str)
    assert json.loads(call["function"]["arguments"]) == {"path": "a.rs"}


def test_prose_and_a_call_in_one_turn_stay_together():
    events = [exchange(1, [{"role": "user", "content": [text("go")]}],
                       [text("writing it now"), tool_use("t1", "write_file", {})])]
    (rec,) = list(records(events))

    last = rec["messages"][-1]
    assert last["content"] == "writing it now"
    assert len(last["tool_calls"]) == 1


def test_tool_results_become_tool_messages_carrying_their_id():
    history = [
        {"role": "user", "content": [text("fix it")]},
        {"role": "assistant", "content": [tool_use("t1", "read_file", {})]},
        {"role": "user", "content": [tool_result("t1", "file contents")]},
    ]
    events = [exchange(2, history, [text("done")])]
    (rec,) = list(records(events))

    tool_msgs = [m for m in rec["messages"] if m["role"] == "tool"]
    assert tool_msgs == [{"role": "tool", "tool_call_id": "t1",
                          "content": "file contents"}]


def test_results_precede_prose_in_the_same_message():
    # OpenAI requires a tool message to follow the assistant turn that asked
    # for it. Prose sharing that message must not be allowed to come between.
    out = to_chat({"role": "user",
                   "content": [tool_result("t1", "ok"), text("also, hurry")]})
    assert [m["role"] for m in out] == ["tool", "user"]


def test_a_reply_with_nothing_in_it_is_not_a_training_example():
    events = [exchange(1, [{"role": "user", "content": [text("go")]}], [])]
    assert list(records(events)) == []


def test_history_is_preserved_so_a_later_step_trains_on_what_it_saw():
    history = [
        {"role": "user", "content": [text("fix it")]},
        {"role": "assistant", "content": [tool_use("t1", "read_file", {})]},
        {"role": "user", "content": [tool_result("t1", "the old contents")]},
    ]
    events = [exchange(4, history, [text("now I understand")])]
    (rec,) = list(records(events))

    rendered = json.dumps(rec)
    assert "the old contents" in rendered, "step 4 must carry what step 4 saw"
    assert rec["step"] == 4


def test_only_a_done_halt_counts_as_success():
    assert succeeded([{"event": "halt", "reason": "done"}])
    assert succeeded([{"event": "task_finished", "outcome": "done"}])

    # The other two terminal halts are failures, and budget_exhausted is the
    # one most likely to be mistaken for a finished run.
    assert not succeeded([{"event": "halt", "reason": "stuck"}])
    assert not succeeded([{"event": "halt", "reason": "budget_exhausted"}])
    assert not succeeded([{"event": "step_started", "index": 1}])


def test_eval_labels_override_a_misleading_done_halt():
    events = [
        {"event": "halt", "reason": "done"},
        {"event": "evaluation_finished", "grader_pass": False,
         "verifier_pass": False, "provider_status": "ok",
         "infrastructure_status": "ok", "tamper": False},
    ]
    assert not succeeded(events)


def test_a_mislabelled_green_eval_cannot_override_a_stuck_agent():
    events = [
        {"event": "halt", "reason": "stuck"},
        {"event": "evaluation_finished", "grader_pass": True,
         "verifier_pass": True, "provider_status": "ok",
         "infrastructure_status": "ok", "tamper": False},
    ]
    assert not succeeded(events)


def test_a_valid_clarification_can_end_without_a_done_halt():
    events = [
        {"event": "experiment_metadata", "expected_action": "clarify"},
        {"event": "halt", "reason": "stuck"},
        {"event": "evaluation_finished", "grader_pass": True,
         "verifier_pass": True, "provider_status": "ok",
         "infrastructure_status": "ok", "tamper": False},
    ]
    assert succeeded(events)


def test_delta_exchanges_reconstruct_the_exact_prefix():
    events = [
        {"event": "exchange_delta", "schema_version": "daedalus-trace/v2",
         "run_id": "r1", "step": 1, "reset": True, "system": "system",
         "messages_start": 0,
         "messages": [{"role": "user", "content": [text("task")]}],
         "response": {"content": [text("first")]}},
        {"event": "exchange_delta", "schema_version": "daedalus-trace/v2",
         "run_id": "r1", "step": 2, "reset": False, "messages_start": 1,
         "messages": [
             {"role": "assistant", "content": [text("first")]},
             {"role": "user", "content": [text("continue")]},
         ],
         "response": {"content": [text("second")]}},
    ]

    first, second = list(records(events))
    assert [m["content"] for m in first["messages"]] == ["system", "task", "first"]
    assert [m["content"] for m in second["messages"]] == [
        "system", "task", "first", "continue", "second"
    ]
    assert second["run_id"] == "r1"
