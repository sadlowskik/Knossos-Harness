"""Isolation tests for the executor loop.

Driven by a scripted engine, so there is no model and no network -- but the
workspace, the tools and the halting policy are all real, and files genuinely
get written.

The claims that matter:

  1. **Completion is decided by the verifier, not the engine.** An engine that
     stops calling tools is requesting verification, not announcing success.
  2. The loop always terminates, even against an engine that never stops.
  3. A failing verification feeds back into the conversation rather than ending
     the run.
  4. The jail holds when the calls come through the loop rather than directly.

    pytest -q tests/test_talos.py
"""
from pathlib import Path
from typing import List, Sequence

import pytest

from harness.ariadne import Ariadne, Halt
from harness.talos import Event, Talos, Verdict
from harness.workspace import Workspace


class ScriptedEngine:
    """Replays a fixed list of replies, one per call."""

    name = "scripted"

    def __init__(self, replies: Sequence[str]) -> None:
        self.replies = list(replies)
        self.prompts: List[str] = []

    def generate(self, prompt, context, cancelled):
        self.prompts.append(prompt)
        if not self.replies:
            yield "I have nothing left to say."
            return
        yield self.replies.pop(0)


def call(tool: str, **args) -> str:
    import json
    return f'```json\n{json.dumps({"tool": tool, "args": args})}\n```'


def always_fails(ws, changed):
    return Verdict(False, "the check never passes", "detail about why")


def passes_when_anything_changed(ws, changed):
    return Verdict(bool(changed), "changed something" if changed else "nothing changed")


@pytest.fixture()
def ws(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "lib.py").write_text("value = 1\n", encoding="utf-8")
    return Workspace(tmp_path)


# ------------------------------------------------------- the load-bearing rule

def test_the_engine_saying_done_does_not_end_the_run(ws):
    """The whole point: only the verifier can produce DONE."""
    engine = ScriptedEngine(["All finished!", "Really, finished.", "Truly done."])
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=3, target_steps=2),
                  verifier=always_fails)

    outcome = talos.run("do something")

    assert outcome.halt is not Halt.DONE
    assert outcome.verdict is not None and not outcome.verdict.passed


def test_a_passing_verdict_ends_the_run(ws):
    engine = ScriptedEngine([
        call("write_file", path="src/new.py", content="x = 1\n"),
        "Done.",
    ])
    talos = Talos(engine, ws, verifier=passes_when_anything_changed)

    outcome = talos.run("add a file")

    assert outcome.halt is Halt.DONE
    assert outcome.succeeded
    assert outcome.steps_used == 2
    assert (ws.root / "src" / "new.py").read_text(encoding="utf-8") == "x = 1\n"


def test_a_failed_verdict_is_fed_back_into_the_conversation(ws):
    engine = ScriptedEngine(["Done.", "Done again.", "Still done."])
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=3, target_steps=2),
                  verifier=always_fails)

    talos.run("do something")

    joined = "\n".join(talos.transcript)
    assert "Verification FAILED" in joined
    assert "detail about why" in joined
    # And the engine actually saw it on a later turn.
    assert "Verification FAILED" in engine.prompts[-1]


# ------------------------------------------------------------------ stopping

def test_the_loop_terminates_against_an_engine_that_never_stops(ws):
    engine = ScriptedEngine([call("read_file", path="src/lib.py")] * 10)
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=3, target_steps=2))

    outcome = talos.run("read forever")

    assert outcome.halt is Halt.BUDGET_EXHAUSTED
    assert outcome.steps_used == 3
    assert "budget" in outcome.summary.lower()


def test_repeated_empty_steps_are_stuck(ws):
    """No tools, no changes, verification failing -- more turns will not help."""
    engine = ScriptedEngine(["thinking", "still thinking", "yet more thinking"])
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=8, target_steps=6),
                  verifier=always_fails)

    outcome = talos.run("achieve nothing")

    assert outcome.halt is Halt.STUCK
    assert outcome.changed == []


def test_budget_pressure_reaches_the_engine(ws):
    engine = ScriptedEngine([call("read_file", path="src/lib.py")] * 6)
    talos = Talos(engine, ws, ariadne=Ariadne(max_steps=4, target_steps=1))

    talos.run("keep going")

    assert any("BUDGET:" in p for p in engine.prompts)


# ------------------------------------------------------------------- safety

def test_the_jail_holds_through_the_loop(ws):
    engine = ScriptedEngine([
        call("write_file", path="../escaped.py", content="pwned"),
        "Done.",
    ])
    talos = Talos(engine, ws, verifier=passes_when_anything_changed)

    outcome = talos.run("try to escape")

    assert not (ws.root.parent / "escaped.py").exists()
    assert outcome.changed == []
    assert "refused" in "\n".join(talos.transcript)


def test_a_disallowed_command_is_refused_and_reported(ws):
    engine = ScriptedEngine([call("run_command", command="rm -rf ."), "Done."])
    talos = Talos(engine, ws, verifier=passes_when_anything_changed)

    talos.run("delete everything")

    assert "not on the allowlist" in "\n".join(talos.transcript)


# ------------------------------------------------------------------ dry runs

def test_a_dry_run_writes_nothing_until_applied(tmp_path):
    dry = Workspace(tmp_path, dry_run=True)
    engine = ScriptedEngine([
        call("write_file", path="a.py", content="staged = True\n"),
        "Done.",
    ])
    talos = Talos(engine, dry, verifier=passes_when_anything_changed)

    outcome = talos.run("propose a file")

    assert outcome.dry_run
    assert not (tmp_path / "a.py").exists()

    written = talos.apply()
    assert len(written) == 1
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "staged = True\n"


# ------------------------------------------------------------------ sessions

def test_resume_keeps_the_conversation(ws):
    engine = ScriptedEngine([
        call("write_file", path="one.py", content="one\n"), "Added one.",
        call("write_file", path="two.py", content="two\n"), "Added two.",
    ])
    talos = Talos(engine, ws, verifier=passes_when_anything_changed)

    first = talos.run("add one")
    assert first.halt is Halt.DONE
    length_after_first = len(talos.transcript)

    second = talos.resume("now add two")
    assert second.halt is Halt.DONE

    assert len(talos.transcript) > length_after_first
    assert talos.task == "add one", "the original task is retained"
    assert (ws.root / "one.py").exists() and (ws.root / "two.py").exists()
    assert second.steps_used == 2, "the budget resets per turn"


def test_resume_on_a_fresh_talos_behaves_like_a_first_task(ws):
    talos = Talos(ScriptedEngine(["Nothing to do."]), ws,
                  verifier=passes_when_anything_changed)
    outcome = talos.resume("look around")
    assert talos.task == "look around"
    assert outcome.steps_used >= 1


# ------------------------------------------------------------------- events

def test_events_describe_the_whole_run(ws):
    engine = ScriptedEngine([
        call("write_file", path="a.py", content="x\n"),
        "Done.",
    ])
    seen: List[Event] = []
    talos = Talos(engine, ws, verifier=passes_when_anything_changed)

    talos.run("add a file", on_event=seen.append)

    kinds = [e.kind for e in seen]
    for expected in ("step", "tool", "verdict", "halt"):
        assert expected in kinds, f"missing {expected}; got {kinds}"

    tool_event = next(e for e in seen if e.kind == "tool")
    assert tool_event.call is not None and tool_event.call.name == "write_file"
    assert tool_event.result is not None and not tool_event.result.is_error

    halt_event = next(e for e in seen if e.kind == "halt")
    assert halt_event.halt is Halt.DONE


def test_the_default_verifier_is_named_to_be_uncomfortable():
    """A run with no verifier has no check on correctness; that should show."""
    from harness.talos import accept_everything
    verdict = accept_everything(None, [])
    assert verdict.passed
    assert "no verifier configured" in verdict.summary
