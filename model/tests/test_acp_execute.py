"""ACP in execute mode: tool calls reach the editor, and edits are staged.

Kept separate from `test_acp.py` because that file drives the retrieval agent
over real pipes. These drive the handlers directly, which is enough — the
transport is already covered, and what needs proving here is different:

  1. Tool calls surface as real ACP `tool_call` updates, not as prose, so the
     editor renders them as actions with status.
  2. Execute mode stages by default. An agent that can write is opted into.
  3. `stopReason` tells the truth: only a verified run says `end_turn`.
  4. A second prompt continues the session rather than restarting it.

    pytest -q tests/test_acp_execute.py
"""
import json
from typing import Any, Dict, List, Sequence

import pytest

from harness.acp import DaedalusAgent
from harness.ariadne import Ariadne
from harness.talos import Talos, Verdict
from harness.workspace import Workspace


class ScriptedEngine:
    name = "scripted"

    def __init__(self, replies: Sequence[str]) -> None:
        self.replies = list(replies)

    def generate(self, prompt, context, cancelled):
        yield self.replies.pop(0) if self.replies else "Nothing further."


def call(tool: str, **args) -> str:
    return f'```json\n{json.dumps({"tool": tool, "args": args})}\n```'


class Recorder:
    """Stands in for the ACP peer, capturing session/update notifications."""

    def __init__(self) -> None:
        self.updates: List[Dict[str, Any]] = []

    def notify(self, method: str, params: Dict[str, Any]) -> None:
        if method == "session/update":
            self.updates.append(params.get("update", params))

    def kinds(self) -> List[str]:
        return [u.get("sessionUpdate", "") for u in self.updates]

    def titles(self) -> List[str]:
        return [u.get("title", "") for u in self.updates if u.get("title")]


@pytest.fixture()
def workspace(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "lib.py").write_text("value = 1\n", encoding="utf-8")
    return tmp_path


def drive(agent, workspace, prompt):
    """initialize -> session/new -> session/prompt, without the transport."""
    agent.initialize({"protocolVersion": 1, "clientCapabilities": {},
                      "clientInfo": {"name": "test", "version": "1"}})
    session_id = agent.session_new({"cwd": str(workspace), "mcpServers": []})["sessionId"]
    result = agent.session_prompt({
        "sessionId": session_id,
        "prompt": [{"type": "text", "text": prompt}],
    })
    return session_id, result


def passes(ws, changed):
    return Verdict(bool(changed), "changed something" if changed else "nothing changed")


# ------------------------------------------------------------------ defaults

def test_execute_is_off_by_default():
    """An executor can damage a workspace; a retrieval agent cannot."""
    assert DaedalusAgent().execute is False


def test_execute_stages_by_default():
    assert DaedalusAgent(execute=True).dry_run is True


# ---------------------------------------------------------------- tool calls

def test_tool_calls_surface_as_acp_updates(workspace):
    agent = DaedalusAgent(
        engine=ScriptedEngine([call("write_file", path="src/new.py", content="x = 1\n"),
                               "Added it."]),
        execute=True, gate=False)
    agent.peer = Recorder()

    session_id, result = drive(agent, workspace, "add a file")
    recorder = agent.peer

    assert "tool_call" in recorder.kinds()
    assert "tool_call_update" in recorder.kinds()
    assert any("Writing src/new.py" in t for t in recorder.titles())


def test_a_refused_tool_call_is_reported_as_failed(workspace):
    agent = DaedalusAgent(
        engine=ScriptedEngine([call("run_command", command="rm -rf ."), "Done."]),
        execute=True, gate=False)
    agent.peer = Recorder()

    drive(agent, workspace, "delete everything")

    failed = [u for u in agent.peer.updates
              if u.get("sessionUpdate") == "tool_call_update"
              and u.get("status") == "failed"]
    assert failed, "a refusal must render as a failed action, not vanish into prose"


def test_verification_is_narrated_as_its_own_step(workspace):
    agent = DaedalusAgent(engine=ScriptedEngine(["Nothing to do."]),
                          execute=True, gate=False)
    agent.peer = Recorder()

    drive(agent, workspace, "check something")

    assert "Verifying" in agent.peer.titles()


# ------------------------------------------------------------------- staging

def test_edits_are_staged_not_written(workspace):
    agent = DaedalusAgent(
        engine=ScriptedEngine([call("write_file", path="staged.py", content="x = 1\n"),
                               "Done."]),
        execute=True, gate=False)
    agent.peer = Recorder()

    session_id, _ = drive(agent, workspace, "propose a file")

    assert not (workspace / "staged.py").exists(), "dry run must not write"
    talos = agent.sessions[session_id].talos
    assert talos is not None and len(talos.ws.staged_paths()) == 1


def test_write_mode_actually_writes(workspace):
    agent = DaedalusAgent(
        engine=ScriptedEngine([call("write_file", path="written.py", content="x = 1\n"),
                               "Done."]),
        execute=True, dry_run=False, gate=False)
    agent.peer = Recorder()

    drive(agent, workspace, "write a file")

    assert (workspace / "written.py").read_text(encoding="utf-8") == "x = 1\n"


# --------------------------------------------------------------- stop reason

def test_an_unverified_run_does_not_report_end_turn(workspace):
    """Reporting a failed task as end_turn would tell the editor it succeeded."""
    agent = DaedalusAgent(engine=ScriptedEngine(["Done.", "Done.", "Done."]),
                          execute=True, gate=False, max_steps=3, target_steps=2)
    agent.peer = Recorder()
    # Oracle over an empty change set passes, so install a failing verifier by
    # pre-seeding the session's executor before the first prompt builds one.
    agent.initialize({"protocolVersion": 1, "clientCapabilities": {},
                      "clientInfo": {"name": "t", "version": "1"}})
    sid = agent.session_new({"cwd": str(workspace), "mcpServers": []})["sessionId"]
    agent.sessions[sid].talos = Talos(
        agent.engine, Workspace(workspace, dry_run=True),
        ariadne=Ariadne(max_steps=3, target_steps=2),
        verifier=lambda ws, changed: Verdict(False, "nope"))

    result = agent.session_prompt({"sessionId": sid,
                                   "prompt": [{"type": "text", "text": "do it"}]})
    assert result["stopReason"] != "end_turn"


# ------------------------------------------------------------------ sessions

def test_a_second_prompt_continues_the_same_session(workspace):
    agent = DaedalusAgent(
        engine=ScriptedEngine([
            call("write_file", path="one.py", content="one\n"), "Added one.",
            call("write_file", path="two.py", content="two\n"), "Added two.",
        ]),
        execute=True, gate=False)
    agent.peer = Recorder()

    session_id, _ = drive(agent, workspace, "add one")
    talos = agent.sessions[session_id].talos
    after_first = len(talos.transcript)

    agent.session_prompt({"sessionId": session_id,
                          "prompt": [{"type": "text", "text": "now add two"}]})

    assert len(talos.transcript) > after_first, "context must carry over"
    assert talos.task == "add one", "the original task is retained"
    assert len(talos.ws.staged_paths()) == 2


def test_retrieval_mode_is_unaffected(workspace):
    """The default path must keep working exactly as before."""
    agent = DaedalusAgent(engine=ScriptedEngine(["Here is an answer."]), gate=False)
    agent.peer = Recorder()

    _, result = drive(agent, workspace, "what does this repo do?")

    assert result["stopReason"] == "end_turn"
    assert "agent_message_chunk" in agent.peer.kinds()
    assert agent.sessions[list(agent.sessions)[0]].talos is None
