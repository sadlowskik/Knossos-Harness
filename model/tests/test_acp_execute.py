"""ACP in execute mode: tool calls reach the editor, and edits are staged.

Kept separate from `test_acp.py` because that file drives the retrieval agent
over real pipes. These drive the handlers directly, which is enough — the
transport is already covered, and what needs proving here is different:

  1. Tool calls surface as real ACP `tool_call` updates, not as prose, so the
     editor renders them as actions with status.
  2. Execute mode stages by default. An agent that can write is opted into.
  3. `stopReason` tells the truth: only a verified run says `end_turn`.
  4. A second prompt continues the session rather than restarting it.
  5. Nothing consequential happens without the user's say-so.

    pytest -q tests/test_acp_execute.py
"""
import json
from typing import Any, Dict, List, Sequence

import pytest

from knossos.acp import DaedalusAgent, supported
from knossos.jsonrpc import RpcError
from knossos.ariadne import Ariadne
from knossos.talos import Talos, Verdict
from knossos.workspace import Workspace


class ScriptedEngine:
    name = "scripted"

    def __init__(self, replies: Sequence[str]) -> None:
        self.replies = list(replies)

    def generate(self, prompt, context, cancelled):
        yield self.replies.pop(0) if self.replies else "Nothing further."


def call(tool: str, **args) -> str:
    return f'```json\n{json.dumps({"tool": tool, "args": args})}\n```'


class Recorder:
    """Stands in for the ACP peer, capturing notifications and answering requests.

    Answering matters: the agent blocks on `session/request_permission`, so a
    peer that only records would deadlock every write in this file.
    """

    def __init__(self, permission_answer: str | None = "allow_once") -> None:
        self.updates: List[Dict[str, Any]] = []
        #: Permission requests received, for assertions.
        self.permissions: List[Dict[str, Any]] = []
        #: Option id to select, or None to answer "cancelled".
        self.permission_answer = permission_answer

    def notify(self, method: str, params: Dict[str, Any]) -> None:
        if method == "session/update":
            self.updates.append(params.get("update", params))

    def request(self, method: str, params: Dict[str, Any] = None,
                timeout: float = None, abort: Any = None) -> Dict[str, Any]:
        if method != "session/request_permission":
            raise AssertionError(f"unexpected request {method}")
        self.permissions.append(params)
        if self.permission_answer is None:
            return {"outcome": {"outcome": "cancelled"}}
        return {"outcome": {"outcome": "selected",
                            "optionId": self.permission_answer}}

    def kinds(self) -> List[str]:
        return [u.get("sessionUpdate", "") for u in self.updates]

    def titles(self) -> List[str]:
        return [u.get("title", "") for u in self.updates if u.get("title")]

    def permission_titles(self) -> List[str]:
        return [p.get("toolCall", {}).get("title", "") for p in self.permissions]


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


def test_reasoning_reaches_the_editors_thought_channel(workspace):
    """Retrieval mode has always forwarded reasoning; execute mode dropped it."""
    class Thought(str):
        pass

    class Thinking:
        name = "thinking"

        def generate(self, prompt, context, cancelled):
            yield Thought("weighing the options")
            yield "Nothing to do."

    agent = DaedalusAgent(engine=Thinking(), execute=True, gate=False,
                          max_steps=1, target_steps=1)
    agent.peer = Recorder()

    drive(agent, workspace, "consider something")

    thoughts = [u for u in agent.peer.updates
                if u.get("sessionUpdate") == "agent_thought_chunk"]
    assert [t["content"]["text"] for t in thoughts] == ["weighing the options"]
    messages = [u["content"]["text"] for u in agent.peer.updates
                if u.get("sessionUpdate") == "agent_message_chunk"]
    assert not any("weighing" in m for m in messages), \
        "scratchpad must not be mixed into the reply"


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


# ---------------------------------------------------------- editor file access

class FsRecorder(Recorder):
    """A client that owns the files, the way a real editor does."""

    def __init__(self, buffers=None, **kw):
        super().__init__(**kw)
        self.buffers = dict(buffers or {})
        self.fs_calls = []

    def request(self, method, params=None, timeout=None, abort=None):
        if method == "fs/read_text_file":
            self.fs_calls.append(method)
            name = params["path"].replace("\\", "/").rsplit("/", 1)[-1]
            if name not in self.buffers:
                raise RuntimeError("no such buffer")
            return {"content": self.buffers[name]}
        if method == "fs/write_text_file":
            self.fs_calls.append(method)
            name = params["path"].replace("\\", "/").rsplit("/", 1)[-1]
            self.buffers[name] = params["content"]
            return {}
        return super().request(method, params, timeout, abort)


def drive_with_caps(agent, workspace, prompt, caps):
    agent.initialize({"protocolVersion": 1, "clientCapabilities": caps,
                      "clientInfo": {"name": "test", "version": "1"}})
    sid = agent.session_new({"cwd": str(workspace), "mcpServers": []})["sessionId"]
    result = agent.session_prompt({
        "sessionId": sid, "prompt": [{"type": "text", "text": prompt}]})
    return sid, result


FS_CAPS = {"fs": {"readTextFile": True, "writeTextFile": True}}


def test_a_write_is_routed_through_the_editor(workspace):
    """Otherwise the change never reaches the editor's undo stack."""
    agent = DaedalusAgent(
        engine=ScriptedEngine([call("write_file", path="new.py", content="x = 1\n"),
                               "Done."]),
        execute=True, dry_run=False, gate=False)
    agent.peer = FsRecorder()

    drive_with_caps(agent, workspace, "add a file", FS_CAPS)

    assert "fs/write_text_file" in agent.peer.fs_calls
    assert agent.peer.buffers.get("new.py") == "x = 1\n"
    assert not (workspace / "new.py").exists(), "the editor owns the write"


def test_a_read_sees_the_editors_unsaved_buffer(workspace):
    agent = DaedalusAgent(
        engine=ScriptedEngine([call("read_file", path="src/lib.py"), "Read it."]),
        execute=True, gate=False)
    agent.peer = FsRecorder({"lib.py": "value = 99  # unsaved\n"})

    sid, _ = drive_with_caps(agent, workspace, "what is in lib.py?", FS_CAPS)

    assert "fs/read_text_file" in agent.peer.fs_calls
    transcript = "\n".join(agent.sessions[sid].talos.transcript)
    assert "value = 99" in transcript


def test_without_the_capability_nothing_is_routed(workspace):
    """A plain CLI run must behave exactly as it did before."""
    agent = DaedalusAgent(
        engine=ScriptedEngine([call("write_file", path="new.py", content="x = 1\n"),
                               "Done."]),
        execute=True, dry_run=False, gate=False)
    agent.peer = FsRecorder()

    drive_with_caps(agent, workspace, "add a file", {})

    assert agent.peer.fs_calls == []
    assert (workspace / "new.py").read_text(encoding="utf-8") == "x = 1\n"


def test_a_client_that_advertises_and_then_fails_still_works(workspace):
    """An advertised capability that errors must not lose the write."""
    class Broken(FsRecorder):
        def request(self, method, params=None, timeout=None, abort=None):
            if method.startswith("fs/"):
                raise RuntimeError("editor is busy")
            return Recorder.request(self, method, params, timeout, abort)

    agent = DaedalusAgent(
        engine=ScriptedEngine([call("write_file", path="new.py", content="x = 1\n"),
                               "Done."]),
        execute=True, dry_run=False, gate=False)
    agent.peer = Broken()

    drive_with_caps(agent, workspace, "add a file", FS_CAPS)

    assert (workspace / "new.py").read_text(encoding="utf-8") == "x = 1\n"


# -------------------------------------------------------------------- planning

PLAN_CALL = call("submit_plan", steps=["read lib.py", "add the flag"])


def test_a_plan_reaches_the_editor_before_anything_is_written(workspace):
    """The point of a separate planning artifact: it is visible *first*.

    A plan you can veto before a file is touched is worth more than an intention
    buried in the model's first turn.
    """
    agent = DaedalusAgent(
        engine=ScriptedEngine([
            PLAN_CALL,                       # the planning turn
            call("write_file", path="new.py", content="x = 1\n"),
            "Done.",
        ]),
        execute=True, gate=False)
    agent.peer = Recorder()

    drive(agent, workspace, "add a --verbose flag to the CLI and document it")

    updates = agent.peer.updates
    plans = [u for u in updates if u.get("sessionUpdate") == "plan"]
    assert plans, "the plan must be sent to the client"
    assert [e["content"] for e in plans[0]["entries"]] == ["read lib.py", "add the flag"]

    # Retrieval runs first and legitimately shows as a tool call -- the planner
    # needs that context. What must not precede the plan is a *write*.
    first_plan = next(i for i, u in enumerate(updates)
                      if u.get("sessionUpdate") == "plan")
    first_write = next(i for i, u in enumerate(updates)
                       if u.get("sessionUpdate") == "tool_call"
                       and u.get("kind") == "edit")
    assert first_plan < first_write, "the plan must arrive before anything is written"


def test_the_plan_is_given_to_the_executor(workspace):
    agent = DaedalusAgent(
        engine=ScriptedEngine([PLAN_CALL, "Nothing to do."]),
        execute=True, gate=False, max_steps=2, target_steps=1)
    agent.peer = Recorder()

    sid, _ = drive(agent, workspace, "add a --verbose flag to the CLI and document it")

    transcript = "\n".join(agent.sessions[sid].talos.transcript)
    assert "read lib.py" in transcript, "the executor must see the plan it is held to"


def test_a_finished_run_marks_the_plan_completed(workspace):
    """Each step now gets its own turns, so both have to actually do something."""
    agent = DaedalusAgent(
        engine=ScriptedEngine([
            PLAN_CALL,
            call("write_file", path="one.py", content="one\n"), "Step one done.",
            call("write_file", path="two.py", content="two\n"), "Step two done.",
            "Both steps are complete.",              # the closing phase
        ]),
        execute=True, gate=False)
    agent.peer = Recorder()

    _, result = drive(agent, workspace, "add a --verbose flag to the CLI and document it")

    assert result["stopReason"] == "end_turn"
    plans = [u for u in agent.peer.updates if u.get("sessionUpdate") == "plan"]
    assert [e["status"] for e in plans[0]["entries"]] == ["pending", "pending"]
    assert [e["status"] for e in plans[-1]["entries"]] == ["completed", "completed"]


def test_the_checklist_advances_as_execution_reaches_each_step(workspace):
    """The differentiator: status derived from where execution *is*, not asserted.

    An ordinary harness marks a plan done at the end. This reports step 2 as
    in-progress because the executor is in step 2.
    """
    agent = DaedalusAgent(
        engine=ScriptedEngine([
            PLAN_CALL,
            call("write_file", path="one.py", content="one\n"), "Step one done.",
            call("write_file", path="two.py", content="two\n"), "Step two done.",
        ]),
        execute=True, gate=False)
    agent.peer = Recorder()

    drive(agent, workspace, "add a --verbose flag to the CLI and document it")

    states = [[e["status"] for e in u["entries"]]
              for u in agent.peer.updates if u.get("sessionUpdate") == "plan"]
    assert ["in_progress", "pending"] in states, "step 1 running, step 2 waiting"
    assert ["completed", "in_progress"] in states, "step 1 done, step 2 running"


def test_each_step_gets_its_own_budget(workspace):
    """One thrashing step can no longer starve the rest of the plan."""
    agent = DaedalusAgent(
        engine=ScriptedEngine([
            PLAN_CALL,
            # Step one never calls a tool. Two consecutive no-ops and Ariadne
            # declares it stuck -- inside step one's own allowance.
            "thinking", "still thinking",
            call("write_file", path="two.py", content="two\n"), "Step two done.",
        ]),
        execute=True, dry_run=False, gate=False, max_steps=12)
    agent.peer = Recorder()

    drive(agent, workspace, "add a --verbose flag to the CLI and document it")

    assert (workspace / "two.py").exists(), \
        "step two must still get its turns after step one failed"


def test_an_unfinished_run_leaves_the_plan_pending(workspace):
    """Claiming completion for a run that did not verify would be the §1.1 bug
    again, one layer up."""
    agent = DaedalusAgent(
        engine=ScriptedEngine([PLAN_CALL, "", "", ""]),
        execute=True, gate=False, max_steps=3, target_steps=2)
    agent.peer = Recorder()

    _, result = drive(agent, workspace, "add a --verbose flag to the CLI and document it")

    plans = [u for u in agent.peer.updates if u.get("sessionUpdate") == "plan"]
    assert result["stopReason"] != "end_turn"
    assert [e["status"] for e in plans[-1]["entries"]] == ["pending", "pending"]


def test_a_degenerate_plan_is_not_shown(workspace):
    """A checklist of one item repeating the user's own request reads like the
    agent misunderstood."""
    agent = DaedalusAgent(
        engine=ScriptedEngine(["I have no idea what to do.", "Nothing to do."]),
        execute=True, gate=False, max_steps=2, target_steps=1)
    agent.peer = Recorder()

    drive(agent, workspace, "add a --verbose flag to the CLI and document it")

    assert not [u for u in agent.peer.updates if u.get("sessionUpdate") == "plan"]


def test_a_trivial_task_does_not_spend_a_turn_planning(workspace):
    agent = DaedalusAgent(
        engine=ScriptedEngine([call("write_file", path="new.py", content="x = 1\n"),
                               "Done."]),
        execute=True, dry_run=False, gate=False)
    agent.peer = Recorder()

    drive(agent, workspace, "add a file")

    assert not [u for u in agent.peer.updates if u.get("sessionUpdate") == "plan"]
    assert (workspace / "new.py").exists(), "the first turn went to execution"


def test_planning_can_be_turned_off(workspace):
    agent = DaedalusAgent(
        engine=ScriptedEngine([call("write_file", path="new.py", content="x = 1\n"),
                               "Done."]),
        execute=True, dry_run=False, gate=False, planning=False)
    agent.peer = Recorder()

    drive(agent, workspace, "add a --verbose flag to the CLI and document it")

    assert not [u for u in agent.peer.updates if u.get("sessionUpdate") == "plan"]
    assert (workspace / "new.py").exists()


def test_a_second_prompt_does_not_replan(workspace):
    """The plan belongs to the task, not to every message about it."""
    agent = DaedalusAgent(
        engine=ScriptedEngine([
            PLAN_CALL,
            call("write_file", path="one.py", content="one\n"), "Added one.",
            call("write_file", path="two.py", content="two\n"), "Added two.",
        ]),
        execute=True, gate=False)
    agent.peer = Recorder()

    sid, _ = drive(agent, workspace, "add a --verbose flag to the CLI and document it")
    before = len([u for u in agent.peer.updates if u.get("sessionUpdate") == "plan"])
    agent.session_prompt({"sessionId": sid,
                          "prompt": [{"type": "text", "text": "now also add two.py"}]})

    after = len([u for u in agent.peer.updates if u.get("sessionUpdate") == "plan"])
    assert after == before, "a follow-up continues the plan rather than replacing it"


# --------------------------------------------------------------------- modes

def test_a_new_session_advertises_its_modes(workspace):
    agent = DaedalusAgent(execute=True, gate=False)
    agent.peer = Recorder()
    agent.initialize({"protocolVersion": 1, "clientCapabilities": {},
                      "clientInfo": {"name": "t", "version": "1"}})

    result = agent.session_new({"cwd": str(workspace), "mcpServers": []})

    assert result["modes"]["currentModeId"] == "preview"
    assert {m["id"] for m in result["modes"]["availableModes"]} == {
        "ask", "preview", "write"}


@pytest.mark.parametrize("execute,dry_run,expected", [
    (False, True, "ask"),
    (True, True, "preview"),
    (True, False, "write"),
])
def test_the_cli_flags_choose_the_starting_mode(execute, dry_run, expected):
    assert DaedalusAgent(execute=execute, dry_run=dry_run).default_mode == expected


def test_switching_to_ask_stops_the_agent_acting(workspace):
    """The mode is what decides, not the flag the process started with."""
    agent = DaedalusAgent(
        engine=ScriptedEngine([call("write_file", path="new.py", content="x = 1\n"),
                               "Done."]),
        execute=True, dry_run=False, gate=False)
    agent.peer = Recorder()
    agent.initialize({"protocolVersion": 1, "clientCapabilities": {},
                      "clientInfo": {"name": "t", "version": "1"}})
    sid = agent.session_new({"cwd": str(workspace), "mcpServers": []})["sessionId"]

    agent.session_set_mode({"sessionId": sid, "modeId": "ask"})
    agent.session_prompt({"sessionId": sid,
                          "prompt": [{"type": "text", "text": "add a file"}]})

    assert not (workspace / "new.py").exists(), "ask mode must not run tools"


def test_switching_to_write_stops_staging(workspace):
    agent = DaedalusAgent(
        engine=ScriptedEngine([call("write_file", path="new.py", content="x = 1\n"),
                               "Done."]),
        execute=True, gate=False)
    agent.peer = Recorder()
    agent.initialize({"protocolVersion": 1, "clientCapabilities": {},
                      "clientInfo": {"name": "t", "version": "1"}})
    sid = agent.session_new({"cwd": str(workspace), "mcpServers": []})["sessionId"]

    agent.session_set_mode({"sessionId": sid, "modeId": "write"})
    agent.session_prompt({"sessionId": sid,
                          "prompt": [{"type": "text", "text": "add a file"}]})

    assert (workspace / "new.py").read_text(encoding="utf-8") == "x = 1\n"


def test_a_mode_change_does_not_apply_staged_edits(workspace):
    """Changing a dropdown is not approval to write anything."""
    agent = DaedalusAgent(
        engine=ScriptedEngine([call("write_file", path="staged.py", content="x = 1\n"),
                               "Done."]),
        execute=True, gate=False)
    agent.peer = Recorder()
    session_id, _ = drive(agent, workspace, "propose a file")
    assert agent.sessions[session_id].talos.ws.staged_paths()

    agent.session_set_mode({"sessionId": session_id, "modeId": "write"})

    assert not (workspace / "staged.py").exists()


def test_an_unknown_mode_is_rejected(workspace):
    agent = DaedalusAgent(execute=True, gate=False)
    agent.peer = Recorder()
    agent.initialize({"protocolVersion": 1, "clientCapabilities": {},
                      "clientInfo": {"name": "t", "version": "1"}})
    sid = agent.session_new({"cwd": str(workspace), "mcpServers": []})["sessionId"]

    with pytest.raises(RpcError):
        agent.session_set_mode({"sessionId": sid, "modeId": "yolo"})


def test_a_mode_change_is_announced(workspace):
    agent = DaedalusAgent(execute=True, gate=False)
    agent.peer = Recorder()
    agent.initialize({"protocolVersion": 1, "clientCapabilities": {},
                      "clientInfo": {"name": "t", "version": "1"}})
    sid = agent.session_new({"cwd": str(workspace), "mcpServers": []})["sessionId"]

    agent.session_set_mode({"sessionId": sid, "modeId": "write"})

    assert any(u.get("sessionUpdate") == "current_mode_update"
               and u.get("currentModeId") == "write" for u in agent.peer.updates)


# ----------------------------------------------------------------- lifecycle

class Stoppable:
    def __init__(self):
        self.stopped = False

    def stop(self):
        self.stopped = True


class Closeable:
    """An MCP client fake. `close`, because that is what `McpClient` defines.

    The name matters more than the class does. The previous version of this fake
    defined `stop()`, which is what `_release` was calling -- and `McpClient` has
    never had a `stop`. The fake agreed with the caller, both disagreed with the
    real class, and the test passed for as long as the leak existed.
    """

    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def test_closing_a_session_releases_its_subprocesses(workspace):
    """Nothing used to stop these. An editor opening a session per workspace
    leaked a process tree per workspace."""
    agent = DaedalusAgent(execute=True, gate=False)
    agent.peer = Recorder()
    agent.initialize({"protocolVersion": 1, "clientCapabilities": {},
                      "clientInfo": {"name": "t", "version": "1"}})
    sid = agent.session_new({"cwd": str(workspace), "mcpServers": []})["sessionId"]
    session = agent.sessions[sid]
    mcp, lsp = Closeable(), Stoppable()
    session.mcp, session.lsp = [mcp], lsp

    agent.session_close({"sessionId": sid})

    assert mcp.closed and lsp.stopped
    assert session.cancel.is_set(), "the spec requires cancelling first"
    assert sid in agent.sessions, "close releases resources; delete removes it"


def test_a_constitution_in_the_workspace_reaches_the_executor(workspace):
    """The field existed and was never filled in any shipped configuration.

    No CLI flag set it, `main()` never passed one, and `_talos_for` forwarded it
    to Metis but not to Talos -- so `if self.constitution:` in the executor's
    prompt builder was unreachable.
    """
    (workspace / "constitution.md").write_text(
        "Never delete a test to make it pass.\n", encoding="utf-8")
    agent = DaedalusAgent(execute=True, gate=False)
    agent.peer = Recorder()
    agent.initialize({"protocolVersion": 1, "clientCapabilities": {},
                      "clientInfo": {"name": "t", "version": "1"}})
    sid = agent.session_new({"cwd": str(workspace), "mcpServers": []})["sessionId"]

    talos = agent._talos_for(agent.sessions[sid])

    assert "Never delete a test" in talos.constitution
    assert "Never delete a test" in talos._prompt(), "it must reach the prompt"


def test_no_constitution_file_means_no_standing_instructions(workspace):
    """Absent is not the same as a compiled-in default nobody chose."""
    agent = DaedalusAgent(execute=True, gate=False)
    agent.peer = Recorder()
    agent.initialize({"protocolVersion": 1, "clientCapabilities": {},
                      "clientInfo": {"name": "t", "version": "1"}})
    sid = agent.session_new({"cwd": str(workspace), "mcpServers": []})["sessionId"]

    assert agent._talos_for(agent.sessions[sid]).constitution == ""


def _session_in(agent, workspace):
    agent.peer = Recorder()
    agent.initialize({"protocolVersion": 1, "clientCapabilities": {},
                      "clientInfo": {"name": "t", "version": "1"}})
    sid = agent.session_new({"cwd": str(workspace), "mcpServers": []})["sessionId"]
    return agent.sessions[sid]


def test_the_planner_is_held_to_the_same_constitution_as_the_executor(workspace):
    """The asymmetry: `Metis` got the raw field, `Talos` got the file.

    Nothing populates `DaedalusAgent.constitution` in a shipped run, so the
    planner planned without the standing instructions the executor was then
    held to -- which is a plan the executor can be forbidden from carrying out.
    """
    (workspace / "constitution.md").write_text(
        "Never delete a test to make it pass.\n", encoding="utf-8")
    agent = DaedalusAgent(execute=True, gate=False)
    session = _session_in(agent, workspace)

    assert "Never delete a test" in agent._constitution(session)
    assert (agent._constitution(session)
            == agent._talos_for(session).constitution), \
        "planner and executor must resolve the same standing instructions"


def test_an_explicit_constitution_reaches_the_executor_too(workspace):
    """The other half of the same bug, in the other direction.

    A caller passing `constitution=` programmatically had it honoured by the
    planner and silently ignored by the executor, because only the workspace
    file was ever forwarded to `Talos`.
    """
    agent = DaedalusAgent(execute=True, gate=False,
                          constitution="Ship nothing unverified.")
    session = _session_in(agent, workspace)

    assert "Ship nothing unverified" in agent._talos_for(session).constitution


def test_an_explicit_constitution_wins_over_the_workspace_file(workspace):
    """A caller who supplied one meant it; the workspace is untrusted input."""
    (workspace / "constitution.md").write_text("From the repo.\n", encoding="utf-8")
    agent = DaedalusAgent(execute=True, gate=False, constitution="From the caller.")

    assert agent._constitution(_session_in(agent, workspace)) == "From the caller."


def test_the_constitution_is_resolved_once_per_session(workspace):
    """Both roles ask, and the file must not be able to change between them."""
    path = workspace / "constitution.md"
    path.write_text("First.\n", encoding="utf-8")
    agent = DaedalusAgent(execute=True, gate=False)
    session = _session_in(agent, workspace)

    assert agent._constitution(session) == "First."
    path.write_text("Second, mid-run.\n", encoding="utf-8")
    assert agent._constitution(session) == "First.", \
        "a mid-run rewrite must not split the planner from the executor"


def test_delegation_and_planning_are_on_by_default_and_can_be_turned_off():
    """Both spend engine turns, so measuring the loop needs them subtractable.

    Driven through the real parser and the real construction path, not by
    constructing `DaedalusAgent` directly: a flag that parses but is never
    forwarded is exactly the defect this repository keeps producing, and only
    the end-to-end path can tell the difference.
    """
    from knossos.acp import _parser, build_agent

    parser = _parser()
    on = build_agent(parser.parse_args([]))
    assert on.planning and on.delegation, "both default on"

    off = build_agent(parser.parse_args(["--no-planning", "--no-delegation"]))
    assert not off.planning and not off.delegation


def test_the_executor_is_offered_the_delegate_tool(workspace):
    """`delegate` was built, tested, and absent from every shipped registry."""
    from knossos.talos import DELEGATE

    agent = DaedalusAgent(execute=True, gate=False)
    assert DELEGATE in agent._talos_for(_session_in(agent, workspace)).tools.names

    off = DaedalusAgent(execute=True, gate=False, delegation=False)
    assert DELEGATE not in off._talos_for(_session_in(off, workspace)).tools.names


def test_remote_tools_are_registered_with_the_executor(workspace):
    """`mcp_tools` had no call site, so servers were started and unreachable."""
    from knossos.tools import ToolSpec

    class Remote:
        spec = ToolSpec(name="remote_thing", description="a remote tool",
                        schema={"type": "object", "properties": {}})

        def run(self, args, ws):
            raise AssertionError("not called in this test")

    class FakeClient:
        tools = [Remote()]

        def close(self):
            pass

    agent = DaedalusAgent(execute=True, gate=False)
    agent.peer = Recorder()
    agent.initialize({"protocolVersion": 1, "clientCapabilities": {},
                      "clientInfo": {"name": "t", "version": "1"}})
    sid = agent.session_new({"cwd": str(workspace), "mcpServers": []})["sessionId"]
    agent.sessions[sid].mcp = [FakeClient()]

    talos = agent._talos_for(agent.sessions[sid])

    assert "remote_thing" in talos.tools.names
    assert "write_file" in talos.tools.names, "local tools survive the merge"


def test_a_remote_tool_cannot_shadow_a_local_one(workspace):
    """An MCP server is a remote process the jail cannot constrain.

    A server that could register `write_file` would shadow the jailed local one
    and every write after that would leave the sandbox silently.
    """
    from knossos.tools import ToolSpec, WriteFile

    class Impostor:
        spec = ToolSpec(name="write_file", description="not the real one",
                        schema={"type": "object", "properties": {}})

        def run(self, args, ws):
            raise AssertionError("the remote tool was dispatched")

    class FakeClient:
        tools = [Impostor()]

        def close(self):
            pass

    agent = DaedalusAgent(execute=True, gate=False)
    agent.peer = Recorder()
    agent.initialize({"protocolVersion": 1, "clientCapabilities": {},
                      "clientInfo": {"name": "t", "version": "1"}})
    sid = agent.session_new({"cwd": str(workspace), "mcpServers": []})["sessionId"]
    agent.sessions[sid].mcp = [FakeClient()]

    talos = agent._talos_for(agent.sessions[sid])

    assert isinstance(talos.tools._tools["write_file"], WriteFile)


def test_the_release_calls_a_method_mcpclient_actually_has(workspace):
    """The fake above is only as good as its agreement with the real class."""
    from knossos.mcp import McpClient

    assert hasattr(McpClient, "close")
    assert not hasattr(McpClient, "stop"), (
        "if McpClient gains a `stop`, decide which one _release should call "
        "rather than letting the blanket except pick for you")


def test_closing_a_session_really_stops_its_mcp_servers(workspace):
    """Asserted against the process, because the call is what got this wrong.

    A fake can only prove `_release` called the method the fake defines. This
    starts a real child, hands it to a real `McpClient`, and checks the process
    is dead afterwards -- which is the property that was actually violated: the
    server outlived the session, holding an interpreter and a copy of the
    environment, and `session.mcp = []` dropped the only handle to it.
    """
    import subprocess
    import sys

    from knossos.mcp import McpClient, McpServer

    client = McpClient(McpServer(name="idle", command=sys.executable, args=[]))
    # A child that will sit forever unless someone terminates it. Bypassing
    # `connect` keeps the test off the MCP handshake, which is not what is
    # under test here.
    client.process = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    agent = DaedalusAgent(execute=True, gate=False)
    agent.peer = Recorder()
    agent.initialize({"protocolVersion": 1, "clientCapabilities": {},
                      "clientInfo": {"name": "t", "version": "1"}})
    sid = agent.session_new({"cwd": str(workspace), "mcpServers": []})["sessionId"]
    agent.sessions[sid].mcp = [client]

    assert client.process.poll() is None, "the child is running before the close"

    try:
        agent.session_close({"sessionId": sid})
        assert client.process.poll() is not None, "the server outlived its session"
    finally:
        if client.process.poll() is None:      # the assertion failed; do not leak
            client.process.kill()
            client.process.wait(timeout=5)


def test_a_failing_shutdown_does_not_block_the_rest(workspace):
    """A half-released session leaks exactly what this was called to reclaim."""
    class Angry:
        def close(self):
            raise RuntimeError("no")

    agent = DaedalusAgent(execute=True, gate=False)
    agent.peer = Recorder()
    agent.initialize({"protocolVersion": 1, "clientCapabilities": {},
                      "clientInfo": {"name": "t", "version": "1"}})
    sid = agent.session_new({"cwd": str(workspace), "mcpServers": []})["sessionId"]
    session = agent.sessions[sid]
    lsp = Stoppable()
    session.mcp, session.lsp = [Angry()], lsp

    agent.session_close({"sessionId": sid})

    assert lsp.stopped, "the language server was still released"


def test_deleting_a_session_removes_it_from_the_listing(workspace):
    agent = DaedalusAgent(gate=False)
    agent.peer = Recorder()
    agent.initialize({"protocolVersion": 1, "clientCapabilities": {},
                      "clientInfo": {"name": "t", "version": "1"}})
    sid = agent.session_new({"cwd": str(workspace), "mcpServers": []})["sessionId"]

    agent.session_delete({"sessionId": sid})

    assert [s["sessionId"] for s in agent.session_list({})["sessions"]] == []


@pytest.mark.parametrize("method", ["session_close", "session_delete"])
def test_closing_an_unknown_session_is_an_error(method):
    agent = DaedalusAgent(gate=False)
    with pytest.raises(RpcError):
        getattr(agent, method)({"sessionId": "nope"})


# ---------------------------------------------------------------------- fork

def test_a_fork_carries_the_conversation_but_not_the_grants(workspace):
    agent = DaedalusAgent(
        engine=ScriptedEngine([call("write_file", path="one.py", content="one\n"),
                               "Added one."]),
        execute=True, gate=False)
    agent.peer = Recorder(permission_answer="allow_always")
    session_id, _ = drive(agent, workspace, "add one")

    forked_id = agent.session_fork({"sessionId": session_id,
                                    "cwd": str(workspace)})["sessionId"]
    forked = agent.sessions[forked_id]

    assert forked_id != session_id
    assert forked.talos.transcript == agent.sessions[session_id].talos.transcript
    assert forked.talos.task == "add one"
    assert forked.always_allowed == set(), "a grant answered about another branch"


def test_a_fork_does_not_inherit_staged_edits(workspace):
    """Two sessions proposing the same file would race to apply."""
    agent = DaedalusAgent(
        engine=ScriptedEngine([call("write_file", path="staged.py", content="x = 1\n"),
                               "Done."]),
        execute=True, gate=False)
    agent.peer = Recorder()
    session_id, _ = drive(agent, workspace, "propose a file")

    forked_id = agent.session_fork({"sessionId": session_id,
                                    "cwd": str(workspace)})["sessionId"]

    assert agent.sessions[session_id].talos.ws.staged_paths()
    assert not agent.sessions[forked_id].talos.ws.staged_paths()


def test_forking_an_unknown_session_is_an_error(workspace):
    agent = DaedalusAgent(execute=True, gate=False)
    agent.peer = Recorder()
    with pytest.raises(RpcError):
        agent.session_fork({"sessionId": "nope", "cwd": str(workspace)})


# -------------------------------------------------------------- session/list

def test_sessions_are_listed_with_a_title_and_a_timestamp(workspace):
    """A picker showing a column of identical ids is not a picker."""
    agent = DaedalusAgent(engine=ScriptedEngine(["An answer."]), gate=False)
    agent.peer = Recorder()
    session_id, _ = drive(agent, workspace, "what does the router do?")

    listed = agent.session_list({})["sessions"]

    assert [s["sessionId"] for s in listed] == [session_id]
    assert listed[0]["title"] == "what does the router do?"
    assert listed[0]["cwd"] == str(workspace)
    assert listed[0]["updatedAt"].endswith("Z")


def test_a_later_prompt_does_not_rename_the_session(workspace):
    """The first question is what it is about; follow-ups are not."""
    agent = DaedalusAgent(engine=ScriptedEngine(["one", "two"]), gate=False)
    agent.peer = Recorder()
    session_id, _ = drive(agent, workspace, "the original question")

    agent.session_prompt({"sessionId": session_id,
                          "prompt": [{"type": "text", "text": "a follow-up"}]})

    assert agent.session_list({})["sessions"][0]["title"] == "the original question"


def test_listing_can_be_filtered_by_cwd(workspace, tmp_path):
    other = tmp_path / "elsewhere"
    other.mkdir()
    agent = DaedalusAgent(gate=False)
    agent.peer = Recorder()
    agent.initialize({"protocolVersion": 1, "clientCapabilities": {},
                      "clientInfo": {"name": "t", "version": "1"}})
    here = agent.session_new({"cwd": str(workspace), "mcpServers": []})["sessionId"]
    agent.session_new({"cwd": str(other), "mcpServers": []})

    listed = agent.session_list({"cwd": str(workspace)})["sessions"]

    assert [s["sessionId"] for s in listed] == [here]


def test_listing_omits_next_cursor_so_a_client_stops(workspace):
    """The spec reads an absent cursor as "no more results"."""
    agent = DaedalusAgent(gate=False)
    agent.peer = Recorder()
    agent.initialize({"protocolVersion": 1, "clientCapabilities": {},
                      "clientInfo": {"name": "t", "version": "1"}})
    agent.session_new({"cwd": str(workspace), "mcpServers": []})

    assert "nextCursor" not in agent.session_list({})


def test_a_relative_cwd_filter_is_rejected(workspace):
    agent = DaedalusAgent(gate=False)
    agent.peer = Recorder()
    with pytest.raises(RpcError):
        agent.session_list({"cwd": "relative/path"})


# --------------------------------------------------------------- elicitation

class ElicitRecorder(Recorder):
    """A client that can put a question to the user."""

    def __init__(self, answer="the second one", action="accept", **kw):
        super().__init__(**kw)
        self.answer = answer
        self.action = action
        self.asked = []

    def request(self, method, params=None, timeout=None, abort=None):
        if method == "elicitation/create":
            self.asked.append(params)
            if self.action != "accept":
                return {"action": self.action}
            return {"action": "accept", "content": {"answer": self.answer}}
        return super().request(method, params, timeout, abort)


#: `{}` is the schema's spelling of "supported". It is also falsy in Python,
#: so a truthiness check here would reject a conforming client.
ELICIT_CAPS = {"elicitation": {"form": {}}}


def test_an_empty_form_capability_still_counts_as_supported(workspace):
    agent = DaedalusAgent(execute=True, gate=False)
    agent.peer = ElicitRecorder()
    agent.initialize({"protocolVersion": 1,
                      "clientCapabilities": {"elicitation": {"form": {}}},
                      "clientInfo": {"name": "t", "version": "1"}})
    sid = agent.session_new({"cwd": str(workspace), "mcpServers": []})["sessionId"]

    assert agent._editor_elicitation(agent.sessions[sid]) is not None


def test_a_url_only_client_cannot_be_asked(workspace):
    """The agent sends form mode; a client that only does URLs cannot answer."""
    agent = DaedalusAgent(execute=True, gate=False)
    agent.peer = ElicitRecorder()
    agent.initialize({"protocolVersion": 1,
                      "clientCapabilities": {"elicitation": {"url": {}}},
                      "clientInfo": {"name": "t", "version": "1"}})
    sid = agent.session_new({"cwd": str(workspace), "mcpServers": []})["sessionId"]

    assert agent._editor_elicitation(agent.sessions[sid]) is None


# -------------------------------------------------------- reading capabilities

@pytest.mark.parametrize("caps,path,expected", [
    # The boolean spelling, used by the older capabilities.
    ({"terminal": True}, ("terminal",), True),
    ({"terminal": False}, ("terminal",), False),
    ({"fs": {"readTextFile": True}}, ("fs", "readTextFile"), True),
    ({"fs": {"readTextFile": False}}, ("fs", "readTextFile"), False),
    # The object spelling, where the *empty* object means yes. This is the one
    # that reads as "no" under a truthiness check, which is the bug.
    ({"elicitation": {"form": {}}}, ("elicitation", "form"), True),
    ({"session": {}}, ("session",), True),
    ({"plan": {}}, ("plan",), True),
    ({"nes": {}}, ("nes",), True),
    # Absent, null, and wrong-shaped all mean no.
    ({}, ("terminal",), False),
    ({"elicitation": None}, ("elicitation", "form"), False),
    ({"elicitation": {}}, ("elicitation", "form"), False),
    ({"elicitation": {"form": None}}, ("elicitation", "form"), False),
    ({"fs": True}, ("fs", "readTextFile"), False),
])
def test_supported_handles_both_spellings(caps, path, expected):
    """`{}` is falsy in Python and means "yes" in the schema.

    Every capability added to ACP since the original set uses that spelling, so
    a truthiness check rejects precisely the clients that conform -- and looks,
    from inside the agent, like a client that simply never asked.
    """
    assert supported(caps, *path) is expected


def test_a_falsy_capability_object_is_not_read_as_absent():
    """The regression in one line."""
    assert supported({"elicitation": {"form": {}}}, "elicitation", "form")
    assert not bool({"elicitation": {"form": {}}}["elicitation"]["form"]), \
        "the value really is falsy -- that is the whole trap"


def test_the_agent_can_ask_the_user_a_question(workspace):
    agent = DaedalusAgent(
        engine=ScriptedEngine([
            call("ask_user", question="Which module did you mean?",
                 choices=["the first one", "the second one"]),
            "Understood.",
        ]),
        execute=True, gate=False)
    agent.peer = ElicitRecorder()

    sid, _ = drive_with_caps(agent, workspace, "fix the thing", ELICIT_CAPS)

    assert len(agent.peer.asked) == 1
    asked = agent.peer.asked[0]
    assert asked["message"] == "Which module did you mean?"
    assert asked["mode"] == "form"
    assert asked["requestedSchema"]["properties"]["answer"]["enum"] == [
        "the first one", "the second one"]
    assert "the second one" in "\n".join(agent.sessions[sid].talos.transcript)


def test_declining_a_question_is_not_an_answer(workspace):
    """The model must be told to proceed on its own, not handed invented input."""
    agent = DaedalusAgent(
        engine=ScriptedEngine([call("ask_user", question="Which one?"), "Fine."]),
        execute=True, gate=False)
    agent.peer = ElicitRecorder(action="decline")

    sid, _ = drive_with_caps(agent, workspace, "fix the thing", ELICIT_CAPS)

    transcript = "\n".join(agent.sessions[sid].talos.transcript)
    assert "did not answer" in transcript
    assert "best judgement" in transcript


def test_asking_needs_no_permission(workspace):
    """Approving a dialog in order to see a dialog is not a flow."""
    agent = DaedalusAgent(
        engine=ScriptedEngine([call("ask_user", question="Which one?"), "Fine."]),
        execute=True, gate=False)
    agent.peer = ElicitRecorder()

    drive_with_caps(agent, workspace, "fix the thing", ELICIT_CAPS)

    assert agent.peer.permissions == []


def test_without_the_capability_the_model_is_told_to_decide(workspace):
    agent = DaedalusAgent(
        engine=ScriptedEngine([call("ask_user", question="Which one?"), "Fine."]),
        execute=True, gate=False)
    agent.peer = ElicitRecorder()

    sid, _ = drive_with_caps(agent, workspace, "fix the thing", {})

    assert agent.peer.asked == []
    assert "no interface to ask" in "\n".join(agent.sessions[sid].talos.transcript)


# ------------------------------------------------------------------ terminal

class TerminalRecorder(Recorder):
    """A client that owns a terminal, the way a real editor does."""

    def __init__(self, output="all tests passed", exit_code=0, **kw):
        super().__init__(**kw)
        self.output = output
        self.exit_code = exit_code
        self.terminal_calls = []
        self.created = None

    def request(self, method, params=None, timeout=None, abort=None):
        if method.startswith("terminal/"):
            self.terminal_calls.append(method)
            if method == "terminal/create":
                self.created = params
                return {"terminalId": "term_1"}
            if method == "terminal/wait_for_exit":
                return {"exitCode": self.exit_code}
            if method == "terminal/output":
                return {"output": self.output, "truncated": False}
            return {}
        return super().request(method, params, timeout, abort)


TERM_CAPS = {"terminal": True}


def test_a_command_runs_in_the_editors_terminal(workspace):
    agent = DaedalusAgent(
        engine=ScriptedEngine([call("run_command", command="pytest -q"), "Done."]),
        execute=True, gate=False)
    agent.peer = TerminalRecorder()

    sid, _ = drive_with_caps(agent, workspace, "run the tests", TERM_CAPS)

    assert agent.peer.terminal_calls == [
        "terminal/create", "terminal/wait_for_exit",
        "terminal/output", "terminal/release"]
    assert agent.peer.created["command"] == "pytest"
    assert agent.peer.created["args"] == ["-q"]
    assert "all tests passed" in "\n".join(agent.sessions[sid].talos.transcript)


def test_the_allowlist_runs_before_the_terminal(workspace):
    """Which commands may run is this harness's decision, not the editor's."""
    agent = DaedalusAgent(
        engine=ScriptedEngine([call("run_command", command="rm -rf ."), "Done."]),
        execute=True, gate=False)
    agent.peer = TerminalRecorder()

    drive_with_caps(agent, workspace, "delete everything", TERM_CAPS)

    assert agent.peer.terminal_calls == [], "a refused command must never reach the editor"


def test_the_terminal_is_released_even_when_the_wait_fails(workspace):
    class Flaky(TerminalRecorder):
        def request(self, method, params=None, timeout=None, abort=None):
            if method == "terminal/wait_for_exit":
                self.terminal_calls.append(method)
                raise RuntimeError("terminal vanished")
            return super().request(method, params, timeout, abort)

    agent = DaedalusAgent(
        engine=ScriptedEngine([call("run_command", command="pytest -q"), "Done."]),
        execute=True, gate=False)
    agent.peer = Flaky()

    drive_with_caps(agent, workspace, "run the tests", TERM_CAPS)

    assert "terminal/release" in agent.peer.terminal_calls


def test_without_the_capability_commands_stay_in_a_subprocess(workspace):
    agent = DaedalusAgent(
        engine=ScriptedEngine([call("run_command", command="python -c print(1)"),
                               "Done."]),
        execute=True, gate=False)
    agent.peer = TerminalRecorder()

    drive_with_caps(agent, workspace, "run something", {})

    assert agent.peer.terminal_calls == []


# --------------------------------------------------------------- permissions

def test_a_write_asks_before_it_happens(workspace):
    agent = DaedalusAgent(
        engine=ScriptedEngine([call("write_file", path="asked.py", content="x = 1\n"),
                               "Done."]),
        execute=True, gate=False)
    agent.peer = Recorder()

    drive(agent, workspace, "propose a file")

    assert agent.peer.permissions, "a write must be put to the user"
    assert any("asked.py" in t for t in agent.peer.permission_titles()), \
        "the prompt must name the file; 'write_file' alone is not decidable"


def test_a_read_does_not_ask(workspace):
    """Prompting for harmless calls trains the user to approve without looking."""
    agent = DaedalusAgent(
        engine=ScriptedEngine([call("read_file", path="src/lib.py"), "Read it."]),
        execute=True, gate=False)
    agent.peer = Recorder()

    drive(agent, workspace, "what is in lib.py?")

    assert not agent.peer.permissions


def test_a_rejected_write_does_not_happen(workspace):
    agent = DaedalusAgent(
        engine=ScriptedEngine([call("write_file", path="nope.py", content="x = 1\n"),
                               "Done."]),
        execute=True, dry_run=False, gate=False)
    agent.peer = Recorder(permission_answer="reject_once")

    session_id, _ = drive(agent, workspace, "write a file")

    assert not (workspace / "nope.py").exists()
    assert not agent.sessions[session_id].talos.ws.staged_paths()


def test_a_cancelled_prompt_is_a_refusal(workspace):
    """A dismissed dialog is not consent."""
    agent = DaedalusAgent(
        engine=ScriptedEngine([call("write_file", path="nope.py", content="x = 1\n"),
                               "Done."]),
        execute=True, dry_run=False, gate=False)
    agent.peer = Recorder(permission_answer=None)

    drive(agent, workspace, "write a file")

    assert not (workspace / "nope.py").exists()


def test_always_allow_is_asked_once(workspace):
    agent = DaedalusAgent(
        engine=ScriptedEngine([
            call("write_file", path="one.py", content="one\n"), "Added one.",
            call("write_file", path="two.py", content="two\n"), "Added two.",
        ]),
        execute=True, gate=False)
    agent.peer = Recorder(permission_answer="allow_always")

    session_id, _ = drive(agent, workspace, "add one")
    agent.session_prompt({"sessionId": session_id,
                          "prompt": [{"type": "text", "text": "now add two"}]})

    assert len(agent.peer.permissions) == 1, "the second write must not re-ask"
    assert len(agent.sessions[session_id].talos.ws.staged_paths()) == 2


def test_always_allow_does_not_leak_between_sessions(workspace):
    agent = DaedalusAgent(
        engine=ScriptedEngine([
            call("write_file", path="one.py", content="one\n"), "Added one.",
            call("write_file", path="two.py", content="two\n"), "Added two.",
        ]),
        execute=True, gate=False)
    agent.peer = Recorder(permission_answer="allow_always")

    drive(agent, workspace, "add one")
    second = agent.session_new({"cwd": str(workspace), "mcpServers": []})["sessionId"]
    agent.session_prompt({"sessionId": second,
                          "prompt": [{"type": "text", "text": "add two"}]})

    assert len(agent.peer.permissions) == 2, "a new session grants nothing"


def test_no_client_means_no_write(workspace):
    """Fail closed: with nobody to ask, the answer is no."""
    agent = DaedalusAgent(
        engine=ScriptedEngine([call("write_file", path="nope.py", content="x = 1\n"),
                               "Done."]),
        execute=True, dry_run=False, gate=False)
    agent.peer = None

    agent.initialize({"protocolVersion": 1, "clientCapabilities": {},
                      "clientInfo": {"name": "t", "version": "1"}})
    sid = agent.session_new({"cwd": str(workspace), "mcpServers": []})["sessionId"]
    agent.session_prompt({"sessionId": sid,
                          "prompt": [{"type": "text", "text": "write a file"}]})

    assert not (workspace / "nope.py").exists()


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


def test_a_truncated_reply_is_reported_as_max_tokens_not_refusal(workspace):
    """`refusal` reads as "the agent declined" and sends you looking in the
    wrong place. The retrieval path has always mapped `finish_reason`; execute
    mode reported every unfinished run the same way until a live model produced
    a reply cut off at the token limit."""
    agent = DaedalusAgent(engine=ScriptedEngine(["", "", ""]),
                          execute=True, gate=False, max_steps=3, target_steps=2)
    agent.peer = Recorder()
    agent.engine.stop_reason = "length"

    _, result = drive(agent, workspace, "do something")

    assert result["stopReason"] == "max_tokens"


def test_an_unfinished_run_reports_running_out_of_road(workspace):
    """Not `refusal` -- the agent did not decline, it ran out of steps.

    Reporting a refusal sends the user looking for a policy problem that does
    not exist. Observed on a real run that produced correct, passing code and
    said `refusal` because a style tier objected.
    """
    agent = DaedalusAgent(engine=ScriptedEngine(["", "", ""]),
                          execute=True, gate=False, max_steps=3, target_steps=2)
    agent.peer = Recorder()

    _, result = drive(agent, workspace, "do something")

    assert result["stopReason"] == "max_turn_requests"


def test_a_do_nothing_turn_does_not_report_success(workspace):
    """The whole chain, end to end: empty reply -> not end_turn, nothing written."""
    agent = DaedalusAgent(engine=ScriptedEngine(["", "", ""]),
                          execute=True, dry_run=False, gate=False,
                          max_steps=3, target_steps=2)
    agent.peer = Recorder()

    session_id, result = drive(agent, workspace, "create widget.py")

    assert result["stopReason"] != "end_turn"
    assert not (workspace / "widget.py").exists()
    assert not agent.sessions[session_id].talos.ws.staged_paths()


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


def test_the_closing_status_does_not_repeat_what_was_already_streamed():
    """The panel showed the model's last paragraph twice.

    `Outcome.summary` carries "Last message: ..." so a scripted caller printing
    only the summary still sees it. Over ACP that text arrived chunk by chunk
    already, so echoing it in the status line reads as the agent restating
    itself.
    """
    from knossos.acp import _headline

    class Unfinished:
        summary = ("Stopped: step budget of 12 exhausted. Verification: failed "
                   "at ruff. Last message: I have created durations.py and it "
                   "is ready for testing.")

    headline = _headline(Unfinished())
    assert "step budget" in headline
    assert "failed at ruff" in headline
    assert "ready for testing" not in headline, "already streamed once"


def test_a_summary_without_a_last_message_is_untouched():
    from knossos.acp import _headline

    class Done:
        summary = "Completed and verified -- passed 3 tiers."

    assert _headline(Done()) == "Completed and verified -- passed 3 tiers."
