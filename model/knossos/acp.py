"""The ACP server -- Daedalus as an agent any compatible editor can spawn.

The Agent Client Protocol (Zed Industries, Aug 2025) is what Claude Code, Gemini
CLI and Codex use to appear inside Zed's agent panel; JetBrains adopted it across
their IDEs. The editor spawns this process, passes it the workspace, streams the
turn back into its own UI. So the harness gets a real front end without forking
anything -- and when the Lapce fork does happen, this same stdio seam is what it
will talk to, because the harness is Python and the editor is Rust either way.

One turn, end to end:

    session/new       index the workspace with Argus (cold scan)
    session/prompt    rescan (incremental) -> retrieve -> engine -> stream back
                      |
                      +-- session/update  tool_call        "searching the repo"
                      +-- session/update  tool_call_update  completed + locations
                      +-- session/update  agent_message_chunk (xN)
                      <-- { "stopReason": "end_turn" }

The retrieval is emitted as a real tool call rather than hidden, so the file
locations Argus chose show up as clickable references in the editor. That is the
same provenance `Retrieved.reason` carries, surfaced one layer up: when the agent
answers from the wrong file you can see that it did, and why.

Run it directly to check it starts:

    python -m knossos --engine retrieval

Then register it with Zed in `settings.json` under `agent_servers`.
"""
from __future__ import annotations

import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from .argus import Argus, Retrieved, render
from .ariadne import Ariadne
from .engine import (PROVIDERS, Engine, OpenAICompatEngine, RetrievalOnlyEngine,
                     Thought)
from .gate import RetrievalGate
from .mcp import McpClient, connect_all
from .lsp import for_workspace as lsp_for_workspace
from .metis import Metis, Plan, worth_planning
from .tools import ToolRegistry
from .jsonrpc import INVALID_PARAMS, METHOD_NOT_FOUND, Peer, RpcError, log
from .oracle import Oracle
from .talos import Event, Talos
from .workspace import Workspace

__all__ = ["DaedalusAgent", "Session", "PROTOCOL_VERSION", "AGENT_NAME"]

PROTOCOL_VERSION = 1
AGENT_NAME = "daedalus"
AGENT_VERSION = "0.1.0"

#: ACP renders tool calls by kind, so a read looks different from an edit.
_TOOL_KINDS = {
    "read_file": "read",
    "list_dir": "read",
    "search": "search",
    "write_file": "edit",
    "edit_file": "edit",
    "run_command": "execute",
}


def _tool_title(name: str, args: Dict[str, Any]) -> str:
    """A human phrase for the editor's activity list."""
    path = args.get("path")
    if name == "read_file" and path:
        return f"Reading {path}"
    if name == "write_file" and path:
        return f"Writing {path}"
    if name == "edit_file" and path:
        return f"Editing {path}"
    if name == "list_dir":
        return f"Listing {path or '.'}"
    if name == "search":
        where = f" in {path}" if path else ""
        return f"Searching for {args.get('pattern', '')}{where}"
    if name == "run_command":
        return f"Running {args.get('command', '')}"
    return name


#: Longest string kept in a replay-history entry. Matches the cap already
#: applied to `tool_call_update` content, so the two halves of a tool call are
#: retained on the same terms.
HISTORY_MAX_STRING = 2_000


def _bounded(value: Any, _depth: int = 0) -> Any:
    """A copy of `value` with long strings truncated, for the replay history.

    Recursive because the size lives in nested structures -- `rawInput` is a
    dict, and a tool result's content is a list of dicts. Depth-limited so a
    pathological payload cannot blow the stack while being stored.
    """
    if _depth > 6:
        return value
    if isinstance(value, str):
        if len(value) <= HISTORY_MAX_STRING:
            return value
        return (f"{value[:HISTORY_MAX_STRING]}\n\n[… {len(value) - HISTORY_MAX_STRING} "
                f"characters not kept for replay …]")
    if isinstance(value, dict):
        return {k: _bounded(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, list):
        return [_bounded(v, _depth + 1) for v in value]
    return value


@dataclass
class Session:
    id: str
    cwd: Path
    argus: Argus
    gate: Optional[RetrievalGate] = None
    cancel: threading.Event = field(default_factory=threading.Event)
    #: Built on first use in execute mode, then kept so the conversation and
    #: any staged edits survive between prompts.
    talos: Optional[Talos] = None
    #: Every `session/update` sent, in order, so `session/load` can replay the
    #: conversation. Stored as the update payloads themselves rather than as
    #: prose: replay must reproduce what the client originally rendered,
    #: including tool calls and their file locations, not a summary of it.
    history: List[Dict[str, Any]] = field(default_factory=list)
    #: Connected MCP servers, kept so they can be shut down with the session.
    mcp: List[McpClient] = field(default_factory=list)
    #: Tool names the user chose to allow for the rest of this session. Scoped
    #: to the session on purpose: a grant given in one workspace should not
    #: silently apply to the next one opened.
    always_allowed: Set[str] = field(default_factory=set)
    #: One of `MODES`. Chosen per session rather than per process: the CLI flag
    #: only says what to *start* in, and asking a question is not a reason to
    #: restart the agent.
    mode: str = "ask"
    #: First thing the user asked, for `session/list`. A session picker showing
    #: a column of identical ids is not a picker.
    title: Optional[str] = None
    #: ISO 8601, last time this session did anything.
    updated_at: str = field(default_factory=lambda: _now())
    #: Language server for exact cross-file references, started on first use.
    lsp: Any = None
    #: Standing instructions for this workspace, resolved once on first use.
    #: `None` means "not looked up yet"; `""` means "looked up, there are none"
    #: -- a distinction that matters because the planner and the executor both
    #: ask, and re-reading the file per role would log it three times a turn and
    #: let the two roles disagree if the file changed mid-run.
    constitution: Optional[str] = None
    #: So a workspace with no server installed is not probed on every turn.
    lsp_tried: bool = False

    def touch(self) -> None:
        self.updated_at = _now()


def supported(capabilities: Dict[str, Any], *path: str) -> bool:
    """Whether a capability is advertised, in either spelling the schema uses.

    ACP says "supported" two ways. Older capabilities are booleans -- `terminal:
    true`, `fs: {readTextFile: true}`. Newer ones are objects where the *empty*
    object means yes: `elicitation: {form: {}}`, and likewise `session`, `plan`
    and `nes`.

    `{}` is falsy in Python, so the obvious `if caps.get(name):` silently
    rejects exactly the clients that conform in the second style. That is not a
    hypothetical -- it was written that way here first, and the only reason it
    surfaced is that the conformance client advertises the real shape. It would
    have looked, from inside the agent, like a client that simply never asked.

    So the question is presence, not truth: a key that is there and is not
    explicitly `false` or `null` means supported.
    """
    node: Any = capabilities or {}
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return False
        node = node[key]
    return node is not False and node is not None


def _now() -> str:
    """UTC, ISO 8601, with the `Z` the spec's examples use."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z")


#: What the agent will do with a prompt. These are the two flags the CLI already
#: had -- execute, and write-vs-stage -- named and made switchable at runtime,
#: which is what turns "restart it with a different flag" into a dropdown.
MODES = [
    {"id": "ask", "name": "Ask",
     "description": "Answer questions about the repository. No tools run and "
                    "nothing is changed."},
    {"id": "preview", "name": "Preview edits",
     "description": "Carry out the task, staging every edit for review instead "
                    "of writing it."},
    {"id": "write", "name": "Write",
     "description": "Carry out the task and write changes to disk, asking "
                    "before each one."},
]

MODE_IDS = {m["id"] for m in MODES}


#: How long a callback into the editor may take. Bounded, unlike a permission
#: prompt: nobody is reading a dialog here, so a client that does not answer is
#: broken rather than slow, and falling back to disk beats stalling the turn.
FS_TIMEOUT = 15.0


class EditorFiles:
    """Reads and writes routed through the client's `fs/*` methods.

    Installed only when the client advertises the capability. Without it the
    agent reads through to disk -- which is not what the user is looking at when
    a buffer has unsaved changes, and which puts writes outside the editor's
    undo stack. Both failure modes are invisible from inside the agent, which is
    why this is wired from the capability rather than from a flag.

    Every method degrades to `None`/`False` instead of raising, so a client that
    advertises the capability and then fails still gets a working agent.
    """

    def __init__(self, agent: "DaedalusAgent", session: "Session",
                 can_read: bool, can_write: bool) -> None:
        self.agent = agent
        self.session = session
        self.can_read = can_read
        self.can_write = can_write

    def read_text_file(self, path: Path) -> Optional[str]:
        if not self.can_read or self.agent.peer is None:
            return None
        try:
            answer = self.agent.peer.request("fs/read_text_file", {
                "sessionId": self.session.id,
                "path": str(path),
            }, timeout=FS_TIMEOUT, abort=self.session.cancel)
        except Exception as exc:
            log(f"[acp] fs/read_text_file failed for {path}, using disk: {exc}")
            return None
        content = (answer or {}).get("content")
        return content if isinstance(content, str) else None

    def write_text_file(self, path: Path, content: str) -> bool:
        if not self.can_write or self.agent.peer is None:
            return False
        try:
            self.agent.peer.request("fs/write_text_file", {
                "sessionId": self.session.id,
                "path": str(path),
                "content": content,
            }, timeout=FS_TIMEOUT, abort=self.session.cancel)
        except Exception as exc:
            log(f"[acp] fs/write_text_file failed for {path}, using disk: {exc}")
            return False
        return True


class EditorTerminal:
    """Runs commands in the editor's terminal rather than a hidden subprocess.

    Installed only when the client advertises `terminal`. The difference is
    visibility: a build in a subprocess cannot be watched, scrolled or killed,
    and its first sign of life is the wall of text the agent pastes back after
    it finishes.

    `terminal/create` returns immediately, so the sequence is create ->
    wait_for_exit -> output -> release. The release is in a `finally` because a
    terminal leaked on every failed command is a resource the editor keeps
    open for the rest of the session.
    """

    def __init__(self, agent: "DaedalusAgent", session: "Session") -> None:
        self.agent = agent
        self.session = session

    def run(self, argv, timeout):
        if self.agent.peer is None or not argv:
            return None

        def call(method: str, params: Dict[str, Any], wait: float):
            return self.agent.peer.request(method, params, timeout=wait,
                                           abort=self.session.cancel)

        try:
            created = call("terminal/create", {
                "sessionId": self.session.id,
                "command": argv[0],
                "args": list(argv[1:]),
                "cwd": str(self.session.cwd),
            }, FS_TIMEOUT)
        except Exception as exc:
            log(f"[acp] terminal/create failed, using a subprocess: {exc}")
            return None

        terminal_id = (created or {}).get("terminalId")
        if not terminal_id:
            return None

        ref = {"sessionId": self.session.id, "terminalId": terminal_id}
        try:
            # The command's own timeout, not the RPC's: waiting for a test suite
            # is the point of this call.
            exit_info = call("terminal/wait_for_exit", ref, timeout + FS_TIMEOUT)
            result = call("terminal/output", ref, FS_TIMEOUT)
        except Exception as exc:
            log(f"[acp] terminal wait/output failed: {exc}")
            return None
        finally:
            try:
                call("terminal/release", ref, FS_TIMEOUT)
            except Exception as exc:
                log(f"[acp] terminal/release failed for {terminal_id}: {exc}")

        output = (result or {}).get("output") or ""
        if (result or {}).get("truncated"):
            output += "\n\n[the editor truncated this output]"
        code = (exit_info or {}).get("exitCode")
        signal = (exit_info or {}).get("signal")
        if signal:
            output += f"\n\n[terminated by signal {signal}]"
        return code, output


class EditorElicitation:
    """Asks the user a structured question through the client.

    Untimed, like a permission prompt and for the same reason: the whole point
    is that the agent waits for a person. Interruptible by cancel, so a turn the
    user gave up on does not pin a worker thread.

    Marked UNSTABLE in the schema, so a client that does not implement it simply
    never advertises the capability and `ask_user` reports that there is nobody
    to ask.
    """

    def __init__(self, agent: "DaedalusAgent", session: "Session") -> None:
        self.agent = agent
        self.session = session

    def ask(self, question: str, choices=None) -> Optional[str]:
        if self.agent.peer is None or self.session.cancel.is_set():
            return None

        field_schema: Dict[str, Any] = {"type": "string", "title": "Answer",
                                        "description": question}
        if choices:
            field_schema["enum"] = list(choices)

        try:
            answer = self.agent.peer.request("elicitation/create", {
                "sessionId": self.session.id,
                "message": question,
                "mode": "form",
                "requestedSchema": {
                    "type": "object",
                    "properties": {"answer": field_schema},
                    "required": ["answer"],
                },
            }, timeout=None, abort=self.session.cancel)
        except Exception as exc:
            log(f"[acp] elicitation/create failed: {exc}")
            return None

        # Anything but an explicit accept is a non-answer. Declining is a
        # decision the user is entitled to make, and inventing content for it
        # would defeat the point of asking.
        if (answer or {}).get("action") != "accept":
            return None
        content = (answer or {}).get("content") or {}
        value = content.get("answer")
        return None if value is None else str(value)


class DaedalusAgent:
    """ACP method handlers. Transport-free, so tests can drive it over a pipe."""

    def __init__(self, engine: Optional[Engine] = None, budget: int = 8000,
                 hops: int = 1, gate: bool = True, execute: bool = False,
                 dry_run: bool = True, max_steps: int = 20,
                 target_steps: int = 6, planning: bool = True,
                 delegation: bool = True,
                 constitution: str = "") -> None:
        self.engine: Engine = engine or RetrievalOnlyEngine()
        self.budget = budget
        self.hops = hops
        #: Whether to let the gate withhold context on general questions.
        self.gate_enabled = gate
        #: Answer questions (False) or carry out tasks with tools (True).
        #: Off by default: a retrieval agent cannot damage a workspace, and an
        #: executor can, so the capability is opted into rather than out of.
        self.execute = execute
        #: In execute mode, stage edits instead of writing them. Also default-on
        #: for the same reason.
        self.dry_run = dry_run
        self.max_steps = max_steps
        self.target_steps = target_steps
        #: Spend one engine turn planning before executing. Costs a turn and
        #: buys a artifact the user can veto before anything is written, so it
        #: is on by default and skipped for tasks too small to need it.
        self.planning = planning
        #: Offer the executor a `delegate` tool for self-contained subtasks.
        #: A flag rather than always-on for the same reason `planning` is one:
        #: it spends engine turns, and a caller measuring the loop needs to be
        #: able to turn it off.
        self.delegation = delegation
        self.constitution = constitution
        self.peer: Optional[Peer] = None
        self.sessions: Dict[str, Session] = {}
        self.client_capabilities: Dict[str, Any] = {}

    # --------------------------------------------------------------- dispatch

    def handle(self, method: str, params: Any, is_request: bool) -> Any:
        handlers = {
            "initialize": self.initialize,
            "authenticate": self.authenticate,
            "session/new": self.session_new,
            "session/load": self.session_load,
            "session/prompt": self.session_prompt,
            "session/cancel": self.session_cancel,
            "session/interject": self.session_interject,
            "session/set_mode": self.session_set_mode,
            "session/fork": self.session_fork,
            "session/list": self.session_list,
            "session/close": self.session_close,
            "session/delete": self.session_delete,
        }
        fn = handlers.get(method)
        if fn is None:
            if not is_request:
                return None                       # unknown notifications are ignorable
            raise RpcError(METHOD_NOT_FOUND, f"method not found: {method}")
        return fn(params or {})

    # ----------------------------------------------------------------- methods

    def initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self.client_capabilities = params.get("clientCapabilities") or {}
        client = params.get("clientInfo") or {}
        log(f"[acp] initialize from {client.get('name', '?')} "
            f"{client.get('version', '')} (protocol {params.get('protocolVersion')})")
        # Never claim a version above our own, whatever the client offers.
        version = min(int(params.get("protocolVersion", PROTOCOL_VERSION)), PROTOCOL_VERSION)
        return {
            "protocolVersion": version,
            "agentCapabilities": {
                "loadSession": True,
                # Named `unstable_` because the schema marks fork that way. A
                # client should not have to read our source to learn that a
                # capability may be withdrawn.
                "unstable_forkSession": True,
                # `{}` is how the schema spells "supported".
                "sessionCapabilities": {"list": {}, "close": {}, "delete": {}},
                "promptCapabilities": {
                    "image": False,
                    "audio": False,
                    "embeddedContext": True,
                },
            },
            "agentInfo": {
                "name": AGENT_NAME,
                "title": "Daedalus",
                "version": AGENT_VERSION,
            },
            "authMethods": [],
        }

    def authenticate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {}                                  # runs locally; nothing to authenticate

    def session_new(self, params: Dict[str, Any]) -> Dict[str, Any]:
        cwd_raw = params.get("cwd")
        if not cwd_raw:
            raise RpcError(INVALID_PARAMS, "session/new requires 'cwd'")
        cwd = Path(cwd_raw)
        if not cwd.is_absolute():
            raise RpcError(INVALID_PARAMS, f"cwd must be an absolute path: {cwd_raw}")
        if not cwd.is_dir():
            raise RpcError(INVALID_PARAMS, f"cwd is not a directory: {cwd_raw}")

        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        argus = Argus(cwd)
        gate = RetrievalGate(argus) if self.gate_enabled else None
        argus.load()                               # a warm index makes the first scan cheap
        report = argus.scan()
        log(f"[acp] session {session_id} on {cwd}: {report}")
        try:
            argus.save()
        except OSError as exc:                     # a read-only workspace is not fatal
            log(f"[acp] could not persist index: {exc}")

        # MCP servers the client declared. Previously accepted and discarded,
        # which meant a configured tool silently never appeared.
        mcp_clients, mcp_errors = connect_all(params.get("mcpServers") or [])
        for message in mcp_errors:
            log(f"[acp] mcp: {message}")
        if mcp_clients:
            total = sum(len(c.tools) for c in mcp_clients)
            log(f"[acp] {len(mcp_clients)} mcp server(s), {total} tool(s)")

        self.sessions[session_id] = Session(id=session_id, cwd=cwd, argus=argus,
                                            gate=gate, mcp=mcp_clients,
                                            mode=self.default_mode)
        return {"sessionId": session_id, "modes": self._mode_state(self.default_mode)}

    @property
    def default_mode(self) -> str:
        """The mode the CLI flags asked for."""
        if not self.execute:
            return "ask"
        return "preview" if self.dry_run else "write"

    @staticmethod
    def _mode_state(current: str) -> Dict[str, Any]:
        return {"currentModeId": current, "availableModes": MODES}

    def session_set_mode(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Switch what a prompt will do, without restarting the agent.

        Staged edits are deliberately *not* applied when moving to `write`.
        Switching a dropdown is not approval to write anything, and a mode
        change that silently committed pending edits would be the exact
        surprise the staging model exists to prevent.
        """
        session = self.sessions.get(params.get("sessionId", ""))
        if session is None:
            raise RpcError(INVALID_PARAMS, f"unknown session: {params.get('sessionId')}")

        mode = params.get("modeId")
        if mode not in MODE_IDS:
            raise RpcError(INVALID_PARAMS,
                           f"unknown mode {mode!r}; expected one of "
                           f"{', '.join(sorted(MODE_IDS))}")

        session.mode = mode
        if session.talos is not None:
            staged = len(session.talos.ws.staged_paths())
            session.talos.ws.dry_run = (mode == "preview")
            if staged and mode == "write":
                log(f"[acp] {session.id}: {staged} edit(s) stay staged across the "
                    f"switch to write; apply them explicitly")
        log(f"[acp] {session.id}: mode -> {mode}")

        # Tell the client, so a mode changed by any other route stays in sync.
        self._update(session, {"sessionUpdate": "current_mode_update",
                               "currentModeId": mode})
        return {}

    def mcp_tools(self, session: Session) -> List[Any]:
        """Every remote tool this session can reach, flattened."""
        return [tool for client in session.mcp for tool in client.tools]

    def session_load(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Resume a session, replaying its conversation to the client.

        The spec is specific about the order: the agent replays the history as
        `session/update` notifications *before* answering this request, so a
        client that renders updates as they arrive rebuilds the transcript and
        only then sees the call succeed.

        Replay is not re-execution. The updates are the ones already sent, so
        nothing runs again -- reopening a panel must not re-edit files.
        """
        session_id = params.get("sessionId") or ""
        session = self.sessions.get(session_id)
        if session is None:
            raise RpcError(INVALID_PARAMS, f"unknown session: {session_id!r}")

        log(f"[acp] replaying {len(session.history)} update(s) for {session_id}")
        for update in session.history:
            # record=False: replaying must not append the history to itself,
            # which would double it on every load.
            self._update(session, update, record=False)
        return {}

    def session_fork(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Branch a session, so one line of work can be tried without losing the other.

        Marked UNSTABLE in the schema, so it is advertised under
        `unstable_forkSession` rather than as a settled capability.

        The fork copies the conversation and the *mode*, and deliberately does
        not copy staged edits: two sessions holding uncommitted proposals for
        the same file would race to apply them, and the resulting file would
        depend on click order. The fork starts from what is on disk.
        """
        source = self.sessions.get(params.get("sessionId", ""))
        if source is None:
            raise RpcError(INVALID_PARAMS, f"unknown session: {params.get('sessionId')}")

        cwd_raw = params.get("cwd") or str(source.cwd)
        cwd = Path(cwd_raw)
        if not cwd.is_absolute():
            raise RpcError(INVALID_PARAMS, f"cwd must be an absolute path: {cwd_raw}")
        if not cwd.is_dir():
            raise RpcError(INVALID_PARAMS, f"cwd is not a directory: {cwd_raw}")

        session_id = f"sess_{uuid.uuid4().hex[:12]}"
        # A fresh index for a fresh root; reusing the source's Argus would make
        # two sessions share a mutable index across different trees.
        argus = Argus(cwd)
        argus.load()
        argus.scan()
        mcp_clients, mcp_errors = connect_all(params.get("mcpServers") or [])
        for message in mcp_errors:
            log(f"[acp] mcp: {message}")

        forked = Session(id=session_id, cwd=cwd, argus=argus,
                         gate=RetrievalGate(argus) if self.gate_enabled else None,
                         mcp=mcp_clients, mode=source.mode,
                         history=list(source.history))
        if source.talos is not None:
            forked.talos = Talos(
                self.engine,
                Workspace(cwd, dry_run=forked.mode == "preview",
                          editor=self._editor_files(forked)),
                ariadne=Ariadne(max_steps=self.max_steps,
                                target_steps=self.target_steps),
                verifier=Oracle(cwd),
                ask_permission=lambda call: self._ask_permission(forked, call))
            # The conversation carries over; the permission grants do not. A
            # branch is a new context, and "always allow" was answered about
            # the other one.
            forked.talos.transcript = list(source.talos.transcript)
            forked.talos.task = source.talos.task

        self.sessions[session_id] = forked
        log(f"[acp] forked {source.id} -> {session_id} on {cwd}")
        return {"sessionId": session_id, "modes": self._mode_state(forked.mode)}

    def session_list(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Every live session, newest activity first.

        No pagination: sessions live in this process's memory and a client that
        has opened thousands of them has a different problem. `nextCursor` is
        omitted, which the spec defines as "there are no more results" -- so a
        paginating client terminates correctly rather than looping.
        """
        wanted = params.get("cwd")
        if wanted is not None:
            root = Path(wanted)
            if not root.is_absolute():
                raise RpcError(INVALID_PARAMS, f"cwd must be absolute: {wanted}")

        entries = []
        for session in self.sessions.values():
            if wanted is not None and session.cwd != Path(wanted):
                continue
            info: Dict[str, Any] = {
                "sessionId": session.id,
                "cwd": str(session.cwd),
                "updatedAt": session.updated_at,
            }
            if session.title:
                info["title"] = session.title
            entries.append(info)

        entries.sort(key=lambda e: e["updatedAt"], reverse=True)
        return {"sessions": entries}

    def session_close(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Cancel any work and release this session's subprocesses.

        The spec requires both, in that order. Until this existed, nothing ever
        stopped a session's MCP servers or its language server -- the `mcp`
        field has said "kept so they can be shut down with the session" since it
        was written, and nothing shut them down. A long-lived editor opening a
        session per workspace leaked a process tree per workspace.
        """
        session = self.sessions.get(params.get("sessionId", ""))
        if session is None:
            raise RpcError(INVALID_PARAMS, f"unknown session: {params.get('sessionId')}")
        session.cancel.set()
        self._release(session)
        return {}

    def session_delete(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Close the session and drop it from `session/list`."""
        session_id = params.get("sessionId", "")
        session = self.sessions.get(session_id)
        if session is None:
            raise RpcError(INVALID_PARAMS, f"unknown session: {session_id!r}")
        session.cancel.set()
        self._release(session)
        del self.sessions[session_id]
        return {}

    @staticmethod
    def _release(session: Session) -> None:
        """Stop everything this session started. Never raises.

        A failure to shut one thing down must not prevent the rest: this runs
        while a client is closing a panel, and a half-released session leaks
        exactly the processes it was called to reclaim.

        The blanket `except` is also why this was broken for so long: it called
        `client.close()`, `McpClient` has no such method, and the resulting
        `AttributeError` was caught and logged as an ordinary shutdown failure.
        Every server survived the session that started it, holding an
        interpreter and a copy of the environment, and `session.mcp = []` then
        dropped the only handle -- so nothing could reach them again even at
        process exit. The swallow stays, because a subprocess declining to die
        must not stop the rest of the release; what guards the method name now
        is `test_closing_a_session_really_stops_its_mcp_servers`, which asserts
        against the process rather than against the call.
        """
        for client in session.mcp:
            try:
                client.close()
            except Exception as exc:
                log(f"[acp] mcp shutdown failed for {session.id}: {exc}")
        session.mcp = []
        if session.lsp is not None:
            try:
                session.lsp.stop()
            except Exception as exc:
                log(f"[acp] lsp shutdown failed for {session.id}: {exc}")
            session.lsp = None
        if session.talos is not None:
            session.talos.ws.symbols = None

    def session_cancel(self, params: Dict[str, Any]) -> None:
        session = self.sessions.get(params.get("sessionId", ""))
        if session:
            session.cancel.set()

    def session_interject(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Say something to a prompt that is already running.

        Handled on the reading thread while `session/prompt` is still executing,
        the same way `session/cancel` is -- a message that had to queue behind
        the very turn it is meant to change would arrive too late to be worth
        sending.

        The difference from cancel is the point: this does not stop the run. The
        text is delivered at the next step boundary, so the agent keeps
        everything it has worked out so far and adjusts. See
        `knossos.interject`.
        """
        session = self.sessions.get(params.get("sessionId", ""))
        if session is None:
            raise RpcError(INVALID_PARAMS,
                           f"unknown session: {params.get('sessionId')}")

        text = _prompt_text(params.get("prompt") or [])
        if not text.strip():
            text = str(params.get("text") or "")

        # No Talos yet means no run to interrupt. Accepting it silently would
        # leave the user believing the agent had been told.
        if session.talos is None:
            return {"accepted": False, "reason": "nothing is running"}

        accepted = session.talos.interjections.push(text)
        return {
            "accepted": accepted,
            **({} if accepted else
               {"reason": "empty, or too many are already queued"}),
        }

    def session_prompt(self, params: Dict[str, Any]) -> Dict[str, Any]:
        session = self.sessions.get(params.get("sessionId", ""))
        if session is None:
            raise RpcError(INVALID_PARAMS, f"unknown session: {params.get('sessionId')}")
        session.cancel.clear()

        prompt = _prompt_text(params.get("prompt") or [])
        if not prompt.strip():
            raise RpcError(INVALID_PARAMS, "prompt contained no text")

        session.touch()
        if session.title is None:
            # The first question is what the session is about; later ones are
            # follow-ups, and renaming a session mid-conversation loses the
            # user's place in a picker.
            session.title = prompt.strip().splitlines()[0][:80]

        hits = self._run_retrieval(session, prompt)
        if session.cancel.is_set():
            return {"stopReason": "cancelled"}

        context = render(hits)

        if session.mode != "ask":
            return self._run_execution(session, prompt, context)

        message_id = f"msg_{uuid.uuid4().hex[:8]}"
        for chunk in self.engine.generate(prompt, context, session.cancel.is_set):
            if session.cancel.is_set():
                return {"stopReason": "cancelled"}
            if not chunk:
                continue
            # A reasoning model's scratchpad is not its answer. ACP has a channel
            # for it, which editors render collapsed.
            thinking = isinstance(chunk, Thought)
            self._update(session, {
                "sessionUpdate": "agent_thought_chunk" if thinking
                                 else "agent_message_chunk",
                "messageId": message_id,
                "content": {"type": "text", "text": str(chunk)},
            })

        if session.cancel.is_set():
            return {"stopReason": "cancelled"}
        return {"stopReason": self._stop_reason()}

    #: Provider `finish_reason` -> ACP `stopReason`. Reporting a reply truncated
    #: at the token limit as "end_turn" would tell the editor the answer was
    #: complete when it was cut off mid-sentence.
    STOP_REASONS = {
        "stop": "end_turn",
        "length": "max_tokens",
        "max_tokens": "max_tokens",
        "content_filter": "refusal",
        "refusal": "refusal",
        "cancelled": "cancelled",
        # The model answered with native OpenAI tool calls. Knossos reads tool
        # calls out of the message content instead, so the reply arrives empty
        # and the turn cannot have finished -- `refusal` at least does not claim
        # it did. The engine logs the actionable detail.
        "tool_calls": "refusal",
    }

    def _stop_reason(self) -> str:
        raw = getattr(self.engine, "stop_reason", None)
        if raw is None:
            return "end_turn"                  # engines that do not report one
        mapped = self.STOP_REASONS.get(str(raw))
        if mapped is None:
            log(f"[acp] unrecognised finish_reason {raw!r}; reporting end_turn")
            return "end_turn"
        return mapped

    # ----------------------------------------------------------------- helpers

    def _update(self, session: Session, update: Dict[str, Any],
                record: bool = True) -> None:
        # Recorded before sending, so a replay contains what a live client saw
        # even if the connection drops mid-turn.
        #
        # The *live* client gets the update whole and the history gets a bounded
        # copy. That split matters: `rawInput` carries a `write_file` call's
        # entire `content` argument, so writing a 100 KB file put 100 KB into a
        # list that is never evicted -- while the `tool_call_update` sent
        # immediately after it has always capped its content at 2000 characters.
        # Truncating what the editor receives would degrade the tool-call view
        # for no reason; truncating what is kept forever costs a replay some
        # fidelity and bounds the session.
        if record:
            session.history.append(_bounded(update))
        if self.peer is not None:
            self.peer.notify("session/update",
                             {"sessionId": session.id, "update": update})

    # ---------------------------------------------------------------- execute

    def _talos_for(self, session: Session) -> Talos:
        """Build the executor once per session, then reuse it.

        Reuse is what makes a second prompt a continuation rather than a restart
        -- the transcript and any staged edits live on the Talos instance.
        """
        if session.talos is None:
            workspace = Workspace(session.cwd, dry_run=session.mode == "preview",
                                  editor=self._editor_files(session),
                                  terminal=self._editor_terminal(session),
                                  elicit=self._editor_elicitation(session))
            oracle = Oracle(session.cwd)
            workspace.symbols = self._symbols(session)
            # Remote tools, registered alongside the local ones. `mcp_tools` had
            # no call site at all, so servers were started, handshaken, their
            # tools listed and logged -- and then never reachable by the agent,
            # which is the opposite of what `mcp.py`'s own docstring claims.
            # Local tools win a name collision: a remote server must not be able
            # to redefine `write_file` and take the jail with it.
            registry = ToolRegistry.default()
            remote = self.mcp_tools(session)
            if remote:
                registry = ToolRegistry.combined(remote, registry)
                log(f"[acp] {session.id}: {len(remote)} remote tool(s) registered")
            session.talos = Talos(
                self.engine, workspace,
                tools=registry,
                ariadne=Ariadne(max_steps=self.max_steps, target_steps=self.target_steps),
                verifier=oracle,
                constitution=self._constitution(session),
                # Between plan steps: tier 0 only. Running the full ladder
                # after every step would cost more than stepwise execution saves.
                interim=oracle.quick,
                # Bound to this session so "always allow" is remembered per
                # session rather than per process.
                ask_permission=lambda call: self._ask_permission(session, call),
                # Without this the plan Metis produced before anything had been
                # read is binding for the whole run.
                replan=lambda task, learned, done, remaining: self._replan(
                    session, task, learned, done, remaining),
                # Without this the `delegate` tool is built, tested, and absent
                # from every shipped registry -- which is what it was until an
                # architecture pass caught it.
                delegation=self.delegation)
        return session.talos

    def _replan(self, session: Session, task: str, learned: str,
                done: Sequence[str], remaining: Sequence[str]) -> List[str]:
        """Ask Metis for a new tail after a plan step failed.

        A fresh `Metis` rather than a retained one, because it "holds no state
        between calls" (`metis.py`) -- so re-planning is an ordinary plan call
        whose prompt happens to describe a run already in progress.

        Returns `[]` on every unhappy path, which Talos reads as "keep the
        current plan". Planning is an optimisation; a failure to re-plan must
        never be a failure of the run.
        """
        if not self.planning or session.cancel.is_set():
            return []

        brief = (
            f"{task}\n\n"
            f"A plan for this is already part-executed. These steps have been "
            f"attempted:\n"
            + "\n".join(f"- {step}" for step in done)
            + f"\n\nThe last of them did not verify. What it produced:\n{learned}\n\n"
            f"The plan from here was:\n"
            + "\n".join(f"- {step}" for step in remaining)
            + "\n\nRe-plan only the remaining work, in light of what that step "
              "found. Keep what still applies.")
        try:
            revised = Metis(self.engine, tools=ToolRegistry.default(),
                            constitution=self._constitution(session)).plan(
                                brief, "", cancelled=session.cancel.is_set)
        except Exception as exc:                      # noqa: BLE001 - reported
            log(f"[acp] {session.id}: re-planning failed: {exc}")
            return []

        if revised.degenerate or not revised.steps:
            return []
        log(f"[acp] {session.id}: plan revised to {len(revised)} step(s)")
        return list(revised.steps)

    def _constitution(self, session: Session) -> str:
        """Standing instructions for this session, or "".

        **One resolution for every role.** An explicit `constitution=` passed to
        the constructor wins -- a caller who supplied one meant it -- and
        otherwise this reads `constitution.md` from the workspace root, which is
        what the Rust harness has always done (`themis/mod.rs`).

        Both halves of that were previously broken, in opposite directions. The
        `DaedalusAgent.constitution` field was never filled (no CLI flag sets it,
        `main()` never passed one), and it was the value handed to `Metis` --
        so the planner ran with an empty constitution. The workspace file was
        read only for `Talos`, so a caller who *did* pass `constitution=`
        programmatically had it silently ignored by the executor. Resolving both
        sources in one place is what makes the planner and the executor run under
        the same standing instructions, which is the only version of this feature
        that means anything.

        Note what this deliberately does *not* do: it does not fall back to a
        compiled-in default the way Rust does. Rust's default is a constitution
        nobody chose; an absent file here means the user has no standing
        instructions, and inventing some would be worse than having none.

        The file lives inside the workspace, which is untrusted input, so this
        is only as trustworthy as the repository. That is a real limitation and
        it is the same one Rust has -- see ARCHITECTURE.md §4.3.
        """
        if session.constitution is not None:
            return session.constitution
        if self.constitution:
            session.constitution = self.constitution
            return session.constitution

        path = session.cwd / "constitution.md"
        try:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            text = ""
        if text:
            log(f"[acp] {session.id}: constitution loaded from {path}")
        session.constitution = text
        return text

    def _editor_files(self, session: Session) -> Optional[EditorFiles]:
        """An `fs/*` delegate, when the client said it has one.

        A client that advertises neither capability gets `None`, and the
        workspace goes to disk exactly as before -- so a plain CLI run is
        unaffected by any of this.
        """
        can_read = supported(self.client_capabilities, "fs", "readTextFile")
        can_write = supported(self.client_capabilities, "fs", "writeTextFile")
        if not (can_read or can_write):
            return None
        log(f"[acp] routing file access through the editor "
            f"(read={can_read}, write={can_write})")
        return EditorFiles(self, session, can_read, can_write)

    def _editor_terminal(self, session: Session) -> Optional[EditorTerminal]:
        """A `terminal/*` delegate, when the client said it has one."""
        if not supported(self.client_capabilities, "terminal"):
            return None
        log("[acp] running commands in the editor's terminal")
        return EditorTerminal(self, session)

    def _plan(self, session: Session, prompt: str, context: str) -> Plan:
        """Plan before acting, when the task is big enough to be worth a turn.

        Emitted to the client as an ACP `plan` update, so it renders as a
        checklist the user can read *before* anything is written -- which is the
        entire reason planning is a separate artifact from execution.

        A degenerate plan (the task restated as one step) is not shown: it tells
        the user nothing they did not just type, and a checklist of one item
        that repeats their own request reads like the agent misunderstood.
        """
        if not self.planning or not worth_planning(prompt):
            return Plan()

        # `self._constitution(session)`, not `self.constitution`: the raw field
        # is never populated in a shipped run, so the planner used to plan
        # without the standing instructions the executor was held to.
        metis = Metis(self.engine, tools=ToolRegistry.default(),
                      constitution=self._constitution(session))
        plan = metis.plan(prompt, context, cancelled=session.cancel.is_set)
        if plan.degenerate or not plan.steps:
            return Plan()

        log(f"[acp] {session.id}: planned {len(plan)} step(s)")
        self._emit_plan(session, plan, "pending")
        return plan

    def _emit_plan(self, session: Session, plan: Plan,
                   status: Optional[str] = None, at: Optional[int] = None) -> None:
        """Send the checklist. Either one status for all, or a position in it.

        `at` is a 1-based step index: everything before it is completed, it is
        in progress, everything after is pending.
        """
        def entry(index: int, step: str) -> Dict[str, Any]:
            if at is None:
                state = status or "pending"
            elif index < at:
                state = "completed"
            elif index == at:
                state = "in_progress"
            else:
                state = "pending"
            return {"content": step, "priority": "medium", "status": state}

        self._update(session, {
            "sessionUpdate": "plan",
            "entries": [entry(i + 1, s) for i, s in enumerate(plan.steps)],
        })

    def _symbols(self, session: Session):
        """A language server for this workspace, started once and reused.

        Started lazily, on the first execute-mode turn, because a retrieval
        session never needs one and starting rust-analyzer to answer a question
        is a poor trade. `None` when none is installed -- `rename_symbol` then
        refuses rather than guessing.
        """
        if session.lsp is None and not session.lsp_tried:
            session.lsp_tried = True
            session.lsp = lsp_for_workspace(session.cwd)
        return session.lsp

    def _editor_elicitation(self, session: Session) -> Optional[EditorElicitation]:
        """An `elicitation/*` delegate, when the client said it has one.

        `form` specifically, because that is the mode this agent sends: a client
        that only does `url` cannot answer a question.
        """
        if not supported(self.client_capabilities, "elicitation", "form"):
            return None
        log("[acp] the agent can ask the user structured questions")
        return EditorElicitation(self, session)

    def _run_execution(self, session: Session, prompt: str, context: str) -> Dict[str, Any]:
        """Carry out a task, narrating each step to the editor.

        Tool calls become real ACP tool calls rather than prose, so the editor
        renders them as actions with status -- and a refused one shows as failed
        rather than being buried in the transcript.
        """
        talos = self._talos_for(session)
        message_id = f"msg_{uuid.uuid4().hex[:8]}"

        def on_event(event: Event) -> None:
            if event.kind == "text" and event.text.strip():
                self._update(session, {
                    "sessionUpdate": "agent_message_chunk",
                    "messageId": message_id,
                    "content": {"type": "text", "text": event.text},
                })
            elif event.kind == "plan_step" and plan.steps:
                # The executor reached step N, so N is in progress and
                # everything before it finished. Derived from where execution
                # actually is rather than asserted, which is why this is honest
                # where an all-at-once "completed" would not be.
                self._emit_plan(session, plan, at=event.step)
            elif event.kind == "thought" and event.text.strip():
                # Same channel the retrieval path uses. Editors render it
                # collapsed, so it explains a long step without burying the
                # answer in scratchpad.
                self._update(session, {
                    "sessionUpdate": "agent_thought_chunk",
                    "messageId": message_id,
                    "content": {"type": "text", "text": event.text},
                })
            elif event.kind == "tool" and event.call and event.result:
                call_id = f"call_{uuid.uuid4().hex[:8]}"
                self._update(session, {
                    "sessionUpdate": "tool_call",
                    "toolCallId": call_id,
                    "title": _tool_title(event.call.name, event.call.args),
                    "kind": _TOOL_KINDS.get(event.call.name, "other"),
                    "status": "in_progress",
                    "rawInput": event.call.args,
                })
                self._update(session, {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": call_id,
                    "status": "failed" if event.result.is_error else "completed",
                    "content": [{"type": "content", "content": {
                        "type": "text", "text": event.result.content[:2000]}}],
                })
            elif event.kind == "verdict" and event.verdict is not None:
                verdict = event.verdict
                call_id = f"call_{uuid.uuid4().hex[:8]}"
                self._update(session, {
                    "sessionUpdate": "tool_call",
                    "toolCallId": call_id,
                    "title": "Verifying",
                    "kind": "think",
                    "status": "in_progress",
                })
                self._update(session, {
                    "sessionUpdate": "tool_call_update",
                    "toolCallId": call_id,
                    "status": "completed" if verdict.passed else "failed",
                    "title": verdict.summary,
                    "content": [{"type": "content", "content": {
                        "type": "text",
                        "text": verdict.detail or verdict.summary}}],
                })

        # A first prompt starts a task; a later one continues the same session.
        started = bool(talos.transcript)
        plan = Plan()
        if not started:
            plan = self._plan(session, prompt, context)

        outcome = (talos.resume(prompt, context, session.cancel.is_set, on_event)
                   if started else
                   talos.run(prompt, context, plan=plan.steps,
                             cancelled=session.cancel.is_set, on_event=on_event))

        if plan.steps:
            # The executor does not report which step it is on, so claiming
            # per-step progress would be invention. The honest signal is the
            # whole plan's fate: completed if the run verified, still pending
            # if it did not.
            self._emit_plan(session, plan,
                            "completed" if outcome.succeeded else "pending")

        self._update(session, {
            "sessionUpdate": "agent_message_chunk",
            "messageId": message_id,
            "content": {"type": "text", "text": f"\n\n{outcome.summary}"},
        })

        if session.cancel.is_set():
            return {"stopReason": "cancelled"}
        # Only a verified run reports end_turn. Anything else says so, rather
        # than letting the editor render an unfinished task as complete.
        if outcome.succeeded:
            return {"stopReason": "end_turn"}
        # Why it did not finish matters. `refusal` reads as "the agent declined",
        # which is wrong and unactionable when the real cause was a reply cut off
        # at the token limit -- the case the retrieval path has always reported
        # correctly and this one did not.
        engine_reason = self._stop_reason()
        return {"stopReason": "refusal" if engine_reason == "end_turn" else engine_reason}

    def _run_retrieval(self, session: Session, prompt: str) -> List[Retrieved]:
        """Retrieve, and narrate it to the editor as a visible tool call."""
        call_id = f"call_{uuid.uuid4().hex[:8]}"
        self._update(session, {
            "sessionUpdate": "tool_call",
            "toolCallId": call_id,
            "title": "Searching the repository",
            "kind": "search",
            "status": "in_progress",
            "rawInput": {"query": prompt, "budget": self.budget, "hops": self.hops},
        })

        try:
            session.argus.scan()                   # incremental; picks up edits since last turn
            hits = session.argus.retrieve(prompt, budget=self.budget, hops=self.hops)
            # Retrieval runs regardless -- it is local and its file locations are
            # useful even when the excerpts are withheld. The gate decides only
            # whether the text reaches the model, because on a small model
            # irrelevant context measurably degrades a general answer.
            if session.gate is not None:
                decision = session.gate.decide(prompt, hits)
                if not decision.inject:
                    log(f"[acp] gate: {decision}")
                    self._update(session, {
                        "sessionUpdate": "tool_call_update",
                        "toolCallId": call_id,
                        "status": "completed",
                        "title": "Skipped repository context â€” general question",
                        "content": [{"type": "content", "content": {
                            "type": "text",
                            "text": f"Context withheld ({decision.confidence:.2f}): "
                                    f"{'; '.join(decision.reasons)}"}}],
                        "rawOutput": {"gated": True,
                                      "confidence": decision.confidence},
                    })
                    return []
        except Exception as exc:
            log(f"[acp] retrieval failed: {exc}")
            self._update(session, {
                "sessionUpdate": "tool_call_update",
                "toolCallId": call_id,
                "status": "failed",
                "content": [{"type": "content",
                             "content": {"type": "text", "text": f"retrieval failed: {exc}"}}],
            })
            return []

        summary = (f"{len(hits)} excerpt(s) from "
                   f"{len({h.file for h in hits})} file(s)") if hits else "no matches"
        self._update(session, {
            "sessionUpdate": "tool_call_update",
            "toolCallId": call_id,
            "status": "completed",
            "title": f"Searched the repository â€” {summary}",
            # Absolute paths: the spec requires them, and the editor needs them to
            # turn these into clickable references.
            "locations": [{"path": str(session.cwd / h.file), "line": h.start_line}
                          for h in hits],
            "content": [{"type": "content", "content": {
                "type": "text",
                "text": "\n".join(f"{h.ref}-{h.end_line}"
                                  f"{'  ' + h.symbol if h.symbol else ''}"
                                  f"  [{h.reason}]" for h in hits) or "no matches"}}],
            "rawOutput": {"hits": len(hits),
                          "chars": sum(len(h.text) for h in hits)},
        })
        return hits

    # ------------------------------------------------------------ permissions

    #: Offered for every consequential call. The ids are ours, and the client
    #: echoes back whichever the user picked.
    PERMISSION_OPTIONS = [
        {"optionId": "allow_once", "name": "Allow once", "kind": "allow_once"},
        {"optionId": "allow_always", "name": "Always allow this tool",
         "kind": "allow_always"},
        {"optionId": "reject_once", "name": "Reject", "kind": "reject_once"},
    ]

    def _ask_permission(self, session: "Session", call: Any) -> bool:
        """Ask the client before a consequential tool call.

        Blocks on the client's answer, which is the point: nothing happens
        until a decision is made. The wait is untimed -- a user may take as
        long as they like to read a diff -- but it is interruptible, so a
        cancelled turn or a departed editor releases it instead of wedging the
        worker thread forever.

        Fails closed. No connected client, a failed request, a cancelled turn,
        or an answer we cannot parse all mean no. An editor that writes files
        because a dialog failed to appear is the worst outcome available.
        """
        name = getattr(call, "name", "?")
        if name in session.always_allowed:
            return True
        if self.peer is None:
            log(f"[acp] no client to ask about {name}; refusing")
            return False
        if session.cancel.is_set():
            return False

        try:
            answer = self.peer.request("session/request_permission", {
                "sessionId": session.id,
                "toolCall": {
                    "toolCallId": f"perm_{uuid.uuid4().hex[:8]}",
                    "title": _tool_title(name, getattr(call, "args", {}) or {}),
                    "kind": _TOOL_KINDS.get(name, "other"),
                },
                "options": self.PERMISSION_OPTIONS,
            }, timeout=None, abort=session.cancel)
        except Exception as exc:
            log(f"[acp] permission request failed, refusing {name}: {exc}")
            return False

        outcome = (answer or {}).get("outcome") or {}
        if outcome.get("outcome") != "selected":
            return False                       # cancelled, or something unknown

        chosen = outcome.get("optionId")
        if chosen == "allow_always":
            session.always_allowed.add(name)
            return True
        return chosen == "allow_once"

    # -------------------------------------------------------------------- run

    #: Handled on the reader thread. `session_cancel` only sets an event, so it
    #: is safe there -- and it has to be, or a cancel could not land mid-turn.
    FAST_PATH = frozenset({"session/cancel"})

    def serve(self, rx=None, tx=None) -> Peer:
        peer = Peer(self.handle, rx=rx, tx=tx,
                    fast_path=lambda method: method in self.FAST_PATH)
        self.peer = peer
        return peer


def _prompt_text(blocks: Any) -> str:
    """Flatten ACP content blocks into plain text.

    Only the parts a structural retriever can use: text, and the paths carried by
    resource links (a mentioned filename is a strong retrieval signal on its own).
    """
    parts: List[str] = []
    for block in blocks if isinstance(blocks, list) else []:
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text" and block.get("text"):
            parts.append(str(block["text"]))
        elif kind == "resource_link" and block.get("uri"):
            parts.append(_basename(str(block["uri"])))
        elif kind == "resource":
            resource = block.get("resource") or {}
            if resource.get("text"):
                parts.append(str(resource["text"]))
            elif resource.get("uri"):
                parts.append(_basename(str(resource["uri"])))
    return "\n".join(parts).strip()


def _basename(uri: str) -> str:
    return uri.rsplit("/", 1)[-1] if "/" in uri else uri


def _parser():
    """The CLI surface, separate from `main` so it can be tested.

    `main` ends in `serve_forever()`, so nothing that only wants to know what a
    flag does can afford to call it. Every flag added here without a test is a
    flag whose wiring is assumed rather than known.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m knossos",
        description="Daedalus as an ACP agent. Speaks JSON-RPC over stdio; "
                    "spawn it from Zed, JetBrains, or any ACP client.")
    parser.add_argument("--engine", choices=("retrieval", "api", "transformers"),
                        default="retrieval", help="what fills the engine slot")
    parser.add_argument("--provider", default="groq",
                        help=f"api engine only: {', '.join(sorted(PROVIDERS))}")
    parser.add_argument("--model", default=None,
                        help="model id; defaults to the provider's preset")
    parser.add_argument("--base-url", default=None,
                        help="override the provider's endpoint")
    parser.add_argument("--adapter", default=None,
                        help="transformers engine only: PEFT/LoRA directory")
    parser.add_argument("--budget", type=int, default=8000,
                        help="retrieval budget in characters")
    parser.add_argument("--hops", type=int, default=1,
                        help="how far to expand along import edges")
    parser.add_argument("--no-gate", action="store_true",
                        help="always inject context, even on general questions")
    parser.add_argument("--list-models", action="store_true",
                        help="ask the provider what it serves, then exit")
    parser.add_argument("--execute", action="store_true",
                        help="carry out tasks with tools instead of only answering "
                             "questions. Edits are staged for review unless --write")
    parser.add_argument("--write", action="store_true",
                        help="with --execute, write changes straight to disk instead "
                             "of staging them")
    parser.add_argument("--max-steps", type=int, default=20,
                        help="execute mode: hard ceiling on engine turns")
    parser.add_argument("--target-steps", type=int, default=6,
                        help="execute mode: where budget pressure begins")
    # Both default on, and both spend engine turns. Anyone measuring the loop
    # itself needs to be able to subtract them; until these flags existed the
    # off-switch was reachable only by constructing `DaedalusAgent` in Python,
    # which is not available to someone running the binary.
    parser.add_argument("--no-planning", action="store_true",
                        help="execute mode: skip the planning turn and execute "
                             "the prompt directly")
    parser.add_argument("--no-delegation", action="store_true",
                        help="execute mode: withhold the `delegate` tool, so the "
                             "executor cannot spawn a subagent")
    return parser


def _engine_from(args) -> Engine:
    """Whatever fills the engine slot for this invocation."""
    if args.engine == "api":
        return OpenAICompatEngine(
            provider=args.provider, model=args.model, base_url=args.base_url)
    if args.engine == "transformers":
        from .engine import TransformersEngine
        return TransformersEngine(
            model_id=args.model or "Qwen/Qwen2.5-Coder-7B-Instruct",
            adapter=args.adapter)
    return RetrievalOnlyEngine()


def build_agent(args) -> "DaedalusAgent":
    """Turn parsed arguments into an agent. Separate from `main` for the same
    reason `_parser` is: this is where a flag stops being a string and starts
    being behaviour, and it is the half that was never covered."""
    return DaedalusAgent(engine=_engine_from(args),
                         budget=args.budget, hops=args.hops,
                         gate=not args.no_gate, execute=args.execute,
                         dry_run=not args.write, max_steps=args.max_steps,
                         target_steps=args.target_steps,
                         planning=not args.no_planning,
                         delegation=not args.no_delegation)


def main(argv: Optional[List[str]] = None) -> int:
    from .jsonrpc import _configure_stdio

    parser = _parser()
    args = parser.parse_args(argv)

    if args.list_models:
        import urllib.error

        eng = OpenAICompatEngine(provider=args.provider, base_url=args.base_url)
        try:
            for model_id in eng.models():
                print(model_id)
        except urllib.error.HTTPError as exc:
            # The provider's own message is the only thing that distinguishes
            # "wrong key" from "right key, wrong provider" from "key not yet
            # activated". Dropping it leaves you guessing at a bare status code.
            body = exc.read().decode("utf-8", errors="replace")[:500]
            log(f"[acp] {eng.base_url}/models returned HTTP {exc.code}\n{body}")
            if eng.key_env:
                raw = os.environ.get(eng.key_env, "")
                log(f"[acp] ${eng.key_env}: {len(raw)} chars, "
                    f"prefix {raw[:4]!r}"
                    + (", HAS SURROUNDING WHITESPACE" if raw != raw.strip() else "")
                    + (", HAS QUOTES" if raw[:1] in "\"'" else ""))
            return 1
        except Exception as exc:
            log(f"[acp] could not list models: {exc}")
            return 1
        return 0

    if args.execute and args.engine == "retrieval":
        # RetrievalOnlyEngine emits a summary of what Argus found. It never
        # produces a tool call, so an executor built on it would burn its whole
        # budget doing nothing. Better to refuse than to look broken.
        parser.error("--execute needs a real engine; pass --engine api or transformers")

    _configure_stdio()
    agent = build_agent(args)
    if args.execute:
        mode = "writing directly" if args.write else "staging edits for review"
        log(f"[acp] execute mode: {mode}, ceiling {args.max_steps} steps, "
            f"planning {'on' if agent.planning else 'off'}, "
            f"delegation {'on' if agent.delegation else 'off'}")
    log(f"[acp] {AGENT_NAME} {AGENT_VERSION} ready on stdio "
        f"(engine={agent.engine.name}, pid={os.getpid()})")
    agent.serve().serve_forever()
    return 0
