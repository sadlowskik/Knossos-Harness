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

    python -m harness --engine retrieval

Then register it with Zed in `settings.json` under `agent_servers`.
"""
from __future__ import annotations

import os
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .argus import Argus, Retrieved, render
from .ariadne import Ariadne
from .engine import (PROVIDERS, Engine, OpenAICompatEngine, RetrievalOnlyEngine,
                     Thought)
from .gate import RetrievalGate
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
    if name == "run_command":
        return f"Running {args.get('command', '')}"
    return name


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


class DaedalusAgent:
    """ACP method handlers. Transport-free, so tests can drive it over a pipe."""

    def __init__(self, engine: Optional[Engine] = None, budget: int = 8000,
                 hops: int = 1, gate: bool = True, execute: bool = False,
                 dry_run: bool = True, max_steps: int = 12,
                 target_steps: int = 6) -> None:
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
        self.peer: Optional[Peer] = None
        self.sessions: Dict[str, Session] = {}
        self.client_capabilities: Dict[str, Any] = {}

    # --------------------------------------------------------------- dispatch

    def handle(self, method: str, params: Any, is_request: bool) -> Any:
        handlers = {
            "initialize": self.initialize,
            "authenticate": self.authenticate,
            "session/new": self.session_new,
            "session/prompt": self.session_prompt,
            "session/cancel": self.session_cancel,
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
                "loadSession": False,
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

        self.sessions[session_id] = Session(id=session_id, cwd=cwd, argus=argus,
                                            gate=gate)
        return {"sessionId": session_id}

    def session_cancel(self, params: Dict[str, Any]) -> None:
        session = self.sessions.get(params.get("sessionId", ""))
        if session:
            session.cancel.set()

    def session_prompt(self, params: Dict[str, Any]) -> Dict[str, Any]:
        session = self.sessions.get(params.get("sessionId", ""))
        if session is None:
            raise RpcError(INVALID_PARAMS, f"unknown session: {params.get('sessionId')}")
        session.cancel.clear()

        prompt = _prompt_text(params.get("prompt") or [])
        if not prompt.strip():
            raise RpcError(INVALID_PARAMS, "prompt contained no text")

        hits = self._run_retrieval(session, prompt)
        if session.cancel.is_set():
            return {"stopReason": "cancelled"}

        context = render(hits)

        if self.execute:
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

    def _update(self, session: Session, update: Dict[str, Any]) -> None:
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
            workspace = Workspace(session.cwd, dry_run=self.dry_run)
            session.talos = Talos(
                self.engine, workspace,
                ariadne=Ariadne(max_steps=self.max_steps, target_steps=self.target_steps),
                verifier=Oracle(session.cwd))
        return session.talos

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
        outcome = (talos.resume(prompt, context, session.cancel.is_set, on_event)
                   if started else
                   talos.run(prompt, context, cancelled=session.cancel.is_set,
                             on_event=on_event))

        self._update(session, {
            "sessionUpdate": "agent_message_chunk",
            "messageId": message_id,
            "content": {"type": "text", "text": f"\n\n{outcome.summary}"},
        })

        if session.cancel.is_set():
            return {"stopReason": "cancelled"}
        # Only a verified run reports end_turn. Anything else says so, rather
        # than letting the editor render an unfinished task as complete.
        return {"stopReason": "end_turn" if outcome.succeeded else "refusal"}

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
                        "title": "Skipped repository context — general question",
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
            "title": f"Searched the repository — {summary}",
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


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    from .jsonrpc import _configure_stdio

    parser = argparse.ArgumentParser(
        prog="python -m harness",
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
    parser.add_argument("--max-steps", type=int, default=12,
                        help="execute mode: hard ceiling on engine turns")
    parser.add_argument("--target-steps", type=int, default=6,
                        help="execute mode: where budget pressure begins")
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

    if args.engine == "api":
        engine: Engine = OpenAICompatEngine(
            provider=args.provider, model=args.model, base_url=args.base_url)
    elif args.engine == "transformers":
        from .engine import TransformersEngine
        engine = TransformersEngine(
            model_id=args.model or "Qwen/Qwen2.5-Coder-7B-Instruct",
            adapter=args.adapter)
    else:
        engine = RetrievalOnlyEngine()

    if args.execute and args.engine == "retrieval":
        # RetrievalOnlyEngine emits a summary of what Argus found. It never
        # produces a tool call, so an executor built on it would burn its whole
        # budget doing nothing. Better to refuse than to look broken.
        parser.error("--execute needs a real engine; pass --engine api or transformers")

    _configure_stdio()
    agent = DaedalusAgent(engine=engine, budget=args.budget, hops=args.hops,
                          gate=not args.no_gate, execute=args.execute,
                          dry_run=not args.write, max_steps=args.max_steps,
                          target_steps=args.target_steps)
    if args.execute:
        mode = "writing directly" if args.write else "staging edits for review"
        log(f"[acp] execute mode: {mode}, ceiling {args.max_steps} steps")
    log(f"[acp] {AGENT_NAME} {AGENT_VERSION} ready on stdio "
        f"(engine={engine.name}, pid={os.getpid()})")
    agent.serve().serve_forever()
    return 0
