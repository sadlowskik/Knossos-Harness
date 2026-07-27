"""MCP: connecting Knossos to tools it did not ship with.

The Model Context Protocol is how an agent reaches things outside its own
process -- databases, browsers, issue trackers, whatever someone has wrapped.
Like ACP it is JSON-RPC 2.0 over newline-delimited stdio, so the transport is
already here: `jsonrpc.Peer` does both.

ACP passes `mcpServers` in `session/new`. Knossos accepted that parameter and
discarded it, which meant a client could configure tools that silently never
appeared. This connects them instead.

The handshake is three calls:

    initialize      exchange protocol version and capabilities
    notifications/initialized   (a notification; the server may wait for it)
    tools/list      what this server can do

Each remote tool is then wrapped as an ordinary `Tool` and registered, so Talos
dispatches it exactly like a local one. Names are prefixed with the server's
label (`github.create_issue`) because two servers may both offer `search`, and
silently shadowing one with the other would be a bug nobody could see.

Failure is contained by design. A server that will not start, times out, or
returns nonsense must not take the session with it -- the agent keeps its local
tools and the failure is reported once. An editor that cannot open because a
side-car is down is worse than one missing a feature.
"""
from __future__ import annotations

import json
import subprocess
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .jsonrpc import Peer, RpcError, log
from .tools import Tool, ToolResult, ToolSpec
from .workspace import Workspace

__all__ = ["McpServer", "McpClient", "McpTool", "connect_all"]

#: MCP's own protocol version, distinct from ACP's.
PROTOCOL_VERSION = "2025-06-18"
CONNECT_TIMEOUT = 30.0
CALL_TIMEOUT = 120.0


@dataclass
class McpServer:
    """A server declaration, as ACP delivers it in `session/new`."""

    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_acp(cls, raw: Dict[str, Any]) -> Optional["McpServer"]:
        """Parse one entry of `mcpServers`, or None if it is unusable.

        ACP carries `env` as a list of {name, value} objects; a plain mapping is
        accepted too because that is what most hand-written configs contain.
        """
        name = raw.get("name")
        command = raw.get("command")
        if not name or not command:
            return None

        env_raw = raw.get("env") or []
        if isinstance(env_raw, dict):
            env = {str(k): str(v) for k, v in env_raw.items()}
        else:
            env = {
                str(e.get("name")): str(e.get("value", ""))
                for e in env_raw
                if isinstance(e, dict) and e.get("name")
            }

        return cls(name=str(name), command=str(command),
                   args=[str(a) for a in raw.get("args") or []], env=env)


class McpTool(Tool):
    """A remote tool, wrapped so Talos cannot tell it apart from a local one."""

    def __init__(self, client: "McpClient", remote_name: str,
                 spec: ToolSpec) -> None:
        self.client = client
        self.remote_name = remote_name
        self.spec = spec

    def run(self, args: Dict[str, Any], ws: Workspace) -> ToolResult:
        # `ws` is unused: a remote tool acts through its own server, so the
        # workspace sandbox cannot constrain it. That is worth stating plainly --
        # connecting an MCP server widens what the agent can reach.
        try:
            return self.client.call_tool(self.remote_name, args)
        except Exception as exc:
            return ToolResult(f"{self.spec.name} failed: {exc}", is_error=True)


class McpClient:
    """One connected MCP server."""

    def __init__(self, server: McpServer) -> None:
        self.server = server
        self.process: Optional[subprocess.Popen] = None
        self.peer: Optional[Peer] = None
        self.tools: List[McpTool] = []
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- lifecycle

    def connect(self) -> List[McpTool]:
        """Start the server and return its tools. Raises on failure."""
        import os

        self.process = subprocess.Popen(
            [self.server.command, *self.server.args],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, encoding="utf-8",
            bufsize=1, env={**os.environ, **self.server.env},
        )
        # Drain stderr, or a chatty server fills the pipe and blocks mid-call.
        threading.Thread(target=self._drain_stderr, daemon=True).start()

        self.peer = Peer(self._handle, rx=self.process.stdout,
                         tx=self.process.stdin)
        self.peer.start()

        self.peer.request("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "knossos", "version": "0.1.0"},
        }, timeout=CONNECT_TIMEOUT)
        # Some servers wait for this before answering anything else.
        self.peer.notify("notifications/initialized", {})

        listed = self.peer.request("tools/list", {}, timeout=CONNECT_TIMEOUT) or {}
        self.tools = [
            McpTool(self, t["name"], ToolSpec(
                name=f"{self.server.name}.{t['name']}",
                description=t.get("description") or f"{t['name']} via {self.server.name}",
                schema=t.get("inputSchema") or {"type": "object", "properties": {}},
            ))
            for t in listed.get("tools", [])
            if isinstance(t, dict) and t.get("name")
        ]
        log(f"[mcp] {self.server.name}: {len(self.tools)} tool(s)")
        return self.tools

    def close(self) -> None:
        if self.peer is not None:
            self.peer.close()
        if self.process is not None:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass

    # -------------------------------------------------------------------- calls

    def call_tool(self, name: str, args: Dict[str, Any]) -> ToolResult:
        if self.peer is None:
            return ToolResult(f"{self.server.name} is not connected", is_error=True)

        # Serialised: one in-flight call per server. MCP servers are not
        # required to handle concurrent requests, and a wrong answer attributed
        # to the wrong call is harder to notice than a slow one.
        with self._lock:
            try:
                raw = self.peer.request(
                    "tools/call", {"name": name, "arguments": args},
                    timeout=CALL_TIMEOUT) or {}
            except RpcError as exc:
                return ToolResult(f"{name}: {exc.message}", is_error=True)

        return ToolResult(_render(raw.get("content")),
                          is_error=bool(raw.get("isError")))

    # ------------------------------------------------------------------ plumbing

    def _handle(self, method: str, params: Any, is_request: bool) -> Any:
        """Server->client requests. Nothing is supported yet, but requests are
        still answered -- silence would block the server indefinitely."""
        if is_request:
            raise RpcError(-32601, f"{method} is not supported by this client")
        return None

    def _drain_stderr(self) -> None:
        if self.process is None or self.process.stderr is None:
            return
        for line in self.process.stderr:
            log(f"[mcp:{self.server.name}] {line.rstrip()}")


def _render(content: Any) -> str:
    """Flatten MCP content blocks into text.

    Non-text blocks are named rather than dropped: an engine told "[image]" can
    ask for something else, while an engine told nothing assumes the call
    returned empty.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    parts: List[str] = []
    for block in content if isinstance(content, list) else [content]:
        if not isinstance(block, dict):
            parts.append(str(block))
            continue
        kind = block.get("type")
        if kind == "text":
            parts.append(str(block.get("text", "")))
        elif kind == "resource":
            resource = block.get("resource") or {}
            parts.append(str(resource.get("text") or resource.get("uri", "")))
        else:
            parts.append(f"[{kind or 'unknown'} content]")
    return "\n".join(p for p in parts if p)


def connect_all(raw_servers: Sequence[Dict[str, Any]]) -> tuple[List[McpClient],
                                                                List[str]]:
    """Connect every declared server. Returns (clients, error messages).

    One bad server must not take the session with it: the agent keeps its local
    tools, and the failure is reported once rather than on every later call.
    """
    clients: List[McpClient] = []
    errors: List[str] = []

    for raw in raw_servers or []:
        server = McpServer.from_acp(raw) if isinstance(raw, dict) else None
        if server is None:
            errors.append(f"unusable mcpServers entry: {json.dumps(raw)[:120]}")
            continue
        client = McpClient(server)
        try:
            client.connect()
            clients.append(client)
        except Exception as exc:
            client.close()
            errors.append(f"{server.name}: {exc}")
            log(f"[mcp] {server.name} failed to start: {exc}")

    return clients, errors
