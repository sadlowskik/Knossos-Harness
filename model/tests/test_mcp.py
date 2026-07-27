"""Tests for the MCP client, against a real server subprocess.

The server is a stdlib Python script written to a temp file, so this exercises
the actual handshake, framing and dispatch rather than a mock of my own reading
of the protocol -- the same reasoning as the ACP integration tests.

Load-bearing claims:

  1. A failing server does not take the session with it. The agent keeps its
     local tools and the failure is reported once.
  2. Tool names are namespaced. Two servers offering `search` must not shadow
     each other, because a silently-wrong tool is unfindable.
  3. A remote error becomes an error *result*, not an exception -- something the
     engine can correct on its next turn.

    pytest -q tests/test_mcp.py
"""
import sys
import textwrap

import pytest

from knossos.mcp import McpClient, McpServer, _render, connect_all

SERVER = textwrap.dedent('''
    import json, sys

    def send(obj):
        sys.stdout.write(json.dumps(obj) + "\\n")
        sys.stdout.flush()

    TOOLS = [
        {"name": "echo", "description": "Echo the input",
         "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}}},
        {"name": "explode", "description": "Always fails", "inputSchema": {}},
    ]

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        method, mid = msg.get("method"), msg.get("id")

        if method == "initialize":
            send({"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": "2025-06-18", "capabilities": {},
                "serverInfo": {"name": "fake", "version": "1"}}})
        elif method == "tools/list":
            send({"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}})
        elif method == "tools/call":
            name = msg["params"]["name"]
            args = msg["params"].get("arguments", {})
            if name == "explode":
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": "boom"}], "isError": True}})
            else:
                send({"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text", "text": "echo: " + args.get("text", "")}]}})
        elif mid is not None:
            send({"jsonrpc": "2.0", "id": mid,
                  "error": {"code": -32601, "message": "no"}})
''').strip()


@pytest.fixture()
def server_script(tmp_path):
    path = tmp_path / "fake_mcp.py"
    path.write_text(SERVER, encoding="utf-8")
    return path


@pytest.fixture()
def client(server_script):
    c = McpClient(McpServer(name="fake", command=sys.executable,
                            args=[str(server_script)]))
    c.connect()
    yield c
    c.close()


# ------------------------------------------------------------------ handshake

def test_tools_are_discovered(client):
    assert len(client.tools) == 2
    assert {t.remote_name for t in client.tools} == {"echo", "explode"}


def test_tool_names_are_namespaced_by_server(client):
    """Two servers may both offer `search`; shadowing one is unfindable."""
    assert {t.spec.name for t in client.tools} == {"fake.echo", "fake.explode"}


def test_the_schema_survives(client):
    echo = next(t for t in client.tools if t.remote_name == "echo")
    assert echo.spec.schema["properties"]["text"]["type"] == "string"


def test_a_missing_schema_becomes_an_empty_object(client):
    """The engine needs *a* schema; None would break call rendering."""
    explode = next(t for t in client.tools if t.remote_name == "explode")
    assert explode.spec.schema.get("type") == "object"


# ---------------------------------------------------------------------- calls

def test_calling_a_remote_tool(client, tmp_path):
    from knossos.workspace import Workspace

    echo = next(t for t in client.tools if t.remote_name == "echo")
    result = echo.run({"text": "hello"}, Workspace(tmp_path))
    assert result.is_error is False
    assert "echo: hello" in result.content


def test_a_remote_failure_is_an_error_result_not_an_exception(client, tmp_path):
    """The engine can correct an error result on its next turn; an exception
    would end the run."""
    from knossos.workspace import Workspace

    explode = next(t for t in client.tools if t.remote_name == "explode")
    result = explode.run({}, Workspace(tmp_path))
    assert result.is_error is True
    assert "boom" in result.content


def test_calling_a_disconnected_client_is_an_error_result(tmp_path):
    from knossos.workspace import Workspace

    from knossos.mcp import McpTool
    from knossos.tools import ToolSpec

    orphan = McpClient(McpServer(name="dead", command="does-not-exist"))
    tool = McpTool(orphan, "x", ToolSpec("dead.x", "", {}))
    assert tool.run({}, Workspace(tmp_path)).is_error is True


# ------------------------------------------------------------------ resilience

def test_a_server_that_cannot_start_does_not_raise(server_script):
    """One bad server must not take the session with it."""
    clients, errors = connect_all([
        {"name": "broken", "command": "definitely-not-a-real-binary-xyz"},
        {"name": "good", "command": sys.executable, "args": [str(server_script)]},
    ])
    try:
        assert len(clients) == 1, "the working server should still connect"
        assert clients[0].server.name == "good"
        assert any("broken" in e for e in errors)
    finally:
        for c in clients:
            c.close()


def test_an_unusable_declaration_is_reported_not_crashed():
    clients, errors = connect_all([{"name": "no-command"}, "not-a-dict"])
    assert clients == []
    assert len(errors) == 2


def test_no_servers_is_not_an_error():
    assert connect_all([]) == ([], [])
    assert connect_all(None) == ([], [])


# -------------------------------------------------------------------- parsing

def test_acp_env_list_form_is_parsed():
    server = McpServer.from_acp({
        "name": "s", "command": "cmd",
        "env": [{"name": "TOKEN", "value": "abc"}],
    })
    assert server.env == {"TOKEN": "abc"}


def test_a_plain_env_mapping_is_also_accepted():
    """Most hand-written configs use a mapping, whatever the spec says."""
    server = McpServer.from_acp({"name": "s", "command": "c", "env": {"A": "1"}})
    assert server.env == {"A": "1"}


def test_a_declaration_without_a_command_is_rejected():
    assert McpServer.from_acp({"name": "s"}) is None
    assert McpServer.from_acp({"command": "c"}) is None


# ------------------------------------------------------------------ rendering

def test_text_blocks_are_joined():
    assert _render([{"type": "text", "text": "a"},
                    {"type": "text", "text": "b"}]) == "a\nb"


def test_non_text_blocks_are_named_not_dropped():
    """An engine told '[image]' can ask for something else; one told nothing
    assumes the call returned empty."""
    assert "[image content]" in _render([{"type": "image", "data": "..."}])


def test_resource_blocks_prefer_text_then_uri():
    assert _render([{"type": "resource",
                     "resource": {"uri": "file:///x", "text": "body"}}]) == "body"
    assert _render([{"type": "resource",
                     "resource": {"uri": "file:///x"}}]) == "file:///x"


def test_empty_content_renders_empty():
    assert _render(None) == ""
    assert _render([]) == ""
