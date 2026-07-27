"""Tests for the LSP client, against a real server subprocess.

The server is a stdlib script speaking genuine `Content-Length` framing --
which is the whole reason this client exists separately from `jsonrpc.Peer`.
Mocking the transport would skip the one part most likely to be wrong: send
newline-framed JSON to a language server and it waits forever for a header,
which looks exactly like a hang.

Load-bearing claims:

  1. A missing language server returns False rather than raising. Most machines
     do not have every server installed; that is normal, not exceptional.
  2. A dead server frees anyone blocked on a request instead of hanging.
  3. URIs round-trip to paths, including on Windows, where `file:///C:/x`
     carries a leading slash that must go.

    pytest -q tests/test_lsp.py
"""
import os
import sys
import textwrap
from pathlib import Path

import pytest

from knossos.lsp import Location, LspClient, path_to_uri, uri_to_path

SERVER = textwrap.dedent('''
    import json, sys

    def read_message():
        length = None
        while True:
            line = sys.stdin.buffer.readline()
            if not line:
                return None
            text = line.decode("ascii").strip()
            if not text:
                break
            if text.lower().startswith("content-length:"):
                length = int(text.split(":", 1)[1])
        if length is None:
            return None
        return json.loads(sys.stdin.buffer.read(length).decode("utf-8"))

    def send(obj):
        body = json.dumps(obj).encode("utf-8")
        sys.stdout.buffer.write(
            ("Content-Length: %d\\r\\n\\r\\n" % len(body)).encode("ascii"))
        sys.stdout.buffer.write(body)
        sys.stdout.buffer.flush()

    def loc(uri, line, ch=0):
        return {"uri": uri, "range": {"start": {"line": line, "character": ch},
                                      "end": {"line": line, "character": ch + 4}}}

    ROOT = ["file:///proj"]

    while True:
        msg = read_message()
        if msg is None:
            break
        method, mid = msg.get("method"), msg.get("id")
        if method == "initialize":
            ROOT[0] = (msg.get("params") or {}).get("rootUri") or ROOT[0]
            send({"jsonrpc": "2.0", "id": mid, "result": {"capabilities": {
                "referencesProvider": True, "definitionProvider": True,
                "workspaceSymbolProvider": True}}})
        elif method == "workspace/symbol":
            send({"jsonrpc": "2.0", "id": mid, "result": [
                {"name": "Router", "kind": 5,
                 "location": loc(ROOT[0] + "/moe.py", 28)}]})
        elif method == "textDocument/references":
            send({"jsonrpc": "2.0", "id": mid, "result": [
                loc(ROOT[0] + "/naiads.py", 41),
                loc(ROOT[0] + "/full.py", 7)]})
        elif method == "textDocument/definition":
            send({"jsonrpc": "2.0", "id": mid, "result": loc(ROOT[0] + "/moe.py", 28)})
        elif method == "textDocument/nothing":
            send({"jsonrpc": "2.0", "id": mid, "result": None})
        elif method == "exit":
            break
        elif mid is not None:
            send({"jsonrpc": "2.0", "id": mid,
                  "error": {"code": -32601, "message": "unsupported"}})
''').strip()


@pytest.fixture()
def server_script(tmp_path):
    path = tmp_path / "fake_lsp.py"
    path.write_text(SERVER, encoding="utf-8")
    return path


@pytest.fixture()
def lsp(server_script, tmp_path):
    client = LspClient([sys.executable, str(server_script)], tmp_path, timeout=15)
    assert client.start() is True
    yield client
    client.stop()


# ------------------------------------------------------------------- lifecycle

def test_a_missing_server_returns_false_rather_than_raising(tmp_path):
    """Not having a language server installed is normal, not exceptional."""
    client = LspClient(["definitely-not-a-real-language-server"], tmp_path)
    assert client.start() is False
    assert client.available() is False


def test_capabilities_are_recorded(lsp):
    assert lsp.available() is True
    assert lsp.capabilities.get("referencesProvider") is True


def test_queries_on_a_stopped_client_return_empty(lsp, tmp_path):
    lsp.stop()
    assert lsp.available() is False
    assert lsp.workspace_symbols("Router") == []
    assert lsp.references(tmp_path / "x.py", 0, 0) == []


def test_a_dead_server_frees_a_blocked_caller(server_script, tmp_path):
    """Otherwise a crashed server hangs whatever asked it a question."""
    client = LspClient([sys.executable, str(server_script)], tmp_path, timeout=15)
    assert client.start()
    client.process.kill()

    # Returns empty rather than blocking to the timeout.
    assert client.references(tmp_path / "x.py", 1, 1) == []


# --------------------------------------------------------------------- queries

def test_workspace_symbols(lsp):
    symbols = lsp.workspace_symbols("Router")
    assert len(symbols) == 1
    assert symbols[0].path.name == "moe.py"
    assert symbols[0].line == 28


def test_references_span_files(lsp, tmp_path):
    """The thing Argus cannot compute: who uses this, across the project."""
    refs = lsp.references(tmp_path / "moe.py", 28, 6)
    assert {r.path.name for r in refs} == {"naiads.py", "full.py"}


def test_definition_accepts_a_single_object_not_just_a_list(lsp, tmp_path):
    """LSP allows either; a client that assumes a list drops the answer."""
    defs = lsp.definition(tmp_path / "naiads.py", 41, 10)
    assert len(defs) == 1
    assert defs[0].path.name == "moe.py"


def test_a_null_result_is_empty_not_an_error(lsp, tmp_path):
    assert lsp._locations("textDocument/nothing", tmp_path / "x.py", 0, 0) == []


def test_an_unsupported_method_does_not_raise(lsp, tmp_path):
    assert lsp._locations("textDocument/whatever", tmp_path / "x.py", 0, 0) == []


# ------------------------------------------------------------------------ uris

def test_uris_round_trip(tmp_path):
    original = (tmp_path / "some file.py").resolve()
    assert uri_to_path(path_to_uri(original)) == original


def test_a_uri_with_spaces_is_decoded():
    assert uri_to_path("file:///proj/my%20file.py").name == "my file.py"


@pytest.mark.skipif(os.name != "nt", reason="windows drive-letter handling")
def test_windows_drive_letters_lose_the_leading_slash():
    assert str(uri_to_path("file:///C:/proj/x.py")).startswith("C:")


def test_a_location_reports_a_one_based_ref():
    """LSP lines are 0-based; humans and editors count from 1."""
    assert Location(Path("/a/b.py"), 27).ref.endswith(":28")


def test_a_location_without_a_range_is_rejected():
    assert Location.from_lsp({"uri": "file:///x"}) is None
    assert Location.from_lsp({"range": {"start": {"line": 1}}}) is None


# ----------------------------------------------------- integration with argus

def test_argus_uses_the_language_server_for_real_references(lsp, tmp_path):
    """`users_of` answers with resolved references, not name matches."""
    from knossos import Argus

    (tmp_path / "moe.py").write_text("class Router:\n    pass\n", encoding="utf-8")
    (tmp_path / "naiads.py").write_text("from moe import Router\n", encoding="utf-8")
    (tmp_path / "full.py").write_text("from moe import Router\n", encoding="utf-8")

    argus = Argus(tmp_path, lsp=lsp)
    argus.scan()
    router = next(s for s in argus.symbols_in("moe.py") if s.name == "Router")

    users = argus.users_of(router)
    assert set(users) == {"naiads.py", "full.py"}
    assert "moe.py" not in users, "a symbol is not its own user"


def test_argus_without_a_server_falls_back_rather_than_failing(tmp_path):
    """No server must mean 'no exact answer', not 'no retrieval'."""
    from knossos import Argus

    (tmp_path / "moe.py").write_text("class Router:\n    pass\n", encoding="utf-8")
    argus = Argus(tmp_path)
    argus.scan()
    router = next(s for s in argus.symbols_in("moe.py") if s.name == "Router")

    assert argus.users_of(router) == []
    # The approximate path still works, which is the point of keeping it.
    assert isinstance(argus.importers_of("moe.py"), list)
