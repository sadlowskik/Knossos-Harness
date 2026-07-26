"""Protocol-level tests for the ACP server.

These drive the agent the way an editor does: real pipes, real JSON-RPC, one
message per line. Nothing is stubbed below the handler, because the parts most
likely to break are the ones a unit test of `DaedalusAgent` would skip --
framing, threading, and whether a cancel can land mid-turn.

The load-bearing claims:

  1. A turn produces the right notifications in the right order and ends with a
     `stopReason`.
  2. `session/cancel` arriving *during* a prompt actually interrupts it. This is
     the one that needs the fast path in `Peer`: the worker thread is inside
     `session/prompt`, so a queued cancel could only be seen after the turn it
     was supposed to stop had already ended.
  3. Every wire message is exactly one line of valid JSON, even when the payload
     is multi-line source code.

No torch import -- the harness does not depend on the model.

    pytest -q tests/test_acp.py
"""
import json
import os
import queue
import textwrap
import threading

import pytest

from harness import DaedalusAgent, StaticEngine, Thought, PROTOCOL_VERSION

TIMEOUT = 10.0


class FakeClient:
    """The editor side of the pipe."""

    def __init__(self, agent: DaedalusAgent):
        agent_r, client_w = os.pipe()
        client_r, agent_w = os.pipe()
        self._agent_rx = os.fdopen(agent_r, "r", encoding="utf-8", newline="\n")
        self._agent_tx = os.fdopen(agent_w, "w", encoding="utf-8", newline="\n")
        self._tx = os.fdopen(client_w, "w", encoding="utf-8", newline="\n")
        self._rx = os.fdopen(client_r, "r", encoding="utf-8", newline="\n")

        self.raw_lines: list[str] = []
        self.notifications: list[dict] = []
        self._inbox: "queue.Queue[dict]" = queue.Queue()
        self._id = 0

        self.peer = agent.serve(rx=self._agent_rx, tx=self._agent_tx)
        self.peer.start()
        threading.Thread(target=self._read_loop, daemon=True).start()

    def _read_loop(self):
        for line in self._rx:
            if not line.strip():
                continue
            self.raw_lines.append(line)
            self._inbox.put(json.loads(line))

    def _write(self, payload: dict):
        self._tx.write(json.dumps(payload) + "\n")
        self._tx.flush()

    def notify(self, method, params=None):
        self._write({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def call(self, method, params=None, timeout=TIMEOUT):
        """Send a request, collect notifications, return the matching response."""
        self._id += 1
        req_id = self._id
        self._write({"jsonrpc": "2.0", "id": req_id, "method": method,
                     "params": params or {}})
        while True:
            msg = self._inbox.get(timeout=timeout)
            if msg.get("id") == req_id:
                return msg
            if "method" in msg:
                self.notifications.append(msg)

    def updates(self, kind=None):
        out = [n["params"]["update"] for n in self.notifications
               if n.get("method") == "session/update"]
        return [u for u in out if kind is None or u.get("sessionUpdate") == kind]

    def close(self):
        self.peer.close()


@pytest.fixture()
def workspace(tmp_path):
    (tmp_path / "app.py").write_text(textwrap.dedent('''
        def balance_experts(scores):
            """Keep one expert from swallowing all the traffic."""
            total = sum(scores)
            return [s / total for s in scores]

        def unrelated(x):
            return x
    ''').strip(), encoding="utf-8")
    return tmp_path


@pytest.fixture()
def client(workspace):
    agent = DaedalusAgent(engine=StaticEngine(["hello ", "world"]))
    c = FakeClient(agent)
    c.agent = agent
    yield c
    c.close()


def _handshake(client, workspace):
    client.call("initialize", {
        "protocolVersion": PROTOCOL_VERSION,
        "clientCapabilities": {"fs": {"readTextFile": True, "writeTextFile": True}},
        "clientInfo": {"name": "test-editor", "version": "1.0"},
    })
    resp = client.call("session/new", {"cwd": str(workspace), "mcpServers": []})
    return resp["result"]["sessionId"]


# --------------------------------------------------------------- initialization

def test_initialize_reports_capabilities(client):
    result = client.call("initialize", {
        "protocolVersion": PROTOCOL_VERSION,
        "clientCapabilities": {},
        "clientInfo": {"name": "test-editor", "version": "1.0"},
    })["result"]
    assert result["protocolVersion"] == PROTOCOL_VERSION
    assert result["agentInfo"]["name"] == "daedalus"
    assert result["agentCapabilities"]["loadSession"] is False
    assert result["authMethods"] == []


def test_never_claims_a_protocol_version_above_its_own(client):
    result = client.call("initialize", {"protocolVersion": 99,
                                        "clientInfo": {"name": "future"}})["result"]
    assert result["protocolVersion"] == PROTOCOL_VERSION


def test_unknown_method_is_a_clean_error_not_a_crash(client):
    resp = client.call("session/teleport", {})
    assert resp["error"]["code"] == -32601
    # the connection survives it
    assert client.call("initialize", {"protocolVersion": 1})["result"]["protocolVersion"] == 1


def test_unknown_notification_is_ignored(client, workspace):
    client.notify("session/nonsense", {"whatever": True})
    assert _handshake(client, workspace).startswith("sess_")


# -------------------------------------------------------------------- sessions

def test_session_new_indexes_the_workspace(client, workspace):
    session_id = _handshake(client, workspace)
    argus = client.agent.sessions[session_id].argus
    assert "app.py" in argus.files
    assert {s.name for s in argus.symbols_in("app.py")} == {"balance_experts", "unrelated"}


def test_relative_cwd_is_rejected(client):
    client.call("initialize", {"protocolVersion": 1})
    resp = client.call("session/new", {"cwd": "./relative"})
    assert resp["error"]["code"] == -32602
    assert "absolute" in resp["error"]["message"]


def test_prompt_on_unknown_session_is_rejected(client):
    client.call("initialize", {"protocolVersion": 1})
    resp = client.call("session/prompt", {
        "sessionId": "sess_nope",
        "prompt": [{"type": "text", "text": "hi"}]})
    assert resp["error"]["code"] == -32602


# ------------------------------------------------------------------ a full turn

def test_full_turn(client, workspace):
    session_id = _handshake(client, workspace)
    resp = client.call("session/prompt", {
        "sessionId": session_id,
        "prompt": [{"type": "text", "text": "where do we balance_experts?"}]})

    assert resp["result"]["stopReason"] == "end_turn"

    started = client.updates("tool_call")
    assert len(started) == 1 and started[0]["kind"] == "search"

    done = client.updates("tool_call_update")[-1]
    assert done["status"] == "completed"
    assert done["locations"], "expected file locations"
    for loc in done["locations"]:
        assert os.path.isabs(loc["path"]), "ACP requires absolute paths"
        assert loc["line"] >= 1

    chunks = client.updates("agent_message_chunk")
    assert "".join(c["content"]["text"] for c in chunks) == "hello world"
    assert len({c["messageId"] for c in chunks}) == 1


def test_retrieved_context_reaches_the_engine(client, workspace):
    session_id = _handshake(client, workspace)
    client.call("session/prompt", {
        "sessionId": session_id,
        "prompt": [{"type": "text", "text": "explain balance_experts"}]})

    prompt, context = client.agent.engine.calls[-1]
    assert "balance_experts" in prompt
    assert "def balance_experts" in context, "the engine got the source, not just a filename"
    assert "app.py:" in context, "provenance travels with the excerpt"


def test_resource_links_become_retrieval_signal(client, workspace):
    session_id = _handshake(client, workspace)
    client.call("session/prompt", {
        "sessionId": session_id,
        "prompt": [{"type": "text", "text": "look here"},
                   {"type": "resource_link", "uri": "file:///x/app.py"}]})
    prompt, _ = client.agent.engine.calls[-1]
    assert "app.py" in prompt


def test_retrieval_report_is_not_confused_by_markdown_in_the_source(tmp_path):
    """Regression: a `# heading` in a retrieved file must not read as provenance.

    `RetrievalOnlyEngine` used to rebuild the excerpt list by scanning the
    rendered context for lines starting with "# ". A README heading inside a
    retrieved slice matched, and leaked into the reply as a phantom location.
    It now reads `Context.hits`, so the file's own content cannot forge one.
    """
    (tmp_path / "README.md").write_text(
        "# Daedalus\n\nAriadne computes halting probabilities.\n", encoding="utf-8")
    (tmp_path / "ariadne.py").write_text(
        "class Ariadne:\n    def halt(self, x):\n        return x\n", encoding="utf-8")

    agent = DaedalusAgent()                    # the real RetrievalOnlyEngine
    client = FakeClient(agent)
    try:
        session_id = _handshake(client, tmp_path)
        client.call("session/prompt", {
            "sessionId": session_id,
            "prompt": [{"type": "text", "text": "where does Ariadne halt?"}]})

        reply = "".join(c["content"]["text"]
                        for c in client.updates("agent_message_chunk"))
        reported = [ln for ln in reply.splitlines()[2:] if ln.strip()]
        assert reported, "expected the retrieval to be itemised"
        for line in reported:
            assert ":" in line.split()[0], f"not a file:line reference: {line!r}"
        assert not any(ln.strip() == "Daedalus" for ln in reported)
    finally:
        client.close()


def test_empty_prompt_is_rejected(client, workspace):
    session_id = _handshake(client, workspace)
    resp = client.call("session/prompt", {"sessionId": session_id, "prompt": []})
    assert resp["error"]["code"] == -32602


def test_gate_withholds_context_on_a_general_question(workspace):
    """The excerpts are withheld, and the editor is told they were."""
    agent = DaedalusAgent(engine=StaticEngine(["ok"]), gate=True)
    client = FakeClient(agent)
    try:
        session_id = _handshake(client, workspace)
        client.call("session/prompt", {
            "sessionId": session_id,
            "prompt": [{"type": "text",
                        "text": "what is expert collapse, in general?"}]})

        _, context = agent.engine.calls[-1]
        assert context == "", "general question should not carry repo excerpts"

        done = client.updates("tool_call_update")[-1]
        assert done["rawOutput"]["gated"] is True
        assert "general" in done["title"].lower()
    finally:
        client.close()


def test_gate_lets_a_repo_question_through(workspace):
    agent = DaedalusAgent(engine=StaticEngine(["ok"]), gate=True)
    client = FakeClient(agent)
    try:
        session_id = _handshake(client, workspace)
        client.call("session/prompt", {
            "sessionId": session_id,
            "prompt": [{"type": "text",
                        "text": "which file defines balance_experts?"}]})
        _, context = agent.engine.calls[-1]
        assert "def balance_experts" in context
    finally:
        client.close()


def test_gate_can_be_disabled(workspace):
    agent = DaedalusAgent(engine=StaticEngine(["ok"]), gate=False)
    client = FakeClient(agent)
    try:
        session_id = _handshake(client, workspace)
        client.call("session/prompt", {
            "sessionId": session_id,
            "prompt": [{"type": "text",
                        "text": "what is expert collapse, in general?"}]})
        _, context = agent.engine.calls[-1]
        assert context != "", "--no-gate should restore unconditional injection"
    finally:
        client.close()


def test_thoughts_go_to_their_own_channel(workspace):
    """A reasoning model's scratchpad must not be delivered as the answer."""
    agent = DaedalusAgent(engine=StaticEngine(
        [Thought("let me look"), "the answer is 42"]))
    client = FakeClient(agent)
    try:
        session_id = _handshake(client, workspace)
        client.call("session/prompt", {
            "sessionId": session_id,
            "prompt": [{"type": "text", "text": "balance_experts"}]})

        thoughts = client.updates("agent_thought_chunk")
        messages = client.updates("agent_message_chunk")
        assert [t["content"]["text"] for t in thoughts] == ["let me look"]
        assert [m["content"]["text"] for m in messages] == ["the answer is 42"]
    finally:
        client.close()


@pytest.mark.parametrize("finish,expected", [
    ("stop", "end_turn"),
    ("length", "max_tokens"),
    ("content_filter", "refusal"),
    (None, "end_turn"),
    ("something_new", "end_turn"),
])
def test_finish_reason_maps_to_stop_reason(workspace, finish, expected):
    """A reply cut off at the token limit must not be reported as complete."""
    engine = StaticEngine(["partial answer"])
    engine.stop_reason = finish
    agent = DaedalusAgent(engine=engine)
    client = FakeClient(agent)
    try:
        session_id = _handshake(client, workspace)
        resp = client.call("session/prompt", {
            "sessionId": session_id,
            "prompt": [{"type": "text", "text": "balance_experts"}]})
        assert resp["result"]["stopReason"] == expected
    finally:
        client.close()


# ---------------------------------------------------------------- cancellation

def test_cancel_lands_mid_turn(workspace):
    """The claim the fast path exists for.

    The engine blocks until the test says otherwise. If `session/cancel` were
    queued behind `session/prompt` on the worker thread, `entered` would be set,
    the cancel would sit unread, and this test would hang until TIMEOUT.
    """
    entered = threading.Event()
    release = threading.Event()

    class BlockingEngine:
        name = "blocking"

        def generate(self, prompt, context, cancelled):
            entered.set()
            while not cancelled():
                if release.wait(0.02):
                    break
            yield ""

    agent = DaedalusAgent(engine=BlockingEngine())
    client = FakeClient(agent)
    try:
        session_id = _handshake(client, workspace)

        result: dict = {}
        def run():
            result["resp"] = client.call("session/prompt", {
                "sessionId": session_id,
                "prompt": [{"type": "text", "text": "balance_experts"}]})

        turn = threading.Thread(target=run, daemon=True)
        turn.start()

        assert entered.wait(TIMEOUT), "engine never started"
        client.notify("session/cancel", {"sessionId": session_id})

        turn.join(TIMEOUT)
        release.set()
        assert not turn.is_alive(), "cancel never reached the running turn"
        assert result["resp"]["result"]["stopReason"] == "cancelled"
    finally:
        release.set()
        client.close()


# --------------------------------------------------------------------- framing

def test_every_message_is_one_line_of_valid_json(workspace):
    """Multi-line output must arrive escaped, not as raw newlines on the wire.

    Model output contains newlines constantly (code blocks, prose), and a single
    raw one desynchronises the stream: the client reads half a message, fails to
    parse it, and drops the connection.
    """
    payload = "here:\n\n```python\ndef balance_experts(scores):\n    ...\n```\n"
    agent = DaedalusAgent(engine=StaticEngine([payload, "\ndone\n"]))
    client = FakeClient(agent)
    try:
        session_id = _handshake(client, workspace)
        resp = client.call("session/prompt", {
            "sessionId": session_id,
            "prompt": [{"type": "text", "text": "balance_experts"}]})
        assert resp["result"]["stopReason"] == "end_turn"

        assert client.raw_lines
        for line in client.raw_lines:
            assert line.endswith("\n")
            assert "\n" not in line[:-1], "embedded newline breaks ndjson framing"
            json.loads(line)                   # each line parses on its own

        assert "\\n" in "".join(client.raw_lines), "newlines should be escaped, not raw"
        chunks = client.updates("agent_message_chunk")
        # and survive the round trip byte for byte
        assert "".join(c["content"]["text"] for c in chunks) == payload + "\ndone\n"
    finally:
        client.close()
