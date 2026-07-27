"""Tests for `session/load` -- resuming a conversation.

The spec's ordering is the load-bearing part: the agent replays history as
`session/update` notifications *before* answering the request, so a client that
renders updates as they arrive rebuilds the transcript and only then sees the
call return.

The other claim is that replay is not re-execution. Reopening a panel must
replay what was said, not run the tools again -- an agent that re-edits files
because the user reopened a tab is worse than one that forgets.

    pytest -q tests/test_acp_load.py
"""
import json

import pytest

from knossos import DaedalusAgent, StaticEngine, PROTOCOL_VERSION

from test_acp import FakeClient, _handshake, workspace  # noqa: F401


@pytest.fixture()
def client(workspace):  # noqa: F811
    agent = DaedalusAgent(engine=StaticEngine(["the answer"]))
    c = FakeClient(agent)
    c.agent = agent
    yield c
    c.close()


def test_load_session_is_advertised(client):
    result = client.call("initialize", {"protocolVersion": PROTOCOL_VERSION})["result"]
    assert result["agentCapabilities"]["loadSession"] is True, (
        "advertising false while implementing it means no client will ever call it"
    )


def test_loading_replays_the_conversation(client, workspace):  # noqa: F811
    session_id = _handshake(client, workspace)
    client.call("session/prompt", {
        "sessionId": session_id,
        "prompt": [{"type": "text", "text": "where is balance_experts?"}]})

    before = client.updates()
    assert before, "the turn should have produced updates"
    client.notifications.clear()

    resp = client.call("session/load", {"sessionId": session_id,
                                        "cwd": str(workspace), "mcpServers": []})
    assert "error" not in resp

    replayed = client.updates()
    assert replayed == before, "replay must reproduce what the client already saw"


def test_replay_includes_tool_calls_not_just_text(client, workspace):  # noqa: F811
    """A transcript without its tool calls loses the file locations.

    Those are the auditable part -- what the agent actually read.
    """
    session_id = _handshake(client, workspace)
    client.call("session/prompt", {
        "sessionId": session_id,
        "prompt": [{"type": "text", "text": "balance_experts"}]})
    client.notifications.clear()

    client.call("session/load", {"sessionId": session_id,
                                 "cwd": str(workspace), "mcpServers": []})

    kinds = {u.get("sessionUpdate") for u in client.updates()}
    assert "tool_call" in kinds
    assert "tool_call_update" in kinds
    assert "agent_message_chunk" in kinds


def test_replay_arrives_before_the_response(client, workspace):  # noqa: F811
    """Ordering is specified: updates first, then the reply.

    A client that renders on arrival must have the transcript rebuilt by the
    time the call returns.
    """
    session_id = _handshake(client, workspace)
    client.call("session/prompt", {
        "sessionId": session_id,
        "prompt": [{"type": "text", "text": "balance_experts"}]})
    client.notifications.clear()
    client.raw_lines.clear()

    client.call("session/load", {"sessionId": session_id,
                                 "cwd": str(workspace), "mcpServers": []})

    # Walk the wire in order: every update must precede the response.
    seen_response = False
    for line in client.raw_lines:
        msg = json.loads(line)
        if "id" in msg and "method" not in msg:
            seen_response = True
        elif msg.get("method") == "session/update":
            assert not seen_response, "an update arrived after the response"


def test_loading_twice_does_not_double_the_history(client, workspace):  # noqa: F811
    """Replay must not append to the history it is replaying."""
    session_id = _handshake(client, workspace)
    client.call("session/prompt", {
        "sessionId": session_id,
        "prompt": [{"type": "text", "text": "balance_experts"}]})
    original = len(client.agent.sessions[session_id].history)

    for _ in range(3):
        client.call("session/load", {"sessionId": session_id,
                                     "cwd": str(workspace), "mcpServers": []})

    assert len(client.agent.sessions[session_id].history) == original


def test_loading_an_unknown_session_is_rejected(client, workspace):  # noqa: F811
    _handshake(client, workspace)
    resp = client.call("session/load", {"sessionId": "sess_nope",
                                        "cwd": str(workspace), "mcpServers": []})
    assert resp["error"]["code"] == -32602


def test_a_loaded_session_still_accepts_prompts(client, workspace):  # noqa: F811
    """Resuming means usable, not merely readable."""
    session_id = _handshake(client, workspace)
    client.call("session/prompt", {
        "sessionId": session_id,
        "prompt": [{"type": "text", "text": "first"}]})
    client.call("session/load", {"sessionId": session_id,
                                 "cwd": str(workspace), "mcpServers": []})

    resp = client.call("session/prompt", {
        "sessionId": session_id,
        "prompt": [{"type": "text", "text": "second"}]})
    assert resp["result"]["stopReason"] == "end_turn"
