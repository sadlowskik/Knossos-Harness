"""`jsonrpc.Peer` request semantics -- specifically, how a wait ends.

An outbound request is the one place the agent hands control to the other side
and waits. `session/request_permission` is untimed by design: a user may take
as long as they like to read a diff before approving it. That makes the exit
conditions load-bearing, because "no deadline" must not mean "unkillable" --
a cancelled turn or a closed pipe has to release the waiting thread, or the
worker wedges and the session is dead with no error to show for it.

    pytest -q tests/test_jsonrpc.py
"""
import json
import os
import threading
import time

import pytest

from knossos.jsonrpc import Peer, RpcError


class Pipes:
    """A Peer wired to a fake counterpart over real pipes."""

    def __init__(self, handler=None):
        agent_rx_fd, self.client_tx_fd = os.pipe()
        client_rx_fd, agent_tx_fd = os.pipe()
        self._agent_rx = os.fdopen(agent_rx_fd, "r", encoding="utf-8")
        self._agent_tx = os.fdopen(agent_tx_fd, "w", encoding="utf-8")
        self._client_tx = os.fdopen(self.client_tx_fd, "w", encoding="utf-8")
        self.client_rx = os.fdopen(client_rx_fd, "r", encoding="utf-8")
        self.peer = Peer(handler or (lambda m, p, r: {}),
                         rx=self._agent_rx, tx=self._agent_tx)
        self.peer.start()

    def read_message(self):
        """The next message the agent sent us."""
        return json.loads(self.client_rx.readline())

    def reply(self, req_id, result):
        self._client_tx.write(
            json.dumps({"jsonrpc": "2.0", "id": req_id, "result": result}) + "\n")
        self._client_tx.flush()

    def hang_up(self):
        self._client_tx.close()

    def close(self):
        for f in (self._client_tx, self.client_rx, self._agent_rx, self._agent_tx):
            try:
                f.close()
            except OSError:
                pass


@pytest.fixture()
def pipes():
    p = Pipes()
    yield p
    p.close()


def call_async(peer, **kwargs):
    """Run `request` on its own thread; returns a getter for the result."""
    box = {}

    def go():
        try:
            box["result"] = peer.request("probe", {}, **kwargs)
        except BaseException as exc:            # noqa: BLE001 - recorded, re-raised below
            box["error"] = exc

    t = threading.Thread(target=go, daemon=True)
    t.start()
    return t, box


# ------------------------------------------------------- surviving a bad message

def send_raw(pipes, text):
    pipes._client_tx.write(text + "\n")
    pipes._client_tx.flush()


@pytest.mark.parametrize("bad", [
    '{"method": []}',                    # unhashable: TypeError inside `in`
    '{"method": {"a": 1}}',
    '{"method": 7}',
    '{"id": [], "result": 1}',           # unhashable: TypeError inside dict.pop
    '{"id": {}, "result": 1}',
    '{"id": 1, "error": "not an object"}',
    'null',
    '[1, 2, 3]',
    'not json at all',
])
def test_one_malformed_message_does_not_kill_the_connection(pipes, bad):
    """A single line from the editor used to end the reader permanently.

    `{"method": []}` is the cheapest case: `_fast_path` tests `method in
    FAST_PATH`, an unhashable key raises `TypeError`, and that escaped the
    loop's `except (OSError, ValueError)`. The `finally` then closed the peer
    and stranded every pending request -- so one bad line took the session's
    in-memory staged edits with it.
    """
    send_raw(pipes, bad)

    # The connection still works: a real request goes out and comes back.
    t, box = call_async(pipes.peer)
    sent = pipes.read_message()
    pipes.reply(sent["id"], {"ok": True})
    t.join(timeout=5)

    assert not t.is_alive(), f"the reader died on {bad!r}"
    assert box.get("result") == {"ok": True}


def test_a_malformed_error_field_still_releases_the_caller(pipes):
    """`_resolve` pops the pending request before it parses the error.

    An exception after that point is worse than a dropped message: nothing is
    left to answer the waiter, so it blocks until its own timeout with no
    error to show.
    """
    t, box = call_async(pipes.peer, timeout=10)
    sent = pipes.read_message()

    send_raw(pipes, json.dumps({"jsonrpc": "2.0", "id": sent["id"],
                                "error": "a string, not an object"}))
    t.join(timeout=5)

    assert not t.is_alive(), "the caller was left waiting"
    assert isinstance(box.get("error"), RpcError)


# --------------------------------------------------------------- the happy path

def test_a_reply_is_returned_to_the_caller(pipes):
    t, box = call_async(pipes.peer)
    sent = pipes.read_message()
    assert sent["method"] == "probe"

    pipes.reply(sent["id"], {"ok": True})
    t.join(timeout=5)

    assert not t.is_alive(), "the wait must end when the reply lands"
    assert box["result"] == {"ok": True}


def test_an_untimed_wait_survives_a_slow_answer(pipes):
    """A user reading a diff is not a timeout."""
    t, box = call_async(pipes.peer, timeout=None)
    sent = pipes.read_message()

    time.sleep(Peer.WAIT_SLICE * 4)
    assert t.is_alive(), "an untimed request must still be waiting"

    pipes.reply(sent["id"], {"outcome": "selected"})
    t.join(timeout=5)
    assert box["result"] == {"outcome": "selected"}


# ------------------------------------------------------------- how a wait ends

def test_abort_releases_an_untimed_wait(pipes):
    """The cancel path: nobody answers, and the turn is cancelled anyway."""
    abort = threading.Event()
    t, box = call_async(pipes.peer, timeout=None, abort=abort)
    pipes.read_message()

    abort.set()
    t.join(timeout=5)

    assert not t.is_alive(), "an aborted request must not hold the thread"
    assert isinstance(box["error"], RpcError)
    assert "cancelled" in str(box["error"])


def test_a_timeout_still_expires(pipes):
    t, box = call_async(pipes.peer, timeout=Peer.WAIT_SLICE * 2)
    pipes.read_message()

    t.join(timeout=5)
    assert isinstance(box["error"], RpcError)
    assert "timed out" in str(box["error"])


def test_a_closed_pipe_releases_an_untimed_wait(pipes):
    """A departed editor must not strand the worker."""
    t, box = call_async(pipes.peer, timeout=None)
    pipes.read_message()

    pipes.hang_up()
    t.join(timeout=5)

    assert not t.is_alive(), "a dead peer must end the wait"
    assert isinstance(box["error"], RpcError)


def test_an_aborted_request_stops_tracking_its_id(pipes):
    """A late answer to an abandoned request must not be mistaken for a live one."""
    abort = threading.Event()
    t, _ = call_async(pipes.peer, timeout=None, abort=abort)
    sent = pipes.read_message()
    abort.set()
    t.join(timeout=5)

    pipes.reply(sent["id"], {"outcome": "selected"})
    time.sleep(Peer.WAIT_SLICE * 3)

    assert not pipes.peer._pending, "the abandoned request must be forgotten"
