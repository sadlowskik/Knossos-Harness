"""A bidirectional JSON-RPC 2.0 peer over newline-delimited stdio.

ACP is symmetric: the editor calls the agent (`session/prompt`), and the agent
calls the editor back mid-turn (`fs/read_text_file`, `session/request_permission`).
That symmetry is the only hard part of the transport, and it is what forces the
threading below.

    reader thread   owns stdin. Parses one message per line. Responses resolve
                    a waiting `Pending`; requests and notifications go on a queue.
    worker thread   drains that queue and runs handlers. Handlers may call
                    `request()`, which blocks on a `Pending` that the *reader*
                    resolves -- so it cannot deadlock against itself.
    any thread      may write, serialised by `_write_lock`.

Doing this on one thread would deadlock the moment a handler called back into
the client: the reader would be sitting inside the handler, so the reply it was
waiting for could never be read.

Framing rules, which are not negotiable:

  * one JSON value per line, `\\n`-terminated, no embedded newlines
  * stdout carries protocol traffic and nothing else -- a stray `print` corrupts
    the stream and the editor drops the connection
  * logging goes to stderr, which the client may show as agent logs

On Windows, text-mode stdout rewrites `\\n` to `\\r\\n`; `_configure_stdio`
turns that off. Getting this wrong produces a stream that works on the user's
Linux CI and fails on their laptop.
"""
from __future__ import annotations

import json
import queue
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, IO, Optional

__all__ = ["Peer", "RpcError", "METHOD_NOT_FOUND", "INVALID_PARAMS", "INTERNAL_ERROR"]

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


class RpcError(Exception):
    """An error to return to the peer, or one received from it."""

    def __init__(self, code: int, message: str, data: Any = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data

    def to_json(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"code": self.code, "message": self.message}
        if self.data is not None:
            out["data"] = self.data
        return out


def log(*parts: Any) -> None:
    """Write to stderr. Never, ever to stdout."""
    print(*parts, file=sys.stderr, flush=True)


def _configure_stdio() -> None:
    for stream, kwargs in ((sys.stdin, {}), (sys.stdout, {"newline": "\n"})):
        try:
            stream.reconfigure(encoding="utf-8", **kwargs)  # type: ignore[union-attr]
        except (AttributeError, ValueError):
            pass


@dataclass
class _Pending:
    event: threading.Event
    result: Any = None
    error: Optional[RpcError] = None


class Peer:
    """One end of a JSON-RPC conversation.

    `handler` is called as `handler(method, params, is_request) -> Any`. Return a
    JSON-serialisable result for requests; raise `RpcError` to send a structured
    error. The return value is ignored for notifications.
    """

    def __init__(self, handler: Callable[[str, Any, bool], Any],
                 rx: Optional[IO[str]] = None, tx: Optional[IO[str]] = None,
                 fast_path: Optional[Callable[[str], bool]] = None) -> None:
        self._handler = handler
        # Methods the worker queue would starve. `session/cancel` is the reason
        # this exists: it arrives *during* a `session/prompt`, so queueing it
        # behind that prompt means it can only be delivered once the turn it was
        # meant to interrupt has already finished. Fast-path handlers run on the
        # reader thread and must therefore never block.
        self._fast_path = fast_path or (lambda _method: False)
        self._rx = rx if rx is not None else sys.stdin
        self._tx = tx if tx is not None else sys.stdout
        self._write_lock = threading.Lock()
        self._pending: Dict[Any, _Pending] = {}
        self._pending_lock = threading.Lock()
        self._next_id = 0
        self._inbox: "queue.Queue[Optional[dict]]" = queue.Queue()
        self._closed = threading.Event()
        self._threads: list[threading.Thread] = []

    # ------------------------------------------------------------------ output

    def _send(self, payload: Dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        assert "\n" not in line, "framing violation: message contains a newline"
        with self._write_lock:
            if self._closed.is_set():
                return
            try:
                self._tx.write(line + "\n")
                self._tx.flush()
            except (BrokenPipeError, ValueError):
                self._closed.set()

    def notify(self, method: str, params: Any = None) -> None:
        msg: Dict[str, Any] = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            msg["params"] = params
        self._send(msg)

    #: How often a waiting `request` looks up to check `abort` and the pipe.
    WAIT_SLICE = 0.1

    def request(self, method: str, params: Any = None, timeout: Optional[float] = None,
                abort: Optional[threading.Event] = None) -> Any:
        """Call the peer and block for its reply. Raises `RpcError` on failure.

        `abort` makes an untimed wait interruptible. Some requests legitimately
        have no deadline -- a permission prompt waits as long as the user takes
        to read it -- but "no deadline" must not mean "unkillable": if the turn
        is cancelled, or the peer goes away, the caller has to get control back.
        """
        with self._pending_lock:
            self._next_id += 1
            req_id = f"h{self._next_id}"
            pending = _Pending(event=threading.Event())
            self._pending[req_id] = pending

        msg: Dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
        if params is not None:
            msg["params"] = params
        self._send(msg)

        if not self._await(pending, timeout, abort):
            with self._pending_lock:
                self._pending.pop(req_id, None)
            if abort is not None and abort.is_set():
                raise RpcError(INTERNAL_ERROR, f"{method} was cancelled")
            if self._closed.is_set():
                raise RpcError(INTERNAL_ERROR, f"peer closed before replying to {method}")
            raise RpcError(INTERNAL_ERROR, f"timed out waiting for reply to {method}")
        if pending.error is not None:
            raise pending.error
        return pending.result

    def _await(self, pending: _Pending, timeout: Optional[float],
               abort: Optional[threading.Event]) -> bool:
        """Wait for `pending`, giving up on timeout, abort, or a closed pipe."""
        if timeout is None and abort is None:
            # Still bounded by the pipe: a dead peer sets `_closed`, and waking
            # to notice that is the difference between a stalled turn and a hung
            # process.
            while not pending.event.wait(self.WAIT_SLICE):
                if self._closed.is_set():
                    return False
            return True

        deadline = None if timeout is None else time.monotonic() + timeout
        while not pending.event.wait(self.WAIT_SLICE):
            if abort is not None and abort.is_set():
                return False
            if self._closed.is_set():
                return False
            if deadline is not None and time.monotonic() >= deadline:
                return False
        return True

    # ------------------------------------------------------------------- input

    def _read_loop(self) -> None:
        try:
            for line in self._rx:
                if self._closed.is_set():
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    self._accept(line)
                except Exception as exc:
                    # One bad message must not take the connection with it.
                    #
                    # This used to catch only `(OSError, ValueError)` around the
                    # whole loop, which is narrower than the ways a *syntactically
                    # valid* JSON object can be malformed. `{"method": []}` is the
                    # cheapest example: `_fast_path` does `method in FAST_PATH`,
                    # and an unhashable key raises `TypeError`, which escaped the
                    # handler and ended the loop -- permanently, because `finally`
                    # then closes the peer and strands every pending request. A
                    # single line from the editor could kill the connection and
                    # take a session's in-memory staged edits with it.
                    log(f"[jsonrpc] dropping malformed message: {exc!r}")
                    continue
        except (OSError, ValueError):
            pass
        finally:
            self._closed.set()
            self._inbox.put(None)
            # Nothing will ever answer an outstanding request now.
            with self._pending_lock:
                stranded = list(self._pending.values())
                self._pending.clear()
            for p in stranded:
                p.error = RpcError(INTERNAL_ERROR, "connection closed")
                p.event.set()

    def _accept(self, line: str) -> None:
        """Route one line. Raises on anything malformed; the caller drops it.

        The shape checks are not decoration. JSON-RPC says `method` is a string
        and `id` is a string, number or null, but nothing stops a peer sending
        `[]` for either, and both then fail deep inside code that assumes
        otherwise -- `method` at the `in` test, `id` at the dict lookup in
        `_resolve`. Rejecting them here keeps the failure at the edge, where it
        is one dropped message rather than a dead connection.
        """
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            log(f"[jsonrpc] dropping unparseable line: {exc}")
            return
        if not isinstance(msg, dict):
            return

        if "method" in msg:
            if not isinstance(msg["method"], str):
                log(f"[jsonrpc] dropping message whose method is "
                    f"{type(msg['method']).__name__}, not a string")
                return
            if self._fast_path(msg["method"]) and "id" not in msg:
                self._dispatch(msg)
            else:
                self._inbox.put(msg)
        elif "id" in msg:
            if not isinstance(msg["id"], (str, int, float, type(None))):
                log(f"[jsonrpc] dropping response whose id is "
                    f"{type(msg['id']).__name__}, which cannot identify a request")
                return
            self._resolve(msg)

    def _resolve(self, msg: dict) -> None:
        with self._pending_lock:
            pending = self._pending.pop(msg["id"], None)
        if pending is None:
            return
        # Everything from here must reach `event.set()`. The request has already
        # been taken off `_pending`, so an exception in between does not merely
        # drop a message -- it leaves the caller blocked until its timeout with
        # nothing left to answer it. A non-dict `error` is enough to do that.
        try:
            err = msg.get("error")
            if err is not None:
                if isinstance(err, dict):
                    pending.error = RpcError(err.get("code", INTERNAL_ERROR),
                                             err.get("message", "unknown error"),
                                             err.get("data"))
                else:
                    pending.error = RpcError(INTERNAL_ERROR,
                                             f"malformed error field: {err!r}")
            else:
                pending.result = msg.get("result")
        finally:
            pending.event.set()

    def _work_loop(self) -> None:
        while True:
            msg = self._inbox.get()
            if msg is None:
                return
            self._dispatch(msg)

    def _dispatch(self, msg: dict) -> None:
        method = msg.get("method", "")
        params = msg.get("params")
        is_request = "id" in msg
        try:
            result = self._handler(method, params, is_request)
            if is_request:
                self._send({"jsonrpc": "2.0", "id": msg["id"],
                            "result": result if result is not None else {}})
        except RpcError as exc:
            if is_request:
                self._send({"jsonrpc": "2.0", "id": msg["id"], "error": exc.to_json()})
            else:
                log(f"[jsonrpc] error in notification {method}: {exc.message}")
        except Exception as exc:                      # a handler bug must not kill the loop
            log(f"[jsonrpc] unhandled error in {method}:\n{traceback.format_exc()}")
            if is_request:
                self._send({"jsonrpc": "2.0", "id": msg["id"],
                            "error": RpcError(INTERNAL_ERROR, str(exc)).to_json()})

    # ----------------------------------------------------------------- control

    def start(self) -> None:
        for name, target in (("acp-reader", self._read_loop), ("acp-worker", self._work_loop)):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)

    def wait(self) -> None:
        for t in self._threads:
            t.join()

    def serve_forever(self) -> None:
        self.start()
        self.wait()

    def close(self) -> None:
        self._closed.set()
        self._inbox.put(None)

    @property
    def closed(self) -> bool:
        return self._closed.is_set()
