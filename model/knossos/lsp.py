"""An LSP client: exact relationships between files.

tree-sitter made Argus exact about what each file *defines*. It still cannot say
what *uses* it -- `importers_of` matches module paths by name, which is a guess
that works until two crates both have a `config` module. A language server
already knows the answer: rust-analyzer, pyright and the rest maintain a real
cross-file index with type resolution.

So this asks them. Three queries carry almost all the value:

    workspace/symbol            every symbol in the project, exactly
    textDocument/references     who uses this, across files
    textDocument/definition     where does this actually come from

LSP is JSON-RPC like ACP and MCP, but **framed differently**: `Content-Length`
headers rather than one message per line. That single difference is why this
cannot reuse `jsonrpc.Peer`, and getting it wrong produces a client that
appears to hang -- the server is waiting for a header that never arrives.

Everything here is optional and failure-tolerant. A language server that is not
installed, is slow to index, or does not implement a capability must degrade to
what Argus already does rather than break retrieval. `available()` says whether
anything is actually connected.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse
from urllib.request import pathname2url

from .jsonrpc import log

__all__ = ["LspClient", "Location", "path_to_uri", "uri_to_path"]

DEFAULT_TIMEOUT = 30.0
#: Servers index in the background and answer emptily until they are ready.
#: An empty answer is indistinguishable from "no references", so give it a
#: moment before believing one.
INDEX_SETTLE = 2.0


def path_to_uri(path: Path) -> str:
    return "file:" + pathname2url(str(Path(path).resolve()))


def uri_to_path(uri: str) -> Path:
    parsed = urlparse(uri)
    raw = unquote(parsed.path)
    # Windows: file:///C:/x -> /C:/x, and the leading slash must go.
    if os.name == "nt" and raw.startswith("/") and len(raw) > 2 and raw[2] == ":":
        raw = raw[1:]
    return Path(raw)


@dataclass(frozen=True)
class Location:
    """A place in the project. Lines are 0-based, as LSP sends them."""

    path: Path
    line: int
    character: int = 0

    @property
    def ref(self) -> str:
        return f"{self.path}:{self.line + 1}"

    @classmethod
    def from_lsp(cls, raw: Dict[str, Any]) -> Optional["Location"]:
        uri = raw.get("uri") or raw.get("targetUri")
        rng = raw.get("range") or raw.get("targetSelectionRange") or raw.get("targetRange")
        if not uri or not rng:
            return None
        start = rng.get("start") or {}
        return cls(uri_to_path(uri), int(start.get("line", 0)),
                   int(start.get("character", 0)))


#: Language servers worth trying, per file extension, cheapest first. Each entry
#: is the argv to launch. Nothing is installed on the user's behalf: an absent
#: server means the exact-reference tools report that they are unavailable,
#: which is a better answer than guessing at symbol positions.
#: `[sys.executable, "-m", ...]` entries matter more than they look. `pip install
#: python-lsp-server` puts `pylsp.exe` in a Scripts directory that is frequently
#: not on PATH -- the default on Windows -- so a `shutil.which("pylsp")` check
#: reports "no language server" for a server that is installed and importable.
#: Observed on this machine, where it would have made `rename_symbol` refuse
#: forever with a message blaming the user's setup.
SERVERS: Dict[str, List[List[str]]] = {
    ".py": [["pyright-langserver", "--stdio"],
            ["pylsp"],
            [sys.executable, "-m", "pylsp"],
            ["jedi-language-server"],
            [sys.executable, "-m", "jedi_language_server"]],
    ".rs": [["rust-analyzer"]],
    ".ts": [["typescript-language-server", "--stdio"]],
    ".go": [["gopls"]],
}


def _launchable(argv: List[str]) -> bool:
    """Whether this command can actually be started.

    A `-m` form is available when the module imports, which is the question
    that matters -- not whether someone put a wrapper script on PATH.
    """
    if len(argv) >= 3 and argv[1] == "-m":
        return importlib.util.find_spec(argv[2]) is not None
    return shutil.which(argv[0]) is not None


def for_workspace(root: str | Path, timeout: float = DEFAULT_TIMEOUT
                  ) -> Optional["LspClient"]:
    """Start a language server suited to what is actually in `root`.

    Picks by counting extensions rather than by configuration, so a mixed tree
    gets the server for its majority language and a tree with no recognised
    source gets nothing. Returns None -- never raises -- when no candidate is
    installed or the handshake fails: exact references are an enhancement, and
    an agent that cannot start without one is worse than one that says so.
    """
    root = Path(root).resolve()
    counts: Dict[str, int] = {}
    for suffix in SERVERS:
        # `rglob` on a large tree is slow; stop as soon as the answer is clear.
        found = 0
        for _ in root.rglob(f"*{suffix}"):
            found += 1
            if found >= 25:
                break
        if found:
            counts[suffix] = found
    if not counts:
        return None

    suffix = max(counts, key=lambda s: counts[s])
    for argv in SERVERS[suffix]:
        if not _launchable(argv):
            continue
        client = LspClient(argv, root, timeout=timeout)
        try:
            if client.start():
                log(f"[lsp] {argv[0]} for {suffix} in {root}")
                return client
        except Exception as exc:
            log(f"[lsp] {argv[0]} failed to start: {exc}")
        client.stop()
    log(f"[lsp] no language server available for {suffix}")
    return None


class LspClient:
    """One language server, spoken to over Content-Length framed JSON-RPC."""

    def __init__(self, command: List[str], root: str | Path,
                 timeout: float = DEFAULT_TIMEOUT) -> None:
        self.command = command
        self.root = Path(root).resolve()
        self.timeout = timeout
        self.process: Optional[subprocess.Popen] = None
        self.capabilities: Dict[str, Any] = {}
        self._id = 0
        self._pending: Dict[int, Any] = {}
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._ready = False

    # ---------------------------------------------------------------- lifecycle

    def start(self) -> bool:
        """Launch and initialise. Returns False instead of raising.

        A missing language server is a normal condition -- most machines do not
        have every one installed -- so it must not be an exception that callers
        have to guard.
        """
        try:
            self.process = subprocess.Popen(
                self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, bufsize=0,
            )
        except (OSError, ValueError) as exc:
            log(f"[lsp] {self.command[0]} did not start: {exc}")
            return False

        threading.Thread(target=self._read_loop, daemon=True).start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()

        try:
            result = self.request("initialize", {
                "processId": os.getpid(),
                "rootUri": path_to_uri(self.root),
                "workspaceFolders": [
                    {"uri": path_to_uri(self.root), "name": self.root.name}
                ],
                "capabilities": {
                    "workspace": {"symbol": {"dynamicRegistration": False}},
                    "textDocument": {
                        "references": {"dynamicRegistration": False},
                        "definition": {"dynamicRegistration": False},
                    },
                },
            })
        except Exception as exc:
            log(f"[lsp] initialize failed: {exc}")
            self.stop()
            return False

        self.capabilities = (result or {}).get("capabilities", {})
        self.notify("initialized", {})
        self._ready = True
        return True

    def available(self) -> bool:
        return self._ready and self.process is not None and self.process.poll() is None

    def stop(self) -> None:
        self._ready = False
        if self.process is None:
            return
        try:
            self.notify("exit", None)
        except Exception:
            pass
        try:
            self.process.terminate()
            self.process.wait(timeout=5)
        except Exception:
            try:
                self.process.kill()
            except Exception:
                pass

    # ------------------------------------------------------------------ queries

    def workspace_symbols(self, query: str = "") -> List[Location]:
        """Every symbol matching `query`. An empty query means all of them."""
        if not self.available():
            return []
        try:
            raw = self.request("workspace/symbol", {"query": query}) or []
        except Exception as exc:
            log(f"[lsp] workspace/symbol failed: {exc}")
            return []
        out = []
        for item in raw if isinstance(raw, list) else []:
            location = item.get("location") or item
            parsed = Location.from_lsp(location)
            if parsed:
                out.append(parsed)
        return out

    def references(self, path: Path, line: int, character: int,
                   include_declaration: bool = False) -> List[Location]:
        """Who uses the symbol at this position, across the whole project.

        This is the thing Argus cannot compute. `importers_of` matches module
        names and is right until two modules share one.
        """
        return self._locations("textDocument/references", path, line, character, {
            "context": {"includeDeclaration": include_declaration},
        })

    def definition(self, path: Path, line: int, character: int) -> List[Location]:
        return self._locations("textDocument/definition", path, line, character)

    def _locations(self, method: str, path: Path, line: int, character: int,
                   extra: Optional[Dict[str, Any]] = None) -> List[Location]:
        if not self.available():
            return []
        params = {
            "textDocument": {"uri": path_to_uri(path)},
            "position": {"line": line, "character": character},
        }
        params.update(extra or {})
        try:
            raw = self.request(method, params)
        except Exception as exc:
            log(f"[lsp] {method} failed: {exc}")
            return []

        if raw is None:
            return []
        items = raw if isinstance(raw, list) else [raw]
        return [loc for loc in (Location.from_lsp(i) for i in items) if loc]

    def wait_until_indexed(self, seconds: float = INDEX_SETTLE) -> None:
        """Give a background indexer a moment.

        Servers answer emptily while still indexing, and an empty answer is
        indistinguishable from "no references" -- which would silently look
        like a correct result.
        """
        time.sleep(seconds)

    # ----------------------------------------------------------------- protocol

    def request(self, method: str, params: Any) -> Any:
        with self._lock:
            self._id += 1
            request_id = self._id
            event = threading.Event()
            self._pending[request_id] = [event, None, None]

        self._send({"jsonrpc": "2.0", "id": request_id,
                    "method": method, "params": params})

        slot = self._pending[request_id]
        if not slot[0].wait(self.timeout):
            with self._lock:
                self._pending.pop(request_id, None)
            raise TimeoutError(f"{method} timed out after {self.timeout}s")

        with self._lock:
            _, result, error = self._pending.pop(request_id)
        if error is not None:
            raise RuntimeError(error.get("message", "unknown LSP error"))
        return result

    def notify(self, method: str, params: Any) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def _send(self, message: Dict[str, Any]) -> None:
        if self.process is None or self.process.stdin is None:
            return
        body = json.dumps(message).encode("utf-8")
        # Content-Length framing, not newline framing. Sending ndjson here makes
        # the server wait forever for a header.
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        with self._write_lock:
            try:
                self.process.stdin.write(header + body)
                self.process.stdin.flush()
            except (BrokenPipeError, ValueError):
                self._ready = False

    def _read_loop(self) -> None:
        stream = self.process.stdout if self.process else None
        if stream is None:
            return
        while True:
            try:
                length = self._read_header(stream)
                if length is None:
                    break
                body = stream.read(length)
                if not body:
                    break
                message = json.loads(body.decode("utf-8", errors="replace"))
            except Exception:
                break

            request_id = message.get("id")
            if request_id is not None and "method" not in message:
                with self._lock:
                    slot = self._pending.get(request_id)
                if slot is not None:
                    slot[1] = message.get("result")
                    slot[2] = message.get("error")
                    slot[0].set()
            # Server->client requests are ignored: none of the capabilities we
            # advertise require answering one.

        self._ready = False
        # Nothing will answer outstanding calls now.
        with self._lock:
            stranded = list(self._pending.values())
        for slot in stranded:
            slot[2] = {"message": "language server closed the connection"}
            slot[0].set()

    @staticmethod
    def _read_header(stream) -> Optional[int]:
        length: Optional[int] = None
        while True:
            line = stream.readline()
            if not line:
                return None
            text = line.decode("ascii", errors="replace").strip()
            if not text:                        # blank line ends the header
                return length
            if text.lower().startswith("content-length:"):
                try:
                    length = int(text.split(":", 1)[1].strip())
                except ValueError:
                    return None

    def _drain_stderr(self) -> None:
        if self.process is None or self.process.stderr is None:
            return
        for line in self.process.stderr:
            text = line.decode("utf-8", errors="replace").rstrip()
            if text:
                log(f"[lsp:{Path(self.command[0]).name}] {text}")
