"""What Talos is allowed to do, and how a text-only engine asks for it.

# Why prompted JSON rather than native tool calling

`Engine` is deliberately narrow: `generate(prompt, context, cancelled)` yielding
text. Most hosted providers offer a native `tools` parameter, and using it would
be more reliable -- but it would also put tool calling *inside* the engine slot,
and the slot has to hold a scaled-up Daedalus core one day. That model will emit
bytes and nothing else. An architecture whose executor only works with providers
that implement OpenAI's function-calling shape is one the from-scratch engine can
never fill.

So the protocol lives here, in the harness: tools are described in the prompt,
the engine emits fenced JSON, and `parse_calls` reads it back. Any engine that
can produce text can drive tools -- including, eventually, yours.

This is worse than native tool calling on reliability, and that is the honest
trade. `parse_calls` is written to fail closed: anything it cannot parse as a
call stays prose, so a malformed reply is a wasted turn rather than a wrong
action.

# Safety

Two properties are structural, not advisory:

  * Every path goes through `Workspace`, which refuses anything outside the root.
  * `run_command` splits a command into program and arguments and executes it
    directly. **No shell is involved**, so `&&`, `|`, `;`, backticks and
    redirection arrive at the program as literal arguments rather than being
    executed. On top of that the program must be on an allowlist.

Removing shell semantics entirely is what makes "the model appended `&& rm -rf`"
a non-event instead of something to pattern-match for.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .workspace import PathEscape, Workspace

__all__ = [
    "ToolSpec", "ToolResult", "ToolCall", "Tool", "ToolRegistry",
    "ReadFile", "WriteFile", "EditFile", "ListDir", "RunCommand",
    "parse_calls", "tokenize",
]

#: Cap on any single tool's output, so one `read_file` of a generated file
#: cannot swallow the context window.
MAX_OUTPUT = 20_000

#: How long any single command may run before being abandoned.
COMMAND_TIMEOUT = 300


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    #: JSON Schema for the arguments object.
    schema: Dict[str, Any]


@dataclass
class ToolResult:
    content: str
    is_error: bool = False
    #: Files this call created or modified. Feeds the halting policy's
    #: "did this step achieve anything" check.
    changed: List[Path] = field(default_factory=list)


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: Dict[str, Any]
    #: Where in the reply it was found, for reporting.
    raw: str = ""


class Tool:
    """A capability. Subclasses set `spec` and implement `run`."""

    spec: ToolSpec

    def run(self, args: Dict[str, Any], ws: Workspace) -> ToolResult:
        raise NotImplementedError


def _cap(text: str, note: str = "") -> str:
    if len(text) <= MAX_OUTPUT:
        return text
    tail = f"\n\n[truncated at {MAX_OUTPUT} characters{'; ' + note if note else ''}]"
    return text[:MAX_OUTPUT] + tail


def _need(args: Dict[str, Any], key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str):
        raise ValueError(f"missing required string argument {key!r}")
    return value


# --------------------------------------------------------------------- files

class ReadFile(Tool):
    spec = ToolSpec(
        name="read_file",
        description="Read a file from the workspace. Returns 1-indexed numbered lines.",
        schema={"type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative to the workspace root"},
                    "offset": {"type": "integer", "description": "1-indexed first line"},
                    "limit": {"type": "integer", "description": "Maximum lines to return"}},
                "required": ["path"]},
    )

    def run(self, args, ws):
        path = _need(args, "path")
        try:
            text = ws.read(path)
        except OSError as exc:
            return ToolResult(f"cannot read {path}: {exc}", is_error=True)

        offset = max(1, int(args.get("offset") or 1))
        limit = int(args.get("limit") or 2000)
        lines = text.splitlines()[offset - 1: offset - 1 + limit]
        if not lines:
            return ToolResult(f"{path} is empty or the offset is past its end")

        numbered = "\n".join(f"{i + offset:>6}\t{line}" for i, line in enumerate(lines))
        return ToolResult(_cap(numbered, "use offset/limit"))


class WriteFile(Tool):
    spec = ToolSpec(
        name="write_file",
        description=("Write a file, creating parent directories and replacing any existing "
                     "content. For a partial change prefer edit_file."),
        schema={"type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"]},
    )

    def run(self, args, ws):
        path = _need(args, "path")
        content = _need(args, "content")
        written = ws.write(path, content)
        verb = "staged" if ws.dry_run else "wrote"
        return ToolResult(f"{verb} {ws.display(written)} ({len(content.splitlines())} lines)",
                          changed=[written])


class EditFile(Tool):
    spec = ToolSpec(
        name="edit_file",
        description=("Replace an exact string in a file. old_string must appear exactly once "
                     "-- include surrounding context to make it unique."),
        schema={"type": "object",
                "properties": {"path": {"type": "string"},
                               "old_string": {"type": "string"},
                               "new_string": {"type": "string"}},
                "required": ["path", "old_string", "new_string"]},
    )

    def run(self, args, ws):
        path = _need(args, "path")
        try:
            edited = ws.edit(path, _need(args, "old_string"), _need(args, "new_string"))
        except (OSError, ValueError) as exc:
            # A failed edit is information the engine can act on, not a crash.
            return ToolResult(str(exc), is_error=True)
        verb = "staged edit to" if ws.dry_run else "edited"
        return ToolResult(f"{verb} {ws.display(edited)}", changed=[edited])


class ListDir(Tool):
    spec = ToolSpec(
        name="list_dir",
        description="List a directory. Directories are suffixed with /.",
        schema={"type": "object",
                "properties": {"path": {"type": "string", "description": "Defaults to the root"}}},
    )

    def run(self, args, ws):
        requested = args.get("path") or "."
        try:
            target = ws.resolve(requested)
            entries = sorted(target.iterdir(), key=lambda p: p.name)
        except (OSError, PathEscape) as exc:
            return ToolResult(f"cannot list {requested}: {exc}", is_error=True)

        if not entries:
            return ToolResult(f"{requested} is empty")
        names = [f"{e.name}/" if e.is_dir() else e.name for e in entries]
        return ToolResult(_cap("\n".join(names), "narrow the path"))


# ------------------------------------------------------------------ commands

#: Programs the executor may invoke. `python` is resolved to the running
#: interpreter so a venv is honoured and PATH lookup cannot be redirected.
ALLOWED_PROGRAMS = {"python", "pytest", "ruff", "mypy", "git"}

#: `git` subcommands that cannot modify the repository or reach the network.
GIT_READONLY = {"status", "diff", "log", "show", "ls-files", "blame", "rev-parse", "branch"}


class RunCommand(Tool):
    spec = ToolSpec(
        name="run_command",
        description=("Run a build, test or inspection command in the workspace. Allowed: "
                     "python, pytest, ruff, mypy, git (read-only subcommands). No shell is "
                     "used, so operators like && and | do not work -- issue one command."),
        schema={"type": "object",
                "properties": {"command": {"type": "string",
                                           "description": "e.g. `pytest -q tests/test_argus.py`"}},
                "required": ["command"]},
    )

    def __init__(self, allowed: Optional[Iterable[str]] = None,
                 timeout: int = COMMAND_TIMEOUT) -> None:
        self.allowed = set(allowed) if allowed is not None else set(ALLOWED_PROGRAMS)
        self.timeout = timeout

    def check(self, argv: Sequence[str]) -> Optional[str]:
        """Return a refusal reason, or None if the command may run."""
        if not argv:
            return "empty command"

        program = Path(argv[0]).name
        if program.lower().endswith(".exe"):
            program = program[:-4]
        if program not in self.allowed:
            return (f"`{program}` is not on the allowlist "
                    f"({', '.join(sorted(self.allowed))})")

        sub = next((a for a in argv[1:] if not a.startswith("-")), None)
        if program == "git":
            if sub is None:
                return "git needs a subcommand"
            if sub not in GIT_READONLY:
                return (f"git {sub} is not allowed; read-only subcommands only "
                        f"({', '.join(sorted(GIT_READONLY))})")
        return None

    def run(self, args, ws):
        argv = tokenize(_need(args, "command"))
        refusal = self.check(argv)
        if refusal:
            return ToolResult(refusal, is_error=True)

        # Resolve `python` to this interpreter rather than whatever PATH offers.
        if Path(argv[0]).name.lower().rstrip(".exe") == "python":
            argv = [sys.executable, *argv[1:]]

        try:
            proc = subprocess.run(
                argv, cwd=ws.root, capture_output=True, text=True,
                timeout=self.timeout, stdin=subprocess.DEVNULL, shell=False)
        except FileNotFoundError:
            return ToolResult(f"`{argv[0]}` is not installed", is_error=True)
        except subprocess.TimeoutExpired:
            return ToolResult(f"command timed out after {self.timeout}s", is_error=True)

        body = "\n".join(part for part in (proc.stdout, proc.stderr) if part.strip())
        body = body or "(no output)"
        text = f"exit {proc.returncode}\n\n{_cap(body)}"
        # A failing command is information, not a harness fault: the engine
        # should see the compiler or test output and react to it.
        return ToolResult(text, is_error=proc.returncode != 0)


def tokenize(command: str) -> List[str]:
    """Split a command line into argv, honouring double quotes only.

    Deliberately not `shlex`: on Windows its POSIX mode eats backslashes in
    paths, and its non-POSIX mode keeps the quotes in the token. More
    importantly this performs no expansion, substitution or operator handling --
    there is no shell here, and the tokenizer should not imply one.
    """
    out: List[str] = []
    current: List[str] = []
    quoted = False

    for ch in command:
        if ch == '"':
            quoted = not quoted
        elif ch.isspace() and not quoted:
            if current:
                out.append("".join(current))
                current = []
        else:
            current.append(ch)
    if current:
        out.append("".join(current))
    return out


# ------------------------------------------------------------------ registry

_FENCE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)

PROTOCOL = """
# Calling tools

To use a tool, emit a fenced JSON block:

```json
{"tool": "<name>", "args": { ... }}
```

Rules:
- One tool call per block. Several blocks in one reply is fine.
- `args` must satisfy that tool's schema exactly.
- When the task is complete and you need no more tools, reply in prose with no
  JSON block. Verification runs automatically at that point.
"""


class ToolRegistry:
    def __init__(self, tools: Sequence[Tool]) -> None:
        self._tools: Dict[str, Tool] = {t.spec.name: t for t in tools}

    @classmethod
    def default(cls) -> "ToolRegistry":
        return cls([ReadFile(), WriteFile(), EditFile(), ListDir(), RunCommand()])

    @property
    def names(self) -> List[str]:
        return sorted(self._tools)

    def specs(self) -> List[ToolSpec]:
        return [self._tools[n].spec for n in self.names]

    def dispatch(self, call: ToolCall, ws: Workspace) -> ToolResult:
        """Run a call. Failures become error results, never exceptions.

        An unknown tool or a bad argument is something the engine can correct on
        its next turn; raising would end the run instead.
        """
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(
                f"unknown tool `{call.name}`; available: {', '.join(self.names)}",
                is_error=True)
        try:
            return tool.run(call.args, ws)
        except PathEscape as exc:
            return ToolResult(f"refused: {exc}", is_error=True)
        except Exception as exc:                     # noqa: BLE001 - see docstring
            return ToolResult(f"{call.name} failed: {exc}", is_error=True)

    def render(self) -> str:
        """The prompt block describing every tool and the call protocol."""
        parts = ["# Available tools"]
        for spec in self.specs():
            parts.append(
                f"\n## {spec.name}\n{spec.description}\n\nArguments:\n"
                f"```json\n{json.dumps(spec.schema, indent=2)}\n```")
        parts.append(PROTOCOL)
        return "\n".join(parts)


def parse_calls(reply: str) -> Tuple[str, List[ToolCall]]:
    """Split a reply into (prose, tool calls).

    Fails closed: a fenced block that is not valid JSON, or lacks a `tool` key,
    is left in the prose rather than guessed at. A malformed reply then costs a
    turn instead of triggering the wrong action.
    """
    calls: List[ToolCall] = []
    prose_parts: List[str] = []
    cursor = 0

    for match in _FENCE.finditer(reply):
        body = match.group(1)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            continue                                  # not a call; leave as prose
        if not isinstance(payload, dict) or not isinstance(payload.get("tool"), str):
            continue
        args = payload.get("args")
        if not isinstance(args, dict):
            args = {}

        prose_parts.append(reply[cursor:match.start()])
        cursor = match.end()
        calls.append(ToolCall(name=payload["tool"], args=args, raw=body.strip()))

    prose_parts.append(reply[cursor:])
    return "".join(prose_parts).strip(), calls
