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

import fnmatch
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from . import hooks, sandbox
from .jsonrpc import log
from .workspace import PathEscape, StaleWrite, Workspace

__all__ = [
    "ToolSpec", "ToolResult", "ToolCall", "Tool", "ToolRegistry",
    "ReadFile", "WriteFile", "EditFile", "ListDir", "Search", "RunCommand",
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
    #: The provider's own identifier for this call, when it made one.
    #:
    #: Native tool calling pairs a call with its result by id, and a provider
    #: given results it cannot match to calls will either error or quietly lose
    #: the association. Knossos normalises native calls into its fenced-block
    #: convention, which used to discard the id -- so a multi-call turn could not
    #: be sent back in the shape it arrived in.
    #:
    #: Empty when the model wrote the fenced block itself, since there was no id
    #: to keep -- and that emptiness is load-bearing rather than a gap. It is
    #: exactly the signal "this provider does not do native tool calling", so
    #: Talos uses it to decide which conversation shape to send back: an id means
    #: native `tool_calls`, no id means the text rendering that has always been
    #: used. Nothing has to negotiate a capability, and the prompted path cannot
    #: regress into a shape its model never produced.
    #:
    #: Deliberately excluded from `_signature`: two attempts at the same call get
    #: different ids and are still the same call.
    id: str = ""


class Tool:
    """A capability. Subclasses set `spec` and implement `run`."""

    spec: ToolSpec

    #: Whether several calls to this tool, or to it and its peers, may run at
    #: the same time.
    #:
    #: **Default `False`, and the default is the point.** A frontier model
    #: answers with several independent reads in one turn and expects them to
    #: overlap; running them one after another both wastes wall-clock and
    #: teaches the model to emit one call per step, which spends the step budget
    #: instead. But the tempting shortcut -- "parallelise anything not in
    #: `CONSEQUENTIAL`" -- is wrong, and `ask_user` is the counterexample: it
    #: changes nothing in the workspace and so is not consequential, yet two of
    #: them at once puts two questions to one person through one channel.
    #: Side-effect-freedom and concurrency-safety are different properties and
    #: only look alike until something asks a question.
    #:
    #: So this is opt-in per tool. A tool nobody has thought about -- including
    #: every tool arriving from an MCP server, whose behaviour is defined
    #: somewhere else entirely -- runs on its own, which is what the harness did
    #: for all of them before this existed.
    parallel_safe: bool = False

    def run(self, args: Dict[str, Any], ws: Workspace) -> ToolResult:
        raise NotImplementedError


def _cap(text: str, note: str = "", keep: str = "head") -> str:
    """Bound a tool's output. `keep` decides which end survives.

    `head` suits output the caller is reading positionally -- numbered file
    lines, a directory listing, search hits -- where the start is the answer and
    the rest is more of the same.

    `both` is for command output, and the difference is not cosmetic. A test
    runner puts its verdict at the *end*: `pytest` prints the failure summary
    and the exit line last, `mypy` its error count, a compiler its final error.
    Head-only truncation on a long run therefore drops precisely the lines the
    model needs and keeps the collection preamble it does not, which reads as a
    command that produced nothing useful.
    """
    if len(text) <= MAX_OUTPUT:
        return text
    suffix = f"; {note}" if note else ""
    if keep == "both":
        half = MAX_OUTPUT // 2
        dropped = len(text) - 2 * half
        return (f"{text[:half]}\n\n[… {dropped} characters elided from the middle"
                f"{suffix} …]\n\n{text[-half:]}")
    return text[:MAX_OUTPUT] + f"\n\n[truncated at {MAX_OUTPUT} characters{suffix}]"


def _need(args: Dict[str, Any], key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str):
        raise ValueError(f"missing required string argument {key!r}")
    return value


# --------------------------------------------------------------------- files

class ReadFile(Tool):
    # Reads consult the staging dict and then the filesystem, neither of which
    # this mutates.
    parallel_safe = True
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
    parallel_safe = True
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


# -------------------------------------------------------------------- search

#: Directories never worth walking. Without pruning, one `search` on a repo with
#: a `.git` or `node_modules` in it spends most of its time reading blobs.
SEARCH_EXCLUDE = frozenset({
    ".git", ".argus", "__pycache__", ".pytest_cache", ".mypy_cache", ".venv",
    "venv", "node_modules", "target", "build", "dist", ".ipynb_checkpoints",
})

#: Files above this are treated as data, not source, and skipped.
SEARCH_MAX_BYTES = 1_000_000

#: Cap on matches returned. A pattern like `.` matches every line in the repo;
#: the useful answer and the useless one are distinguished by stopping early.
SEARCH_MAX_MATCHES = 100


class Search(Tool):
    """Regex search across the workspace.

    The gap this fills: retrieval ran once, in the ACP layer, before the agent
    had read anything, and its result was passed unchanged on every step. The
    agent could not ask a second question. Everything it learned mid-run --
    the real name of a helper, the module something actually lives in -- had no
    way of turning into a new query. Reading files one at a time to find a
    symbol is the alternative, and it is how a step budget gets spent.

    Reads through the workspace rather than off disk, so a staged edit is
    searchable in the same turn it was made. In a dry run that is the difference
    between searching the proposal and searching the code it replaces.
    """

    parallel_safe = True
    spec = ToolSpec(
        name="search",
        description=("Search file contents with a regular expression. Returns "
                     "`path:line: text` for each match. Use this to find where "
                     "something is defined or used before reading whole files."),
        schema={"type": "object",
                "properties": {
                    "pattern": {"type": "string",
                                "description": "Python regular expression"},
                    "path": {"type": "string",
                             "description": "Subtree to search; defaults to the root"},
                    "glob": {"type": "string",
                             "description": "Filename filter, e.g. `*.py`"},
                    "ignore_case": {"type": "boolean"},
                    "max_results": {"type": "integer",
                                    "description": f"Default {SEARCH_MAX_MATCHES}"}},
                "required": ["pattern"]},
    )

    def run(self, args, ws):
        pattern = _need(args, "pattern")
        flags = re.IGNORECASE if args.get("ignore_case") else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as exc:
            return ToolResult(f"invalid regular expression {pattern!r}: {exc}",
                              is_error=True)

        requested = args.get("path") or "."
        try:
            root = ws.resolve(requested)
        except PathEscape as exc:
            return ToolResult(f"refused: {exc}", is_error=True)
        if not root.is_dir():
            return ToolResult(f"cannot search {requested}: not a directory",
                              is_error=True)

        glob = args.get("glob") or "*"
        limit = max(1, int(args.get("max_results") or SEARCH_MAX_MATCHES))

        hits: List[str] = []
        truncated = False
        for path in self._walk(root, glob):
            if len(hits) >= limit:
                truncated = True
                break
            try:
                text = ws.read(path)
            except (OSError, PathEscape):
                # Unreadable, vanished mid-walk, or somehow outside the jail.
                # One bad file must not fail the whole search.
                continue
            for number, line in enumerate(text.splitlines(), start=1):
                if len(hits) >= limit:
                    truncated = True
                    break
                if regex.search(line):
                    hits.append(f"{ws.display(path)}:{number}: {line.strip()}")

        if not hits:
            return ToolResult(f"no match for {pattern!r} under {requested}")
        note = (f"\n\n[stopped at {limit} matches; narrow the pattern, the glob "
                f"or the path]") if truncated else ""
        return ToolResult(_cap("\n".join(hits) + note))

    @staticmethod
    def _walk(root: Path, glob: str) -> Iterable[Path]:
        """Filenames under `root` matching `glob`, pruning excluded directories.

        `os.walk` with in-place pruning rather than `rglob`, so an excluded
        directory is never descended into. Filtering `rglob`'s output instead
        walks `.git` in full and then throws the results away.
        """
        for current, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in SEARCH_EXCLUDE)
            here = Path(current)
            for name in sorted(filenames):
                if not fnmatch.fnmatch(name, glob):
                    continue
                path = here / name
                try:
                    if path.stat().st_size > SEARCH_MAX_BYTES:
                        continue
                except OSError:
                    continue
                yield path


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

        # After the allowlist, never before it: which commands may run is this
        # harness's decision, and handing an unchecked argv to the editor would
        # move that decision somewhere it is not being made.
        delegated = self._via_terminal(argv, ws)
        if delegated is not None:
            code, body = delegated
            text = (f"exit {code if code is not None else 'signal'}\n\n"
                    f"{_cap(body or '(no output)', keep='both')}")
            return ToolResult(text, is_error=code != 0)

        try:
            # Through the sandbox: the allowlist admits `pytest`, which runs
            # whatever is in the workspace, so what that code can *see* is a
            # separate question from what may be started. See `sandbox`.
            proc = sandbox.DEFAULT.run(argv, cwd=ws.root, timeout=self.timeout)
        except FileNotFoundError:
            return ToolResult(f"`{argv[0]}` is not installed", is_error=True)
        except subprocess.TimeoutExpired:
            return ToolResult(f"command timed out after {self.timeout}s", is_error=True)

        body = "\n".join(part for part in (proc.stdout, proc.stderr) if part.strip())
        body = body or "(no output)"
        text = f"exit {proc.returncode}\n\n{_cap(body, keep='both')}"
        # A failing command is information, not a harness fault: the engine
        # should see the compiler or test output and react to it.
        return ToolResult(text, is_error=proc.returncode != 0)


    def _via_terminal(self, argv, ws):
        """Run in the editor's terminal, if there is one and it works."""
        terminal = getattr(ws, "terminal", None)
        if terminal is None:
            return None
        try:
            return terminal.run(argv, self.timeout)
        except Exception:                            # a broken client is not a
            return None                              # reason to skip the command


@dataclass(frozen=True)
class _Position:
    """A seed position, shaped like `lsp.Location` without importing it."""

    path: Path
    line: int
    character: int


class RenameSymbol(Tool):
    """Rename a symbol everywhere it is used, using the language server.

    The reason this is a tool rather than a suggestion to use `edit_file`: an
    agent asked to rename something will otherwise search for the name and
    replace what it finds. That edits comments, string literals, substrings of
    longer identifiers, and unrelated locals that happen to share the spelling
    -- and it misses uses the search pattern did not anticipate. The failures
    are silent and they are spread across files.

    A language server knows which occurrences *are* the symbol. Every edit still
    goes through the workspace, so the jail applies per file and each write is
    put to the permission gate exactly as a hand-written edit would be.

    Refuses when no server is running rather than degrading to text
    substitution: a rename that quietly becomes find-and-replace is worse than
    one that did not happen.
    """

    spec = ToolSpec(
        name="rename_symbol",
        description=("Rename a symbol across every file that uses it, resolved by "
                     "the language server rather than by text search. Use this for "
                     "renames instead of editing files one at a time. Requires a "
                     "running language server; returns an error if there is none."),
        schema={"type": "object",
                "properties": {
                    "symbol": {"type": "string",
                               "description": "The current name, exactly."},
                    "new_name": {"type": "string",
                                 "description": "What to call it instead."}},
                "required": ["symbol", "new_name"]},
    )

    #: Files worth scanning for a starting position, when the server cannot
    #: answer `workspace/symbol`.
    SOURCE_SUFFIXES = (".py", ".rs", ".ts", ".tsx", ".js", ".go")
    #: Cap on that scan. It only needs to find *one* position.
    SCAN_LIMIT = 4000

    def _locate(self, name, symbols, ws):
        """Find any one position of `name`, to ask the server about.

        Two strategies, because one server in common use answers neither.
        `workspace/symbol` is the direct question but pylsp does not implement
        it -- it replies `Method Not Found`, which is how this tool was found to
        be built on a capability it cannot rely on.

        The fallback scans for the identifier as a whole word. That is a text
        search, and it would be an unsafe way to *choose edits* -- but it is not
        choosing edits. It only supplies a seed position, and the server's
        reference list still decides what gets renamed. `definition` then
        normalises the seed to the declaration so references are asked for from
        the right place.
        """
        try:
            for found in symbols.workspace_symbols(name):
                if found.path.suffix:
                    return found
        except Exception:
            pass                                     # unimplemented: use the scan

        word = re.compile(rf"\b{re.escape(name)}\b")
        scanned = 0
        for path in sorted(ws.root.rglob("*")):
            if path.suffix not in self.SOURCE_SUFFIXES or not path.is_file():
                continue
            scanned += 1
            if scanned > self.SCAN_LIMIT:
                break
            try:
                lines = ws.read(path).splitlines()
            except (OSError, PathEscape):
                continue
            for number, line in enumerate(lines):
                match = word.search(line)
                if not match:
                    continue
                seed = _Position(path, number, match.start())
                try:
                    for defined in symbols.definition(path, number, match.start()):
                        if defined.path.suffix:
                            return defined
                except Exception:
                    pass
                return seed
        return None

    def run(self, args, ws):
        old = _need(args, "symbol")
        new = _need(args, "new_name")
        if old == new:
            return ToolResult("the new name is the old name; nothing to do",
                              is_error=True)
        if not new.isidentifier():
            return ToolResult(f"`{new}` is not a valid identifier", is_error=True)

        symbols = getattr(ws, "symbols", None)
        if symbols is None:
            return ToolResult(
                "no language server is running, so the uses of this symbol cannot "
                "be resolved. Do not fall back to searching for the name -- that "
                "edits comments and unrelated identifiers. Edit the specific "
                "files you can identify instead.", is_error=True)

        try:
            target = self._locate(old, symbols, ws)
        except Exception as exc:
            return ToolResult(f"the language server failed: {exc}", is_error=True)
        if target is None:
            return ToolResult(f"no declaration of `{old}` was found", is_error=True)

        try:
            # `include_declaration=True`: the definition is a use like any
            # other, and renaming every call while leaving `def total` behind
            # produces a file that does not import.
            uses = symbols.references(target.path, target.line, target.character,
                                      include_declaration=True)
        except Exception as exc:
            return ToolResult(f"the language server failed: {exc}", is_error=True)
        if not uses:
            uses = [target]
        elif not any(u.path == target.path and u.line == target.line for u in uses):
            uses = list(uses) + [target]             # server omitted the declaration

        # Group by file and apply bottom-up: editing a later line first keeps
        # every earlier line number valid, so no offset bookkeeping is needed.
        by_file: Dict[Path, List[Any]] = {}
        for use in uses:
            by_file.setdefault(use.path, []).append(use)

        changed: List[Path] = []
        skipped: List[str] = []
        for path, locations in sorted(by_file.items()):
            try:
                resolved = ws.resolve(path)
                lines = ws.read(resolved).splitlines(keepends=True)
            except (OSError, PathEscape) as exc:
                skipped.append(f"{path}: {exc}")
                continue

            edited = False
            for loc in sorted(locations, key=lambda l: (-l.line, -l.character)):
                if loc.line >= len(lines):
                    continue
                line = lines[loc.line]
                start = loc.character
                if line[start:start + len(old)] != old:
                    # The server's position and the file disagree: a stale index,
                    # or content changed underneath. Skipping is right -- writing
                    # at a position we cannot confirm is how a rename corrupts.
                    skipped.append(f"{ws.display(path)}:{loc.line + 1}: "
                                   f"`{old}` is not at the reported position")
                    continue
                lines[loc.line] = line[:start] + new + line[start + len(old):]
                edited = True

            if edited:
                ws.write(resolved, "".join(lines))
                changed.append(resolved)

        if not changed:
            detail = "; ".join(skipped) or "no usable positions were reported"
            return ToolResult(f"nothing was renamed: {detail}", is_error=True)

        body = (f"renamed `{old}` to `{new}` in {len(changed)} file(s), "
                f"{len(uses)} occurrence(s):\n"
                + "\n".join(f"  {ws.display(p)}" for p in changed))
        if skipped:
            body += "\n\nskipped:\n" + "\n".join(f"  {s}" for s in skipped)
        return ToolResult(body, changed=changed)


class AskUser(Tool):
    """Put a question to the user instead of guessing.

    Deliberately *not* consequential: asking changes nothing, and routing it
    through the permission gate would mean approving a dialog in order to see a
    dialog.
    """

    spec = ToolSpec(
        name="ask_user",
        description=("Ask the user a question when the task is genuinely ambiguous "
                     "and guessing would waste the turn. Returns their answer. If "
                     "no interface is available this returns an error -- proceed "
                     "with your best judgement rather than asking again."),
        schema={"type": "object",
                "properties": {
                    "question": {"type": "string",
                                 "description": "What you need to know, in one sentence."},
                    "choices": {"type": "array", "items": {"type": "string"},
                                "description": "Optional fixed options to choose between."}},
                "required": ["question"]},
    )

    def run(self, args, ws):
        question = _need(args, "question")
        choices = args.get("choices")
        if not isinstance(choices, list) or not all(isinstance(c, str) for c in choices):
            choices = None

        elicit = getattr(ws, "elicit", None)
        if elicit is None:
            return ToolResult(
                "there is no interface to ask the user through; proceed with "
                "your best judgement and say what you assumed", is_error=True)
        try:
            answer = elicit.ask(question, choices)
        except Exception as exc:                     # a broken client is not a
            return ToolResult(f"could not ask the user: {exc}",  # reason to raise
                              is_error=True)
        if answer is None:
            return ToolResult(
                "the user did not answer; proceed with your best judgement and "
                "say what you assumed", is_error=True)
        return ToolResult(f"The user answered: {answer}")


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
    def __init__(self, tools: Sequence[Tool],
                 hook_list: Optional[Sequence["hooks.Hook"]] = None) -> None:
        self._tools: Dict[str, Tool] = {t.spec.name: t for t in tools}
        #: Policy that runs around every call. Defaults to protecting `.git`,
        #: because the hole it closes -- `write_file` into `.git/`, which the
        #: path jail permits since `.git` is under the root -- is present in
        #: every workspace and is not something a caller should have to know to
        #: opt into.
        self._hooks: List["hooks.Hook"] = (
            [hooks.ProtectPaths()] if hook_list is None else list(hook_list))

    def with_hook(self, hook: "hooks.Hook") -> "ToolRegistry":
        """Add a policy hook. They run in the order they are added."""
        self._hooks.append(hook)
        return self

    @classmethod
    def default(cls) -> "ToolRegistry":
        return cls([ReadFile(), WriteFile(), EditFile(), ListDir(), Search(),
                    RunCommand(), RenameSymbol(), AskUser()])

    @classmethod
    def combined(cls, extra: Sequence[Tool],
                 base: Optional["ToolRegistry"] = None) -> "ToolRegistry":
        """`base` plus `extra`, with **base winning any name collision**.

        The precedence is the point. `extra` is where MCP tools arrive, and an
        MCP server is a remote process the workspace jail cannot constrain --
        `McpTool.run` says so itself. If a server could register a tool named
        `write_file`, it would shadow the jailed local one and every subsequent
        write would leave the sandbox without anything appearing to change.
        Local tools are therefore applied last and overwrite.
        """
        base = base or cls.default()
        merged: Dict[str, Tool] = {t.spec.name: t for t in extra}
        shadowed = sorted(set(merged) & set(base._tools))
        merged.update(base._tools)
        if shadowed:
            log(f"[tools] ignoring remote tool(s) shadowing local names: "
                f"{', '.join(shadowed)}")
        return cls(list(merged.values()))

    def without(self, *names: str) -> "ToolRegistry":
        """This registry minus `names`. Absent names are not an error.

        Used to hand a child agent everything the parent has except the ability
        to spawn further children -- a capability that has to be removed by
        construction rather than by asking the model not to use it.
        """
        # The hooks come too: a child agent losing the policy its parent ran
        # under would be a capability *gained* by delegation.
        return ToolRegistry([tool for name, tool in self._tools.items()
                             if name not in names], self._hooks)

    @property
    def names(self) -> List[str]:
        return sorted(self._tools)

    def specs(self) -> List[ToolSpec]:
        return [self._tools[n].spec for n in self.names]

    def parallel_safe(self, name: str) -> bool:
        """Whether this call may overlap with its neighbours.

        An unknown name is not safe: it will fail in `dispatch`, and doing that
        on its own keeps the error attributable to the call that caused it.
        """
        tool = self._tools.get(name)
        return bool(tool is not None and tool.parallel_safe)

    def dispatch(self, call: ToolCall, ws: Workspace) -> ToolResult:
        """Run a call. Failures become error results, never exceptions.

        An unknown tool or a bad argument is something the engine can correct on
        its next turn; raising would end the run instead.

        Policy hooks run around the call -- see `knossos.hooks`. There is one
        exit point on purpose: an early return for any outcome would make the
        hook contract "every call except the ones we forgot", which is not a
        contract an audit hook can be built on.
        """
        args, denial = hooks.before_chain(self._hooks, call.name, call.args)

        if denial is not None:
            result = ToolResult(denial, is_error=True)
        else:
            tool = self._tools.get(call.name)
            if tool is None:
                result = ToolResult(
                    f"unknown tool `{call.name}`; "
                    f"available: {', '.join(self.names)}",
                    is_error=True)
            else:
                try:
                    result = tool.run(args, ws)
                except PathEscape as exc:
                    result = ToolResult(f"refused: {exc}", is_error=True)
                except StaleWrite as exc:
                    # Its own arm rather than the catch-all below, because the
                    # message already says what to do about it and "write_file
                    # failed:" in front would read like a fault in the harness.
                    result = ToolResult(str(exc), is_error=True)
                except Exception as exc:             # noqa: BLE001 - see docstring
                    result = ToolResult(f"{call.name} failed: {exc}",
                                        is_error=True)

        for hook in self._hooks:
            hook.after(call.name, args, result)
        return result

    def openai_schema(self) -> List[Dict[str, Any]]:
        """The same tools in OpenAI's `tools` format.

        Sent alongside the prose protocol rather than instead of it. A model
        given the schema in the request stops inventing argument names -- the
        first live model to reach execute mode emitted `"file"` where the schema
        says `"path"`, having only ever seen the schema described in prose.
        """
        return [{"type": "function",
                 "function": {"name": spec.name,
                              "description": spec.description,
                              "parameters": spec.schema}}
                for spec in self.specs()]

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

    An optional `id` is read alongside `tool` and `args`. The engine puts the
    provider's own call id there when it normalises a native tool call into this
    convention, so the pairing survives a round trip through text. A model that
    writes the block itself supplies no id, and Talos assigns one.
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

        call_id = payload.get("id")
        prose_parts.append(reply[cursor:match.start()])
        cursor = match.end()
        calls.append(ToolCall(name=payload["tool"], args=args, raw=body.strip(),
                              id=call_id if isinstance(call_id, str) else ""))

    prose_parts.append(reply[cursor:])
    return "".join(prose_parts).strip(), calls
