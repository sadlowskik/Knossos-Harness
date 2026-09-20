"""Isolation tests for the tool layer.

The claims that matter, in order:

  1. `run_command` cannot be talked into running a shell. Operators arrive as
     literal argv entries, and the program must be on an allowlist.
  2. `parse_calls` fails closed -- anything it cannot read as a call stays prose,
     so a malformed reply wastes a turn rather than triggering a wrong action.
  3. A tool that fails returns an error result the engine can read, rather than
     raising and ending the run.

No torch, no network, no model.

    pytest -q tests/test_tools.py
"""
import json

import pytest

from knossos.tools import RenameSymbol


class FakeSymbols:
    """Stands in for a language server, returning fixed positions."""

    def __init__(self, declarations=(), uses=(), explode=False):
        self._declarations = list(declarations)
        self._uses = list(uses)
        self.explode = explode

    def workspace_symbols(self, query=""):
        if self.explode:
            raise RuntimeError("server died")
        return list(self._declarations)

    def references(self, path, line, character, include_declaration=True):
        if self.explode:
            raise RuntimeError("server died")
        return list(self._uses)


class At:
    """A minimal `lsp.Location`."""

    def __init__(self, path, line, character=0):
        self.path, self.line, self.character = path, line, character

from knossos.tools import (EditFile, ListDir, ReadFile, RunCommand, Search,
                           ToolCall, ToolRegistry, WriteFile, parse_calls,
                           tokenize)
from knossos.workspace import Workspace


@pytest.fixture()
def ws(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "lib.py").write_text("value = 1\n", encoding="utf-8")
    return Workspace(tmp_path)


# ------------------------------------------------------------ rename_symbol

@pytest.fixture()
def project(tmp_path):
    """Two files using `total`, plus every trap a text search falls into."""
    (tmp_path / "a.py").write_text(
        "def total(xs):\n"
        "    return sum(xs)\n"
        "# total is documented here\n"
        'LABEL = "total"\n'
        "subtotal = 1\n",
        encoding="utf-8")
    (tmp_path / "b.py").write_text(
        "from a import total\n"
        "print(total([1, 2]))\n",
        encoding="utf-8")
    return tmp_path


def rename(ws, symbol="total", new_name="summed"):
    return RenameSymbol().run({"symbol": symbol, "new_name": new_name}, ws)


def test_a_rename_touches_only_real_uses(project):
    """The whole reason this is a tool: a text search would also hit the
    comment, the string literal, and `subtotal`."""
    a, b = project / "a.py", project / "b.py"
    symbols = FakeSymbols(
        declarations=[At(a, 0, 4)],
        uses=[At(a, 0, 4), At(b, 0, 14), At(b, 1, 6)])
    ws = Workspace(project, symbols=symbols)

    result = rename(ws)

    assert not result.is_error, result.content
    text_a = a.read_text(encoding="utf-8")
    assert text_a.startswith("def summed(xs):")
    assert "# total is documented here" in text_a, "a comment is not a use"
    assert 'LABEL = "total"' in text_a, "a string literal is not a use"
    assert "subtotal = 1" in text_a, "a longer identifier is not a use"
    assert b.read_text(encoding="utf-8") == "from a import summed\nprint(summed([1, 2]))\n"


def test_every_touched_file_is_reported_as_changed(project):
    """So the permission gate and the verifier both see the real blast radius."""
    a, b = project / "a.py", project / "b.py"
    ws = Workspace(project, symbols=FakeSymbols(
        declarations=[At(a, 0, 4)], uses=[At(a, 0, 4), At(b, 1, 6)]))

    result = rename(ws)

    assert {p.name for p in result.changed} == {"a.py", "b.py"}


def test_several_uses_on_one_line_all_move(project):
    """Applied bottom-up, so earlier positions stay valid as the line grows."""
    (project / "c.py").write_text("total = total + total\n", encoding="utf-8")
    c = project / "c.py"
    ws = Workspace(project, symbols=FakeSymbols(
        declarations=[At(c, 0, 0)],
        uses=[At(c, 0, 0), At(c, 0, 8), At(c, 0, 16)]))

    rename(ws)

    assert c.read_text(encoding="utf-8") == "summed = summed + summed\n"


def test_a_stale_position_is_skipped_not_written(project):
    """If the server and the file disagree, writing anyway is how a rename
    corrupts a file."""
    a = project / "a.py"
    ws = Workspace(project, symbols=FakeSymbols(
        declarations=[At(a, 0, 4)], uses=[At(a, 0, 4), At(a, 1, 0)]))

    result = rename(ws)

    assert "not at the reported position" in result.content
    assert a.read_text(encoding="utf-8").splitlines()[1] == "    return sum(xs)"


def test_without_a_language_server_it_refuses(project):
    """Degrading to find-and-replace would be worse than not renaming."""
    result = rename(Workspace(project))

    assert result.is_error
    assert "no language server" in result.content
    assert "Do not fall back to searching" in result.content


def test_a_dead_server_is_an_error_not_a_crash(project):
    result = rename(Workspace(project, symbols=FakeSymbols(explode=True)))

    assert result.is_error and "language server failed" in result.content


def test_an_unknown_symbol_is_reported(project):
    """A name that appears nowhere -- not merely one the server cannot resolve.

    `_locate` falls back to a whole-word text scan when `workspace/symbol` is
    unimplemented, so a symbol present in the source is locatable even from a
    server that answers nothing. Asking about `total` here would exercise the
    fallback and rename it; only a name absent from every file reaches the
    "no declaration" branch.
    """
    result = rename(Workspace(project, symbols=FakeSymbols()),
                    symbol="nowhere_at_all")

    assert result.is_error and "no declaration" in result.content


@pytest.mark.parametrize("new_name", ["not an identifier", "2bad", ""])
def test_an_invalid_new_name_is_refused(project, new_name):
    a = project / "a.py"
    ws = Workspace(project, symbols=FakeSymbols(declarations=[At(a, 0, 4)]))

    assert rename(ws, new_name=new_name).is_error


def test_renaming_to_the_same_name_is_refused(project):
    a = project / "a.py"
    ws = Workspace(project, symbols=FakeSymbols(declarations=[At(a, 0, 4)]))

    assert rename(ws, new_name="total").is_error


def test_the_jail_holds_for_a_rename(tmp_path):
    """A server reporting a path outside the workspace must not be followed."""
    inside = tmp_path / "work"
    inside.mkdir()
    (inside / "a.py").write_text("total = 1\n", encoding="utf-8")
    outside = tmp_path / "elsewhere.py"
    outside.write_text("total = 1\n", encoding="utf-8")

    ws = Workspace(inside, symbols=FakeSymbols(
        declarations=[At(inside / "a.py", 0, 0)],
        uses=[At(inside / "a.py", 0, 0), At(outside, 0, 0)]))

    result = rename(ws)

    assert outside.read_text(encoding="utf-8") == "total = 1\n", "jail must hold"
    assert (inside / "a.py").read_text(encoding="utf-8") == "summed = 1\n"
    assert "escapes the workspace" in result.content


def test_a_dry_run_stages_the_rename(project):
    a, b = project / "a.py", project / "b.py"
    ws = Workspace(project, dry_run=True, symbols=FakeSymbols(
        declarations=[At(a, 0, 4)], uses=[At(a, 0, 4), At(b, 1, 6)]))

    rename(ws)

    assert len(ws.staged_paths()) == 2
    assert "summed" not in a.read_text(encoding="utf-8"), "nothing on disk yet"


# ------------------------------------------------------------------ no shell

def test_shell_operators_are_not_interpreted():
    """They become literal argv entries. Nothing executes them."""
    argv = tokenize("pytest -q && rm -rf /")
    assert argv[0] == "pytest"
    assert "&&" in argv, "the operator must survive as a plain argument"


def test_tokenizer_keeps_quoted_arguments_together():
    assert tokenize('pytest -k "my test name"') == ["pytest", "-k", "my test name"]


def test_tokenizer_keeps_windows_paths_intact():
    """shlex's POSIX mode would eat these backslashes."""
    assert tokenize(r"python C:\tools\run.py") == ["python", r"C:\tools\run.py"]


def test_allowlist_rejects_arbitrary_programs():
    run = RunCommand()
    assert run.check(["rm", "-rf", "."]) is not None
    assert run.check(["powershell", "-c", "x"]) is not None
    assert run.check(["curl", "http://x"]) is not None
    assert run.check(["pytest", "-q"]) is None


def test_git_is_limited_to_readonly_subcommands():
    run = RunCommand()
    assert run.check(["git", "status"]) is None
    assert run.check(["git", "diff"]) is None
    assert run.check(["git", "push"]) is not None
    assert run.check(["git", "reset", "--hard"]) is not None
    assert run.check(["git"]) is not None


def test_a_refused_command_is_an_error_result_not_an_exception(ws):
    out = RunCommand().run({"command": "rm -rf ."}, ws)
    assert out.is_error
    assert "not on the allowlist" in out.content


def test_an_allowed_command_actually_runs(ws):
    out = RunCommand().run({"command": 'python -c "print(6*7)"'}, ws)
    assert not out.is_error, out.content
    assert "42" in out.content
    assert out.content.startswith("exit 0")


def test_a_failing_command_is_reported_not_hidden(ws):
    out = RunCommand().run({"command": 'python -c "raise SystemExit(3)"'}, ws)
    assert out.is_error
    assert "exit 3" in out.content


# --------------------------------------------------------------- parse_calls

def test_a_well_formed_call_is_extracted():
    prose, calls = parse_calls(
        'Let me look.\n```json\n{"tool": "read_file", "args": {"path": "a.py"}}\n```')
    assert [c.name for c in calls] == ["read_file"]
    assert calls[0].args["path"] == "a.py"
    assert prose == "Let me look."


def test_several_calls_in_one_reply():
    reply = ('```json\n{"tool":"read_file","args":{"path":"a"}}\n```\n'
             '```json\n{"tool":"read_file","args":{"path":"b"}}\n```')
    _, calls = parse_calls(reply)
    assert [c.args["path"] for c in calls] == ["a", "b"]


def test_a_plain_json_block_stays_prose():
    """A code block that is not a call must not be guessed at."""
    prose, calls = parse_calls('Config looks like:\n```json\n{"debug": true}\n```')
    assert calls == []
    assert "debug" in prose


def test_malformed_json_stays_prose():
    prose, calls = parse_calls('```json\n{"tool": "read_file", oops\n```')
    assert calls == []
    assert "oops" in prose


def test_a_call_without_args_gets_an_empty_dict():
    _, calls = parse_calls('```json\n{"tool": "list_dir"}\n```')
    assert calls[0].args == {}


def test_a_reply_with_no_blocks_is_all_prose():
    prose, calls = parse_calls("I think we are done here.")
    assert calls == []
    assert prose == "I think we are done here."


# --------------------------------------------------------------------- search

@pytest.fixture()
def repo(tmp_path):
    """A small tree, plus the directories a search must never descend into."""
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "core.py").write_text(
        "def handle_retry(n):\n    return n\n", encoding="utf-8")
    (tmp_path / "pkg" / "util.py").write_text(
        "from .core import handle_retry\n\nhandle_retry(3)\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("handle_retry is the entry point\n",
                                       encoding="utf-8")
    junk = tmp_path / "node_modules" / "dep"
    junk.mkdir(parents=True)
    (junk / "bundle.py").write_text("handle_retry\n" * 500, encoding="utf-8")
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "COMMIT_EDITMSG").write_text("handle_retry\n",
                                                      encoding="utf-8")
    return Workspace(tmp_path)


def search(ws, **args):
    return Search().run(args, ws)


def test_search_finds_a_symbol_across_files(repo):
    """The gap this fills: without it the agent reads files one at a time."""
    out = search(repo, pattern=r"def handle_retry")

    assert not out.is_error
    assert "core.py:1:" in out.content
    assert "def handle_retry(n):" in out.content


def test_search_reports_every_use_with_line_numbers(repo):
    out = search(repo, pattern=r"handle_retry", glob="*.py")

    lines = out.content.splitlines()
    assert any(line.startswith(("pkg\\util.py:1:", "pkg/util.py:1:"))
               for line in lines), out.content
    assert any(":3:" in line for line in lines)


def test_search_never_descends_into_excluded_directories(repo):
    """`node_modules` holds 500 matches. Finding none of them is the point."""
    out = search(repo, pattern=r"handle_retry")

    assert "node_modules" not in out.content
    assert "COMMIT_EDITMSG" not in out.content


def test_search_honours_the_glob(repo):
    out = search(repo, pattern=r"handle_retry", glob="*.md")

    assert "notes.md" in out.content
    assert "core.py" not in out.content


def test_search_is_scoped_to_a_subtree(repo):
    out = search(repo, pattern=r"handle_retry", path="pkg")

    assert "notes.md" not in out.content
    assert "core.py" in out.content


def test_search_sees_staged_content_in_a_dry_run(tmp_path):
    """Otherwise a dry run searches the code the proposal replaces."""
    (tmp_path / "a.py").write_text("old_name = 1\n", encoding="utf-8")
    ws = Workspace(tmp_path, dry_run=True)
    ws.write("a.py", "new_name = 1\n")

    assert "new_name" in search(ws, pattern="new_name").content
    assert "no match" in search(ws, pattern="old_name").content


def test_search_cannot_leave_the_workspace(repo):
    out = search(repo, pattern="anything", path="../..")

    assert out.is_error
    assert "refused" in out.content


def test_a_bad_regex_is_an_error_the_engine_can_read(repo):
    """Not an exception: the engine fixes it on the next step."""
    out = search(repo, pattern="handle_retry(")

    assert out.is_error
    assert "invalid regular expression" in out.content


def test_search_stops_rather_than_returning_the_whole_repository(repo):
    out = search(repo, pattern=".", max_results=3)

    assert len([ln for ln in out.content.splitlines() if ":" in ln]) <= 3
    assert "stopped at 3 matches" in out.content


def test_hitting_the_limit_exactly_is_not_reported_as_truncation(repo):
    """Nothing was dropped, so saying otherwise would send the agent re-querying."""
    everything = search(repo, pattern=".", max_results=1000)
    total = len(everything.content.splitlines())

    out = search(repo, pattern=".", max_results=total)

    assert "stopped at" not in out.content


def test_no_match_says_so_rather_than_returning_nothing(repo):
    out = search(repo, pattern="definitely_not_present")

    assert not out.is_error
    assert "no match" in out.content


def test_search_is_registered_by_default():
    assert "search" in ToolRegistry.default().names


# ------------------------------------------------------------------ registry

def test_unknown_tool_is_an_error_result(ws):
    out = ToolRegistry.default().dispatch(ToolCall("nope", {}), ws)
    assert out.is_error
    assert "unknown tool" in out.content


def test_a_path_escape_is_refused_through_the_registry(ws):
    out = ToolRegistry.default().dispatch(
        ToolCall("write_file", {"path": "../escaped.py", "content": "x"}), ws)
    assert out.is_error
    assert "refused" in out.content
    assert not (ws.root.parent / "escaped.py").exists()


def test_a_missing_argument_is_an_error_result(ws):
    out = ToolRegistry.default().dispatch(ToolCall("read_file", {}), ws)
    assert out.is_error
    assert "path" in out.content


def test_render_documents_every_tool_and_the_protocol():
    rendered = ToolRegistry.default().render()
    for name in ToolRegistry.default().names:
        assert f"## {name}" in rendered
    assert "Calling tools" in rendered
    assert '"tool"' in rendered


def test_every_spec_schema_is_valid_json():
    for spec in ToolRegistry.default().specs():
        json.dumps(spec.schema)                      # raises if not serialisable
        assert spec.schema["type"] == "object"
        assert spec.description


# --------------------------------------------------------------------- files

def test_read_returns_numbered_lines(ws):
    out = ReadFile().run({"path": "src/lib.py"}, ws)
    assert not out.is_error
    assert "     1\tvalue = 1" in out.content


def test_write_reports_the_change(ws):
    out = WriteFile().run({"path": "src/new.py", "content": "x = 1\n"}, ws)
    assert not out.is_error
    assert len(out.changed) == 1
    assert (ws.root / "src" / "new.py").exists()


def test_edit_refuses_an_ambiguous_match(ws):
    ws.write("dup.py", "a = 1\na = 1\n")
    out = EditFile().run({"path": "dup.py", "old_string": "a = 1", "new_string": "b = 2"}, ws)
    assert out.is_error
    assert "appears 2 times" in out.content


def test_list_dir_marks_directories(ws):
    out = ListDir().run({"path": "."}, ws)
    assert "src/" in out.content


def test_tools_stage_rather_than_write_in_dry_run(tmp_path):
    dry = Workspace(tmp_path, dry_run=True)
    out = WriteFile().run({"path": "a.py", "content": "staged = True\n"}, dry)
    assert "staged" in out.content
    assert not (tmp_path / "a.py").exists()
    assert len(dry.staged_paths()) == 1
