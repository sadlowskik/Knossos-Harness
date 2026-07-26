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

from harness.tools import (EditFile, ListDir, ReadFile, RunCommand, ToolCall,
                           ToolRegistry, WriteFile, parse_calls, tokenize)
from harness.workspace import Workspace


@pytest.fixture()
def ws(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "lib.py").write_text("value = 1\n", encoding="utf-8")
    return Workspace(tmp_path)


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
