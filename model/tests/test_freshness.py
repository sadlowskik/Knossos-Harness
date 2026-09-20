"""Not overwriting a change nobody in this run made.

An unattended eval turn can read a file, spend a minute composing a
replacement, and write it back over an edit that arrived in between -- from the
user, a formatter, a rebase, a rerun of the test suite. The write succeeds, the
transcript records success, and the only evidence is that someone's work is
gone.

The claim: a whole-file write whose target changed since it was read is refused,
while an `edit` that still matches goes through. The second half matters as much
as the first -- a rule that blocked every write to a file the user is also
touching would make the agent useless in exactly the situation where someone is
watching it.

No torch, no network.

    pytest -q tests/test_freshness.py
"""
import pytest

from knossos.tools import EditFile, ToolCall, ToolRegistry, WriteFile
from knossos.workspace import StaleWrite, Workspace


def ws(tmp_path):
    return Workspace(tmp_path)


def test_a_write_after_an_outside_change_is_refused(tmp_path):
    (tmp_path / "a.py").write_text("def one(): ...\n", encoding="utf-8")
    w = ws(tmp_path)

    w.read("a.py")
    # The user saves the file while the model is composing.
    (tmp_path / "a.py").write_text("def one(): ...\ndef theirs(): ...\n",
                                   encoding="utf-8")

    with pytest.raises(StaleWrite):
        w.write("a.py", "def one(): ...\ndef mine(): ...\n")

    assert (tmp_path / "a.py").read_text(encoding="utf-8") == (
        "def one(): ...\ndef theirs(): ...\n"), "their edit must survive"


def test_creating_a_file_never_read_is_not_a_conflict(tmp_path):
    w = ws(tmp_path)
    w.write("new.py", "fresh\n")
    assert (tmp_path / "new.py").read_text(encoding="utf-8") == "fresh\n"


def test_consecutive_writes_by_the_agent_do_not_conflict(tmp_path):
    # Without re-baselining on write, the second write to any file would be
    # refused as somebody else's edit.
    w = ws(tmp_path)
    for content in ("one\n", "two\n", "three\n"):
        w.write("a.py", content)
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "three\n"


def test_a_file_deleted_after_being_read_is_reported_as_such(tmp_path):
    (tmp_path / "a.py").write_text("gone soon\n", encoding="utf-8")
    w = ws(tmp_path)
    w.read("a.py")
    (tmp_path / "a.py").unlink()

    with pytest.raises(StaleWrite, match="deleted"):
        w.write("a.py", "back\n")


def test_an_edit_whose_region_changed_fails_on_the_match(tmp_path):
    """`edit` needs no freshness check; this pins the reasoning."""
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    w = ws(tmp_path)
    w.read("a.py")

    (tmp_path / "a.py").write_text("x = 99\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not found"):
        w.edit("a.py", "x = 1", "x = 2")


def test_an_edit_elsewhere_in_a_changed_file_still_applies(tmp_path):
    """The other half: an outside change must not block an edit that matches.

    Refusing here would make the agent unable to work in any file the user has
    open, which is most of them.
    """
    (tmp_path / "a.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
    w = ws(tmp_path)
    w.read("a.py")

    (tmp_path / "a.py").write_text("x = 1\ny = 2\nz = 3\n", encoding="utf-8")
    w.edit("a.py", "x = 1", "x = 7")

    after = (tmp_path / "a.py").read_text(encoding="utf-8")
    assert "x = 7" in after, "the edit applied"
    assert "z = 3" in after, "their addition survived"


def test_accepting_the_current_state_clears_the_conflict(tmp_path):
    (tmp_path / "a.py").write_text("one\n", encoding="utf-8")
    w = ws(tmp_path)
    w.read("a.py")
    (tmp_path / "a.py").write_text("theirs\n", encoding="utf-8")

    assert w.conflict("a.py") is not None
    w.accept_current("a.py")
    assert w.conflict("a.py") is None
    w.write("a.py", "mine\n")


def test_a_dry_run_write_is_not_checked_against_disk(tmp_path):
    # Nothing reaches disk, so nothing can be clobbered; the staged content is
    # reviewed before it is applied.
    (tmp_path / "a.py").write_text("one\n", encoding="utf-8")
    w = Workspace(tmp_path, dry_run=True)
    w.read("a.py")
    (tmp_path / "a.py").write_text("theirs\n", encoding="utf-8")

    w.write("a.py", "staged\n")
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "theirs\n"


def test_a_rewound_file_can_be_written_again(tmp_path):
    # rewind changes files behind the check's back; without re-baselining every
    # write after an undo would be refused.
    (tmp_path / "a.py").write_text("original\n", encoding="utf-8")
    w = ws(tmp_path)

    w.checkpoint("turn")
    w.read("a.py")
    w.write("a.py", "attempt\n")
    w.rewind("turn")

    assert w.conflict("a.py") is None
    w.write("a.py", "second attempt\n")


def test_the_editor_buffer_is_what_is_compared_against(tmp_path):
    """Disk is not what the user is looking at.

    An unsaved buffer is the real current state, so comparing against disk
    would report a conflict for content the user is still typing and miss one
    they have only saved into the editor.
    """
    class Editor:
        def __init__(self):
            self.content = "from the buffer\n"

        def read_text_file(self, path):
            return self.content

        def write_text_file(self, path, content):
            return False

    (tmp_path / "a.py").write_text("on disk\n", encoding="utf-8")
    editor = Editor()
    w = Workspace(tmp_path, editor=editor)

    assert w.read("a.py") == "from the buffer\n"
    assert w.conflict("a.py") is None, "nothing changed yet"

    editor.content = "the user typed more\n"
    assert w.conflict("a.py") is not None, "a buffer change is a change"


# ------------------------------------------------------------- through a tool


def test_the_write_tool_reports_a_refusal_the_agent_can_act_on(tmp_path):
    (tmp_path / "a.py").write_text("one\n", encoding="utf-8")
    w = ws(tmp_path)
    registry = ToolRegistry([WriteFile(), EditFile()])

    w.read("a.py")
    (tmp_path / "a.py").write_text("theirs\n", encoding="utf-8")

    result = registry.dispatch(
        ToolCall(name="write_file", args={"path": "a.py", "content": "mine\n"}), w)

    assert result.is_error
    assert "Read it again" in result.content, result.content
    # It has to stay a tool result, not an exception: the loop must continue so
    # the agent can re-read and try again.
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "theirs\n"
