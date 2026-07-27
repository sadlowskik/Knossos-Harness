"""Tests for the workspace undo journal.

Staging protects a dry run. Once `apply` has written, only a journal can put
things back -- and that is exactly when it matters, because a bad turn has
already touched the disk.

The claims that have to hold:

  1. Rewinding restores content, not just "something".
  2. A file the agent *created* is deleted on rewind, not left behind empty.
     Restoring "" would leave a plausible-looking wrong file in the tree.
  3. A path written several times lands on its state at the mark, not on an
     intermediate version.
  4. Files the agent never touched are never touched by the rewind either.

    pytest -q tests/test_checkpoint.py
"""
import pytest

from knossos.workspace import Workspace


@pytest.fixture()
def ws(tmp_path):
    (tmp_path / "kept.py").write_text("original kept\n", encoding="utf-8")
    (tmp_path / "edited.py").write_text("original edited\n", encoding="utf-8")
    return Workspace(tmp_path)


def test_rewind_restores_previous_content(ws, tmp_path):
    ws.checkpoint("before")
    ws.write("edited.py", "changed\n")
    assert (tmp_path / "edited.py").read_text() == "changed\n"

    restored = ws.rewind("before")
    assert (tmp_path / "edited.py").read_text() == "original edited\n"
    assert (tmp_path / "edited.py") in restored


def test_a_created_file_is_deleted_not_blanked(ws, tmp_path):
    """`before is None` means it did not exist.

    Restoring empty content would leave a plausible-looking wrong file behind,
    which is harder to notice than a missing one.
    """
    ws.checkpoint("before")
    ws.write("brand_new.py", "invented\n")
    assert (tmp_path / "brand_new.py").exists()

    ws.rewind("before")
    assert not (tmp_path / "brand_new.py").exists()


def test_repeated_writes_rewind_to_the_mark_not_an_intermediate(ws, tmp_path):
    ws.checkpoint("before")
    for text in ("first\n", "second\n", "third\n"):
        ws.write("edited.py", text)

    ws.rewind("before")
    assert (tmp_path / "edited.py").read_text() == "original edited\n"


def test_untouched_files_are_left_alone(ws, tmp_path):
    ws.checkpoint("before")
    ws.write("edited.py", "changed\n")
    ws.rewind("before")
    assert (tmp_path / "kept.py").read_text() == "original kept\n"


def test_marks_are_independent(ws, tmp_path):
    ws.write("edited.py", "one\n")
    ws.checkpoint("after_one")
    ws.write("edited.py", "two\n")

    ws.rewind("after_one")
    assert (tmp_path / "edited.py").read_text() == "one\n"


def test_rewinding_discards_later_marks(ws):
    ws.checkpoint("early")
    ws.write("edited.py", "a\n")
    ws.checkpoint("late")
    ws.write("edited.py", "b\n")

    ws.rewind("early")
    # "late" pointed into history that no longer exists; keeping it would let a
    # caller rewind to a state that was never restorable.
    with pytest.raises(KeyError):
        ws.rewind("late")


def test_an_unknown_mark_is_an_error_not_a_silent_no_op(ws):
    with pytest.raises(KeyError):
        ws.rewind("never_taken")


def test_the_journal_records_what_was_replaced(ws):
    ws.write("edited.py", "changed\n")
    ws.write("brand_new.py", "new\n")

    entries = {e.path.name: e.before for e in ws.journal}
    assert entries["edited.py"] == "original edited\n"
    assert entries["brand_new.py"] is None


def test_applying_staged_changes_is_journalled(tmp_path):
    """Dry-run then apply must be as undoable as a direct write."""
    (tmp_path / "f.py").write_text("before\n", encoding="utf-8")
    ws = Workspace(tmp_path, dry_run=True)
    ws.checkpoint("mark")
    ws.write("f.py", "after\n")
    assert (tmp_path / "f.py").read_text() == "before\n", "dry run must not write"

    ws.apply()
    assert (tmp_path / "f.py").read_text() == "after\n"

    ws.rewind("mark")
    assert (tmp_path / "f.py").read_text() == "before\n"


def test_rewind_with_nothing_written_changes_nothing(ws, tmp_path):
    ws.checkpoint("before")
    assert ws.rewind("before") == []
    assert (tmp_path / "edited.py").read_text() == "original edited\n"
