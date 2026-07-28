"""Isolation tests for the workspace boundary.

Two claims, and they are the reason this module exists at all:

  1. Nothing outside the root is reachable -- not by `../`, not by an absolute
     path, not through a symlink.
  2. In dry-run mode nothing reaches disk, and reads see staged content, so a
     multi-step preview is what would actually have happened.

Argus only reads, so the harness has not needed this. Talos will write, and
these are the properties that make that safe to allow.

No torch, no network.

    pytest -q tests/test_workspace.py
"""
import os
from pathlib import Path

import pytest

from knossos.workspace import PathEscape, Workspace


class FakeEditor:
    """An editor with unsaved buffers, and a record of what it was asked."""

    def __init__(self, buffers=None, accept_writes=True, explode=False):
        self.buffers = dict(buffers or {})
        self.accept_writes = accept_writes
        self.explode = explode
        self.reads = []
        self.writes = []

    def read_text_file(self, path):
        if self.explode:
            raise RuntimeError("editor went away")
        self.reads.append(path)
        return self.buffers.get(path.name)

    def write_text_file(self, path, content):
        if self.explode:
            raise RuntimeError("editor went away")
        self.writes.append((path, content))
        if not self.accept_writes:
            return False
        self.buffers[path.name] = content
        return True


@pytest.fixture()
def ws(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "lib.py").write_text("value = 1\n", encoding="utf-8")
    return Workspace(tmp_path)


# -------------------------------------------------------------- the editor

def test_an_unsaved_buffer_is_what_gets_read(tmp_path):
    """Disk is not what the user is looking at.

    Reading through to disk means reasoning about a version of the file the
    user cannot see, and possibly editing away changes they just made.
    """
    (tmp_path / "lib.py").write_text("value = 1\n", encoding="utf-8")
    editor = FakeEditor({"lib.py": "value = 99  # unsaved\n"})
    ws = Workspace(tmp_path, editor=editor)

    assert ws.read("lib.py") == "value = 99  # unsaved\n"


def test_the_editor_is_asked_before_disk_but_disk_still_backs_it(tmp_path):
    """A path the editor has no buffer for still resolves."""
    (tmp_path / "lib.py").write_text("value = 1\n", encoding="utf-8")
    ws = Workspace(tmp_path, editor=FakeEditor())

    assert ws.read("lib.py") == "value = 1\n"


def test_writes_go_through_the_editor(tmp_path):
    """So the change lands on the editor's undo stack, not behind its back."""
    editor = FakeEditor()
    ws = Workspace(tmp_path, editor=editor)

    ws.write("new.py", "x = 1\n")

    assert [p.name for p, _ in editor.writes] == ["new.py"]
    assert not (tmp_path / "new.py").exists(), "the editor owns the write"


def test_a_refusing_editor_falls_back_to_disk(tmp_path):
    """Losing the write would be worse than bypassing the undo stack."""
    ws = Workspace(tmp_path, editor=FakeEditor(accept_writes=False))

    ws.write("new.py", "x = 1\n")

    assert (tmp_path / "new.py").read_text(encoding="utf-8") == "x = 1\n"


def test_an_editor_that_raises_never_breaks_the_workspace(tmp_path):
    (tmp_path / "lib.py").write_text("value = 1\n", encoding="utf-8")
    ws = Workspace(tmp_path, editor=FakeEditor(explode=True))

    assert ws.read("lib.py") == "value = 1\n"
    ws.write("new.py", "x = 1\n")
    assert (tmp_path / "new.py").exists()


def test_the_jail_still_holds_with_an_editor(tmp_path):
    """Delegation changes where bytes come from, never which paths are allowed."""
    editor = FakeEditor()
    ws = Workspace(tmp_path, editor=editor)

    with pytest.raises(PathEscape):
        ws.write("../escaped.py", "pwned")

    assert editor.writes == [], "the jail must run before the editor is consulted"


def test_staged_content_outranks_the_editor(tmp_path):
    """A staged edit is this run's own proposal and nobody else has it yet."""
    (tmp_path / "lib.py").write_text("value = 1\n", encoding="utf-8")
    ws = Workspace(tmp_path, dry_run=True,
                   editor=FakeEditor({"lib.py": "value = 99\n"}))

    ws.write("lib.py", "value = 3\n")

    assert ws.read("lib.py") == "value = 3\n"


def test_undo_restores_the_buffer_not_the_saved_file(tmp_path):
    """What the user had may never have been written to disk."""
    (tmp_path / "lib.py").write_text("saved\n", encoding="utf-8")
    editor = FakeEditor({"lib.py": "unsaved\n"})
    ws = Workspace(tmp_path, editor=editor)

    ws.checkpoint("before")
    ws.write("lib.py", "agent wrote this\n")
    ws.rewind("before")

    assert editor.buffers["lib.py"] == "unsaved\n"


# ----------------------------------------------------------------- the jail

def test_a_plain_relative_path_resolves_inside(ws):
    p = ws.resolve("src/lib.py")
    assert p.is_file()
    assert str(p).startswith(str(ws.root))


def test_parent_traversal_is_refused(ws):
    for attempt in ("../secrets", "src/../../secrets", "a/b/../../../../out"):
        with pytest.raises(PathEscape):
            ws.resolve(attempt)


def test_absolute_paths_outside_the_root_are_refused(ws):
    outside = "C:\\Windows\\System32\\drivers\\etc\\hosts" if os.name == "nt" else "/etc/passwd"
    with pytest.raises(PathEscape):
        ws.resolve(outside)


def test_absolute_paths_inside_the_root_are_allowed(ws):
    assert ws.resolve(str(ws.root / "src" / "lib.py")).is_file()


def test_interior_parent_segments_that_stay_inside_are_fine(ws):
    assert ws.resolve("src/deep/../lib.py") == ws.root / "src" / "lib.py"


def test_a_path_that_does_not_exist_yet_still_resolves(ws):
    """A file about to be written has no filesystem entry to canonicalise."""
    p = ws.resolve("src/new/deeply/nested.py")
    assert not p.exists()
    assert str(p).startswith(str(ws.root))


@pytest.mark.skipif(os.name == "nt", reason="symlink creation needs privilege on Windows")
def test_a_symlink_pointing_out_of_the_tree_is_refused(ws, tmp_path):
    outside = tmp_path.parent / "outside_target"
    outside.mkdir(exist_ok=True)
    (ws.root / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(PathEscape):
        ws.resolve("escape/file.txt")


@pytest.mark.skipif(os.name == "nt", reason="symlink creation needs privilege on Windows")
def test_a_dangling_symlink_out_of_the_tree_is_refused(ws, tmp_path):
    """The case the existing test did not cover, and the one that got through.

    `exists()` follows the link, so a link to a path that does not exist yet
    reports False, the probe walks past it to a parent inside the root, and the
    jail approves a path that `write_text` will then create outside the
    workspace. The target here deliberately does not exist -- that is the whole
    point of the case.
    """
    outside = tmp_path.parent / "not_created_yet.txt"
    assert not outside.exists()
    (ws.root / "escape.txt").symlink_to(outside)

    with pytest.raises(PathEscape):
        ws.resolve("escape.txt")


@pytest.mark.skipif(os.name == "nt", reason="symlink creation needs privilege on Windows")
def test_a_write_through_a_dangling_symlink_creates_nothing_outside(ws, tmp_path):
    """The consequence, asserted where it actually matters: on the filesystem."""
    outside = tmp_path.parent / "escaped_payload.txt"
    if outside.exists():
        outside.unlink()
    (ws.root / "innocent.txt").symlink_to(outside)

    with pytest.raises(PathEscape):
        ws.write("innocent.txt", "payload")

    assert not outside.exists(), "the write followed the link out of the workspace"


@pytest.mark.skipif(os.name == "nt", reason="symlink creation needs privilege on Windows")
def test_a_symlinked_directory_cannot_seed_an_outside_tree(ws, tmp_path):
    """`mkdir(parents=True)` through a dangling directory link builds the lot."""
    outside = tmp_path.parent / "outside_tree"
    (ws.root / "sub").symlink_to(outside, target_is_directory=True)

    with pytest.raises(PathEscape):
        ws.resolve("sub/a/b/c.txt")

    assert not outside.exists()


@pytest.mark.skipif(os.name == "nt", reason="symlink creation needs privilege on Windows")
def test_a_symlink_that_stays_inside_the_tree_still_works(ws):
    """The fix must refuse escapes, not symlinks."""
    (ws.root / "inside_link.py").symlink_to(ws.root / "src" / "lib.py")

    assert ws.resolve("inside_link.py") == ws.root / "inside_link.py"


def test_the_dangling_symlink_branch_refuses_without_a_real_symlink(ws, monkeypatch):
    """Platform-independent cover for the branch above.

    Creating a symlink is privileged on Windows, so every test in this group
    skips there -- which is how the dangling case went untested long enough to
    become a bypass. This one fakes the two filesystem answers the jail depends
    on (`is_symlink` says yes, `resolve` names an outside path) and asserts the
    control flow, so the branch is covered on every platform. It does not
    replace the tests above: they are what check the real semantics.
    """
    escape = ws.root / "escape.txt"
    outside = ws.root.parent / "outside.txt"
    real_is_symlink, real_resolve = Path.is_symlink, Path.resolve

    monkeypatch.setattr(
        Path, "is_symlink",
        lambda self: self == escape or real_is_symlink(self))
    monkeypatch.setattr(
        Path, "resolve",
        lambda self, *a, **k: outside if self == escape else real_resolve(self, *a, **k))

    with pytest.raises(PathEscape, match="symlink"):
        ws.resolve("escape.txt")


def test_the_probe_still_accepts_an_ordinary_new_file(ws, monkeypatch):
    """The `is_symlink` check must not reject paths that are simply absent."""
    calls = []
    real_is_symlink = Path.is_symlink
    monkeypatch.setattr(
        Path, "is_symlink",
        lambda self: calls.append(self) or real_is_symlink(self))

    resolved = ws.resolve("src/brand/new/file.py")

    assert str(resolved).startswith(str(ws.root))
    assert calls, "the probe consulted is_symlink rather than exists alone"


# -------------------------------------------------------------------- edits

def test_edit_requires_a_unique_match(ws):
    ws.write("dup.py", "x = 1\nx = 1\n")
    with pytest.raises(ValueError, match="appears 2 times"):
        ws.edit("dup.py", "x = 1", "y = 2")


def test_edit_reports_a_missing_match(ws):
    with pytest.raises(ValueError, match="not found"):
        ws.edit("src/lib.py", "nothing like this", "x")


def test_edit_applies_a_unique_match(ws):
    ws.edit("src/lib.py", "value = 1", "value = 2")
    assert (ws.root / "src" / "lib.py").read_text(encoding="utf-8") == "value = 2\n"


# ------------------------------------------------------------------ staging

def test_a_dry_run_write_never_touches_disk(tmp_path):
    ws = Workspace(tmp_path, dry_run=True)
    ws.write("a.py", "created = True\n")
    assert not (tmp_path / "a.py").exists()
    assert ws.staged_paths() == [tmp_path.resolve() / "a.py"]


def test_staged_content_is_visible_to_later_reads_and_edits(tmp_path):
    """Without this, a previewed multi-step change is a guess, not a preview."""
    (tmp_path / "a.py").write_text("step = 0\n", encoding="utf-8")
    ws = Workspace(tmp_path, dry_run=True)

    ws.edit("a.py", "step = 0", "step = 1")
    assert ws.read("a.py") == "step = 1\n"

    ws.edit("a.py", "step = 1", "step = 2")          # chains off the first
    assert ws.read("a.py") == "step = 2\n"

    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "step = 0\n"


def test_apply_writes_everything_staged(tmp_path):
    ws = Workspace(tmp_path, dry_run=True)
    ws.write("x/a.py", "a\n")
    ws.write("x/b.py", "b\n")

    written = ws.apply()
    assert len(written) == 2
    assert (tmp_path / "x" / "a.py").read_text(encoding="utf-8") == "a\n"
    assert ws.staged_paths() == [], "staging clears after apply"


def test_apply_can_take_a_subset(tmp_path):
    """Partial review: accept one file, leave the other staged."""
    ws = Workspace(tmp_path, dry_run=True)
    ws.write("a.py", "a\n")
    ws.write("b.py", "b\n")

    written = ws.apply(only=[tmp_path.resolve() / "a.py"])
    assert [p.name for p in written] == ["a.py"]
    assert (tmp_path / "a.py").exists()
    assert not (tmp_path / "b.py").exists()
    assert [p.name for p in ws.staged_paths()] == ["b.py"]


def test_discard_leaves_nothing_behind(tmp_path):
    ws = Workspace(tmp_path, dry_run=True)
    ws.write("a.py", "nope\n")
    ws.discard()
    assert ws.staged_paths() == []
    assert not (tmp_path / "a.py").exists()


def test_the_jail_still_applies_in_dry_run(tmp_path):
    ws = Workspace(tmp_path, dry_run=True)
    with pytest.raises(PathEscape):
        ws.write("../escaped.py", "x")


def test_original_reports_none_for_a_new_file(tmp_path):
    ws = Workspace(tmp_path, dry_run=True)
    p = ws.write("fresh.py", "new\n")
    assert ws.original(p) is None

    (tmp_path / "existing.py").write_text("old\n", encoding="utf-8")
    q = ws.write("existing.py", "changed\n")
    assert ws.original(q) == "old\n"
