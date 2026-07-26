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

import pytest

from harness.workspace import PathEscape, Workspace


@pytest.fixture()
def ws(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "lib.py").write_text("value = 1\n", encoding="utf-8")
    return Workspace(tmp_path)


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
