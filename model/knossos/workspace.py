"""The workspace: everything an executor is allowed to touch, and nothing else.

Argus only ever *reads*, so until now the harness has needed no such boundary.
Talos will write, and the moment it does, two properties have to be structural
rather than advisory:

  1. **Path jail.** Every path resolves through `Workspace.resolve`, which
     refuses anything landing outside the root -- `../` traversal, absolute
     paths, and symlinks pointing out of the tree.
  2. **Staging.** In `dry_run` mode nothing reaches disk. Reads consult the
     staging area first, so a sequence of edits to one file behaves exactly as
     it would on disk and a previewed multi-step change is what would actually
     have happened, rather than a guess about it.

Both are enforced here rather than in each tool, because a guard that every
caller has to remember is a guard that one caller will forget.

Stdlib only, like the rest of the harness.
"""
from __future__ import annotations

import time
import os
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Dict, List, Optional, Protocol, Sequence, Tuple


class PathEscape(PermissionError):
    """A tool asked for a path outside the workspace."""


class EditorFiles(Protocol):
    """The editor's view of the tree, when there is an editor.

    Disk is not what the user is looking at. An open buffer with unsaved changes
    is, and an agent that reads through to disk reasons about a version of the
    file the user cannot see. Writing back the same way puts the change on the
    editor's undo stack instead of surprising it.

    Both methods report failure rather than raising, because falling back to
    disk is always better than losing the operation.
    """

    def read_text_file(self, path: Path) -> Optional[str]:
        """The editor's content for `path`, or None if it cannot supply it."""

    def write_text_file(self, path: Path, content: str) -> bool:
        """Write through the editor. False means the caller should use disk."""


class Symbols(Protocol):
    """Exact cross-file symbol knowledge, when a language server is running.

    The difference between this and searching for a name is the difference
    between a rename that works and one that quietly edits a comment, a string
    literal, and an unrelated variable that happens to share the spelling.
    """

    def workspace_symbols(self, query: str = ""):
        """Declarations matching `query`, as `lsp.Location`s."""

    def references(self, path: Path, line: int, character: int,
                   include_declaration: bool = True):
        """Every use of the symbol at that position, across the project."""


class Elicitor(Protocol):
    """A way to put a question to the user mid-turn, when there is one.

    Without this an agent that does not know which of two things was meant has
    only one move: guess, and spend the turn finding out it guessed wrong.
    """

    def ask(self, question: str,
            choices: Optional[Sequence[str]]) -> Optional[str]:
        """The user's answer, or None if they declined or nobody could ask."""


class Terminals(Protocol):
    """The editor's terminal, when there is one.

    A build run in a hidden subprocess is invisible: the user cannot watch it,
    scroll it, or kill it. Run in the editor's own terminal it is an ordinary
    terminal they already know how to use.
    """

    def run(self, argv: Sequence[str],
            timeout: int) -> Optional[Tuple[Optional[int], str]]:
        """`(exit code, output)`, or None to fall back to a subprocess.

        The exit code may itself be None when a signal ended the process, which
        is not the same as exiting zero.
        """


@dataclass(frozen=True)
class JournalEntry:
    """What a path looked like immediately before Knossos changed it.

    `before is None` means the file did not exist, so undoing that entry means
    deleting it rather than restoring empty content -- a distinction that
    matters the first time an agent creates a file you did not want.
    """

    path: Path
    before: Optional[str]


class Workspace:
    """A rooted, optionally write-staged view of a directory tree."""

    def __init__(self, root: str | os.PathLike, dry_run: bool = False,
                 editor: Optional[EditorFiles] = None,
                 terminal: Optional[Terminals] = None,
                 elicit: Optional[Elicitor] = None,
                 symbols: Optional[Symbols] = None) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(f"workspace root is not a directory: {self.root}")
        self.dry_run = dry_run
        #: absolute path -> proposed content. Empty unless `dry_run`.
        self._staged: Dict[Path, str] = {}
        #: Set when a client advertises the ACP `fs` capabilities. The jail runs
        #: first either way -- delegation changes *where* the bytes come from,
        #: never *which paths* may be touched.
        self.editor = editor
        #: Set when a client advertises `terminal`. The allowlist still runs
        #: first: delegation changes where a command runs, never which commands
        #: are allowed to.
        self.terminal = terminal
        #: Set when a client advertises `elicitation`. `None` means there is
        #: nobody to ask, and `ask_user` says so rather than blocking.
        self.elicit = elicit
        #: A language server, when one is running. `None` means `rename_symbol`
        #: refuses rather than falling back to text substitution -- a rename
        #: that silently becomes find-and-replace is worse than no rename.
        self.symbols = symbols

    # ------------------------------------------------------------------ jail

        #: Every disk write, oldest first, with the content it replaced. This
        #: is what makes a bad turn undoable: staging protects a dry run, but
        #: once `apply` has written, only a journal can put it back.
        self._journal: List[JournalEntry] = []
        #: label -> journal length when the mark was taken.
        self._marks: Dict[str, int] = {}

    # ------------------------------------------------------------ checkpoints

    def checkpoint(self, label: str) -> str:
        """Mark a point the workspace can be rewound to.

        Cheap: it records a position in the journal rather than copying files,
        so marking before every turn costs nothing.
        """
        self._marks[label] = len(self._journal)
        return label

    def rewind(self, label: str) -> List[Path]:
        """Restore every file to its state at `label`. Returns what changed.

        Replayed newest-first so a path written several times lands on the
        content it had at the mark, not on an intermediate version.
        """
        if label not in self._marks:
            raise KeyError(f"no checkpoint named {label!r}")
        mark = self._marks[label]
        restored: List[Path] = []

        for entry in reversed(self._journal[mark:]):
            if entry.before is None:
                # It did not exist before; undoing means removing it.
                try:
                    entry.path.unlink()
                    restored.append(entry.path)
                except FileNotFoundError:
                    pass
                except OSError:
                    pass
            else:
                try:
                    self._write_through(entry.path, entry.before)
                    restored.append(entry.path)
                except OSError:
                    pass

        del self._journal[mark:]
        # Marks taken after this one no longer refer to anything real.
        self._marks = {k: v for k, v in self._marks.items() if v <= mark}
        # Deduplicate while preserving order.
        seen, out = set(), []
        for path in restored:
            if path not in seen:
                seen.add(path)
                out.append(path)
        return out

    @property
    def journal(self) -> List[JournalEntry]:
        return list(self._journal)

    def _record(self, path: Path) -> None:
        """Capture a path's current content before it is overwritten.

        Through the editor when there is one: undo has to restore what the user
        had, and what the user had may never have been saved.
        """
        try:
            before: Optional[str] = self._read_through(path)
        except (FileNotFoundError, NotADirectoryError):
            before = None
        except OSError:
            before = None
        self._journal.append(JournalEntry(path=path, before=before))

    def resolve(self, requested: str | os.PathLike) -> Path:
        """Resolve `requested` against the root, refusing anything that escapes.

        Must work for paths that do not exist yet -- a file about to be written
        -- so it cannot lean on `Path.resolve(strict=True)`. Two checks run: a
        lexical one on the requested path, and a real one on the nearest
        ancestor that *does* exist, which is what catches a symlink pointing
        out of the tree.

        The probe tests `is_symlink() or exists()`, and the first half is
        load-bearing. `exists()` follows the link and reports False for a
        *broken* one, so a symlink to a path that does not exist yet used to
        slip through the whole loop: the probe walked past it to a parent that
        resolves cleanly inside the root, the check passed, and `write_text`
        then followed the link and created the file outside the workspace. A
        symlinked directory did worse, because `mkdir(parents=True)` would build
        the entire outside tree. `is_symlink` uses `lstat`, so it sees the link
        itself rather than its target and closes that gap. Git preserves
        symlinks, so this arrived in an ordinary checkout rather than needing
        the agent to create one.
        """
        candidate = Path(requested)
        joined = candidate if candidate.is_absolute() else self.root / candidate
        normalized = _normalize(joined)

        if not _is_within(normalized, self.root):
            raise PathEscape(
                f"path escapes the workspace: {requested!r} resolves outside {self.root}")

        probe = normalized
        while True:
            if probe.is_symlink() or probe.exists():
                # Non-strict `resolve` still follows a dangling link to the
                # target it names, which is exactly the path that would be
                # written, so this compares the right thing.
                if not _is_within(probe.resolve(), self.root):
                    raise PathEscape(
                        f"path escapes the workspace via a symlink: {requested!r}")
                break
            if probe.parent == probe:            # reached the filesystem root
                break
            probe = probe.parent

        return normalized

    def display(self, path: str | os.PathLike) -> str:
        """Render a path relative to the root where possible, for messages."""
        p = Path(path)
        try:
            return str(p.relative_to(self.root))
        except ValueError:
            return str(p)

    # ------------------------------------------------------------------- io

    def read(self, requested: str | os.PathLike) -> str:
        """Read a file: staged content first, then the editor, then disk.

        Staging outranks the editor because a staged edit is this run's own
        proposal and has not been offered to anyone yet.
        """
        path = self.resolve(requested)
        staged = self._staged.get(path)
        if staged is not None:
            return staged
        return self._read_through(path)

    def _read_through(self, path: Path) -> str:
        """Editor content if there is any, else disk."""
        if self.editor is not None:
            try:
                content = self.editor.read_text_file(path)
            except Exception:                        # never let a peer break a read
                content = None
            if content is not None:
                return content
        return path.read_text(encoding="utf-8", errors="replace")

    def exists(self, requested: str | os.PathLike) -> bool:
        path = self.resolve(requested)
        return path in self._staged or path.exists()

    def write(self, requested: str | os.PathLike, content: str) -> Path:
        """Write a file, or stage it when running dry. Returns the resolved path."""
        path = self.resolve(requested)
        if self.dry_run:
            self._staged[path] = content
            return path
        self._commit(path, content)
        return path

    def _commit(self, path: Path, content: str) -> None:
        """Journal the old content, then write it."""
        self._record(path)
        self._write_through(path, content)

    def _write_through(self, path: Path, content: str) -> None:
        """Write via the editor if there is one, else straight to disk."""
        if self.editor is not None:
            try:
                if self.editor.write_text_file(path, content):
                    return
            except Exception:
                pass                                 # fall through to disk
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    def edit(self, requested: str | os.PathLike, old: str, new: str) -> Path:
        """Replace `old` with `new`, requiring it to appear exactly once.

        Uniqueness is enforced rather than assumed. A silent multi-replace is
        how an edit tool corrupts a file in a way nobody notices for hours.
        """
        path = self.resolve(requested)
        text = self.read(path)
        count = text.count(old)
        if count == 0:
            raise ValueError(f"old text not found in {self.display(path)}")
        if count > 1:
            raise ValueError(
                f"old text appears {count} times in {self.display(path)}; "
                f"include surrounding context to make it unique")
        self.write(path, text.replace(old, new, 1))
        return path

    # -------------------------------------------------------------- staging

    def staged(self) -> List[Tuple[Path, str]]:
        """Proposed (path, content) pairs. Empty unless running dry."""
        return sorted(self._staged.items())

    def staged_paths(self) -> List[Path]:
        return sorted(self._staged)

    def original(self, path: Path) -> Optional[str]:
        """What is on disk for a staged path, or None if it is a new file."""
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return None

    def apply(self, only: Optional[List[Path]] = None) -> List[Path]:
        """Write staged changes to disk and drop them from staging.

        `only` limits the write to selected paths, so a partial review can be
        applied without discarding what has not been reviewed yet.
        """
        targets = self.staged_paths() if only is None else [
            p for p in self.staged_paths() if p in set(only)]

        written: List[Path] = []
        for path in targets:
            content = self._staged.pop(path)
            self._commit(path, content)
            written.append(path)
        return written

    def discard(self) -> None:
        self._staged.clear()

    def __repr__(self) -> str:
        mode = "dry-run" if self.dry_run else "live"
        return f"Workspace({self.root}, {mode}, staged={len(self._staged)})"


def _normalize(p: Path) -> Path:
    """Resolve `.` and `..` lexically, without consulting the filesystem.

    `Path.resolve()` would also follow symlinks and, on some versions, fail on
    paths that do not exist. Normalising by hand keeps this usable for a file
    that is about to be created.
    """
    parts: List[str] = []
    root = PurePath(p).anchor
    for part in PurePath(p).parts:
        if part == root:
            continue
        if part == os.curdir:
            continue
        if part == os.pardir:
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return Path(root).joinpath(*parts)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
