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

import os
from pathlib import Path, PurePath
from typing import Dict, List, Optional, Tuple


class PathEscape(PermissionError):
    """A tool asked for a path outside the workspace."""


class Workspace:
    """A rooted, optionally write-staged view of a directory tree."""

    def __init__(self, root: str | os.PathLike, dry_run: bool = False) -> None:
        self.root = Path(root).resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(f"workspace root is not a directory: {self.root}")
        self.dry_run = dry_run
        #: absolute path -> proposed content. Empty unless `dry_run`.
        self._staged: Dict[Path, str] = {}

    # ------------------------------------------------------------------ jail

    def resolve(self, requested: str | os.PathLike) -> Path:
        """Resolve `requested` against the root, refusing anything that escapes.

        Must work for paths that do not exist yet -- a file about to be written
        -- so it cannot lean on `Path.resolve(strict=True)`. Two checks run: a
        lexical one on the requested path, and a real one on the nearest
        ancestor that *does* exist, which is what catches a symlink pointing
        out of the tree.
        """
        candidate = Path(requested)
        joined = candidate if candidate.is_absolute() else self.root / candidate
        normalized = _normalize(joined)

        if not _is_within(normalized, self.root):
            raise PathEscape(
                f"path escapes the workspace: {requested!r} resolves outside {self.root}")

        probe = normalized
        while True:
            if probe.exists():
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
        """Read a file, preferring staged content over what is on disk."""
        path = self.resolve(requested)
        staged = self._staged.get(path)
        if staged is not None:
            return staged
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
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

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
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
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
