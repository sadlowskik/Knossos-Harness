#!/usr/bin/env python3
"""Create a deterministic Knossos Field runtime archive from verified build inputs."""

from __future__ import annotations

import argparse
import hashlib
import stat
import zipfile
from pathlib import Path


ZIP_TIMESTAMP = (2026, 1, 1, 0, 0, 0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_file(archive: zipfile.ZipFile, source: Path, name: str) -> None:
    info = zipfile.ZipInfo(name, ZIP_TIMESTAMP)
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    archive.writestr(info, source.read_bytes())


def release_files(root: Path) -> list[tuple[Path, Path]]:
    field = root / "field"
    fixed = [
        field / "package.json",
        field / "package-lock.json",
        field / "README.md",
        field / "CHANGELOG.md",
        field / "asset-manifest.json",
        field / "ASSET_PROVENANCE.md",
        field / "server" / "package.json",
        field / "web" / "package.json",
        field / "web" / "index.html",
        field / "web" / "vite.config.js",
        root / "LICENSE",
    ]
    trees = [
        field / "server" / "src",
        field / "server" / "test",
        field / "field",
        field / "scripts",
        field / "web" / "dist",
        field / "web" / "public",
        field / "web" / "src",
    ]
    files = [(path, path.relative_to(field)) for path in fixed[:-1]]
    files.append((fixed[-1], Path("LICENSE")))
    for tree in trees:
        if not tree.is_dir():
            raise FileNotFoundError(f"release input directory is missing: {tree}")
        files.extend(
            (path, path.relative_to(field))
            for path in tree.rglob("*")
            if path.is_file()
        )
    missing = [str(path) for path, _ in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"release inputs are missing: {', '.join(missing)}")
    unsafe = [str(path) for path, _ in files if path.is_symlink()]
    if unsafe:
        raise ValueError(f"release inputs must not be symbolic links: {', '.join(unsafe)}")
    return sorted(files, key=lambda item: item[1].as_posix())


def package(version: str, out_dir: Path) -> Path:
    root = Path(__file__).resolve().parents[1]
    archive_stem = f"knossos-field-{version}"
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / f"{archive_stem}.zip"
    temp_archive = archive_path.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(
            temp_archive,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for source, relative in release_files(root):
                add_file(archive, source, f"{archive_stem}/{relative.as_posix()}")
        temp_archive.replace(archive_path)
    finally:
        temp_archive.unlink(missing_ok=True)

    checksum = sha256(archive_path)
    archive_path.with_suffix(".zip.sha256").write_text(
        f"{checksum}  {archive_path.name}\n",
        encoding="utf-8",
        newline="\n",
    )
    return archive_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    archive = package(args.version, args.out)
    print(archive)
    print(archive.with_suffix(".zip.sha256"))


if __name__ == "__main__":
    main()
