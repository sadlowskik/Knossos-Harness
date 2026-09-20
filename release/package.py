#!/usr/bin/env python3
"""Create a reproducible, self-contained Knossos release archive."""

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


def add_file(archive: zipfile.ZipFile, source: Path, name: str, executable: bool) -> None:
    info = zipfile.ZipInfo(name, ZIP_TIMESTAMP)
    mode = 0o755 if executable else 0o644
    info.external_attr = (stat.S_IFREG | mode) << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    archive.writestr(info, source.read_bytes())


def package(binary: Path, version: str, platform: str, out_dir: Path) -> Path:
    root = Path(__file__).resolve().parents[1]
    binary = binary.resolve()
    if not binary.is_file():
        raise FileNotFoundError(f"binary does not exist: {binary}")

    archive_stem = f"knossos-{version}-{platform}"
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / f"{archive_stem}.zip"

    files = [
        (binary, binary.name, True),
        (root / "README.md", "README.md", False),
        (root / "knossos-rs" / "constitution.md", "constitution.md", False),
        (root / "knossos-rs" / "LICENSE", "LICENSE", False),
    ]
    missing = [str(source) for source, _, _ in files if not source.is_file()]
    if missing:
        raise FileNotFoundError(f"release inputs are missing: {', '.join(missing)}")

    temp_archive = archive_path.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(
            temp_archive,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
        ) as archive:
            for source, relative_name, executable in sorted(files, key=lambda item: item[1]):
                add_file(
                    archive,
                    source,
                    f"{archive_stem}/{relative_name}",
                    executable,
                )
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
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--version", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    archive = package(args.binary, args.version, args.platform, args.out)
    print(archive)
    print(archive.with_suffix(".zip.sha256"))


if __name__ == "__main__":
    main()
