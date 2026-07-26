"""Fetch code data by shallow-cloning GitHub repos, then build byte-level splits.

Requires internet + git. Clones each repo, collects source files by extension,
caps the total at --max-bytes, splits BY FILE, and saves train/val/test tensors.

Example:
    python scripts/fetch_rust.py --out ./data --ext rs --max-bytes 400000000
"""
from __future__ import annotations
import argparse
import glob
import os
import random
import subprocess

import torch

from daedalus import ByteTokenizer

# A default set of substantial, idiomatic Rust projects.
DEFAULT_REPOS = [
    "tokio-rs/tokio", "serde-rs/serde", "clap-rs/clap", "BurntSushi/ripgrep",
    "sharkdp/bat", "alacritty/alacritty", "actix/actix-web", "hyperium/hyper",
    "seanmonstar/reqwest", "diesel-rs/diesel", "bevyengine/bevy", "denoland/deno",
    "nushell/nushell", "rust-lang/rust-analyzer", "pola-rs/polars",
    "meilisearch/meilisearch", "sharkdp/fd", "starship/starship", "tokio-rs/axum",
    "rayon-rs/rayon", "crossbeam-rs/crossbeam", "dtolnay/anyhow", "tokio-rs/tracing",
    "image-rs/image", "hyperium/tonic",
]


def _read(path):
    try:
        return open(path, encoding="utf-8", errors="replace").read()
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./data")
    ap.add_argument("--src", default="./rust_src")
    ap.add_argument("--ext", default="rs")
    ap.add_argument("--max-bytes", type=int, default=400_000_000)
    ap.add_argument("--max-file-bytes", type=int, default=400_000)
    ap.add_argument("--repos", nargs="*", default=DEFAULT_REPOS)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    os.makedirs(args.src, exist_ok=True)
    for repo in args.repos:
        dst = os.path.join(args.src, repo.split("/")[-1])
        if not os.path.exists(dst):
            subprocess.run(["git", "clone", "--depth", "1",
                            f"https://github.com/{repo}", dst], check=False)

    files = [f for f in glob.glob(os.path.join(args.src, "**", f"*.{args.ext}"), recursive=True)
             if os.path.isfile(f) and 0 < os.path.getsize(f) < args.max_file_bytes]
    random.seed(args.seed)
    random.shuffle(files)

    kept, total = [], 0
    for f in files:
        kept.append(f)
        total += os.path.getsize(f)
        if total >= args.max_bytes:
            break
    print(f"kept {len(kept)} files, ~{total/1e6:.0f}M bytes")

    tok = ByteTokenizer()
    os.makedirs(args.out, exist_ok=True)
    n_val = max(1, len(kept) // 20)
    n_test = max(1, len(kept) // 20)
    splits = {
        "test": kept[:n_test],
        "val": kept[n_test:n_test + n_val],
        "train": kept[n_test + n_val:],
    }
    for name, flist in splits.items():
        text = "\n\n".join(_read(f) for f in flist)
        torch.save(torch.tensor(tok.encode(text), dtype=torch.uint8),
                   os.path.join(args.out, f"{name}.pt"))
        print(f"{name:5s}: {len(flist):5d} files, {len(text):>12,d} tokens")


if __name__ == "__main__":
    main()
