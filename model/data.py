"""Byte-level data pipeline.

Gathers Python source files, splits them BY FILE (so no file's content leaks
across the train/val/test boundary), tokenizes with the byte tokenizer, and
saves each split as a uint8 tensor.

Usage:
    python data.py --source /usr/lib/python3.12 --out ./data
"""
from __future__ import annotations
import argparse
import glob
import os
import random
from typing import Dict

import torch

from daedalus import ByteTokenizer


def build_splits(source_dir: str, out_dir: str, max_bytes: int = 8_000_000,
                 seed: int = 1337) -> None:
    tok = ByteTokenizer()
    files, total = [], 0
    for f in sorted(glob.glob(os.path.join(source_dir, "**", "*.py"), recursive=True)):
        try:
            n = os.path.getsize(f)
        except OSError:
            continue
        if 0 < n < 200_000:
            files.append(f)
            total += n
            if total >= max_bytes:
                break

    random.seed(seed)
    random.shuffle(files)                       # deterministic

    n = len(files)
    n_val = max(1, n // 20)
    n_test = max(1, n // 20)
    splits = {
        "test": files[:n_test],
        "val": files[n_test:n_test + n_val],
        "train": files[n_test + n_val:],
    }

    os.makedirs(out_dir, exist_ok=True)
    for name, flist in splits.items():
        text = "\n\n".join(
            open(f, encoding="utf-8", errors="replace").read() for f in flist
        )
        ids = torch.tensor(tok.encode(text), dtype=torch.uint8)
        torch.save(ids, os.path.join(out_dir, f"{name}.pt"))
        print(f"{name:5s}: {len(flist):4d} files, {len(ids):>10,d} tokens")


def load_splits(out_dir: str) -> Dict[str, torch.Tensor]:
    return {name: torch.load(os.path.join(out_dir, f"{name}.pt"))
            for name in ("train", "val", "test")}


def get_batch(data: Dict[str, torch.Tensor], split: str, batch_size: int,
              block_size: int, device: str):
    """Random contiguous windows; targets are inputs shifted by one."""
    stream = data[split]
    ix = torch.randint(0, len(stream) - block_size - 1, (batch_size,))
    x = torch.stack([stream[i:i + block_size].long() for i in ix])
    y = torch.stack([stream[i + 1:i + block_size + 1].long() for i in ix])
    return x.to(device), y.to(device)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="directory of .py files")
    ap.add_argument("--out", default="./data")
    ap.add_argument("--max-bytes", type=int, default=8_000_000)
    args = ap.parse_args()
    build_splits(args.source, args.out, args.max_bytes)
