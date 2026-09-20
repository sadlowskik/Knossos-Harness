"""Byte-level data pipeline.

Gathers Python source files, splits them BY FILE (so no file's content leaks
across the train/val/test boundary), tokenizes with the byte tokenizer, and
saves each split as a uint8 tensor.

Usage:
    python data.py --source /usr/lib/python3.12 --out ./data
"""
from __future__ import annotations
import argparse
import bisect
import glob
import json
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


# --------------------------------------------------------------------------
# Sharded memory-mapped corpora (for runs too big to hold in RAM)
# --------------------------------------------------------------------------

class Corpus:
    """A split stored as one or more memory-mapped `.bin` shards.

    `load_splits` above reads the whole split into a tensor, which is fine up to
    ~100M tokens and impossible past a few billion. A memmap leaves the data on
    disk and lets the OS page in only the windows actually sampled, so corpus
    size stops being bounded by RAM. Shards are sampled in proportion to their
    length, so every token remains equally likely regardless of how the corpus
    was chunked at write time.

    Written by `scripts/prepare_corpus.py`; `meta.json` records the dtype and
    the tokenizer the ids belong to.
    """

    def __init__(self, data_dir: str, split: str):
        import numpy as np
        with open(os.path.join(data_dir, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.dtype = np.dtype(self.meta.get("dtype", "uint16"))
        paths = sorted(glob.glob(os.path.join(data_dir, f"{split}_*.bin")))
        if not paths:
            single = os.path.join(data_dir, f"{split}.bin")
            paths = [single] if os.path.exists(single) else []
        if not paths:
            raise FileNotFoundError(f"no shards for split {split!r} in {data_dir}")
        self.shards = [np.memmap(p, dtype=self.dtype, mode="r") for p in paths]
        self.lengths = [len(s) for s in self.shards]
        self.total = sum(self.lengths)
        # Cumulative lengths let a single uniform draw pick a shard in proportion
        # to its size without materialising a per-token probability vector.
        self.cum = []
        running = 0
        for n in self.lengths:
            running += n
            self.cum.append(running)

    def __len__(self) -> int:
        return self.total

    @property
    def vocab_size(self) -> int:
        return int(self.meta["vocab_size"])

    def batch(self, batch_size: int, block_size: int, device: str,
              generator: "torch.Generator | None" = None):
        """Random contiguous windows; targets are inputs shifted by one."""
        import numpy as np
        xs, ys = [], []
        for _ in range(batch_size):
            r = int(torch.randint(0, self.total, (1,), generator=generator).item())
            si = bisect.bisect_right(self.cum, r)
            shard = self.shards[si]
            hi = len(shard) - block_size - 1
            if hi <= 0:                       # shard shorter than a window
                continue
            off = int(torch.randint(0, hi, (1,), generator=generator).item())
            win = np.asarray(shard[off:off + block_size + 1], dtype=np.int64)
            xs.append(torch.from_numpy(win[:-1]))
            ys.append(torch.from_numpy(win[1:]))
        if not xs:
            raise RuntimeError("every shard is shorter than block_size + 1")
        x, y = torch.stack(xs), torch.stack(ys)
        if device.startswith("cuda"):
            # pin + non_blocking overlaps the host->device copy with compute
            return (x.pin_memory().to(device, non_blocking=True),
                    y.pin_memory().to(device, non_blocking=True))
        return x.to(device), y.to(device)


class PairCorpus:
    """A prompt/completion corpus stored as pre-packed fixed-size windows.

    `Corpus` samples a random offset into a flat token stream, which is right
    for plain language modelling and wrong for a transduction task: a window
    routinely starts in the middle of one example's target and ends in the
    middle of the next one's prompt, so the model is asked to predict a
    correction whose input it cannot see. Here the packing is done once, at
    build time, by `scripts/make_gec.py`: examples are laid end to end into
    windows of exactly `block_size + 1` tokens, never straddling a boundary,
    and the leftover tail of each window is padded.

    A parallel uint8 `.mask` file marks which tokens are *completion* tokens.
    Targets outside the mask are set to -100 so `F.cross_entropy` ignores them,
    which is what stops the model spending half its capacity learning to
    generate the corrupted input it is supposed to be fixing.

    Examples are packed several to a window, so a window's later examples attend
    to earlier ones. That is the same compromise every packed-pretraining setup
    makes; the separator token is what marks the boundary.
    """

    IGNORE_INDEX = -100

    def __init__(self, data_dir: str, split: str):
        import numpy as np
        with open(os.path.join(data_dir, "meta.json"), encoding="utf-8") as f:
            self.meta = json.load(f)
        self.dtype = np.dtype(self.meta.get("dtype", "uint16"))
        self.block_size = int(self.meta["block_size"])
        self.window = self.block_size + 1
        # `{split}_*.bin` cannot match `{split}_*.mask` -- keep it that way, or a
        # mask file would be memmapped as tokens and silently train on garbage.
        paths = sorted(glob.glob(os.path.join(data_dir, f"{split}_*.bin")))
        if not paths:
            raise FileNotFoundError(f"no shards for split {split!r} in {data_dir}")
        self.shards, self.masks, self.counts = [], [], []
        for p in paths:
            mp = p[:-len(".bin")] + ".mask"
            if not os.path.exists(mp):
                raise FileNotFoundError(
                    f"{p} has no matching {os.path.basename(mp)}. This corpus "
                    "says format=pairs but was not written by make_gec.py.")
            toks = np.memmap(p, dtype=self.dtype, mode="r")
            mask = np.memmap(mp, dtype=np.uint8, mode="r")
            if len(toks) != len(mask):
                raise ValueError(f"{p}: {len(toks)} tokens vs {len(mask)} mask bytes")
            if len(toks) % self.window:
                raise ValueError(
                    f"{p}: {len(toks)} tokens is not a multiple of block_size+1 "
                    f"({self.window}). The corpus was built for a different "
                    "--block-size.")
            self.shards.append(toks)
            self.masks.append(mask)
            self.counts.append(len(toks) // self.window)
        self.total_windows = sum(self.counts)
        if self.total_windows == 0:
            raise ValueError(f"split {split!r} in {data_dir} has zero windows")
        self.total = sum(len(s) for s in self.shards)
        self.cum, running = [], 0
        for n in self.counts:
            running += n
            self.cum.append(running)

    def __len__(self) -> int:
        return self.total

    @property
    def vocab_size(self) -> int:
        return int(self.meta["vocab_size"])

    def batch(self, batch_size: int, block_size: int, device: str,
              generator: "torch.Generator | None" = None):
        """Whole aligned windows; targets outside the completion mask are -100."""
        import numpy as np
        if block_size != self.block_size:
            raise ValueError(
                f"this corpus was packed for --block-size {self.block_size}, "
                f"but training asked for {block_size}. Repacking is required: "
                "the windows are physically laid out at the build-time size.")
        xs, ys = [], []
        for _ in range(batch_size):
            w = int(torch.randint(0, self.total_windows, (1,), generator=generator).item())
            si = bisect.bisect_right(self.cum, w)
            local = w - (self.cum[si - 1] if si else 0)
            off = local * self.window
            win = np.array(self.shards[si][off:off + self.window], dtype=np.int64)
            msk = np.array(self.masks[si][off:off + self.window], dtype=np.int64)
            y = win[1:].copy()
            # mask[i] describes token i, and y[j] is token j+1 -- so the target
            # at position j is supervised iff mask[j+1] is set. Off-by-one here
            # would train on the prompt and ignore the answer.
            y[msk[1:] == 0] = self.IGNORE_INDEX
            xs.append(torch.from_numpy(win[:-1]))
            ys.append(torch.from_numpy(y))
        x, y = torch.stack(xs), torch.stack(ys)
        if (y != self.IGNORE_INDEX).sum() == 0:
            raise RuntimeError(
                "sampled a batch with no supervised targets; the corpus is "
                "malformed (every window must contain at least one example)")
        if device.startswith("cuda"):
            return (x.pin_memory().to(device, non_blocking=True),
                    y.pin_memory().to(device, non_blocking=True))
        return x.to(device), y.to(device)


def load_corpus(data_dir: str) -> "Dict[str, object]":
    """Load whichever of train/val/test exist, as the format `meta.json` names."""
    with open(os.path.join(data_dir, "meta.json"), encoding="utf-8") as f:
        fmt = json.load(f).get("format", "flat")
    cls = PairCorpus if fmt == "pairs" else Corpus
    out = {}
    for split in ("train", "val", "test"):
        try:
            out[split] = cls(data_dir, split)
        except FileNotFoundError:
            pass
    if "train" not in out:
        raise FileNotFoundError(f"no train shards in {data_dir}")
    return out


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
