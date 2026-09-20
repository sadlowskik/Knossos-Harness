#!/usr/bin/env python
"""Reproducible CPU inference smoke benchmark for the Daedalus architecture.

This intentionally uses deterministic random weights: it measures model shape
and generation mechanics without requiring a checkpoint or network access.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from daedalus import ByteTokenizer  # noqa: E402
from generate import build, generate, resolve_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="adaptive", choices=["dense", "labyrinth", "moe", "unified", "full", "adaptive", "proteus"])
    parser.add_argument("--tokens", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--threads", type=int, default=0, help="0 keeps PyTorch's default")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    cfg = resolve_config({}, {"model": args.model})
    model = build(cfg, "cpu")
    tokenizer = ByteTokenizer()
    params = sum(p.numel() for p in model.parameters())
    print(f"model={args.model} params={params:,}")
    print(
        "weight-only memory: "
        f"fp32={params * 4 / 2**20:.1f} MiB "
        f"fp16={params * 2 / 2**20:.1f} MiB "
        f"int8={params / 2**20:.1f} MiB"
    )

    # Warm allocations and kernels before timing.
    generate(model, tokenizer, "def ", "cpu", n=1, top_k=0, rep_pen=1.0)
    for count in args.tokens:
        started = time.perf_counter()
        generate(model, tokenizer, "def ", "cpu", n=count, top_k=0, rep_pen=1.0)
        elapsed = time.perf_counter() - started
        print(f"{count:4d} tokens  {elapsed:8.3f}s  {count / elapsed:8.2f} tok/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
