"""Echo acceptance gate: loop-count vs val-loss, with and without self-distillation.

Trains two Labyrinth models that differ in exactly one thing -- `--echo-weight` --
then sweeps loop count 1..R at eval and prints both curves plus the delta. The
claim under test is that Echo lowers loss at loop counts *below* the training
depth, i.e. that the model can be forced into fewer loops without falling apart.

    python scripts/echo_sweep.py                      # synthetic corpus, fast
    python scripts/echo_sweep.py --data ./data        # the real byte-level corpus
"""
from __future__ import annotations
import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from daedalus import Labyrinth, echo_step        # noqa: E402

BLOCK, BATCH, VOCAB = 64, 8, 256


def synthetic_corpus(n: int = 200_000, seed: int = 0):
    """A cheap byte stream with real structure: repeated short motifs.

    Enough for the mechanism to be measurable without downloading anything.
    """
    g = torch.Generator().manual_seed(seed)
    motifs = [torch.randint(32, 127, (k,), generator=g) for k in (3, 5, 8, 13)]
    out = []
    while sum(len(m) for m in out) < n:
        out.append(motifs[torch.randint(0, len(motifs), (1,), generator=g).item()])
    stream = torch.cat(out)[:n]
    cut = int(len(stream) * 0.9)
    return stream[:cut], stream[cut:]


def _batch(stream, batch_size, block_size, device, generator=None):
    ix = torch.randint(0, len(stream) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([stream[i:i + block_size].long() for i in ix])
    y = torch.stack([stream[i + 1:i + block_size + 1].long() for i in ix])
    return x.to(device), y.to(device)


def train_one(train_stream, echo_weight: float, steps: int = 300, seed: int = 0,
              n_loops: int = 4, n_embd: int = 64, device: str = "cpu") -> Labyrinth:
    """Train one Labyrinth. Identical seed/data order for both arms of the comparison."""
    torch.manual_seed(seed)
    model = Labyrinth(VOCAB, n_embd=n_embd, n_head=4, core_layers=2,
                      n_loops=n_loops, block_size=BLOCK).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(seed)          # same batches in both arms
    ge = torch.Generator().manual_seed(seed)         # same sampled student depths
    for _ in range(steps):
        x, y = _batch(train_stream, BATCH, BLOCK, device, g)
        loss = echo_step(model, x, y, model.n_loops, echo_weight, generator=ge)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


@torch.no_grad()
def sweep(model, val_stream, max_loops: int | None = None, iters: int = 20,
          device: str = "cpu", seed: int = 1234) -> dict:
    """Val loss at each loop count 1..R. Returns {n_loops: loss}."""
    model.eval()
    r = max_loops or model.n_loops
    out = {}
    for n in range(1, r + 1):
        g = torch.Generator().manual_seed(seed)      # same eval batches at every depth
        total = 0.0
        for _ in range(iters):
            x, y = _batch(val_stream, BATCH, BLOCK, device, g)
            logits = model(x, n_loops=n)[0]
            b, t, v = logits.shape
            total += F.cross_entropy(logits.reshape(b * t, v), y.reshape(b * t)).item()
        out[n] = total / iters
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="byte-corpus dir (default: synthetic)")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--n-loops", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=64)
    ap.add_argument("--echo-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.data:
        from data import load_splits
        d = load_splits(args.data)
        train_stream, val_stream = d["train"], d["val"]
    else:
        train_stream, val_stream = synthetic_corpus(seed=args.seed)

    curves = {}
    for label, w in (("echo off", 0.0), (f"echo {args.echo_weight}", args.echo_weight)):
        model = train_one(train_stream, w, args.steps, args.seed,
                          args.n_loops, args.n_embd, device)
        curves[label] = sweep(model, val_stream, args.n_loops, device=device)

    off, on = list(curves.values())
    print(f"\nloop-count vs val loss ({args.steps} steps, seed {args.seed})")
    print(f"{'loops':>6} | {'echo off':>9} | {'echo on':>9} | {'delta':>7} | bits/byte on")
    for n in sorted(off):
        d = on[n] - off[n]
        print(f"{n:>6} | {off[n]:>9.4f} | {on[n]:>9.4f} | {d:>+7.4f} | {on[n]/math.log(2):>6.2f}")
    print(f"\nloop-1 delta: {on[1] - off[1]:+.4f} nats "
          f"({'Echo helps' if on[1] < off[1] else 'Echo does not help'} at the shallow end)")


if __name__ == "__main__":
    main()
