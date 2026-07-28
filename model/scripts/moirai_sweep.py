"""Moirai acceptance gate: do decoupled erase/write gates earn their keep?

Moirai is Gated DeltaNet (Yang et al. 2024) with one change: the erase gate
(Clotho) and the write gate (Lachesis) are separate projections rather than one
shared gate. The argument is that this lets the model drop stale content
*before* deciding how hard to write, so new writes can cannibalize the space
that low-value ones held.

That argument is the only original claim in the module, so it is what this
measures. Three arms, identical everywhere else:

    softmax       ordinary causal self-attention -- the floor
    moirai-tied   one gate drives both -- plain Gated DeltaNet
    moirai        erase and write decoupled -- the claim

`moirai - moirai-tied` is the contribution. `moirai-tied - softmax` says whether
the family is viable here at all, which is worth knowing but is not Moirai's
result to claim.

The arms are **not** parameter-matched, and the printed counts show it: decoupling
costs one extra projection per block. That is the honest framing -- the question
is whether the second gate buys more than the same parameters would elsewhere,
so a win smaller than the parameter difference would suggest is not a win.

A second, cheaper claim also gets tested. Moirai's state is O(n_head * d_head^2)
regardless of sequence length, so it should degrade more gracefully past the
training length than a model with a learned absolute position table. Note the
asymmetry honestly: the softmax baseline uses `Embeddings`, whose `pos_emb` rows
beyond the training length are still at their random initialisation. Its
collapse past that point is a property of *that positional scheme*, not of
attention -- `rope.py` exists in this repo and would extrapolate far better. The
number is reported for what it is.

    python scripts/moirai_sweep.py                    # synthetic, ~minutes on CPU
    python scripts/moirai_sweep.py --data ./data      # the real byte corpus
    python scripts/moirai_sweep.py --seeds 5 --steps 1500
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from daedalus import Labyrinth                                  # noqa: E402
from scripts.echo_sweep import _batch, synthetic_corpus         # noqa: E402
from scripts.seeds import DEFAULT_SEEDS, delta, run_arm, table  # noqa: E402

VOCAB = 256
ARMS = ("softmax", "moirai", "moirai-untied")


def build(mixer: str, n_embd: int, n_head: int, core_layers: int,
          n_loops: int, block_size: int, device: str) -> Labyrinth:
    """Same model everywhere but the mixer.

    `block_size` is the *largest* length that will ever be evaluated, not the
    training length: the softmax arm needs a position table that reaches, and
    sizing it here rather than at eval keeps every arm's parameter count fixed
    across the extrapolation sweep.
    """
    return Labyrinth(VOCAB, n_embd=n_embd, n_head=n_head, core_layers=core_layers,
                     n_loops=n_loops, block_size=block_size, mixer=mixer).to(device)


def train_one(model, stream, steps: int, block: int, batch: int, seed: int,
              lr: float, device: str):
    """Identical data order across arms: the generator is seeded from the seed."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(steps):
        x, y = _batch(stream, batch, block, device, g)
        loss = model(x, y)[1]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


@torch.no_grad()
def evaluate(model, stream, block: int, batch: int, iters: int, device: str,
             seed: int = 1234) -> float:
    """Mean val loss at one sequence length, on identical batches for every arm."""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    total = 0.0
    for _ in range(iters):
        x, y = _batch(stream, batch, block, device, g)
        logits = model(x)[0]
        b, t, v = logits.shape
        total += F.cross_entropy(logits.reshape(b * t, v), y.reshape(b * t)).item()
    model.train()
    return total / iters


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=None, help="byte-corpus dir (default: synthetic)")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--block", type=int, default=64, help="training sequence length")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-embd", type=int, default=64)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--core-layers", type=int, default=2)
    ap.add_argument("--n-loops", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--eval-iters", type=int, default=20)
    ap.add_argument("--extrapolate", type=int, nargs="*", default=None,
                    help="eval lengths (default: train length, 2x, 4x)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = list(range(args.seeds))
    lengths = args.extrapolate if args.extrapolate is not None else [
        args.block, args.block * 2, args.block * 4]
    longest = max(lengths)

    if args.data:
        from data import load_splits
        splits = load_splits(args.data)
        train_stream, val_stream = splits["train"], splits["val"]
        corpus = args.data
    else:
        train_stream, val_stream = synthetic_corpus(seed=0)
        corpus = "synthetic (repeated motifs)"

    print(f"\nMoirai sweep -- {args.steps} steps, {len(seeds)} seeds, device={device}")
    print(f"corpus: {corpus}")
    print(f"train length {args.block}, eval lengths {lengths}\n")

    # Cache each trained model so the extrapolation sweep does not retrain.
    trained: dict = {}
    params: dict = {}

    def make_runner(mixer: str, at_length: int):
        def run(seed: int) -> float:
            key = (mixer, seed)
            if key not in trained:
                torch.manual_seed(seed)
                model = build(mixer, args.n_embd, args.n_head, args.core_layers,
                              args.n_loops, longest, device)
                params[mixer] = sum(p.numel() for p in model.parameters())
                trained[key] = train_one(model, train_stream, args.steps, args.block,
                                         args.batch_size, seed, args.lr, device)
            return evaluate(trained[key], val_stream, at_length,
                            args.batch_size, args.eval_iters, device)
        return run

    print(f"[1] val loss at the training length ({args.block})")
    at_train = [run_arm(m, make_runner(m, args.block), seeds,
                        note=f"{params.get(m, 0):,} params" if m in params else "")
                for m in ARMS]
    for arm, mixer in zip(at_train, ARMS):
        arm.note = f"{params[mixer]:,} params"
    print("\n" + table(at_train, "losses"))

    by_label = {a.label: a for a in at_train}
    print("\n" + delta(by_label["moirai"], by_label["moirai-untied"]))
    print("    ^ the contribution: decoupling the gates, against the mechanism it extends")
    print(delta(by_label["softmax"], by_label["moirai"]))
    print("    ^ context only: whether gated fast weights are viable on this corpus")

    if len(lengths) > 1:
        print(f"\n[2] extrapolation past the training length ({args.block})")
        print("    softmax uses a learned absolute position table, so its rows past")
        print("    the training length are still randomly initialised. Read its")
        print("    collapse as a fact about that scheme, not about attention.\n")
        # Mean *and* spread: a bare mean here would be the same rule-4 violation
        # the rest of this script exists to avoid, and the tied/untied gap at
        # length is small enough that it matters.
        header = f"{'arm':<12} | " + " | ".join(f"{n:>17}" for n in lengths)
        print(header)
        print("-" * len(header))
        for mixer in ARMS:
            cells = []
            for n in lengths:
                arm = run_arm(mixer, make_runner(mixer, n), seeds, progress=False)
                cells.append(f"{arm.mean:>8.4f}+/-{arm.stdev:<6.4f}"
                             if arm.values else f"{'--':>17}")
            print(f"{mixer:<12} | " + " | ".join(cells))

    best = min((a for a in at_train if a.values), key=lambda a: a.mean, default=None)
    if best is not None:
        print(f"\nbest at the training length: {best.label} "
              f"({best.mean:.4f} nats, {best.mean / math.log(2):.3f} bits/byte)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
