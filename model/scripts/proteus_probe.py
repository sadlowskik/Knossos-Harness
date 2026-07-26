"""Proteus acceptance gate: does the self-modification do anything, and is it stable?

Two probes, both cheap, both run before trusting Proteus on real code:

1. Stability -- train on a fixed toy batch and log the Frobenius norm of the
   self-written matrix alongside the loss. Divergence shows up in the norm first.

2. Sequential adaptation -- the test the original fast-weight / SRWM work used.
   The model reads a sequence built from a repeating rule; if the fast weights
   are genuinely storing that rule, the second half of the sequence should be
   predicted better than the first. `adaptation_gap` returns
   (loss on first half) - (loss on second half): positive means it adapted
   within the sequence, ~0 means the mechanism is idle.

    python scripts/proteus_probe.py
    python scripts/proteus_probe.py --self-referential      # the full SRWM
"""
from __future__ import annotations
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from daedalus import DaedalusProteus                # noqa: E402

VOCAB, BLOCK = 256, 64


def repeating_task(batch: int, block: int, period: int = 8, seed: int = 0):
    """Each row is one random motif tiled to fill the block.

    The rule ("token t equals token t-period") is discoverable inside a single
    sequence, so any within-sequence adaptation shows up as a first-half /
    second-half gap. No gradient step can memorise it: the motif is redrawn
    every call.
    """
    g = torch.Generator().manual_seed(seed)
    motif = torch.randint(0, VOCAB, (batch, period), generator=g)
    reps = block // period + 2
    stream = motif.repeat(1, reps)
    return stream[:, :block], stream[:, 1:block + 1]


@torch.no_grad()
def adaptation_gap(steps: int = 400, seed: int = 0, self_referential: bool = False,
                   device: str = "cpu") -> float:
    """Train briefly on the repeating task, then measure the first/second-half gap."""
    model = _train_on_task(steps, seed, self_referential, device)
    model.eval()
    x, y = repeating_task(16, BLOCK, seed=seed + 999)
    x, y = x.to(device), y.to(device)
    logits = model(x)[0]
    half = BLOCK // 2
    first = F.cross_entropy(logits[:, :half].reshape(-1, VOCAB), y[:, :half].reshape(-1))
    second = F.cross_entropy(logits[:, half:].reshape(-1, VOCAB), y[:, half:].reshape(-1))
    return (first - second).item()


def _train_on_task(steps: int, seed: int, self_referential: bool, device: str):
    torch.manual_seed(seed)
    model = DaedalusProteus(VOCAB, n_embd=64, n_head=4, n_layer=2, block_size=BLOCK,
                            self_referential=self_referential).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with torch.enable_grad():
        for step in range(steps):
            x, y = repeating_task(16, BLOCK, seed=seed * 100000 + step)
            loss = model(x.to(device), y.to(device))[1]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


def stability_run(steps: int, self_referential: bool, seed: int, device: str) -> None:
    torch.manual_seed(seed)
    model = DaedalusProteus(VOCAB, n_embd=64, n_head=4, n_layer=2, block_size=BLOCK,
                            self_referential=self_referential).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x, y = repeating_task(8, BLOCK, seed=seed)
    x, y = x.to(device), y.to(device)
    print(f"{'step':>6} | {'loss':>8} | {'||W|| final token':>18}")
    for step in range(steps + 1):
        if step % max(1, steps // 10) == 0:
            with torch.no_grad():
                norm = model.weight_norms(x[:1])[-1]
            print(f"{step:>6} | {model(x, y)[1].item():>8.4f} | {norm:>18.4f}")
        loss = model(x, y)[1]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--self-referential", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    level = "SRWM (fully self-referential)" if args.self_referential else "single-level"
    print(f"Proteus probe -- {level}, {args.steps} steps, device={device}\n")
    print("[1] stability: watch the weight norm, not just the loss")
    stability_run(args.steps, args.self_referential, args.seed, device)

    print("\n[2] sequential adaptation on a repeating-motif task")
    gap = adaptation_gap(args.steps, args.seed, args.self_referential, device)
    verdict = ("adapts within the sequence" if gap > 0.05
               else "no measurable within-sequence adaptation")
    print(f"first-half minus second-half loss: {gap:+.4f} nats -> {verdict}")


if __name__ == "__main__":
    main()
