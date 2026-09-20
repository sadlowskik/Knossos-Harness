"""Proteus acceptance gate: does the self-modification do anything attention can't?

Two questions, and the second is the one the old version of this script could not
answer.

**[1] Adaptation.** The model reads a sequence built from a repeating rule
("token t equals token t-period"). If the fast weights are genuinely storing that
rule, the second half of the sequence should be predicted better than the first.
`adaptation_gap` returns (loss on first half) - (loss on second half): positive
means it adapted inside the sequence.

That number on its own proves nothing. **Ordinary causal attention solves this
task too** -- it is what induction heads do, and a transformer can attend
directly to position t-period. So a positive gap is only interesting if it beats
a dense softmax baseline with the same budget on the identical task. That
baseline is the third arm here, and it is the entire point of this rewrite.

**[2] Stability.** The known failure mode is unbounded growth of the self-written
matrix: the layer teaches itself to write ever harder, and the norm diverges
before the loss does. Reported as its own table, per seed. A diverged seed is a
*result* and is printed as one -- not dropped, because a failure that correlates
with the arm is the finding, not noise.

    python scripts/proteus_probe.py                  # all three arms, 5 seeds
    python scripts/proteus_probe.py --trace          # the old per-step norm log
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from daedalus import Daedalus, DaedalusProteus        # noqa: E402
from scripts.seeds import DEFAULT_SEEDS, delta, run_arm, table   # noqa: E402

VOCAB, BLOCK = 256, 64

#: Above this the self-written matrix is considered to have run away. Chosen an
#: order of magnitude above anything a healthy level-1 run reaches, so it flags
#: divergence rather than ordinary growth.
NORM_CEILING = 1e4

ARMS = ("dense", "proteus", "proteus-srwm")


def repeating_task(batch: int, block: int, period: int = 8, seed: int = 0):
    """Each row is one random motif tiled to fill the block.

    The rule is discoverable inside a single sequence, so within-sequence
    adaptation shows up as a first-half / second-half gap. No gradient step can
    memorise it: the motif is redrawn every call.
    """
    g = torch.Generator().manual_seed(seed)
    motif = torch.randint(0, VOCAB, (batch, period), generator=g)
    reps = block // period + 2
    stream = motif.repeat(1, reps)
    return stream[:, :block], stream[:, 1:block + 1]


def build(arm: str, device: str, n_layer: int = 2, n_embd: int = 64, n_head: int = 4):
    """One model per arm, matched in depth, width and heads."""
    if arm == "dense":
        return Daedalus(VOCAB, n_embd=n_embd, n_head=n_head, n_layer=n_layer,
                        block_size=BLOCK).to(device)
    return DaedalusProteus(VOCAB, n_embd=n_embd, n_head=n_head, n_layer=n_layer,
                           block_size=BLOCK,
                           self_referential=arm == "proteus-srwm").to(device)


def train_on_task(arm: str, steps: int, seed: int, device: str, lr: float = 1e-3):
    torch.manual_seed(seed)
    model = build(arm, device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(steps):
        x, y = repeating_task(16, BLOCK, seed=seed * 100_000 + step)
        loss = model(x.to(device), y.to(device))[1]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


@torch.no_grad()
def measure_gap(model, seed: int, device: str) -> float:
    """First-half minus second-half loss on a sequence the model has never seen."""
    model.eval()
    x, y = repeating_task(16, BLOCK, seed=seed + 999)
    x, y = x.to(device), y.to(device)
    logits = model(x)[0]
    half = BLOCK // 2
    first = F.cross_entropy(logits[:, :half].reshape(-1, VOCAB), y[:, :half].reshape(-1))
    second = F.cross_entropy(logits[:, half:].reshape(-1, VOCAB), y[:, half:].reshape(-1))
    return (first - second).item()


@torch.no_grad()
def final_norm(model, device: str) -> float:
    """Frobenius norm of the self-written matrix at the last token."""
    x, _ = repeating_task(1, BLOCK, seed=7)
    return model.weight_norms(x.to(device))[-1]


def trace(arm: str, steps: int, seed: int, device: str) -> None:
    """The original per-step log: loss beside the norm, on a fixed batch."""
    torch.manual_seed(seed)
    model = build(arm, device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x, y = repeating_task(8, BLOCK, seed=seed)
    x, y = x.to(device), y.to(device)
    print(f"{'step':>6} | {'loss':>8} | {'||W|| final token':>18}")
    for step in range(steps + 1):
        if step % max(1, steps // 10) == 0:
            with torch.no_grad():
                norm = model.weight_norms(x[:1])[-1] if arm != "dense" else float("nan")
            print(f"{step:>6} | {model(x, y)[1].item():>8.4f} | {norm:>18.4f}")
        loss = model(x, y)[1]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--trace", action="store_true",
                    help="also print the per-step norm log for each Proteus arm")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = list(range(args.seeds))
    print(f"\nProteus probe -- {args.steps} steps, {len(seeds)} seeds, device={device}")
    print("task: predict a tiled random motif; the rule is learnable within one "
          "sequence\n")

    # Train once per (arm, seed); both tables read the same models.
    models: dict = {}
    params: dict = {}

    def gap_runner(arm: str):
        def go(seed: int) -> float:
            key = (arm, seed)
            if key not in models:
                models[key] = train_on_task(arm, args.steps, seed, device)
                params[arm] = sum(p.numel() for p in models[key].parameters())
            return measure_gap(models[key], seed, device)
        return go

    print("[1] within-sequence adaptation (first half minus second half, nats)")
    gaps = [run_arm(a, gap_runner(a), seeds) for a in ARMS]
    for arm, name in zip(gaps, ARMS):
        arm.note = f"{params[name]:,} params"
    print("\n" + table(gaps, "gaps"))

    by = {a.label: a for a in gaps}
    print("\n" + delta(by["dense"], by["proteus"]))
    print("    ^ the claim: does self-modification adapt better than attention does")
    print(delta(by["proteus"], by["proteus-srwm"]))
    print("    ^ whether the fully self-referential level buys anything over level 1")

    print("\n[2] stability of the self-written matrix (final ||W||)")

    def norm_runner(arm: str):
        def go(seed: int):
            value = final_norm(models[(arm, seed)], device)
            if not math.isfinite(value) or value > NORM_CEILING:
                return None            # diverged: recorded as an exclusion, not a number
            return value
        return go

    norms = [run_arm(a, norm_runner(a), seeds, progress=False)
             for a in ("proteus", "proteus-srwm")]
    print("\n" + table(norms, "norms"))
    for arm in norms:
        if arm.excluded:
            print(f"\n{arm.label}: {len(arm.excluded)}/{len(seeds)} seed(s) exceeded "
                  f"||W|| = {NORM_CEILING:g}. That is the documented failure mode, "
                  f"and it is the result -- not a lost sample.")

    if args.trace:
        for arm in ("proteus", "proteus-srwm"):
            print(f"\n[trace] {arm}")
            trace(arm, args.steps, seeds[0], device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
