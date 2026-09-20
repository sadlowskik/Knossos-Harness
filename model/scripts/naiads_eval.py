"""Naiads acceptance gate: is it the routing, or is it the parameters?

The segment-prediction check the README reports -- predict segment B from the
compressed gist of the preceding segment A -- run four ways:

    no memory       gist zeroed (the floor: how much does memory matter at all)
    1 bank          Mnemosyne, the number already in the README
    n banks, all    n banks, every one updated every segment, mean readout
    n banks, top-k  Naiads: routed, and unselected banks copied through untouched

The third arm is the one this script exists for. Comparing Naiads against a
single bank confounds two different things, because n banks is also n times the
memory parameters -- "4 banks beat 1 bank" and "more parameters beat fewer" are
the same measurement. So:

    (n banks, all) - (1 bank)        what the extra capacity buys
    (n banks, top-k) - (n banks, all)  what *routing and isolation* buy

Only the second is Naiads' claim. The module's docstring is explicit that the
bit-exactness of unselected banks "is the whole point" -- so the control is a
model with the same banks and the same capacity that simply does not route.
The two n-bank arms differ only by the router, whose parameter count is printed.

    python scripts/naiads_eval.py --data ./data --n-banks 4
    python scripts/naiads_eval.py --data ./data --seeds 5 --steps 2000
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from daedalus import Embeddings, Block, Mnemosyne, Naiads       # noqa: E402
from scripts.seeds import DEFAULT_SEEDS, delta, run_arm, table  # noqa: E402

VOCAB = 256


def multi_source_corpus(n: int = 200_000, n_sources: int = 4, run: int = 4096,
                        seed: int = 0):
    """A byte stream that switches between `n_sources` distinguishable subjects.

    Naiads' claim is about interference between *unrelated* context, so the
    corpus has to contain unrelated context. `echo_sweep.synthetic_corpus` draws
    every motif from one pool: a memory that smears has nothing to smear into,
    and both n-bank arms would look identical for reasons that say nothing about
    routing.

    Here each source owns a disjoint slice of the byte range and holds the
    stream for `run` bytes at a time, so a segment pair usually falls inside one
    subject and consecutive pairs often do not.
    """
    g = torch.Generator().manual_seed(seed)
    span = max(4, (127 - 32) // n_sources)
    pools = []
    for s in range(n_sources):
        lo = 32 + s * span
        pools.append([torch.randint(lo, lo + span, (k,), generator=g) for k in (3, 5, 8)])

    out, total = [], 0
    while total < n:
        pool = pools[torch.randint(0, n_sources, (1,), generator=g).item()]
        written = 0
        while written < run and total < n:
            motif = pool[torch.randint(0, len(pool), (1,), generator=g).item()]
            out.append(motif)
            written += len(motif)
            total += len(motif)
    stream = torch.cat(out)[:n]
    cut = int(len(stream) * 0.9)
    return stream[:cut], stream[cut:]


class UnroutedBanks(nn.Module):
    """`n_banks` gist banks with no routing: all updated, mean readout.

    The capacity-matched control. Same number of `Mnemosyne` compressors as
    Naiads and the same state shape, minus the router and minus the pass-through
    of unselected banks -- which is precisely the mechanism under test.

    Its interface matches `Naiads.forward` so `SegmentPredictor` treats the two
    identically; `scores` is None because there is nothing to balance.
    """

    def __init__(self, n_embd: int, n_gist: int, n_head: int, n_banks: int):
        super().__init__()
        self.n_banks, self.n_gist = n_banks, n_gist
        self.compress = nn.ModuleList(
            [Mnemosyne(n_embd, n_gist, n_head) for _ in range(n_banks)])
        self.init_state = nn.Parameter(torch.randn(n_banks, n_gist, n_embd) * 0.02)
        self.read = nn.MultiheadAttention(n_embd, n_head, batch_first=True)
        self.ln = nn.LayerNorm(n_embd)

    def forward(self, x, state=None):
        b = x.shape[0]
        if state is None:
            state = self.init_state.unsqueeze(0).expand(b, -1, -1, -1).to(x.device, x.dtype)
        banks = [self.ln(state[:, e] + self.compress[e](x)) for e in range(self.n_banks)]
        new_state = torch.stack(banks, dim=1)
        # Every bank contributes equally: no router, so no selection.
        readout = sum(self.read(x, bank, bank)[0] for bank in banks) / self.n_banks
        return readout, new_state, None


class SegmentPredictor(nn.Module):
    """Encode segment A -> memory -> predict segment B.

    Everything outside the memory is held identical across arms, so the
    comparison is about the memory and nothing else.
    """

    KINDS = ("one", "unrouted", "naiads")

    def __init__(self, kind: str, n_embd=128, n_head=4, n_gist=16, block=128,
                 n_banks=4, top_k=2):
        super().__init__()
        assert kind in self.KINDS, kind
        self.kind, self.n_gist = kind, n_gist
        self.emb = Embeddings(VOCAB, n_embd, block + n_gist)
        self.encoder = Block(n_embd, n_head, block + n_gist)
        if kind == "one":
            self.mem = Mnemosyne(n_embd, n_gist, n_head)
        elif kind == "unrouted":
            self.mem = UnroutedBanks(n_embd, n_gist, n_head, n_banks)
        else:
            self.mem = Naiads(n_embd, n_gist, n_head, n_banks, min(top_k, n_banks))
        self.decoder = Block(n_embd, n_head, block + n_gist)
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, VOCAB)

    def forward(self, seg_a, seg_b, targets_b, use_memory=True):
        enc = self.encoder(self.emb(seg_a))
        b_emb = self.emb(seg_b)

        if self.kind == "one":
            gist, aux = self.mem(enc), enc.new_zeros(())
        else:
            _read, state, scores = self.mem(enc)
            gist = state.mean(dim=1)
            aux = (self.mem.aux_loss(scores) if scores is not None
                   else enc.new_zeros(()))

        if not use_memory:
            gist = torch.zeros_like(gist)
        seq = torch.cat([gist, b_emb], dim=1)
        logits = self.lm_head(self.ln_f(self.decoder(seq)))[:, self.n_gist:, :]
        ce = F.cross_entropy(logits.reshape(-1, VOCAB), targets_b.reshape(-1))
        return ce, aux


def batches(stream, batch_size, seg, device, generator=None):
    ix = torch.randint(0, len(stream) - 2 * seg - 1, (batch_size,), generator=generator)
    a = torch.stack([stream[i:i + seg].long() for i in ix])
    b = torch.stack([stream[i + seg:i + 2 * seg].long() for i in ix])
    t = torch.stack([stream[i + seg + 1:i + 2 * seg + 1].long() for i in ix])
    return a.to(device), b.to(device), t.to(device)


def run(kind, train_stream, val_stream, args, device, seed):
    """Train one arm and return (val loss with memory, without memory, params)."""
    torch.manual_seed(seed)
    model = SegmentPredictor(kind, block=args.seg, n_banks=args.n_banks,
                             top_k=args.top_k).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    g = torch.Generator().manual_seed(seed)          # identical data order per seed
    for _ in range(args.steps):
        ce, aux = model(*batches(train_stream, args.batch_size, args.seg, device, g))
        loss = ce + args.alpha * aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    with torch.no_grad():
        out = {}
        for use_mem in (True, False):
            ge = torch.Generator().manual_seed(seed + 7)   # identical eval batches
            total = sum(
                model(*batches(val_stream, args.batch_size, args.seg, device, ge),
                      use_memory=use_mem)[0].item()
                for _ in range(args.eval_iters))
            out["with" if use_mem else "without"] = total / args.eval_iters
    return out, sum(p.numel() for p in model.parameters())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=None,
                    help="byte-corpus dir (default: a synthetic multi-source stream)")
    ap.add_argument("--n-sources", type=int, default=4,
                    help="distinguishable subjects in the synthetic corpus")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--seg", type=int, default=128)
    ap.add_argument("--n-banks", type=int, default=4)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--eval-iters", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--alpha", type=float, default=0.01)
    args = ap.parse_args()

    if args.data:
        from data import load_splits
        splits = load_splits(args.data)
        train_stream, val_stream = splits["train"], splits["val"]
        corpus = args.data
    else:
        train_stream, val_stream = multi_source_corpus(n_sources=args.n_sources)
        corpus = f"synthetic, {args.n_sources} distinguishable sources"
    d = {"train": train_stream, "val": val_stream}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = list(range(args.seeds))

    print(f"\nNaiads sweep -- predict {args.seg} tokens from the gist of the prior "
          f"{args.seg}")
    print(f"{args.steps} steps, {len(seeds)} seeds, {args.n_banks} banks, "
          f"top-{args.top_k}, device={device}")
    print(f"corpus: {corpus}\n")

    labels = {"one": "1 bank", "unrouted": f"{args.n_banks} banks, all",
              "naiads": f"{args.n_banks} banks, top-{args.top_k}"}
    params: dict = {}
    floors: dict = {}

    def runner(kind: str):
        def go(seed: int) -> float:
            out, n = run(kind, d["train"], d["val"], args, device, seed)
            params[kind] = n
            floors.setdefault(kind, []).append(out["without"])
            return out["with"]
        return go

    arms = [run_arm(labels[k], runner(k), seeds) for k in SegmentPredictor.KINDS]
    for arm, kind in zip(arms, SegmentPredictor.KINDS):
        floor = sum(floors[kind]) / len(floors[kind]) if floors.get(kind) else float("nan")
        arm.note = (f"{params[kind]:,} params | memory-off floor {floor:.4f} "
                    f"(gain {floor - arm.mean:+.4f})")

    print("\n" + table(arms, "losses"))

    by = {a.label: a for a in arms}
    print("\n" + delta(by[labels["one"]], by[labels["unrouted"]]))
    print("    ^ capacity only: more banks, no routing")
    print(delta(by[labels["unrouted"]], by[labels["naiads"]]))
    print("    ^ the claim: routing and bit-exact isolation, at matched capacity")
    print("\n" + delta(by[labels["one"]], by[labels["naiads"]]))
    print("    ^ the headline the README reports -- confounds both of the above")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
