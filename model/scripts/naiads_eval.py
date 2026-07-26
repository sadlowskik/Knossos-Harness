"""Naiads acceptance gate: multi-bank memory vs the single Mnemosyne bank.

Runs the same segment-prediction check the README already reports -- predict
segment B from the compressed gist of the preceding 128 tokens -- three ways:

    no memory      gist zeroed (the ablation floor)
    1 bank         Mnemosyne, i.e. the number already in the README
    n banks        Naiads with top-k segment routing

and prints the deltas. The interesting number is (1 bank - n banks): whether
splitting the memory and routing to it actually reduces interference, or just
adds parameters.

    python scripts/naiads_eval.py --data ./data --n-banks 4
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

VOCAB = 256


class SegmentPredictor(nn.Module):
    """Encode segment A -> memory -> predict segment B.

    `n_banks=1` is Mnemosyne (the MemoryModel in daedalus/memory.py); `n_banks>1`
    routes through Naiads. Everything else is held identical so the comparison is
    about the memory and nothing else.
    """

    def __init__(self, n_embd=128, n_head=4, n_gist=16, block=128, n_banks=1, top_k=2):
        super().__init__()
        self.n_banks, self.n_gist = n_banks, n_gist
        self.emb = Embeddings(VOCAB, n_embd, block + n_gist)
        self.encoder = Block(n_embd, n_head, block + n_gist)
        if n_banks == 1:
            self.mem = Mnemosyne(n_embd, n_gist, n_head)
        else:
            self.mem = Naiads(n_embd, n_gist, n_head, n_banks, min(top_k, n_banks))
        self.decoder = Block(n_embd, n_head, block + n_gist)
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, VOCAB)

    def forward(self, seg_a, seg_b, targets_b, use_memory=True):
        enc = self.encoder(self.emb(seg_a))
        b_emb = self.emb(seg_b)
        if self.n_banks == 1:
            gist = self.mem(enc)
            if not use_memory:
                gist = torch.zeros_like(gist)
            seq = torch.cat([gist, b_emb], dim=1)
            logits = self.lm_head(self.ln_f(self.decoder(seq)))[:, self.n_gist:, :]
            aux = enc.new_zeros(())
        else:
            _read, state, scores = self.mem(enc)
            # flatten the routed banks into the same "prepend gists" interface
            gist = state.mean(dim=1)
            if not use_memory:
                gist = torch.zeros_like(gist)
            seq = torch.cat([gist, b_emb], dim=1)
            logits = self.lm_head(self.ln_f(self.decoder(seq)))[:, self.n_gist:, :]
            aux = self.mem.aux_loss(scores)
        ce = F.cross_entropy(logits.reshape(-1, VOCAB), targets_b.reshape(-1))
        return ce, aux


def batches(stream, batch_size, seg, device, generator=None):
    ix = torch.randint(0, len(stream) - 2 * seg - 1, (batch_size,), generator=generator)
    a = torch.stack([stream[i:i + seg].long() for i in ix])
    b = torch.stack([stream[i + seg:i + 2 * seg].long() for i in ix])
    t = torch.stack([stream[i + seg + 1:i + 2 * seg + 1].long() for i in ix])
    return a.to(device), b.to(device), t.to(device)


def run(train_stream, val_stream, n_banks, steps, seg, device, seed=0, alpha=0.01):
    torch.manual_seed(seed)
    model = SegmentPredictor(block=seg, n_banks=n_banks).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        ce, aux = model(*batches(train_stream, 16, seg, device, g))
        loss = ce + alpha * aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    with torch.no_grad():
        out = {}
        for use_mem in (True, False):
            g = torch.Generator().manual_seed(seed + 7)
            total = sum(model(*batches(val_stream, 16, seg, device, g), use_memory=use_mem)[0].item()
                        for _ in range(30))
            out["with" if use_mem else "without"] = total / 30
    params = sum(p.numel() for p in model.parameters())
    return out, params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--seg", type=int, default=128)
    ap.add_argument("--n-banks", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from data import load_splits
    d = load_splits(args.data)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"segment prediction: predict 128 tokens from the gist of the prior {args.seg}\n")
    print(f"{'memory':>10} | {'params':>10} | {'with mem':>9} | {'without':>9} | {'gain':>7}")
    results = {}
    for banks in (1, args.n_banks):
        r, params = run(d["train"], d["val"], banks, args.steps, args.seg, device, args.seed)
        results[banks] = r
        label = "1 (Mnemo)" if banks == 1 else f"{banks} (Naiads)"
        print(f"{label:>10} | {params:>10,} | {r['with']:>9.4f} | {r['without']:>9.4f} | "
              f"{r['without'] - r['with']:>+7.4f}")

    delta = results[args.n_banks]["with"] - results[1]["with"]
    print(f"\n{args.n_banks} banks vs 1 bank: {delta:+.4f} nats "
          f"({'Naiads helps' if delta < 0 else 'no improvement from splitting the memory'})")


if __name__ == "__main__":
    main()
