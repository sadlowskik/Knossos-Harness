"""Train a Daedalus model on byte-level code data, with checkpoint-and-resume.

Prepare data first (data.py or scripts/fetch_rust.py), then e.g.:

    python train.py --model dense     --steps 3000
    python train.py --model labyrinth --steps 3000 --variable-loops
    python train.py --model moe       --steps 3000
    python train.py --model adaptive  --n-embd 512 --core-layers 3 --steps 40000 --resume
    python train.py --model labyrinth --steps 3000 --mixer moirai         # fast-weight core
    python train.py --model full      --steps 3000 --n-mem-banks 4        # mixture-of-memories
    python train.py --model labyrinth --steps 3000 --echo-weight 0.5      # loop self-distillation
    python train.py --model proteus   --steps 3000                        # self-modifying weights

`--resume` loads the checkpoint at --out and continues, so long runs can span
multiple sessions. The checkpoint is saved every --eval-interval steps.
"""
from __future__ import annotations
import argparse
import math
import os
import random
import time

import torch

from daedalus import (Daedalus, Labyrinth, DaedalusMoE, UnifiedDaedalus,
                      Ariadne, ponder_loss, DaedalusFull, DaedalusFullAdaptive,
                      DaedalusProteus, echo_step, echo_from_steps)
from data import load_splits, get_batch


def build_model(name, args, device):
    c = dict(vocab_size=256, n_embd=args.n_embd, n_head=args.n_head, block_size=args.block_size)
    if name == "dense":     return Daedalus(**c, n_layer=args.n_layer).to(device)
    if name == "labyrinth": return Labyrinth(**c, core_layers=args.core_layers, n_loops=args.n_loops, mixer=args.mixer).to(device)
    if name == "moe":       return DaedalusMoE(**c, n_layer=args.n_layer, n_experts=args.n_experts).to(device)
    if name == "unified":   return UnifiedDaedalus(**c, core_layers=args.core_layers, n_loops=args.n_loops, n_experts=args.n_experts).to(device)
    if name == "ariadne":   return Ariadne(**c, core_layers=args.core_layers, max_loops=args.max_loops).to(device)
    if name == "full":      return DaedalusFull(**c, core_layers=args.core_layers, n_loops=args.n_loops, n_experts=args.n_experts, n_gist=args.n_gist, n_stages=args.n_stages, n_mem_banks=args.n_mem_banks).to(device)
    if name == "adaptive":  return DaedalusFullAdaptive(**c, core_layers=args.core_layers, max_loops=args.max_loops, n_experts=args.n_experts, n_gist=args.n_gist, n_stages=args.n_stages, n_mem_banks=args.n_mem_banks).to(device)
    if name == "proteus":   return DaedalusProteus(**c, n_layer=args.n_layer, self_referential=args.self_referential).to(device)
    raise ValueError(f"unknown model: {name}")


def train_loss(name, model, x, y, args):
    if name in ("dense", "proteus"):
        return model(x, y)[1]
    if name == "labyrinth":
        if args.echo_weight > 0.0:
            # Echo: a shallow pass is trained to agree with the full-depth pass.
            return echo_step(model, x, y, model.n_loops, args.echo_weight,
                             kind=args.echo_kind, temperature=args.echo_temp)
        if args.variable_loops:
            r = model.n_loops if random.random() < 0.5 else random.randint(2, model.n_loops * 2)
            logits = model(x, n_loops=r)[0]
            b, t, v = logits.shape
            return torch.nn.functional.cross_entropy(logits.view(b * t, v), y.view(b * t))
        return model(x, y)[1]
    if name in ("moe", "unified", "full"):
        if name == "full" and args.echo_weight > 0.0:
            # echo_step supplies the CE (at the sampled shallow depth) + distillation;
            # the aux term still comes from a full-depth pass.
            _, _, aux = model(x, y)
            return echo_step(model, x, y, model.n_loops, args.echo_weight,
                             kind=args.echo_kind, temperature=args.echo_temp) + args.alpha * aux
        _, ce, aux = model(x, y)
        return ce + args.alpha * aux
    if name == "ariadne":
        p, logits = model(x)
        loss = ponder_loss(p, logits, y, args.lambda_prior, args.beta)[0]
        # per-loop logits already exist for the halting loss -- the teacher is free
        return loss + echo_from_steps(logits, args.echo_k, args.echo_weight,
                                      args.echo_kind, args.echo_temp)
    if name == "adaptive":
        _, loss, extras = model(x, y, lambda_prior=args.lambda_prior,
                                beta=args.beta, alpha=args.alpha)
        if args.echo_weight > 0.0 and "step_logits" in extras:
            loss = loss + echo_from_steps(extras["step_logits"], args.echo_k,
                                          args.echo_weight, args.echo_kind, args.echo_temp)
        return loss


@torch.no_grad()
def val_loss(name, model, data, args, device, iters=30):
    model.eval()
    losses = []
    for _ in range(iters):
        x, y = get_batch(data, "val", args.batch_size, args.block_size, device)
        if name in ("dense", "labyrinth", "moe", "unified", "full", "proteus"):
            losses.append(model(x, y)[1].item())
        elif name == "ariadne":
            p, logits = model(x)
            losses.append(ponder_loss(p, logits, y, args.lambda_prior, args.beta)[1].item())
        elif name == "adaptive":
            losses.append(model(x, y, beta=args.beta)[2]["l_rec"].item())
    model.train()
    return sum(losses) / len(losses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="adaptive",
                    choices=["dense", "labyrinth", "moe", "unified", "ariadne", "full",
                             "adaptive", "proteus"])
    ap.add_argument("--data", default="./data")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--n-embd", type=int, default=512)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--n-layer", type=int, default=3)
    ap.add_argument("--core-layers", type=int, default=3)
    ap.add_argument("--n-loops", type=int, default=4)
    ap.add_argument("--max-loops", type=int, default=6)
    ap.add_argument("--n-stages", type=int, default=2)
    ap.add_argument("--n-experts", type=int, default=8)
    ap.add_argument("--n-gist", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--alpha", type=float, default=0.01, help="MoE aux-loss weight")
    ap.add_argument("--beta", type=float, default=0.1, help="adaptive-halting ponder weight")
    ap.add_argument("--lambda-prior", type=float, default=0.2)
    ap.add_argument("--variable-loops", action="store_true")
    ap.add_argument("--mixer", default="softmax", choices=["softmax", "moirai"],
                    help="how the Labyrinth core mixes tokens (Moirai = gated fast weights)")
    ap.add_argument("--n-mem-banks", type=int, default=1,
                    help="Naiads memory banks in full/adaptive (1 = single Mnemosyne bank)")
    ap.add_argument("--echo-weight", type=float, default=0.0,
                    help="loop self-distillation weight (0 = off)")
    ap.add_argument("--echo-k", type=int, default=1, help="Echo student loop count")
    ap.add_argument("--echo-kind", default="kl", choices=["kl", "mse"])
    ap.add_argument("--echo-temp", type=float, default=1.0)
    ap.add_argument("--self-referential", action="store_true",
                    help="proteus: full SRWM (the update rule modifies itself)")
    ap.add_argument("--eval-interval", type=int, default=500)
    ap.add_argument("--out", default="./checkpoint.pt")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_splits(args.data)
    model = build_model(args.model, args, device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    start, best = 0, float("inf")
    if args.resume and os.path.exists(args.out):
        ck = torch.load(args.out, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start, best = ck["step"], ck["best"]
        print(f"resumed from step {start} (best {best:.4f})")

    n = sum(p.numel() for p in model.parameters())
    print(f"model={args.model}  params={n:,} (~{n/1e6:.1f}M)  device={device}")

    t0 = time.time()
    for step in range(start, args.steps + 1):
        if step % args.eval_interval == 0:
            v = val_loss(args.model, model, data, args, device)
            best = min(best, v)
            print(f"step {step:6d} | val {v:.4f} | {v/math.log(2):.2f} bits/byte | {time.time()-t0:.0f}s")
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "best": best, "args": vars(args)}, args.out)

        x, y = get_batch(data, "train", args.batch_size, args.block_size, device)
        loss = train_loss(args.model, model, x, y, args)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    print(f"\ndone. best val {best:.4f} ({best/math.log(2):.2f} bits/byte) -> {args.out}")


if __name__ == "__main__":
    main()
