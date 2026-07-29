"""Train a Daedalus model, at toy scale or for a real multi-billion-token run.

Prepare data first, then train:

    # toy scale (byte-level, in-RAM splits) -- unchanged from before
    python data.py --source /usr/lib/python3.12 --out ./data
    python train.py --model labyrinth --steps 3000 --variable-loops

    # real scale (BPE, memmapped shards from scripts/prepare_corpus.py)
    python scripts/prepare_corpus.py --preset fineweb-edu --out ./corpus/nl \
        --train-tokenizer --target-tokens 2_000_000_000
    python train.py --model adaptive --data ./corpus/nl --max-tokens 2e9 \
        --n-embd 768 --core-layers 4 --batch-size 16 --grad-accum 8 --compile

The data directory decides the mode: a `meta.json` means memmapped BPE shards,
otherwise the legacy `{train,val,test}.pt` byte tensors are used.

Phase 2 (code) continues from phase 1's weights with a fresh schedule:

    python train.py --data ./corpus/code --init-from ./ckpt/nl_best.pt \
        --lr 1e-4 --max-tokens 5e8 --out ./ckpt/code.pt

`--resume` restarts an interrupted run of the *same* phase, restoring the
optimizer, schedule position and RNG, so long runs can span many sessions.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import shutil
import time

import torch
import torch.nn as nn

from daedalus import (Daedalus, Labyrinth, DaedalusMoE, UnifiedDaedalus,
                      Ariadne, ponder_loss, DaedalusFull, DaedalusFullAdaptive,
                      DaedalusProteus, echo_step, echo_from_steps)
from data import load_splits, get_batch, load_corpus


def build_model(name, args, device):
    c = dict(vocab_size=args.vocab_size, n_embd=args.n_embd, n_head=args.n_head,
             block_size=args.block_size)
    if name == "dense":     return Daedalus(**c, n_layer=args.n_layer).to(device)
    if name == "labyrinth": return Labyrinth(**c, core_layers=args.core_layers, n_loops=args.n_loops, mixer=args.mixer).to(device)
    if name == "moe":       return DaedalusMoE(**c, n_layer=args.n_layer, n_experts=args.n_experts).to(device)
    if name == "unified":   return UnifiedDaedalus(**c, core_layers=args.core_layers, n_loops=args.n_loops, n_experts=args.n_experts).to(device)
    if name == "ariadne":   return Ariadne(**c, core_layers=args.core_layers, max_loops=args.max_loops).to(device)
    if name == "full":      return DaedalusFull(**c, core_layers=args.core_layers, n_loops=args.n_loops, n_experts=args.n_experts, n_gist=args.n_gist, n_stages=args.n_stages, n_mem_banks=args.n_mem_banks, loop_embed=args.loop_embed).to(device)
    if name == "adaptive":  return DaedalusFullAdaptive(**c, core_layers=args.core_layers, max_loops=args.max_loops, n_experts=args.n_experts, n_gist=args.n_gist, n_stages=args.n_stages, n_mem_banks=args.n_mem_banks, loop_embed=args.loop_embed, inject_gate=args.inject_gate, qk_norm=args.qk_norm, n_kv_head=args.n_kv_head, bias_update=args.bias_update, mtp=args.mtp).to(device)
    if name == "proteus":   return DaedalusProteus(**c, n_layer=args.n_layer, self_referential=args.self_referential).to(device)
    raise ValueError(f"unknown model: {name}")


# --------------------------------------------------------------------------
# Initialisation, weight tying, optimizer groups
# --------------------------------------------------------------------------

def token_embedding(model):
    """The token embedding, wherever this model keeps it."""
    emb = getattr(model, "tok_emb", None)
    if emb is None:
        emb = getattr(getattr(model, "emb", None), "tok_emb", None)
    return emb


def init_weights(model, n_effective_layers: int) -> None:
    """GPT-2 style init: N(0, 0.02), with residual projections scaled down.

    Without the 1/sqrt(2N) scaling on the projections that write into the
    residual stream, the stream's variance grows with depth and deep models
    start unstable (Radford et al. 2019, section 2.3). Recurrent-depth models
    make this *worse*, because the effective depth is core_layers x n_loops --
    which is what `n_effective_layers` accounts for.
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
    scale = 0.02 / math.sqrt(2 * max(n_effective_layers, 1))
    for name, p in model.named_parameters():
        # `proj` = attention output projection; `net.2` = the MLP's second layer.
        # Both write directly into the residual stream.
        if name.endswith("proj.weight") or name.endswith("net.2.weight"):
            nn.init.normal_(p, mean=0.0, std=scale)


def tie_weights(model) -> bool:
    """Share the token embedding with the output head (Press & Wolf 2017).

    At vocab 32768 x n_embd 768 the embedding is 25M parameters. Untied, the
    model pays for it twice -- on a 100M-parameter budget that is half the model
    spent on two copies of the same table. Tying also tends to help perplexity
    slightly at small scale.
    """
    emb, head = token_embedding(model), getattr(model, "lm_head", None)
    if emb is None or head is None or emb.weight.shape != head.weight.shape:
        return False
    head.weight = emb.weight
    return True


def make_optimizer(model, lr, weight_decay, betas):
    """AdamW with decay on matrices only.

    Weight-decaying LayerNorm gains and biases shrinks parameters that have no
    business being shrunk -- a normalisation gain pulled toward zero just
    fights the normalisation. Standard practice is to decay >=2D tensors only.
    """
    decay, no_decay = [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    fused = torch.cuda.is_available()
    try:
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=fused)
    except (RuntimeError, TypeError):
        return torch.optim.AdamW(groups, lr=lr, betas=betas)


def lr_at(step: int, args) -> float:
    """Linear warmup, then cosine decay to --min-lr."""
    if step < args.warmup:
        return args.lr * (step + 1) / max(args.warmup, 1)
    total = args.decay_steps or args.steps
    if step >= total:
        return args.min_lr
    ratio = (step - args.warmup) / max(total - args.warmup, 1)
    return args.min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (args.lr - args.min_lr)


# --------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------

def document_ids(x, eot_id):
    """Segment ids for block-diagonal attention, derived from EOT positions.

    The separator is subtracted back out so the <|endoftext|> token belongs to
    the document it *terminates* rather than opening the next one -- otherwise
    every document would start with a token it cannot attend to.
    """
    if eot_id is None:
        return None
    is_eot = (x == eot_id)
    return is_eot.cumsum(dim=1) - is_eot.long()


def z_loss(logits):
    """logsumexp(logits)^2, averaged. Computed in fp32 -- the quantity being
    controlled is exactly the one that overflows in fp16."""
    return torch.logsumexp(logits.float(), dim=-1).pow(2).mean()


def train_loss(name, model, x, y, args, doc_ids=None):
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
            return torch.nn.functional.cross_entropy(logits.view(b * t, v), y.reshape(b * t))
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
        logits, loss, extras = model(x, y, lambda_prior=args.lambda_prior,
                                     beta=args.beta, alpha=args.alpha,
                                     doc_ids=doc_ids, mtp_weight=args.mtp_weight)
        if args.z_weight > 0.0:
            loss = loss + args.z_weight * z_loss(logits)
        if args.echo_weight > 0.0 and "step_logits" in extras:
            loss = loss + echo_from_steps(extras["step_logits"], args.echo_k,
                                          args.echo_weight, args.echo_kind, args.echo_temp)
        return loss
    raise ValueError(f"unknown model: {name}")


@torch.no_grad()
def evaluate(name, model, batcher, args, ctx, iters=50, eot_id=None):
    """Validation loss on a *fixed* set of windows.

    The generator is re-seeded every call so successive evaluations score the
    same windows -- otherwise the eval noise is comparable to the improvement
    being measured and the curve is unreadable.
    """
    model.eval()
    gen = torch.Generator().manual_seed(args.seed + 777)
    losses = []
    for _ in range(iters):
        x, y = batcher("val", gen)
        with ctx:
            if name in ("dense", "labyrinth", "moe", "unified", "full", "proteus"):
                losses.append(model(x, y)[1].item())
            elif name == "ariadne":
                p, logits = model(x)
                losses.append(ponder_loss(p, logits, y, args.lambda_prior, args.beta)[1].item())
            elif name == "adaptive":
                d = document_ids(x, eot_id) if eot_id is not None else None
                losses.append(model(x, y, beta=args.beta,
                                    doc_ids=d)[2]["l_rec"].item())
    model.train()
    return sum(losses) / len(losses)


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="adaptive",
                    choices=["dense", "labyrinth", "moe", "unified", "ariadne", "full",
                             "adaptive", "proteus"])
    ap.add_argument("--data", default="./data")

    budget = ap.add_argument_group("budget")
    budget.add_argument("--steps", type=int, default=40000)
    budget.add_argument("--max-tokens", type=float, default=0,
                        help="stop after this many tokens (overrides --steps)")
    budget.add_argument("--batch-size", type=int, default=8)
    budget.add_argument("--block-size", type=int, default=256)
    budget.add_argument("--grad-accum", type=int, default=1,
                        help="micro-batches per optimizer step")

    arch = ap.add_argument_group("architecture")
    arch.add_argument("--vocab-size", type=int, default=0,
                      help="0 = read from the corpus meta.json (else 256)")
    arch.add_argument("--n-embd", type=int, default=512)
    arch.add_argument("--n-head", type=int, default=8)
    arch.add_argument("--n-layer", type=int, default=3)
    arch.add_argument("--core-layers", type=int, default=3)
    arch.add_argument("--n-loops", type=int, default=4)
    arch.add_argument("--max-loops", type=int, default=6)
    arch.add_argument("--n-stages", type=int, default=2)
    arch.add_argument("--n-experts", type=int, default=8)
    arch.add_argument("--n-gist", type=int, default=32)
    arch.add_argument("--inject-gate", action="store_true",
                      help="learned per-iteration gate on the input injection, so "
                           "the residual stream stops growing unchecked across loops")
    arch.add_argument("--qk-norm", action="store_true",
                      help="RMSNorm on q and k; caps attention-logit growth across "
                           "a shared-weight recurrence")
    arch.add_argument("--n-kv-head", type=int, default=None,
                      help="grouped-query attention: fewer KV heads than query "
                           "heads, shrinking the KV cache at inference")
    arch.add_argument("--bias-update", type=float, default=0.0,
                      help="aux-loss-free MoE balancing rate (e.g. 1e-3). Steers "
                           "expert selection by a bias instead of taxing the loss; "
                           "pair with a lower --alpha")
    arch.add_argument("--mtp", action="store_true",
                      help="multi-token prediction head (predicts t+2 as well)")
    arch.add_argument("--loop-embed", action="store_true",
                      help="learned per-iteration embedding on each recurrent "
                           "core (Universal-Transformer style). Off by default: "
                           "it changes parameter shapes, so a checkpoint trained "
                           "without it cannot be --init-from'd into one with it.")
    arch.add_argument("--mixer", default="softmax", choices=["softmax", "moirai"],
                      help="how the Labyrinth core mixes tokens (Moirai = gated fast weights)")
    arch.add_argument("--n-mem-banks", type=int, default=1,
                      help="Naiads memory banks in full/adaptive (1 = single Mnemosyne bank)")
    arch.add_argument("--no-tie-weights", action="store_true")

    opt_g = ap.add_argument_group("optimization")
    opt_g.add_argument("--lr", type=float, default=6e-4)
    opt_g.add_argument("--min-lr", type=float, default=6e-5)
    opt_g.add_argument("--warmup", type=int, default=500)
    opt_g.add_argument("--decay-steps", type=int, default=0,
                       help="cosine floor is reached here (0 = --steps)")
    opt_g.add_argument("--weight-decay", type=float, default=0.1)
    opt_g.add_argument("--beta1", type=float, default=0.9)
    opt_g.add_argument("--beta2", type=float, default=0.95)
    opt_g.add_argument("--grad-clip", type=float, default=1.0)
    opt_g.add_argument("--precision", default="auto",
                       choices=["auto", "bf16", "fp16", "fp32"])
    opt_g.add_argument("--compile", action="store_true",
                       help="torch.compile (expect graph breaks on MoE/variable loops)")

    aux = ap.add_argument_group("model-specific")
    aux.add_argument("--alpha", type=float, default=0.01, help="MoE aux-loss weight")
    aux.add_argument("--beta", type=float, default=0.1, help="adaptive-halting ponder weight")
    aux.add_argument("--lambda-prior", type=float, default=0.2)
    aux.add_argument("--z-weight", type=float, default=0.0,
                     help="z-loss coefficient (e.g. 1e-4). Penalises logsumexp of "
                          "the logits, keeping them from drifting large -- cheap "
                          "insurance against an fp16 run diverging mid-schedule")
    aux.add_argument("--mtp-weight", type=float, default=0.0,
                     help="weight on the multi-token-prediction loss (needs --mtp)")
    aux.add_argument("--doc-mask", action="store_true",
                     help="block-diagonal attention within packed documents, so a "
                          "token cannot attend past an <|endoftext|> into an "
                          "unrelated document")
    aux.add_argument("--variable-loops", action="store_true")
    aux.add_argument("--echo-weight", type=float, default=0.0,
                     help="loop self-distillation weight (0 = off)")
    aux.add_argument("--echo-k", type=int, default=1, help="Echo student loop count")
    aux.add_argument("--echo-kind", default="kl", choices=["kl", "mse"])
    aux.add_argument("--echo-temp", type=float, default=1.0)
    aux.add_argument("--self-referential", action="store_true",
                     help="proteus: full SRWM (the update rule modifies itself)")

    io = ap.add_argument_group("io")
    io.add_argument("--eval-interval", type=int, default=500)
    io.add_argument("--eval-batches", type=int, default=50)
    io.add_argument("--log-interval", type=int, default=20)
    io.add_argument("--out", default="./checkpoint.pt")
    io.add_argument("--resume", action="store_true")
    io.add_argument("--init-from", default=None,
                    help="load weights from a checkpoint but start a fresh schedule")
    io.add_argument("--mirror", default=None,
                    help="second directory to copy checkpoints into, e.g. a "
                         "mounted Google Drive. Train against fast local disk "
                         "and keep the durable copy elsewhere.")
    io.add_argument("--mirror-every", type=int, default=4,
                    help="mirror every N evals (the copy is slow over FUSE)")
    io.add_argument("--max-hours", type=float, default=0,
                    help="save and exit cleanly after this long (0 = no limit). "
                         "Set it below the platform's session cap so a run ends "
                         "at a checkpoint instead of being killed between them.")
    io.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------------------------------------------------------- data
    if os.path.exists(os.path.join(args.data, "meta.json")):
        corpus = load_corpus(args.data)
        args.vocab_size = args.vocab_size or corpus["train"].vocab_size
        def batcher(split, gen=None):
            return corpus[split].batch(args.batch_size, args.block_size, device, gen)
        n_train = len(corpus["train"])
        packed = getattr(corpus["train"], "block_size", None)
        if packed is not None and packed != args.block_size:
            # Caught here rather than on the first batch, because the first batch
            # happens after the model, optimizer and any --resume have loaded.
            raise SystemExit(
                f"\n{args.data} is a packed prompt/completion corpus built for "
                f"--block-size {packed}, but this run passes {args.block_size}.\n"
                f"  Pass --block-size {packed}, or rebuild with "
                f"make_gec.py --block-size {args.block_size}.")
        eot_id = corpus["train"].meta.get("eot_id") if args.doc_mask else None
        if args.doc_mask and eot_id is None:
            raise SystemExit(
                "--doc-mask needs an eot_id in the corpus meta.json; this corpus "
                "has none, so document boundaries cannot be recovered.")
        kind = "prompt/completion pairs" if packed is not None else "flat"
        print(f"corpus: {n_train/1e9:.2f}B train tokens ({kind}), "
              f"vocab {args.vocab_size:,}")
    else:
        legacy = load_splits(args.data)
        eot_id = None
        args.vocab_size = args.vocab_size or 256
        def batcher(split, gen=None):
            return get_batch(legacy, split, args.batch_size, args.block_size, device)
        n_train = len(legacy["train"])
        print(f"corpus: {n_train/1e6:.1f}M train bytes (legacy byte-level)")

    # ------------------------------------------------------------- precision
    if args.precision == "auto":
        if device == "cuda":
            # Compute capability, NOT torch.cuda.is_bf16_supported(). That call
            # counts *emulated* bf16 and returns True on Turing (T4, sm_75) and
            # Pascal (P100, sm_60), neither of which has bf16 hardware -- so on
            # a Kaggle T4 it selects an emulated path that is far slower than
            # fp16 and, because the bf16 branch sets scaler=None, also drops
            # loss scaling. Real bf16 tensor cores start at Ampere (sm_80).
            args.precision = ("bf16" if torch.cuda.get_device_capability()[0] >= 8
                              else "fp16")
        else:
            args.precision = "fp32"
    if args.precision == "fp32" or device == "cpu":
        ctx, scaler = torch.autocast(device_type=device, enabled=False), None
    else:
        dt = torch.bfloat16 if args.precision == "bf16" else torch.float16
        ctx = torch.autocast(device_type="cuda", dtype=dt)
        # fp16 needs loss scaling to keep small gradients from flushing to zero;
        # bf16 has fp32's exponent range and does not.
        scaler = torch.amp.GradScaler("cuda") if args.precision == "fp16" else None

    # ----------------------------------------------------------------- model
    model = build_model(args.model, args, device)
    depth = {"dense": args.n_layer, "moe": args.n_layer, "proteus": args.n_layer}.get(
        args.model, args.core_layers * max(args.n_loops, args.max_loops) * args.n_stages)
    init_weights(model, depth)
    tied = (not args.no_tie_weights) and tie_weights(model)
    opt = make_optimizer(model, args.lr, args.weight_decay, (args.beta1, args.beta2))

    tokens_per_step = args.batch_size * args.block_size * args.grad_accum
    if args.max_tokens:
        args.steps = int(args.max_tokens / tokens_per_step)
    args.decay_steps = args.decay_steps or args.steps

    start, best = 0, float("inf")
    if args.init_from:
        ck = torch.load(args.init_from, map_location=device, weights_only=False)
        state = ck["model"] if "model" in ck else ck
        here = model.state_dict()
        bad = {k: (tuple(v.shape), tuple(here[k].shape)) for k, v in state.items()
               if k in here and v.shape != here[k].shape}
        if bad:
            old_v = ck.get("args", {}).get("vocab_size", "?")
            lines = "\n".join(f"    {k}: checkpoint {a} vs model {b}"
                              for k, (a, b) in list(bad.items())[:6])
            raise SystemExit(
                f"\n--init-from {args.init_from} is not compatible with this "
                f"configuration:\n{lines}\n\n"
                f"  The checkpoint was trained with vocab_size={old_v}; this run "
                f"uses {args.vocab_size}.\n"
                "  Embeddings are not transferable across tokenizers -- row 42 of "
                "the old table\n"
                "  means a raw byte, row 42 of the new one means a BPE token. "
                "There is nothing\n"
                "  to carry over, so train this phase from scratch (drop "
                "--init-from).\n\n"
                "  --init-from is for continuing a run onto NEW DATA with the "
                "SAME tokenizer,\n"
                "  which is exactly the phase-1 -> phase-2 handoff in "
                "TRAINING.md.")
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"init-from {args.init_from}: {len(missing)} missing, "
              f"{len(unexpected)} unexpected keys; fresh optimizer + schedule")
    if args.resume and os.path.exists(args.out):
        ck = torch.load(args.out, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if scaler is not None and ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        if ck.get("rng") is not None:
            torch.set_rng_state(ck["rng"].cpu().to(torch.uint8))
        start, best = ck["step"], ck["best"]
        print(f"resumed from step {start} (best {best:.4f})")

    raw_model = model
    if args.compile:
        model = torch.compile(model)

    n = sum(p.numel() for p in raw_model.parameters())
    emb = token_embedding(raw_model)
    n_emb = emb.weight.numel() if emb is not None else 0
    print(f"model={args.model}  params={n:,} (~{n/1e6:.1f}M, {n-n_emb*(1 if tied else 2):,} "
          f"non-embedding)  tied={tied}  device={device}  precision={args.precision}")
    print(f"budget: {args.steps:,} steps x {tokens_per_step:,} tok/step "
          f"= {args.steps*tokens_per_step/1e9:.2f}B tokens")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    log_path = os.path.splitext(args.out)[0] + ".jsonl"
    best_path = os.path.splitext(args.out)[0] + ".best.pt"

    def save(path, step, best):
        torch.save({"model": raw_model.state_dict(), "opt": opt.state_dict(),
                    "scaler": scaler.state_dict() if scaler else None,
                    "rng": torch.get_rng_state(), "step": step, "best": best,
                    "args": vars(args)}, path)

    def mirror(paths):
        """Copy checkpoints to durable storage.

        Written to a .tmp name and then moved, because a session that dies
        mid-copy would otherwise leave a truncated checkpoint where the good
        one used to be -- losing the run rather than the session.
        """
        if not args.mirror:
            return
        os.makedirs(args.mirror, exist_ok=True)
        for p in paths:
            if not os.path.exists(p):
                continue
            dst = os.path.join(args.mirror, os.path.basename(p))
            tmp = dst + ".tmp"
            try:
                shutil.copyfile(p, tmp)
                os.replace(tmp, dst)
            except OSError as e:
                print(f"  mirror failed for {os.path.basename(p)}: {e}")

    # ------------------------------------------------------------------ loop
    t0 = time.time()
    t_last, tok_last, n_evals = t0, 0, 0
    for step in range(start, args.steps + 1):
        lr = lr_at(step, args)
        for g in opt.param_groups:
            g["lr"] = lr

        if step % args.eval_interval == 0:
            v = evaluate(args.model, model, batcher, args, ctx, args.eval_batches,
                         eot_id if args.doc_mask else None)
            improved = v < best
            best = min(best, v)
            tokens = step * tokens_per_step
            print(f"step {step:7d} | val {v:.4f} | {v/math.log(2):.3f} bits/tok | "
                  f"lr {lr:.2e} | {tokens/1e9:.3f}B tok | {time.time()-t0:.0f}s"
                  + ("  *best*" if improved else ""))
            save(args.out, step, best)
            if improved:
                save(best_path, step, best)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, "val": v, "lr": lr,
                                    "tokens": tokens, "elapsed": time.time() - t0}) + "\n")
            n_evals += 1
            if args.mirror and n_evals % max(args.mirror_every, 1) == 0:
                mirror([args.out, best_path, log_path])

        opt.zero_grad(set_to_none=True)
        total = 0.0
        for micro in range(args.grad_accum):
            x, y = batcher("train")
            doc_ids = document_ids(x, eot_id) if args.doc_mask else None
            with ctx:
                loss = train_loss(args.model, model, x, y, args,
                                  doc_ids=doc_ids) / args.grad_accum
            (scaler.scale(loss) if scaler else loss).backward()
            total += loss.item()
        if args.grad_clip > 0:
            if scaler:
                scaler.unscale_(opt)          # clip real gradients, not scaled ones
            torch.nn.utils.clip_grad_norm_(raw_model.parameters(), args.grad_clip)
        if scaler:
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()

        if args.max_hours and (time.time() - t0) > args.max_hours * 3600:
            v = evaluate(args.model, model, batcher, args, ctx, args.eval_batches,
                         eot_id if args.doc_mask else None)
            best = min(best, v)
            save(args.out, step, best)
            mirror([args.out, best_path, log_path])
            print(f"\nreached --max-hours at step {step:,} (val {v:.4f}). "
                  f"Resume with:  --resume --out {args.out}")
            return

        if step % args.log_interval == 0 and step > start:
            now = time.time()
            done = (step - start) * tokens_per_step
            tps = (done - tok_last) / max(now - t_last, 1e-9)
            t_last, tok_last = now, done
            left = (args.steps - step) * tokens_per_step / max(tps, 1e-9)
            print(f"  step {step:7d} | loss {total:.4f} | {tps:,.0f} tok/s | "
                  f"eta {left/3600:.1f}h", flush=True)

    save(args.out, args.steps, best)
    mirror([args.out, best_path, log_path])
    print(f"\ndone. best val {best:.4f} ({best/math.log(2):.3f} bits/tok) -> {best_path}")


if __name__ == "__main__":
    main()
