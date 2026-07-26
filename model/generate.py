"""Generate text from a trained Daedalus checkpoint.

Example:
    python generate.py --checkpoint checkpoint.pt --model adaptive \
        --prompt "fn " --n-embd 512 --core-layers 3 --n-stages 2 --n-gist 32 \
        --block-size 256 --rep-pen 1.4

Architecture flags must match the training run or the checkpoint will not load.
That includes the newer shape-changing ones: --mixer (softmax/moirai),
--n-mem-banks (Naiads) and --self-referential (Proteus). --echo-weight is a
training-only loss term, adds no parameters, and has no counterpart here.
"""
from __future__ import annotations
import argparse
import torch
import torch.nn.functional as F

from daedalus import (ByteTokenizer, Daedalus, Labyrinth, DaedalusMoE,
                      UnifiedDaedalus, DaedalusFull, DaedalusFullAdaptive,
                      DaedalusProteus)


def build(name, args, device):
    c = dict(vocab_size=256, n_embd=args.n_embd, n_head=args.n_head, block_size=args.block_size)
    if name == "dense":     return Daedalus(**c, n_layer=args.n_layer).to(device)
    if name == "labyrinth": return Labyrinth(**c, core_layers=args.core_layers, n_loops=args.n_loops, mixer=args.mixer).to(device)
    if name == "moe":       return DaedalusMoE(**c, n_layer=args.n_layer, n_experts=args.n_experts).to(device)
    if name == "unified":   return UnifiedDaedalus(**c, core_layers=args.core_layers, n_loops=args.n_loops, n_experts=args.n_experts).to(device)
    if name == "full":      return DaedalusFull(**c, core_layers=args.core_layers, n_loops=args.n_loops, n_experts=args.n_experts, n_gist=args.n_gist, n_stages=args.n_stages, n_mem_banks=args.n_mem_banks).to(device)
    if name == "adaptive":  return DaedalusFullAdaptive(**c, core_layers=args.core_layers, max_loops=args.max_loops, n_experts=args.n_experts, n_gist=args.n_gist, n_stages=args.n_stages, n_mem_banks=args.n_mem_banks).to(device)
    if name == "proteus":   return DaedalusProteus(**c, n_layer=args.n_layer, self_referential=args.self_referential).to(device)
    raise ValueError(name)


@torch.no_grad()
def generate(model, tok, prompt, device, n=400, temp=0.9, top_k=50, rep_pen=1.3):
    """Autoregressive sampling with temperature, top-k, and a repetition penalty.

    The repetition penalty divides the logits of recently-used bytes, which
    prevents the model from collapsing into a run of one token (e.g. whitespace).
    """
    model.eval()
    idx = torch.tensor([tok.encode(prompt)], device=device)
    for _ in range(n):
        cond = idx[:, -model.block_size:]
        logits = model(cond)[0][:, -1, :]                    # (1, vocab)
        for t in set(idx[0, -80:].tolist()):
            logits[0, t] /= rep_pen
        logits = logits / temp
        v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
        logits[logits < v[:, [-1]]] = -float("inf")
        idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], 1)
    return tok.decode(idx[0].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--model", default="adaptive",
                    choices=["dense", "labyrinth", "moe", "unified", "full", "adaptive",
                             "proteus"])
    ap.add_argument("--prompt", default="fn ")
    ap.add_argument("--n-embd", type=int, default=512)
    ap.add_argument("--n-head", type=int, default=8)
    ap.add_argument("--n-layer", type=int, default=3)
    ap.add_argument("--core-layers", type=int, default=3)
    ap.add_argument("--n-loops", type=int, default=4)
    ap.add_argument("--max-loops", type=int, default=6)
    ap.add_argument("--n-stages", type=int, default=2)
    ap.add_argument("--n-experts", type=int, default=8)
    ap.add_argument("--n-gist", type=int, default=32)
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--rep-pen", type=float, default=1.3)
    # must match the training run, or the checkpoint will not load
    ap.add_argument("--mixer", default="softmax", choices=["softmax", "moirai"])
    ap.add_argument("--n-mem-banks", type=int, default=1)
    ap.add_argument("--self-referential", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build(args.model, args, device)
    ck = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
    tok = ByteTokenizer()
    print(generate(model, tok, args.prompt, device, args.tokens, args.temp, args.top_k, args.rep_pen))


if __name__ == "__main__":
    main()
