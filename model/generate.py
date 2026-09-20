"""Generate text from a trained Daedalus checkpoint.

The architecture and tokenizer are read from the checkpoint itself, so a plain

    python generate.py --checkpoint ./ckpt/nl.best.pt --prompt "def add("

is enough -- no need to re-type the training flags and no way to silently load
a mismatched shape. Any flag you do pass overrides what the checkpoint says
(useful for `--max-loops`, the test-time depth dial, which is *meant* to differ
from training).

Older checkpoints that predate the saved config still work; pass the
architecture flags by hand for those.
"""
from __future__ import annotations
import argparse
import json
import os
import torch
import torch.nn.functional as F

from daedalus import (ByteTokenizer, BPETokenizer, Daedalus, Labyrinth, DaedalusMoE,
                      UnifiedDaedalus, DaedalusFull, DaedalusFullAdaptive,
                      DaedalusProteus)

ARCH_KEYS = ("model", "vocab_size", "n_embd", "n_head", "n_layer", "core_layers",
             "n_loops", "max_loops", "n_stages", "n_experts", "n_gist",
             "block_size", "mixer", "n_mem_banks", "self_referential",
             "loop_embed", "inject_gate", "qk_norm", "n_kv_head",
             "bias_update", "mtp")


def build(cfg, device):
    name = cfg["model"]
    c = dict(vocab_size=cfg["vocab_size"], n_embd=cfg["n_embd"],
             n_head=cfg["n_head"], block_size=cfg["block_size"])
    if name == "dense":     return Daedalus(**c, n_layer=cfg["n_layer"]).to(device)
    if name == "labyrinth": return Labyrinth(**c, core_layers=cfg["core_layers"], n_loops=cfg["n_loops"], mixer=cfg["mixer"]).to(device)
    if name == "moe":       return DaedalusMoE(**c, n_layer=cfg["n_layer"], n_experts=cfg["n_experts"]).to(device)
    if name == "unified":   return UnifiedDaedalus(**c, core_layers=cfg["core_layers"], n_loops=cfg["n_loops"], n_experts=cfg["n_experts"]).to(device)
    if name == "full":      return DaedalusFull(**c, core_layers=cfg["core_layers"], n_loops=cfg["n_loops"], n_experts=cfg["n_experts"], n_gist=cfg["n_gist"], n_stages=cfg["n_stages"], n_mem_banks=cfg["n_mem_banks"], loop_embed=bool(cfg["loop_embed"])).to(device)
    if name == "adaptive":  return DaedalusFullAdaptive(**c, core_layers=cfg["core_layers"], max_loops=cfg["max_loops"], n_experts=cfg["n_experts"], n_gist=cfg["n_gist"], n_stages=cfg["n_stages"], n_mem_banks=cfg["n_mem_banks"], loop_embed=bool(cfg["loop_embed"]), inject_gate=bool(cfg["inject_gate"]), qk_norm=bool(cfg["qk_norm"]), n_kv_head=cfg["n_kv_head"], bias_update=float(cfg["bias_update"]), mtp=bool(cfg["mtp"])).to(device)
    if name == "proteus":   return DaedalusProteus(**c, n_layer=cfg["n_layer"], self_referential=cfg["self_referential"]).to(device)
    raise ValueError(name)


@torch.inference_mode()
def generate(model, tok, prompt, device, n=400, temp=0.9, top_k=50, top_p=0.0,
             rep_pen=1.15, rep_window=128):
    """Autoregressive sampling with temperature, top-k/top-p and a repetition penalty.

    The repetition penalty divides the logits of recently-emitted tokens, which
    is what stops the model collapsing into a run of one token -- the whitespace
    collapse seen in the 37M Rust run. Note it is a blunt instrument: with BPE it
    also penalises legitimately repeated syntax, so keep it mild (~1.1-1.2) and
    lower it as the model gets better rather than leaving it cranked up.
    """
    model.eval()
    idx = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    if idx.numel() == 0:
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    start = idx.shape[1]
    tokens = torch.empty((1, start + n), dtype=torch.long, device=device)
    tokens[:, :start] = idx
    length = start
    for _ in range(n):
        cond = tokens[:, max(0, length - model.block_size):length]
        output = (model(cond, return_step_logits=False)
                  if isinstance(model, DaedalusFullAdaptive) else model(cond))
        logits = output[0][:, -1, :].float()                 # (1, vocab)
        if rep_pen != 1.0:
            recent = torch.unique(tokens[0, max(0, length - rep_window):length])
            selected = logits[0, recent]
            logits[0, recent] = torch.where(selected > 0, selected / rep_pen,
                                             selected * rep_pen)
        logits = logits / max(temp, 1e-6)
        if top_k:
            v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
            logits[logits < v[:, [-1]]] = -float("inf")
        if top_p:
            srt, si = torch.sort(logits, descending=True)
            cum = torch.cumsum(F.softmax(srt, -1), -1)
            drop = cum - F.softmax(srt, -1) > top_p         # keep the token that crosses p
            logits[0, si[0][drop[0]]] = -float("inf")
        tokens[:, length:length + 1] = torch.multinomial(F.softmax(logits, -1), 1)
        length += 1
    return prompt + tok.decode(tokens[0, start:length].tolist())


def resolve_config(ck, overrides=None):
    """Merge a checkpoint's recorded architecture with any explicit overrides."""
    cfg = dict(ck.get("args") or {}) if isinstance(ck, dict) else {}
    defaults = dict(model="adaptive", vocab_size=256, n_embd=512, n_head=8, n_layer=3,
                    core_layers=3, n_loops=4, max_loops=6, n_stages=2, n_experts=8,
                    n_gist=32, block_size=256, mixer="softmax", n_mem_banks=1,
                    self_referential=False, loop_embed=0, inject_gate=0,
                    qk_norm=0, n_kv_head=None, bias_update=0.0, mtp=0)
    overrides = overrides or {}
    for k in ARCH_KEYS:
        if overrides.get(k) is not None:
            cfg[k] = overrides[k]
        elif k not in cfg:
            cfg[k] = defaults[k]
        if k not in ("model", "mixer", "self_referential",
                     "n_kv_head", "bias_update"):
            cfg[k] = int(cfg[k])
    return cfg


def load_model(path, device, overrides=None):
    """Rebuild a model from a checkpoint. Returns (model, cfg)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    cfg = resolve_config(ck, overrides)
    model = build(cfg, device)
    model.load_state_dict(state)
    model.eval()
    return model, cfg


def load_tokenizer(path_hint, cfg):
    """Find the tokenizer this checkpoint was trained with."""
    for cand in (path_hint,
                 os.path.join(cfg.get("data", ""), "tokenizer.json") if cfg.get("data") else None):
        if cand and os.path.exists(cand):
            tok = BPETokenizer.load(cand)
            print(f"tokenizer: {cand} (vocab {tok.vocab_size:,})")
            return tok
    if cfg.get("vocab_size", 256) > 256:
        raise SystemExit(
            f"this checkpoint has vocab {cfg['vocab_size']:,}, so it needs its BPE "
            "tokenizer.json -- pass --tokenizer /path/to/tokenizer.json")
    print("tokenizer: byte-level (vocab 256)")
    return ByteTokenizer()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--prompt", default="def ")
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.0)
    ap.add_argument("--rep-pen", type=float, default=1.15)
    ap.add_argument("--seed", type=int, default=None)
    # overrides; None means "use whatever the checkpoint recorded"
    for k in ARCH_KEYS:
        if k in ("model", "mixer"):
            ap.add_argument(f"--{k.replace('_', '-')}", default=None)
        elif k == "self_referential":
            ap.add_argument("--self-referential", action="store_true", default=None)
        else:
            ap.add_argument(f"--{k.replace('_', '-')}", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.seed is not None:
        torch.manual_seed(args.seed)

    overrides = {k: getattr(args, k) for k in ARCH_KEYS}
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = resolve_config(ck, overrides)
    tok = load_tokenizer(args.tokenizer, cfg)
    model, cfg = load_model(args.checkpoint, device, overrides)
    print(f"model: {cfg['model']}  "
          + json.dumps({k: cfg[k] for k in ("n_embd", "block_size", "vocab_size")}))
    print("-" * 60)
    print(generate(model, tok, args.prompt, device, args.tokens, args.temp,
                   args.top_k, args.top_p, args.rep_pen))


if __name__ == "__main__":
    main()
