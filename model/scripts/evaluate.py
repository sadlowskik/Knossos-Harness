"""Score a checkpoint on the things that decide whether a run was worth it.

    python scripts/evaluate.py --checkpoint ./ckpt/nl.best.pt --data ./corpus/nl
    python scripts/evaluate.py --checkpoint ./ckpt/code.best.pt --data ./corpus/code \
        --evals ./evals --code-language python

Three sections, and they are meant to be read together rather than any one of
them alone:

    compression   bits per byte on the **held-out test split**, which is
                  tokenizer-independent and therefore the one number that can be
                  compared against other models or your own byte-level runs.
    reasoning     zero-shot multiple choice from whatever .jsonl files are in
                  --evals (see scripts/fetch_evals.py). Near chance below ~100M
                  parameters -- that is the scale, not a broken harness.
    generation    does the model actually emit valid code, and does it emit
                  anything at all. This is the section that catches the failure
                  the 37M Rust run had: excellent loss, whitespace output.

The `repetition` figure is the guard against exactly that collapse: the fraction
of generated tokens that are the single most common token. Above ~0.5 the model
is degenerate no matter how good the loss looks.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from daedalus.benchmarks import (evaluate_bits_per_byte, halting_stats,    # noqa: E402
                                 multiple_choice, parse_rate)
from daedalus.gec import build_prompt, gec_score, parse_completion         # noqa: E402
from generate import load_model, load_tokenizer, generate                  # noqa: E402
from data import Corpus                                                    # noqa: E402

CODE_PROMPTS = [
    "def add(a, b):\n", "def main():\n", "class Node:\n",
    "import os\n\ndef ", "def parse(text):\n    \"\"\"", "for i in range(",
    "if __name__ ==", "    return ",
]
PROSE_PROMPTS = ["The ", "In 1943, ", "She said that ", "Water is "]


def load_split_ids(data_dir: str, split: str, max_tokens: int) -> np.ndarray:
    """Read the first `max_tokens` ids of a split, across shards."""
    corpus = Corpus(data_dir, split)
    out, need = [], max_tokens
    for shard in corpus.shards:
        take = min(len(shard), need)
        out.append(np.asarray(shard[:take]))
        need -= take
        if need <= 0:
            break
    return np.concatenate(out).astype(np.int64)


def run_multiple_choice(model, tok, cfg, evals_dir, device, limit, batch_size):
    results = {}
    for path in sorted(glob.glob(os.path.join(evals_dir, "*.jsonl"))):
        name = os.path.splitext(os.path.basename(path))[0]
        examples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
                if limit and len(examples) >= limit:
                    break
        if not examples:
            continue
        print(f"  {name}: {len(examples)} examples...", flush=True)
        results[name] = multiple_choice(model, tok, examples, cfg["block_size"],
                                        device, batch_size, cfg["vocab_size"])
    return results


def run_generation(model, tok, cfg, device, prompts, n_samples, tokens, temp, language):
    samples, all_ids = [], []
    for prompt in prompts:
        for _ in range(max(1, n_samples // max(len(prompts), 1))):
            text = generate(model, tok, prompt, device, n=tokens, temp=temp,
                            top_k=50, rep_pen=1.0)
            samples.append(text)
            all_ids.extend(tok.encode(text[len(prompt):]))
    counts = Counter(all_ids)
    top = counts.most_common(1)[0][1] if counts else 0
    return {
        **parse_rate(samples, language),
        "repetition": top / max(len(all_ids), 1),
        "distinct_tokens": len(counts),
        "samples": samples[:3],
    }


def run_gec(model, tok, cfg, device, rows, max_new):
    """Greedy-decode a correction for each row and score it.

    Greedy, and with the repetition penalty **off**. Both matter. Sampling makes
    a corrector paraphrase, and the default `rep_pen=1.15` is actively harmful
    here: correction is mostly copying, so penalising tokens the model just
    emitted -- which are exactly the input tokens it is supposed to reproduce --
    would manufacture errors the model did not make.
    """
    corrupted = [r["corrupt"] for r in rows]
    clean = [r["clean"] for r in rows]
    predicted = []
    for i, text in enumerate(corrupted):
        prompt = build_prompt(text)
        # Budget from the INPUT, never the reference: a correction is about as
        # long as what it corrects, and sizing the decode from `clean[i]` would
        # leak the answer's length into the metric.
        budget = min(len(tok.encode(text)) + 16, max_new)
        out = generate(model, tok, prompt, device, n=budget, temp=0.0,
                       top_k=1, rep_pen=1.0)
        predicted.append(parse_completion(out[len(prompt):]))
        if (i + 1) % 100 == 0:
            print(f"    {i+1}/{len(corrupted)}...", flush=True)
    scored = gec_score(corrupted, clean, predicted)
    scored["samples"] = [{"corrupt": corrupted[i], "clean": clean[i],
                          "predicted": predicted[i]}
                         for i in range(min(3, len(rows)))]
    return scored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", default=None, help="corpus dir (for bits/byte)")
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--evals", default="./evals", help="dir of multiple-choice .jsonl")
    ap.add_argument("--eval-tokens", type=int, default=2_000_000)
    ap.add_argument("--limit", type=int, default=0, help="cap examples per benchmark")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--gen-samples", type=int, default=32)
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--code-language", default="python")
    ap.add_argument("--gec", default=None,
                    help="gec_eval.jsonl to score correction on; defaults to "
                         "the one in --data if it exists")
    ap.add_argument("--gec-limit", type=int, default=500,
                    help="rows to score. This section decodes one example at a "
                         "time, so it is the slow one -- raise it deliberately.")
    ap.add_argument("--gec-max-new", type=int, default=128)
    ap.add_argument("--depth-batches", type=int, default=16,
                    help="batches to sample for the adaptive-depth report")
    ap.add_argument("--max-loops", type=int, default=None,
                    help="test-time depth dial; defaults to the trained value")
    ap.add_argument("--out", default=None, help="write results JSON here")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(args.checkpoint, device, {"max_loops": args.max_loops})
    tok_hint = args.tokenizer or (os.path.join(args.data, "tokenizer.json")
                                  if args.data else None)
    tok = load_tokenizer(tok_hint, cfg)
    results = {"checkpoint": args.checkpoint,
               "config": {k: cfg[k] for k in ("model", "n_embd", "block_size",
                                              "vocab_size", "max_loops")},
               "params": sum(p.numel() for p in model.parameters())}
    print(f"\n{'='*64}\n{cfg['model']}  "
          f"{results['params']/1e6:.1f}M params  block {cfg['block_size']}  "
          f"device {device}\n{'='*64}")

    # ------------------------------------------------------------ compression
    data_format = "flat"
    if args.data and os.path.exists(os.path.join(args.data, "meta.json")):
        with open(os.path.join(args.data, "meta.json"), encoding="utf-8") as f:
            data_format = json.load(f).get("format", "flat")
    if args.data and data_format == "pairs":
        # Bits/byte over a packed pair corpus would score the corrupted prompts
        # and the padding alongside the targets, which is not a language-modelling
        # number and is not comparable to phase 1's. Silently reporting it would
        # be worse than not reporting it.
        print(f"\ncompression: skipped, {args.data} is a packed pair corpus.\n"
              "  Point --data at the prose corpus to compare bits/byte across phases.")
    elif args.data:
        print(f"\ncompression ({args.split} split)")
        ids = load_split_ids(args.data, args.split, args.eval_tokens)
        bpb = evaluate_bits_per_byte(model, ids, tok, cfg["block_size"],
                                     batch_size=args.batch_size, device=device,
                                     vocab_size=cfg["vocab_size"])
        results["compression"] = bpb
        print(f"  bits/byte      {bpb['bits_per_byte']:.4f}   <- compare this one")
        print(f"  bits/token     {bpb['bits_per_token']:.4f}")
        print(f"  perplexity     {bpb['perplexity']:.2f}")
        print(f"  scored         {bpb['tokens']:,} tokens / {bpb['bytes']:,} bytes")

    # ------------------------------------------------------------------ depth
    # Reported before the slower sections deliberately: it is the check on
    # whether the architecture's central claim holds, and it costs one forward
    # pass. Only the halting models return a distribution; None for the rest.
    if args.data and data_format != "pairs":
        ids_d = load_split_ids(args.data, args.split,
                               cfg["block_size"] * args.depth_batches * 8)
        h = halting_stats(model, ids_d, cfg["block_size"], device,
                          batch_size=args.batch_size, max_batches=args.depth_batches)
        if h:
            results["halting"] = h
            print(f"\nadaptive depth (max_loops={h['max_loops']}, "
                  f"{h['tokens']:,} tokens)")
            print(f"  mean depth     {h['depth_mean']:.3f}  "
                  f"(min {h['depth_min']:.2f}, max {h['depth_max']:.2f})")
            print(f"  depth std      {h['depth_std']:.4f}  "
                  f"{'<- DEAD: depth never varies' if h['depth_std'] < 0.01 else ''}")
            print(f"  at floor       {h['frac_at_floor']:.3f}   "
                  f"at ceiling {h['frac_at_ceiling']:.3f}")
            # The correlation is only meaningful once the depth actually varies.
            # A model whose halting has collapsed still produces a nonzero
            # correlation from float noise, and reporting that as "supported"
            # would be the exact self-deception this section exists to prevent.
            alive = h["depth_std"] >= 0.01
            verdict = ("meaningless: depth is constant" if not alive else
                       "supported" if h["depth_nll_corr"] > 0.05 else "NOT supported")
            print(f"  corr(depth,nll){h['depth_nll_corr']:+.4f}  <- the thesis: "
                  f"{verdict}")
            bars = "  ".join(f"{i+1}:{f:.2f}" for i, f in enumerate(h["histogram"]))
            print(f"  depth hist     {bars}")

    # -------------------------------------------------------------- reasoning
    if os.path.isdir(args.evals):
        print("\nreasoning (zero-shot multiple choice)")
        mc = run_multiple_choice(model, tok, cfg, args.evals, device,
                                 args.limit, args.batch_size)
        if mc:
            results["multiple_choice"] = mc
            print(f"  {'benchmark':<16}{'acc':>8}{'acc_norm':>10}{'chance':>9}{'n':>7}")
            for name, r in mc.items():
                print(f"  {name:<16}{r['acc']:>8.3f}{r['acc_norm']:>10.3f}"
                      f"{r['chance']:>9.3f}{r['n']:>7}")
        else:
            print(f"  no .jsonl files in {args.evals} "
                  "(run scripts/fetch_evals.py first)")
    else:
        print(f"\nreasoning: skipped, {args.evals} does not exist")

    # ------------------------------------------------------------- generation
    print("\ngeneration")
    prompts = CODE_PROMPTS if args.code_language else PROSE_PROMPTS
    gen = run_generation(model, tok, cfg, device, prompts, args.gen_samples,
                         args.gen_tokens, args.temp, args.code_language)
    results["generation"] = gen
    print(f"  parse rate     {gen['parse_rate']:.3f}  ({gen['method']}, n={gen['n']})")
    print(f"  repetition     {gen['repetition']:.3f}  "
          f"{'<- DEGENERATE' if gen['repetition'] > 0.5 else ''}")
    print(f"  distinct toks  {gen['distinct_tokens']}")
    print("\n  --- sample ---")
    # Model output routinely contains U+FFFD from partial UTF-8 sequences, and a
    # Windows console defaults to cp1252, which cannot encode it -- that crashes
    # the whole eval after the expensive sections have already run.
    enc = sys.stdout.encoding or "utf-8"
    sample = gen["samples"][0].replace("\n", "\n  ")[:600]
    print("  " + sample.encode(enc, errors="replace").decode(enc, errors="replace"))

    # -------------------------------------------------------------- correction
    gec_path = args.gec or (os.path.join(args.data, "gec_eval.jsonl")
                            if args.data else None)
    if gec_path and os.path.exists(gec_path):
        rows = []
        with open(gec_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
                if args.gec_limit and len(rows) >= args.gec_limit:
                    break
        if rows:
            print(f"\ncorrection ({len(rows)} examples from "
                  f"{os.path.basename(gec_path)})")
            g = run_gec(model, tok, cfg, device, rows, args.gec_max_new)
            results["gec"] = g
            print(f"  fixed (errored)   {g['exact_match_errored']:.3f}  "
                  f"n={g['n_errored']}   <- the headline")
            print(f"  copy rate         {g['copy_rate']:.3f}  "
                  f"{'<- DEGENERATE: echoing the input' if g['copy_rate'] > 0.9 else ''}")
            print(f"  kept (clean in)   {g['exact_match_clean']:.3f}  n={g['n_clean']}")
            print(f"  false positives   {g['false_positive_rate']:.3f}  "
                  f"{'<- damaging correct text' if g['false_positive_rate'] > 0.2 else ''}")
            print(f"  word similarity   {g['word_similarity']:.3f}")
            for s in g["samples"][:2]:
                print(f"\n    in   {s['corrupt']}\n    out  {s['predicted']}"
                      f"\n    ref  {s['clean']}")
    elif args.gec:
        print(f"\ncorrection: skipped, {args.gec} does not exist")

    out = args.out or os.path.splitext(args.checkpoint)[0] + ".eval.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nresults -> {out}")


if __name__ == "__main__":
    main()
