"""Build a packed prompt/completion corpus for grammatical error correction.

Clean text in, `corrupted -> corrected` pairs out, tokenized and packed into
fixed-size windows that `data.PairCorpus` memory-maps.

    python scripts/make_gec.py --preset fineweb-edu \
        --out ./corpus/gec --tokenizer ./corpus/main/tokenizer.json \
        --block-size 1024 --target-examples 2_000_000

The tokenizer is **required** and must be phase 1's. This is the same rule as
everywhere else in the pipeline: `--init-from` carries embeddings across, and an
embedding row only means anything relative to the tokenizer that produced it.

Two properties of the output are worth understanding, because they are what
distinguish this from `prepare_corpus.py`:

**Windows are packed, never straddled.** Examples are laid end to end and a new
window is started rather than splitting one across the boundary. `Corpus` picks
a random offset into a flat stream, which for a transduction task means routinely
asking the model to produce a correction whose input scrolled off the left edge.

**Only the completion is supervised.** The parallel `.mask` file marks target
tokens; everything else becomes -100 and is dropped by the loss. Without it half
the gradient signal teaches the model to *generate* the corrupted text.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from typing import List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from daedalus.bpe import BPETokenizer                                # noqa: E402
from daedalus.gec import (GEC_END, GEC_SEP, corrupt_sentence,        # noqa: E402
                          split_sentences)
from scripts.prepare_corpus import (PRESETS, build_sources,          # noqa: E402
                                   interleave)

EOT = "<|endoftext|>"


class PackedWriter:
    """Accumulates examples into `block_size + 1` token windows and shards them.

    The window is one token longer than the block because training consumes it
    as `x = window[:-1]`, `y = window[1:]` -- the extra token is a target only.
    """

    def __init__(self, out_dir: str, split: str, window: int, dtype,
                 pad_id: int, windows_per_shard: int):
        self.out_dir, self.split, self.window = out_dir, split, window
        self.dtype, self.pad_id = dtype, pad_id
        self.windows_per_shard = windows_per_shard
        self.cur_t: List[int] = []
        self.cur_m: List[int] = []
        self.out_t: List[int] = []
        self.out_m: List[int] = []
        self.index = self.n_windows = self.examples = 0
        self.supervised = self.dropped = 0

    def add(self, prompt_ids: List[int], target_ids: List[int]) -> bool:
        n = len(prompt_ids) + len(target_ids)
        if n > self.window:
            self.dropped += 1
            return False
        if len(self.cur_t) + n > self.window:
            self._close_window()
        self.cur_t.extend(prompt_ids)
        self.cur_m.extend([0] * len(prompt_ids))
        self.cur_t.extend(target_ids)
        self.cur_m.extend([1] * len(target_ids))
        self.examples += 1
        self.supervised += len(target_ids)
        return True

    def _close_window(self) -> None:
        if not self.cur_t:
            return
        pad = self.window - len(self.cur_t)
        self.cur_t.extend([self.pad_id] * pad)
        self.cur_m.extend([0] * pad)
        self.out_t.extend(self.cur_t)
        self.out_m.extend(self.cur_m)
        self.cur_t, self.cur_m = [], []
        self.n_windows += 1
        if self.n_windows % self.windows_per_shard == 0:
            self._flush()

    def _flush(self) -> None:
        if not self.out_t:
            return
        stem = os.path.join(self.out_dir, f"{self.split}_{self.index:05d}")
        np.asarray(self.out_t, dtype=self.dtype).tofile(stem + ".bin")
        np.asarray(self.out_m, dtype=np.uint8).tofile(stem + ".mask")
        self.out_t, self.out_m = [], []
        self.index += 1

    def close(self) -> None:
        self._close_window()
        self._flush()

    @property
    def tokens(self) -> int:
        return self.n_windows * self.window


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("source (same flags as prepare_corpus.py)")
    src.add_argument("--preset", choices=sorted(PRESETS))
    src.add_argument("--local")
    src.add_argument("--ext", nargs="+", default=[".txt", ".md"])
    src.add_argument("--hf-dataset")
    src.add_argument("--hf-config", default=None)
    src.add_argument("--hf-split", default="train")
    src.add_argument("--text-field", default="text")
    src.add_argument("--mix", nargs="+", metavar="SPEC=WEIGHT", default=None)

    out = ap.add_argument_group("output")
    out.add_argument("--out", required=True)
    out.add_argument("--tokenizer", required=True,
                     help="phase-1 tokenizer.json -- must be the SAME one")
    out.add_argument("--block-size", type=int, default=1024,
                     help="windows are packed at this size; training must match")
    out.add_argument("--target-examples", type=int, default=1_000_000)
    out.add_argument("--val-examples", type=int, default=5_000)
    out.add_argument("--test-examples", type=int, default=5_000)
    out.add_argument("--windows-per-shard", type=int, default=100_000)
    out.add_argument("--eval-jsonl-limit", type=int, default=2_000)

    cor = ap.add_argument_group("corruption")
    cor.add_argument("--clean-frac", type=float, default=0.25,
                     help="fraction left uncorrupted, so the model learns not to "
                          "'correct' text that is already right")
    cor.add_argument("--max-edits", type=int, default=2)
    cor.add_argument("--min-chars", type=int, default=40)
    cor.add_argument("--max-chars", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    if not os.path.exists(args.tokenizer):
        raise SystemExit(
            f"no tokenizer at {args.tokenizer!r}. Point --tokenizer at phase 1's "
            "tokenizer.json; a corrector trained on a different tokenizer cannot "
            "be initialised from the phase-1 checkpoint.")
    tok = BPETokenizer.load(args.tokenizer)
    eot_id = tok.specials.get(EOT)
    pad_id = eot_id if eot_id is not None else 0
    dtype, dtype_name = ((np.uint32, "uint32") if tok.vocab_size > 65535
                         else (np.uint16, "uint16"))
    os.makedirs(args.out, exist_ok=True)
    # Copy the tokenizer in, so the corpus directory is self-contained. Every
    # downstream tool looks for `<data>/tokenizer.json`, and a corpus that
    # points at one living somewhere else breaks as soon as the two directories
    # are moved or mounted separately -- which on Kaggle is the normal case.
    local_tok = os.path.join(args.out, "tokenizer.json")
    if os.path.abspath(local_tok) != os.path.abspath(args.tokenizer):
        shutil.copyfile(args.tokenizer, local_tok)

    sources = build_sources(args)
    total_w = sum(w for _, w, _ in sources)
    print("sources:")
    for label, w, _ in sources:
        print(f"  {w/total_w:6.1%}  {label}")
    print(f"tokenizer: {args.tokenizer} (vocab {tok.vocab_size:,})")

    window = args.block_size + 1
    writers = {s: PackedWriter(args.out, s, window, dtype, pad_id,
                               args.windows_per_shard)
               for s in ("val", "test", "train")}
    rng = random.Random(args.seed)
    doc_counts: dict = {}
    eval_rows: List[dict] = []
    n_docs, t0 = 0, time.time()

    for doc in interleave(sources, args.seed, doc_counts):
        # The split is chosen per DOCUMENT, not per sentence. Sentences from one
        # document are near-duplicates of each other often enough that splitting
        # within a document leaks the answer from train into val.
        if writers["val"].examples < args.val_examples:
            split = "val"
        elif writers["test"].examples < args.test_examples:
            split = "test"
        else:
            split = "train"
        n_docs += 1

        for sent in split_sentences(doc, args.min_chars, args.max_chars):
            if rng.random() < args.clean_frac:
                corrupted, _ = sent, 0
            else:
                corrupted, _ = corrupt_sentence(sent, rng, args.max_edits)
            prompt_ids = tok.encode(corrupted + GEC_SEP)
            target_ids = tok.encode(sent + GEC_END)
            if eot_id is not None:
                target_ids = target_ids + [eot_id]
            writers[split].add(prompt_ids, target_ids)
            if split == "test" and len(eval_rows) < args.eval_jsonl_limit:
                eval_rows.append({"corrupt": corrupted, "clean": sent})

        if n_docs % 2000 == 0:
            ex = writers["train"].examples
            print(f"  {n_docs:,} docs | {ex:,} train examples | "
                  f"{ex/max(time.time()-t0, 1e-9):,.0f} ex/s", flush=True)
        if writers["train"].examples >= args.target_examples:
            break

    for w in writers.values():
        w.close()

    starved = [label for label, _, _ in sources if doc_counts.get(label, 0) == 0]
    if starved:
        print("\n" + "!" * 70)
        print(f"WARNING: these sources contributed ZERO documents: {', '.join(starved)}")
        print("!" * 70)

    meta = {
        "format": "pairs",
        "block_size": args.block_size,
        "vocab_size": tok.vocab_size,
        "dtype": dtype_name,
        "tokenizer": "tokenizer.json",
        "tokenizer_source": args.tokenizer,
        "eot_id": eot_id,
        "pad_id": pad_id,
        "separator": GEC_SEP,
        "documents": n_docs,
        "examples": {s: w.examples for s, w in writers.items()},
        "windows": {s: w.n_windows for s, w in writers.items()},
        "tokens": {s: w.tokens for s, w in writers.items()},
        "supervised_tokens": {s: w.supervised for s, w in writers.items()},
        "dropped_too_long": {s: w.dropped for s, w in writers.items()},
        "clean_frac": args.clean_frac,
        "max_edits": args.max_edits,
        "sources": {label: {"weight": w / total_w,
                            "documents": doc_counts.get(label, 0)}
                    for label, w, _ in sources},
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if eval_rows:
        path = os.path.join(args.out, "gec_eval.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for row in eval_rows:
                f.write(json.dumps(row) + "\n")
        n_err = sum(1 for r in eval_rows if r["corrupt"] != r["clean"])
        print(f"\neval set -> {path} ({len(eval_rows):,} rows, {n_err:,} with errors)")

    tr = writers["train"]
    frac = tr.supervised / max(tr.tokens, 1)
    print("\n" + " | ".join(f"{s} {w.examples:,} ex / {w.tokens/1e6:.1f}M tok"
                            for s, w in writers.items()))
    print(f"supervised fraction (train): {frac:.1%} "
          f"-- the rest is prompt and padding, and is not trained on")
    print(f"meta -> {os.path.join(args.out, 'meta.json')}")


if __name__ == "__main__":
    main()
