"""Build a BPE-tokenized, sharded corpus for real training runs.

`data.py` was written for the toy scale: read a directory of files, hold the
whole thing in one uint8 tensor. That caps you at roughly 100M tokens and at
one byte per token. This script is the scaled version:

  * documents come from local files **or** a streamed Hugging Face dataset,
  * they are tokenized with a byte-level BPE (~4x fewer tokens per byte),
  * output is written as uint16 `.bin` shards that training memory-maps,
  * val/test are held out **by document**, so no document spans the boundary.

Examples
--------
Natural language (phase 1), 2B tokens from FineWeb-Edu::

    python scripts/prepare_corpus.py --preset fineweb-edu \
        --out ./corpus/nl --target-tokens 2_000_000_000 --train-tokenizer

Code (phase 2), from local repos already on disk::

    python scripts/prepare_corpus.py --local ./rust_repos --ext .rs .toml .md \
        --out ./corpus/code --tokenizer ./corpus/nl/tokenizer.json

Mixed shards (recommended over a hard NL->code switch; see --mix)::

    python scripts/prepare_corpus.py --preset fineweb-edu --mix code=0.3 \
        --mix-local ./rust_repos --out ./corpus/mixed --target-tokens 5_000_000_000

The tokenizer must be **the same** across phases. Train it once, on a sample
that already contains both kinds of text, and reuse the file everywhere --
retokenizing between phases would invalidate every learned embedding.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
import time
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from daedalus.bpe import BPETokenizer                        # noqa: E402

EOT = "<|endoftext|>"

# Streamed Hugging Face presets.
#
# Dataset ids, config names and field names DO drift, and a preset that 404s is
# not evidence that the pipeline is broken. Run `--inspect <source>` before
# committing to a multi-hour build: it pulls a couple of rows and prints the
# available fields, which is the 30-second version of finding out the hard way.
# `--hf-dataset/--hf-config/--text-field` covers anything without a preset.
#
# `quality` is the editorial note that matters most at small scale. A 45M model
# has room to learn either the structure of good text or the boilerplate of
# scraped junk, and not both -- filtered corpora (FineWeb-Edu, Cosmopedia,
# python-edu) buy far more per token than raw CommonCrawl or unfiltered GitHub.
PRESETS = {
    # ---- natural language -------------------------------------------------
    "fineweb-edu": dict(
        path="HuggingFaceFW/fineweb-edu", name="sample-10BT", field="text",
        split="train", kind="nl", license="ODC-By",
        quality="classifier-filtered educational web text; the default choice"),
    "fineweb-edu-dedup": dict(
        path="HuggingFaceTB/smollm-corpus", name="fineweb-edu-dedup", field="text",
        split="train", kind="nl", license="ODC-By",
        quality="SmolLM's deduplicated FineWeb-Edu; ~220B tokens"),
    "cosmopedia-v2": dict(
        path="HuggingFaceTB/smollm-corpus", name="cosmopedia-v2", field="text",
        split="train", kind="nl", license="Apache-2.0",
        quality="synthetic textbooks/stories; punches above its weight for small models"),
    "cosmopedia-100k": dict(
        path="HuggingFaceTB/cosmopedia-100k", name=None, field="text",
        split="train", kind="nl", license="Apache-2.0",
        quality="100k-row taster of Cosmopedia; good for a pipeline smoke test"),
    "fineweb": dict(
        path="HuggingFaceFW/fineweb", name="sample-10BT", field="text",
        split="train", kind="nl", license="ODC-By",
        quality="unfiltered web; larger but markedly worse per token than -edu"),
    "wikipedia": dict(
        path="wikimedia/wikipedia", name="20231101.en", field="text",
        split="train", kind="nl", license="CC-BY-SA",
        quality="clean encyclopedic prose; note the share-alike licence"),
    "tinystories": dict(
        path="roneneldan/TinyStories", name=None, field="text",
        split="train", kind="nl", license="CDLA-Sharing-1.0",
        quality="tiny synthetic stories; use to validate the pipeline, not to train"),
    # ---- code -------------------------------------------------------------
    "codeparrot-clean": dict(
        path="codeparrot/codeparrot-clean", name=None, field="content",
        split="train", kind="code", license="mixed permissive",
        quality="deduplicated Python, ~50GB, ungated; the practical default"),
    "python-edu": dict(
        path="HuggingFaceTB/smollm-corpus", name="python-edu", field="text",
        split="train", kind="code", license="mixed permissive",
        quality="NOT USABLE DIRECTLY -- verified 2026-07-28 to ship blob_id/path/"
                "score only, no text. The content lives in Software Heritage S3 "
                "and needs credentials + a separate fetch. Use codeparrot-clean "
                "or starcoderdata instead"),
    "stack-python": dict(
        path="bigcode/the-stack-smol", name="data/python", field="content",
        split="train", kind="code", license="mixed permissive",
        quality="only ~10k Python files (~25M tokens) -- fine for a smoke test, "
                "will run dry as a real phase-2 source"),
    "stack-rust": dict(
        path="bigcode/the-stack-smol", name="data/rust", field="content",
        split="train", kind="code", license="mixed permissive",
        quality="~10k Rust files; the language the earlier flagship run used"),
    "starcoderdata": dict(
        path="bigcode/starcoderdata", name="python", field="content",
        split="train", kind="code", license="mixed permissive",
        quality="the StarCoder training set; GATED -- accept terms + set HF_TOKEN"),
    "github-code": dict(
        path="codeparrot/github-code-clean", name="Python-all", field="code",
        split="train", kind="code", license="mixed permissive",
        quality="ungated GitHub scrape; least filtered, expect boilerplate"),
}


# --------------------------------------------------------------- doc sources

def iter_local(root: str, exts: Iterable[str], max_bytes: int = 2_000_000
               ) -> Iterator[str]:
    """Yield the contents of every matching file under `root`."""
    exts = tuple(exts)
    for path in sorted(glob.glob(os.path.join(root, "**", "*"), recursive=True)):
        if not path.endswith(exts) or not os.path.isfile(path):
            continue
        try:
            if os.path.getsize(path) > max_bytes:
                continue
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
        except OSError:
            continue
        if text.strip():
            yield text


def iter_hf(path: str, name: Optional[str], split: str, field: str) -> Iterator[str]:
    """Stream a Hugging Face dataset without downloading it in full."""
    try:
        from datasets import load_dataset
    except ImportError:                                      # pragma: no cover
        raise SystemExit(
            "streaming a HF dataset needs `datasets`:  pip install datasets\n"
            "(or use --local to build from files already on disk)")
    ds = load_dataset(path, name=name, split=split, streaming=True)
    for row in ds:
        text = row.get(field)
        if text:
            yield text


def parse_spec(spec: str, args) -> tuple:
    """`"fineweb-edu=0.7"` or `"local:./repos=0.3"` -> (label, weight, factory).

    A factory rather than an iterator, because a mixture may need to restart a
    stream that ran dry, and because the tokenizer sample must not consume
    documents that the writer is about to need.
    """
    body, _, weight = spec.rpartition("=")
    if not body:                                   # no "=" -> weight defaults to 1
        body, weight = spec, "1"
    try:
        w = float(weight)
    except ValueError:
        raise SystemExit(f"bad weight in --mix spec {spec!r}: expected name=weight")

    if body.startswith("local:"):
        root = body[len("local:"):]
        if not os.path.isdir(root):
            raise SystemExit(f"--mix local source {root!r} is not a directory")
        return body, w, lambda: iter_local(root, args.ext)
    if body.startswith("hf:"):
        # hf:<path>[:<config>][#<field>]
        rest, _, field = body[len("hf:"):].partition("#")
        path, _, config = rest.partition(":")
        return body, w, (lambda: iter_hf(path, config or None, args.hf_split,
                                         field or args.text_field))
    if body in PRESETS:
        p = PRESETS[body]
        return body, w, lambda: iter_hf(p["path"], p["name"], p["split"], p["field"])
    raise SystemExit(f"unknown source {body!r}. Presets: {', '.join(sorted(PRESETS))}\n"
                     "or use local:<dir> / hf:<path>[:<config>][#<field>]")


def build_sources(args) -> List[tuple]:
    """Resolve every requested source into (label, weight, factory) triples."""
    if args.mix:
        return [parse_spec(s, args) for s in args.mix]
    if args.local:
        return [(f"local:{args.local}", 1.0, lambda: iter_local(args.local, args.ext))]
    if args.preset:
        p = PRESETS[args.preset]
        return [(args.preset, 1.0,
                 lambda: iter_hf(p["path"], p["name"], p["split"], p["field"]))]
    if args.hf_dataset:
        return [(args.hf_dataset, 1.0,
                 lambda: iter_hf(args.hf_dataset, args.hf_config, args.hf_split,
                                 args.text_field))]
    raise SystemExit("pick a source: --preset, --local, --hf-dataset, or --mix")


def interleave(sources: Sequence[tuple], seed: int = 1337,
               counts: Optional[Dict[str, int]] = None) -> Iterator[str]:
    """Randomly interleave N weighted document streams.

    Mixing beats a hard phase switch: training purely on prose and *then* purely
    on code causes the model to forget the prose (catastrophic forgetting,
    McCloskey & Cohen 1989). Keeping a fraction of each throughout, and shifting
    the ratio between phases, keeps both.

    Weights are per *document*, not per token, so a source of long documents
    contributes more tokens than its weight suggests. The realised token shares
    are recorded in meta.json -- read those rather than assuming the weights.

    A stream that runs dry is dropped and the remaining weights renormalise, so
    a small source cannot silently truncate the whole build.
    """
    rng = random.Random(seed)
    live = [(label, w, factory()) for label, w, factory in sources if w > 0]
    # Caller-supplied so the tallies survive an early break on --target-tokens;
    # a generator that never runs to completion cannot report them itself.
    counts = {} if counts is None else counts
    counts.update({label: 0 for label, _, _ in live})
    while live:
        total = sum(w for _, w, _ in live)
        pick = rng.random() * total
        idx, running = len(live) - 1, 0.0
        for i, (_, w, _) in enumerate(live):
            running += w
            if pick < running:
                idx = i
                break
        label, _, stream = live[idx]
        try:
            doc = next(stream)
        except StopIteration:
            live.pop(idx)
            continue
        counts[label] += 1
        yield doc


# -------------------------------------------------------------- introspection

def list_sources() -> None:
    for kind in ("nl", "code"):
        print(f"\n{'NATURAL LANGUAGE' if kind == 'nl' else 'CODE'}")
        for name, p in sorted(PRESETS.items()):
            if p["kind"] != kind:
                continue
            cfg = f"[{p['name']}]" if p["name"] else ""
            print(f"  {name:<20} {p['path']} {cfg}")
            print(f"  {'':<20} {p['quality']}")
            print(f"  {'':<20} licence: {p['license']}")


def inspect_source(spec: str, args, n: int = 2) -> None:
    """Pull a couple of rows and show what is actually in them.

    Worth 30 seconds before any multi-hour build. The two failures this catches
    are a dataset whose text lives under a different field name (you would
    otherwise write an empty corpus and not find out until the loss refused to
    move) and a gated dataset that needs HF_TOKEN.
    """
    # tolerate a copy-pasted "name=weight" from a --mix line
    head, sep, tail = spec.rpartition("=")
    if sep and head:
        try:
            float(tail)
            spec = head
        except ValueError:
            pass

    if spec in PRESETS:
        p = PRESETS[spec]
        print(f"{spec}: {p['path']} config={p['name']} field={p['field']}")
        print(f"  {p['quality']}\n  licence: {p['license']}")

    label, _, factory = parse_spec(spec, args)
    if spec in PRESETS or spec.startswith("hf:"):
        try:
            from datasets import load_dataset
        except ImportError:
            raise SystemExit("--inspect on a HF source needs `datasets`")
        if spec in PRESETS:
            p = PRESETS[spec]
            path, cfg, split, field = p["path"], p["name"], p["split"], p["field"]
        else:
            rest, _, field = spec[len("hf:"):].partition("#")
            path, _, cfg = rest.partition(":")
            cfg, split = cfg or None, args.hf_split
            field = field or args.text_field
        try:
            ds = load_dataset(path, name=cfg, split=split, streaming=True)
            row = next(iter(ds))
        except Exception as e:                       # noqa: BLE001 - diagnostic path
            print(f"  FAILED: {type(e).__name__}: {e}")
            print("  If this mentions auth/gated: accept the terms on the dataset "
                  "page and set HF_TOKEN.")
            return
        print(f"  fields: {sorted(row.keys())}")
        if field not in row:
            print(f"  !! field {field!r} IS NOT PRESENT -- pass --text-field, "
                  f"or use hf:{path}#<field>")
        else:
            sample = str(row[field])[:300].replace("\n", "\\n")
            print(f"  {field}[:300]: {sample}")
        return

    docs = factory()
    for i, doc in enumerate(docs):
        if i >= n:
            break
        print(f"  doc {i}: {doc[:300]!r}")


# ------------------------------------------------------------------ tokenizer

def get_tokenizer(args, sample_source) -> BPETokenizer:
    if args.tokenizer and os.path.exists(args.tokenizer):
        tok = BPETokenizer.load(args.tokenizer)
        print(f"tokenizer: loaded {args.tokenizer} (vocab {tok.vocab_size:,})")
        return tok
    if not args.train_tokenizer:
        raise SystemExit(
            f"no tokenizer at {args.tokenizer!r}. Pass --train-tokenizer to learn "
            "one, or point --tokenizer at an existing file.")

    print(f"training tokenizer on ~{args.tokenizer_sample_mb}MB sample...")
    budget = args.tokenizer_sample_mb * 1_000_000
    sample, total = [], 0
    for doc in sample_source:
        sample.append(doc)
        total += len(doc)
        if total >= budget:
            break
    t0 = time.time()
    tok = BPETokenizer.train(sample, vocab_size=args.vocab_size, verbose=True)
    path = args.tokenizer or os.path.join(args.out, "tokenizer.json")
    tok.save(path)
    n_bytes = sum(len(s.encode("utf-8")) for s in sample[:200])
    n_ids = sum(len(tok.encode(s)) for s in sample[:200])
    print(f"tokenizer: {tok.vocab_size:,} tokens, trained in {time.time()-t0:.0f}s, "
          f"{n_bytes/max(n_ids,1):.2f} bytes/token -> {path}")
    return tok


# --------------------------------------------------------------------- writer

# ------------------------------------------------------------- parallel encode
# Pure-Python BPE encoding runs about 6 MB/s on one core, so a 2B-token corpus
# is ~20 minutes single-threaded and a few minutes across cores. Workers load
# the tokenizer once via the pool initializer rather than pickling it per task.

_WORKER_TOK: Optional[BPETokenizer] = None


def _init_worker(tokenizer_path: str) -> None:
    global _WORKER_TOK
    _WORKER_TOK = BPETokenizer.load(tokenizer_path)


def _encode_batch(docs: List[str]) -> List[List[int]]:
    tok = _WORKER_TOK
    eot = tok.specials.get(EOT)
    out = []
    for doc in docs:
        ids = tok.encode(doc)
        if eot is not None:
            ids.append(eot)
        out.append(ids)
    return out


def batched(it: Iterator[str], n: int) -> Iterator[List[str]]:
    batch: List[str] = []
    for item in it:
        batch.append(item)
        if len(batch) >= n:
            yield batch
            batch = []
    if batch:
        yield batch


def encoded_documents(docs: Iterator[str], tok: BPETokenizer, tokenizer_path: str,
                      workers: int, batch: int = 256) -> Iterator[List[int]]:
    """Yield one id-list per document, in order, optionally across processes."""
    if workers <= 1:
        eot = tok.specials.get(EOT)
        for doc in docs:
            ids = tok.encode(doc)
            if eot is not None:
                ids.append(eot)
            yield ids
        return
    import multiprocessing as mp
    with mp.Pool(workers, initializer=_init_worker, initargs=(tokenizer_path,)) as pool:
        # imap (not imap_unordered) so shard contents stay reproducible
        for group in pool.imap(_encode_batch, batched(docs, batch)):
            yield from group


class ShardWriter:
    """Accumulates ids and flushes fixed-size uint16 shards to disk."""

    def __init__(self, out_dir: str, split: str, shard_tokens: int, dtype):
        self.out_dir, self.split = out_dir, split
        self.shard_tokens, self.dtype = shard_tokens, dtype
        self.buf: List[int] = []
        self.index = 0
        self.total = 0

    def add(self, ids: List[int]) -> None:
        self.buf.extend(ids)
        self.total += len(ids)
        while len(self.buf) >= self.shard_tokens:
            self._flush(self.buf[:self.shard_tokens])
            del self.buf[:self.shard_tokens]

    def close(self) -> None:
        if self.buf:
            self._flush(self.buf)
            self.buf = []

    def _flush(self, chunk: List[int]) -> None:
        path = os.path.join(self.out_dir, f"{self.split}_{self.index:05d}.bin")
        np.asarray(chunk, dtype=self.dtype).tofile(path)
        self.index += 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("source")
    src.add_argument("--preset", choices=sorted(PRESETS))
    src.add_argument("--local", help="directory of source files")
    src.add_argument("--ext", nargs="+", default=[".py", ".rs", ".md", ".txt"])
    src.add_argument("--hf-dataset", help="any HF dataset id (streamed)")
    src.add_argument("--hf-config", default=None)
    src.add_argument("--hf-split", default="train")
    src.add_argument("--text-field", default="text")

    mix = ap.add_argument_group("mixing")
    mix.add_argument("--mix", nargs="+", metavar="SPEC=WEIGHT", default=None,
                     help="weighted blend, e.g. --mix fineweb-edu=0.7 "
                          "cosmopedia-v2=0.15 local:./repos=0.15. Each SPEC is a "
                          "preset name, local:<dir>, or hf:<path>[:<config>][#<field>]")

    ap.add_argument("--list-sources", action="store_true",
                    help="print the available presets and exit")
    ap.add_argument("--inspect", metavar="SPEC",
                    help="show a sample row and its field names, then exit")

    out = ap.add_argument_group("output")
    out.add_argument("--out", default=None)
    out.add_argument("--target-tokens", type=int, default=0,
                     help="stop after roughly this many train tokens (0 = all)")
    out.add_argument("--shard-tokens", type=int, default=100_000_000)
    out.add_argument("--val-tokens", type=int, default=5_000_000)
    out.add_argument("--test-tokens", type=int, default=5_000_000)

    tk = ap.add_argument_group("tokenizer")
    tk.add_argument("--tokenizer", default=None, help="path to tokenizer.json")
    tk.add_argument("--train-tokenizer", action="store_true")
    tk.add_argument("--vocab-size", type=int, default=32768)
    tk.add_argument("--tokenizer-sample-mb", type=int, default=200)

    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    if args.list_sources:
        list_sources()
        return
    if args.inspect:
        inspect_source(args.inspect, args)
        return
    if not args.out:
        raise SystemExit("--out is required (unless --list-sources / --inspect)")

    os.makedirs(args.out, exist_ok=True)
    args.tokenizer = args.tokenizer or os.path.join(args.out, "tokenizer.json")

    sources = build_sources(args)
    total_w = sum(w for _, w, _ in sources)
    print("sources:")
    for label, w, _ in sources:
        print(f"  {w/total_w:6.1%}  {label}")

    # The tokenizer sample is drawn from a *separate* pass over the sources, so
    # it does not consume documents the writer is about to need. It samples the
    # same mixture, which matters: a tokenizer fitted on prose alone handles
    # `__init__` and `=>` badly.
    tok = get_tokenizer(args, interleave(sources, args.seed))
    if tok.vocab_size > 65535:
        dtype, dtype_name = np.uint32, "uint32"
    else:
        dtype, dtype_name = np.uint16, "uint16"
    eot_id = tok.specials.get(EOT)

    doc_counts: Dict[str, int] = {}
    docs = interleave(sources, args.seed, doc_counts)

    writers = {s: ShardWriter(args.out, s, args.shard_tokens, dtype)
               for s in ("val", "test", "train")}
    quotas = {"val": args.val_tokens, "test": args.test_tokens}

    print(f"writing {dtype_name} shards to {args.out} "
          f"(target {args.target_tokens or 'all'} train tokens)")
    t0, n_docs = time.time(), 0
    for ids in encoded_documents(docs, tok, args.tokenizer, args.workers):
        # Fill the held-out splits first, then everything else is train. Because
        # splits are filled by whole document, no document is ever split across
        # the train/val boundary.
        if writers["val"].total < quotas["val"]:
            writers["val"].add(ids)
        elif writers["test"].total < quotas["test"]:
            writers["test"].add(ids)
        else:
            writers["train"].add(ids)
        n_docs += 1
        if n_docs % 2000 == 0:
            tt = writers["train"].total
            rate = tt / max(time.time() - t0, 1e-9)
            print(f"  {n_docs:,} docs | {tt/1e6:.1f}M train tokens | "
                  f"{rate/1e3:.0f}k tok/s", flush=True)
        if args.target_tokens and writers["train"].total >= args.target_tokens:
            break

    for w in writers.values():
        w.close()

    # A source that yields nothing is dropped by interleave() and its weight is
    # redistributed, so the build *succeeds* with an ingredient missing. That is
    # the worst kind of failure: a corpus that is 100% prose when you asked for
    # 15% code, discovered weeks later when the model cannot write code.
    starved = [label for label, _, _ in sources if doc_counts.get(label, 0) == 0]
    if starved:
        print("\n" + "!" * 70)
        print(f"WARNING: these sources contributed ZERO documents: {', '.join(starved)}")
        print("Their weight was redistributed across the others, so the mixture you")
        print("got is NOT the mixture you asked for. Usual causes: the text field")
        print("is named something else (run --inspect SPEC), the config/split is")
        print("empty, or the dataset is gated and needs HF_TOKEN.")
        print("!" * 70)

    meta = {
        "vocab_size": tok.vocab_size,
        "dtype": dtype_name,
        "tokenizer": os.path.relpath(args.tokenizer, args.out),
        "eot_id": eot_id,
        "documents": n_docs,
        "tokens": {s: w.total for s, w in writers.items()},
        "shards": {s: w.index for s, w in writers.items()},
        "sources": {label: {"weight": w / total_w,
                            "documents": doc_counts.get(label, 0),
                            "doc_share": doc_counts.get(label, 0) / max(n_docs, 1)}
                    for label, w, _ in sources},
        "args": {k: v for k, v in vars(args).items()
                 if k in ("preset", "local", "hf_dataset", "hf_config", "mix", "ext")},
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\ndone in {time.time()-t0:.0f}s: "
          + " | ".join(f"{s} {w.total/1e6:.1f}M" for s, w in writers.items()))
    print(f"meta -> {os.path.join(args.out, 'meta.json')}")


if __name__ == "__main__":
    main()
