"""Download standard zero-shot benchmarks into the flat JSONL that evaluate.py reads.

    python scripts/fetch_evals.py --out ./evals
    python scripts/fetch_evals.py --out ./evals --only hellaswag --limit 1000

Every benchmark is normalised to one object per line::

    {"context": "...", "choices": ["...", "..."], "label": 0}

so `daedalus/benchmarks.py` never needs to know about any particular dataset's
schema, and adding a benchmark means adding an adapter here and nothing else.

Needs `datasets` (present on Colab and Kaggle; `pip install datasets` otherwise).
Dataset ids and field names were correct when written but do change -- if an
adapter breaks, check the dataset card rather than assuming the harness is wrong.

**These are prose/commonsense benchmarks and they are the right yardstick for
phase 1, not phase 2.** Below roughly 100M parameters expect results at or just
above chance; GPT-2 124M itself scores ~0.31 on HellaSwag against 0.25 chance.
Read them as "is this above chance and rising", not as a capability claim.
"""
from __future__ import annotations

import argparse
import json
import os


def adapt_hellaswag(row):
    """Sentence completion. `label` arrives as a string index."""
    ctx = row.get("ctx") or (row.get("ctx_a", "") + " " + row.get("ctx_b", "")).strip()
    endings = row.get("endings")
    if not ctx or not endings or row.get("label") in (None, ""):
        return None
    return {"context": ctx, "choices": [" " + e.strip() for e in endings],
            "label": int(row["label"])}


def adapt_arc(row):
    """Grade-school science questions; answerKey is a letter or a digit."""
    choices = row.get("choices") or {}
    texts, labels = choices.get("text"), choices.get("label")
    key = row.get("answerKey")
    if not texts or not labels or key is None or key not in labels:
        return None
    return {"context": f"Question: {row['question']}\nAnswer:",
            "choices": [" " + t.strip() for t in texts],
            "label": list(labels).index(key)}


def adapt_piqa(row):
    """Physical commonsense: pick the solution that actually works."""
    if row.get("label") is None:
        return None
    return {"context": f"Question: {row['goal']}\nAnswer:",
            "choices": [" " + row["sol1"].strip(), " " + row["sol2"].strip()],
            "label": int(row["label"])}


def adapt_openbookqa(row):
    choices = row.get("choices") or {}
    texts, labels = choices.get("text"), choices.get("label")
    key = row.get("answerKey")
    if not texts or key is None or key not in labels:
        return None
    return {"context": row["question_stem"], "choices": [" " + t.strip() for t in texts],
            "label": list(labels).index(key)}


BENCHMARKS = {
    "hellaswag":  dict(path="Rowan/hellaswag", name=None, split="validation",
                       adapt=adapt_hellaswag),
    "arc_easy":   dict(path="allenai/ai2_arc", name="ARC-Easy", split="validation",
                       adapt=adapt_arc),
    "arc_challenge": dict(path="allenai/ai2_arc", name="ARC-Challenge",
                          split="validation", adapt=adapt_arc),
    "piqa":       dict(path="ybisk/piqa", name=None, split="validation",
                       adapt=adapt_piqa),
    "openbookqa": dict(path="allenai/openbookqa", name="main", split="validation",
                       adapt=adapt_openbookqa),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./evals")
    ap.add_argument("--only", nargs="+", choices=sorted(BENCHMARKS), default=None)
    ap.add_argument("--limit", type=int, default=0, help="cap examples per benchmark")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("needs `datasets`:  pip install datasets")

    os.makedirs(args.out, exist_ok=True)
    for name in (args.only or sorted(BENCHMARKS)):
        spec = BENCHMARKS[name]
        path = os.path.join(args.out, f"{name}.jsonl")
        try:
            ds = load_dataset(spec["path"], name=spec["name"], split=spec["split"])
        except Exception as e:                       # noqa: BLE001 - report and continue
            print(f"{name:<16} SKIPPED: {type(e).__name__}: {e}")
            continue
        kept = skipped = 0
        with open(path, "w", encoding="utf-8") as f:
            for row in ds:
                ex = spec["adapt"](row)
                if ex is None:
                    skipped += 1
                    continue
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                kept += 1
                if args.limit and kept >= args.limit:
                    break
        note = f" ({skipped} unusable rows dropped)" if skipped else ""
        print(f"{name:<16} {kept:>6} examples -> {path}{note}")


if __name__ == "__main__":
    main()
