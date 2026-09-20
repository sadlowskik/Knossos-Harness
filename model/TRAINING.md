# Training Daedalus for real

The operational plan for taking Daedalus from a validated toy architecture to a
model that produces coherent English and syntactically valid code, on free
compute. Read [README.md](README.md) for what the architecture *is*; this
document is about how to actually run it without wasting a week of GPU quota.

## The honest target

"On the level of other open weights" spans three orders of magnitude of
compute, so it is worth being precise about which rung is which:

| Reference model | Training tokens | What it would cost you |
|---|---:|---|
| **Rung 1 — this plan** | ~2.5B | ~35 T4-hours ≈ 1–1.5 weeks of Kaggle quota (free) |
| GPT-2 124M (2019) | ~10B | ~16 A100-hours ≈ $25–50 rented |
| SmolLM2-135M (2024) | 2T | ~$3–10k |
| Qwen2.5-0.5B | 18T | not reachable solo |

Rung 1 is **not** GPT-2 tier and this document will not pretend otherwise. What
it should produce: fluent-looking English with real syntax and local coherence,
Python that parses, and multiple-choice scores a little above chance. What it
will not produce: factual reliability, working algorithms, or useful long-range
reasoning. Those need rung 2 and beyond.

The point of rung 1 is to prove the *pipeline* end to end on free compute, so
that spending money on rung 2 is a decision made with evidence.

## Configuration

```
DaedalusFullAdaptive   vocab 16384   n_embd 512   n_head 8
core_layers 3   n_stages 2   max_loops 4   n_experts 8   block_size 1024
```

| | |
|---|---:|
| Total parameters | **45.2M** |
| Embedding (tied) | 8.4M |
| MoE experts | 25.2M |
| **Active per token** | **26.3M** |
| Effective depth | 3 × 4 × 2 = **24 layers** |

Two deliberate choices worth understanding:

**Vocab 16384, not 32768.** At 45M parameters a 32k vocab spends 16.8M — over a
third of the model — on the embedding table. Halving it costs about 8% in
compression (roughly 3.7 vs 4.0 bytes/token) and buys back 8.4M parameters for
actual computation. At rung 2 the trade flips and 32k is right.

**24 effective layers from 3 layers of parameters.** This is the architecture's
whole thesis — recurrent depth buys depth with compute instead of memory — and
it is why a 45M model here is not comparable to a 45M dense model. It also means
the FLOPs bill is ~4x what the parameter count suggests. Budget accordingly.

## Where the data comes from

**Filtered beats big, and it is not close.** A 45M model has room to learn
either the structure of good text or the boilerplate of scraped junk, not both.
This is also the diagnosis of the earlier 37M Rust run: a corpus of whole GitHub
repos is dominated by near-duplicate files, generated code and license headers,
so the loss fell while capability did not follow.

`python scripts/prepare_corpus.py --list-sources` prints the full table. The
ones that matter:

| Source | What it is | Use it for |
|---|---|---|
| **`fineweb-edu`** | classifier-filtered educational web text, ODC-By | the phase-1 backbone |
| **`cosmopedia-v2`** | synthetic textbooks and stories, Apache-2.0 | punches far above its weight at small scale |
| `fineweb-edu-dedup` | SmolLM's deduplicated cut, ~220B tokens | swap in if you scale past 10B |
| `wikipedia` | clean encyclopedic prose, **CC-BY-SA** | optional; share-alike is contagious if you release weights |
| **`codeparrot-clean`** | deduplicated Python, ~50GB, ungated | the phase-2 backbone |
| `starcoderdata` | the StarCoder set, better filtered — **gated** | best quality; needs terms accepted + `HF_TOKEN` |
| `stack-python` / `stack-rust` | only ~10k files each (~25M tokens) | smoke tests only — runs dry as a real source |
| ~~`python-edu`~~ | **ships `blob_id`, not text** (verified 2026-07-28) | unusable without a Software Heritage S3 fetch |
| `github-code` | raw ungated GitHub scrape | last resort; expect boilerplate |

The mixture below roughly follows SmolLM's recipe, which is the closest public
precedent to what you are building and was tuned for models in exactly this size
range.

**Verify a source before committing hours to it:**

```bash
python scripts/prepare_corpus.py --inspect fineweb-edu
python scripts/prepare_corpus.py --inspect codeparrot-clean
```

This prints the row's actual field names. The two failures it catches are a
dataset whose text moved to a different field — you would otherwise write an
empty corpus and only find out when the loss refused to move — and a gated
dataset that needs `HF_TOKEN` set.

If you would rather stay offline, `local:<dir>` mixes in any directory of files,
and `scripts/fetch_rust.py` still clones repos the old way.

## Step 0 — corpus and tokenizer (once)

Do this once, save the result, never repeat it. The tokenizer especially:
retokenizing between phases invalidates every embedding the model has learned.

```bash
pip install datasets
python scripts/prepare_corpus.py \
    --mix fineweb-edu=0.70 cosmopedia-v2=0.15 codeparrot-clean=0.15 \
    --out ./corpus/main --train-tokenizer --vocab-size 16384 \
    --tokenizer-sample-mb 300 --target-tokens 2_500_000_000 --workers 4
```

That 15% code in the *prose* phase is deliberate. A tokenizer fitted on prose
alone handles `__init__` and `=>` badly, and a model that has never seen code in
phase 1 has far more to forget in phase 2.

Then the code-heavy phase-2 corpus, **with the same tokenizer**:

```bash
python scripts/prepare_corpus.py \
    --mix codeparrot-clean=0.75 fineweb-edu=0.25 \
    --out ./corpus/code --tokenizer ./corpus/main/tokenizer.json \
    --target-tokens 500_000_000 --workers 4
```

The 25% prose here is the same idea in reverse — it is what stops the model
forgetting English while it learns Python.

Weights are per **document**, not per token, so a source of long documents
contributes more tokens than its weight suggests, and a source that runs dry is
dropped with the rest renormalised. `meta.json` records the realised document
shares; read those rather than trusting the weights you asked for.

**Gate:** `meta.json` should report ~3.5–4.0 bytes/token. Below 3.0 means the
tokenizer did not train properly and every later number will be worse for no
architectural reason.

**On licences, since you intend to release weights.** FineWeb-Edu and Cosmopedia
are ODC-By / Apache-2.0 and unproblematic. Wikipedia is CC-BY-SA, whose
share-alike terms are the one genuinely awkward ingredient here. The Stack
honours opt-outs and filters to permissive licences, which is why it is
preferable to a raw GitHub scrape. None of this is legal advice — just record
your mixture in the model card, which `meta.json` now gives you for free.

## Step 0.5 — measure throughput before committing

Every time estimate in this document is extrapolated from published T4 numbers,
not measured on your hardware. Spend three minutes replacing them with a fact:

```bash
python train.py --model adaptive --data ./corpus/main --out /tmp/probe.pt \
    --vocab-size 16384 --n-embd 512 --n-head 8 --core-layers 3 --n-stages 2 \
    --max-loops 4 --block-size 1024 --batch-size 4 --grad-accum 8 \
    --steps 60 --warmup 10 --eval-interval 1000 --log-interval 10
```

Read the `tok/s` line. Then `2.5e9 / tok_per_s / 3600` is your real phase-1 hour
count. If it lands above ~45 hours, drop `--n-stages` to 1 or `--max-loops` to 3
before starting rather than discovering it 20 hours in.

If you hit OOM, lower `--batch-size` and raise `--grad-accum` by the same factor
— the effective batch, and therefore the training dynamics, stay identical.

## Step 1 — phase 1, natural language

```bash
python train.py --model adaptive --data ./corpus/main --out ./ckpt/nl.pt \
    --vocab-size 16384 --n-embd 512 --n-head 8 --core-layers 3 --n-stages 2 \
    --max-loops 4 --n-experts 8 --block-size 1024 \
    --batch-size 4 --grad-accum 8 --max-tokens 2.5e9 \
    --lr 6e-4 --min-lr 6e-5 --warmup 2000 --weight-decay 0.1 \
    --eval-interval 500 --max-hours 10 --resume
```

`--max-hours 10` stops cleanly below Kaggle's session cap, and `--resume` picks
up the optimizer, schedule position and RNG. Run the identical command each
session; it advances until `--max-tokens` is reached.

### Gates — check these, in this order

| When | Check | Fail means |
|---|---|---|
| step 0 | val ≈ **9.70** (= ln 16384) | init or vocab is wrong; stop immediately |
| step ~200 | val below ~7.5 | LR too high, or the schedule never warmed up |
| step ~2000 | val still falling, no NaN | lower `--lr` to 3e-4 and restart |
| every session | `bits/byte` falling in `evaluate.py` | see below |
| every session | `repetition` < 0.5 | **the whitespace-collapse failure** |

That last row is the one that matters most, and it is the reason this repo now
has an eval harness at all. The 37M Rust run reached 0.46 bits/byte and
generated pure whitespace. **A falling loss is not evidence of a working model.**
Run the eval every session, not at the end:

```bash
python scripts/evaluate.py --checkpoint ./ckpt/nl.best.pt --data ./corpus/main \
    --split val --evals ./evals --code-language ""
```

## Step 2 — the phase-1 gate

Before spending quota on phase 2, confirm phase 1 actually worked:

```bash
python scripts/fetch_evals.py --out ./evals --limit 1000
python scripts/evaluate.py --checkpoint ./ckpt/nl.best.pt --data ./corpus/main \
    --split test --evals ./evals
```

Rough expectations for 45M on 2.5B tokens — these are estimates, and the
*direction* matters more than hitting a number:

| Metric | Good | Concerning |
|---|---|---|
| bits/byte (held-out prose) | 1.2–1.6 | > 2.0 |
| repetition | < 0.2 | > 0.5 |
| distinct tokens generated | > 300 | < 50 |
| HellaSwag acc_norm | 0.26–0.30 vs 0.25 chance | at or below chance |

HellaSwag barely moving is **expected** at this scale and is not a reason to
stop — GPT-2 124M itself only reaches ~0.31. Judge phase 1 on bits/byte and on
whether the generated text looks like English. If repetition is high or distinct
tokens are low, the model has collapsed and phase 2 will not rescue it.

## Step 3 — phase 2, code

A continuation, not a restart: `--init-from` loads the weights and starts a
fresh, shorter, lower-peak schedule.

```bash
python train.py --model adaptive --data ./corpus/code \
    --init-from ./ckpt/nl.best.pt --out ./ckpt/code.pt \
    --vocab-size 16384 --n-embd 512 --n-head 8 --core-layers 3 --n-stages 2 \
    --max-loops 4 --n-experts 8 --block-size 1024 \
    --batch-size 4 --grad-accum 8 --max-tokens 5e8 \
    --lr 1.5e-4 --min-lr 1.5e-5 --warmup 200 \
    --eval-interval 500 --max-hours 10 --resume
```

The peak LR is ~4x lower than phase 1 on purpose. Re-warming to 6e-4 would blow
away most of what phase 1 learned in the first few hundred steps.

**Final gate:**

```bash
python scripts/evaluate.py --checkpoint ./ckpt/code.best.pt --data ./corpus/code \
    --split test --evals ./evals --code-language python
```

`parse_rate` is the headline: the fraction of generated samples that `ast.parse`
accepts. This is the code metric that actually moves at 45M, where HumanEval
pass@1 is a flat zero and tells you nothing. Anything above ~0.3 means real
syntactic structure was learned. Also re-check the phase-1 prose bits/byte — if
it degraded badly, raise the prose fraction in the phase-2 mix and redo.

## Kaggle session mechanics

- 30 GPU-hours/week, ~12h per session. `--max-hours 10` leaves margin.
- Checkpoints go to `/kaggle/working`. To carry state across sessions, save the
  checkpoint as a Kaggle Dataset and mount it read-only next session, then point
  `--out` at a writable copy. Verify the current quotas yourself — they change.
- Do the same with the corpus: build it once, upload as a Dataset, mount it.
  Re-streaming and re-tokenizing 2.5B tokens every session is pure waste.
- T4 is Turing, so it has **no bf16**. `--precision auto` correctly selects
  fp16 + `GradScaler`; do not force `--precision bf16` there.
- Keep Track A installs minimal. The recorded failure mode is Track B packages
  (`transformers`/`trl`/`bitsandbytes`) corrupting `sympy` until AdamW throws
  `module sympy.core has no attribute symbol`. Track A needs only `torch`,
  `numpy` and `datasets`.

## If it goes wrong

| Symptom | Likely cause |
|---|---|
| val stuck at exactly ln(vocab) | LR too high; the model collapsed to uniform |
| NaN loss | fp16 overflow — lower `--lr`, confirm `--grad-clip 1.0` |
| loss great, output is whitespace | the classic. Data is boilerplate-heavy and/or too few epochs |
| experts collapse to one | raise `--alpha` from 0.01 to 0.02 |
| val rises while train falls | overfitting; you have run out of unique data |
| OOM | halve `--batch-size`, double `--grad-accum` |

## After rung 1

If the gates pass, the same pipeline scales to rung 2 with only config changes:
`--n-embd 768`, `--core-layers 4`, vocab 32768, 10B tokens, on a rented A100 for
roughly $25–50. That is the point where "open-weights class" becomes a
defensible claim — and by then you will have measured evidence that the pipeline
earns the money.
