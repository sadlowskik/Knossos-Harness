# Daedalus

A small, **recurrent-depth, mixture-of-experts** language model for code, built
from scratch as a learning-first research project. The goal is not to beat
frontier models — it is to understand, and to make every mechanism inspectable,
hackable, and honestly measured.

Every component is named after Greek myth, and each name describes what the
piece does:

| Name | Component | What it does |
|------|-----------|--------------|
| **Daedalus** | the model | the master craftsman |
| **Labyrinth** | recurrent-depth core | a shared block looped back on itself |
| **Ariadne** | adaptive halting | decides how deep to loop, per token |
| **Muses** | routed experts | specialization emerges from data |
| **Apollo** | the router | picks which Muses speak |
| **Themis** | shared experts | always-on, carry the common ground |
| **Mnemosyne** | gist memory | lossy, high-level recollection |
| **Scribe** | symbol table | exact, never approximated |
| **Moirai** | fast-weight mixer | spin, measure and cut the thread of memory |
| **Naiads** | mixture-of-memories | many springs, each holding its own water |
| **Echo** | loop self-distillation | the shallow pass repeats what the deep one said |
| **Proteus** | self-modifying weights | changes his own shape (experimental) |

## Why this architecture

- **Recurrent depth (Labyrinth).** Loop one shared core `r` times to get the
  effective depth of `r` layers at the parameter cost of one. Decouples *how
  much the model computes* from *how big it is* — ideal when memory, not time,
  is the bottleneck. *(Universal Transformer; Huginn, Geiping et al. 2025; Ouro.)*
- **Adaptive halting (Ariadne).** A PonderNet halting head lets each token
  choose its own depth — more loops on hard tokens, fewer on easy ones.
  *(PonderNet, Banino et al. 2021; ACT, Graves 2016.)*
- **Fine-grained MoE (Muses / Apollo / Themis).** Many small experts, a noisy
  top-k router, and an always-on shared expert. More capacity at similar active
  compute. *(DeepSeekMoE; Switch Transformer; Shazeer et al. 2017.)*
- **Two-tier memory (Mnemosyne + Scribe).** Compress fuzzy context lossily, but
  keep identifiers/signatures/paths bit-exact in an AST-parsed symbol table —
  because a single hallucinated identifier breaks compilation.
- **Unified (Mixture-of-Recursions).** Loop a *shared MoE core*: recurrent depth
  and sparse experts at once. *(Bae et al. 2025.)*
- **DaedalusFull.** The whole architecture in one model: RoPE positions + MoE +
  input injection + interleaved memory (`core -> memory -> core`) + variable-loop
  recurrence. *(RoPE: Su et al. 2021; input injection: Huginn; interleaved
  memory: Block-Recurrent Transformer, Hutchins et al. 2022 / RMT.)*
- **Gated fast weights (Moirai).** An alternative to softmax attention inside the
  core: one fixed-size fast-weight matrix per head, rewritten every token by the
  delta rule, with *decoupled* erase and write gates — erase drops stale content
  before the write decides how hard to commit, so new writes can cannibalize the
  space low-value ones held. O(1) state instead of a growing KV cache, and an
  axis orthogonal to loop depth and expert routing. *(Gated DeltaNet, Yang et al.
  2024; fast-weight programmers, Schlag et al. 2021.)*
- **Mixture-of-memories (Naiads).** Mnemosyne is one bank, so every segment
  writes over every other segment's context. Naiads splits it into `n` banks and
  reuses Apollo to route each segment to its top-k; unselected banks are left
  **bit-identical**, which is what removes the interference. Balanced by the same
  Switch-Transformer aux loss that keeps the Muses honest.
- **Loop self-distillation (Echo).** `--variable-loops` teaches the core to
  *survive* unfamiliar depths; it never teaches a shallow pass to *agree* with a
  deep one. Echo adds `distill(k-loop, stopgrad(R-loop))`, so a 1-loop pass is
  explicitly pulled toward what 4 loops would have produced. For Ariadne and
  DaedalusFullAdaptive the teacher is free — the per-step logits already exist
  for the halting loss. *(Hinton et al. 2015, applied across depth.)*
- **Self-modifying weights (Proteus).** Moirai's delta rule turned on the layer's
  own transform, and optionally on the rows that generate the update too (the
  SRWM of Irie et al. 2022). Deliberately isolated in its own model class — it is
  the least-proven idea here and must not be able to destabilize the main line.

## Repository layout

```
daedalus/
  tokenizer.py   ByteTokenizer -- 256-symbol byte vocabulary
  layers.py      Embeddings, Head, MultiHeadAttention, FeedForward, Block
  models.py      Daedalus (dense baseline), Labyrinth (recurrent depth)
  ariadne.py     Ariadne + ponder_loss + expected_steps (PonderNet halting)
  moe.py         Expert/Muses, Router/Apollo, shared Themis, load_balance_loss
  unified.py     UnifiedDaedalus (MoE inside the looped core)
  memory.py      Mnemosyne (gist memory), MemoryModel, Scribe (AST symbol table)
  rope.py        Rotary positions + RoPEAttention
  full.py        DaedalusFull, DaedalusFullAdaptive, RecurrentMoECore, MemoryLayer
  moirai.py      MoiraiMixer -- gated fast-weight token mixer          (new)
  naiads.py      Naiads -- routed multi-bank gist memory               (new)
  echo.py        echo_loss / echo_step / echo_from_steps               (new)
  proteus.py     SelfModifyingLinear, ProteusBlock, DaedalusProteus    (new)

train.py         one training entry point for every model, checkpoint-and-resume
generate.py      sampling from a checkpoint (temperature, top-k, repetition penalty)
data.py          byte-level corpus builder, split BY FILE
scripts/
  fetch_rust.py     clone a Rust corpus from GitHub
  naiads_eval.py    acceptance gate: n memory banks vs one                (new)
  echo_sweep.py     acceptance gate: loop-count sweep with/without Echo   (new)
  proteus_probe.py  acceptance gate: weight-norm stability + adaptation   (new)
tests/           one isolation test file per component
```

## Results (toy scale)

Byte-level, ~0.68M–0.8M params, trained on the CPython standard library on a
single T4 GPU. These are **learning-scale** numbers — reported honestly, not to
impress:

| Model | Val loss | bits/byte | Note |
|-------|---------:|----------:|------|
| Daedalus (dense, 3 layers) | 1.32 | 1.91 | baseline |
| **Labyrinth** (3-layer core × 4 loops) | **1.19** | **1.72** | beats dense at **equal params** |
| DaedalusMoE (3 MoE blocks) | 1.30 | 1.87 | no expert collapse |
| UnifiedDaedalus (MoE core × 4 loops) | 1.35 | 1.95 | stable fusion (underfit) |
| **Labyrinth + Moirai** (fast-weight core) | — | — | tests pass; **untrained** |
| **DaedalusFull + Naiads** (4 memory banks) | — | — | tests pass; **untrained** |
| **Labyrinth + Echo** (loop distillation) | — | — | tests pass + sweep measured (below); **untrained** |
| **DaedalusProteus** (self-modifying) | — | — | tests pass; **untrained** |

✅ **Verification status of the last four rows (updated 2026-07-26).** These were
written in an environment with no Python interpreter, so for a long time `pytest`
had never been executed against them. **It has now been.** Result: `166 passed`
fast, `4 passed` slow — and the first run surfaced **five real bugs**, exactly as
this section used to warn it would:

| Bug | Where | Effect |
|---|---|---|
| `load_balance_loss` result not unpacked | `full.py` ×2 | `DaedalusFull` and `DaedalusFullAdaptive` raised on every forward |
| `targets.view()` on a non-contiguous tensor | 6 call sites | `RuntimeError` whenever targets were a strided slice |
| `echo_loss` KL scaled by `T` | `echo.py` | `batchmean` divides by `shape[0]`; on `(B,T,V)` the term was **64× too large** and collapsed training |
| `echo_step` moved the CE depth | `echo.py` | `--echo-weight` changed *two* things, so the sweep was not an ablation |
| `Argus.save()` dropped two Counters | `harness/argus.py` | `TypeError` on tuple keys; ACP `session/new` failed |

The KL scale bug is the one worth remembering: **both `echo_loss` unit tests pass
with it in place**, because "zero when they agree" and "positive when they
disagree" are scale-invariant. Only the training-based sweep could catch it. That
is the argument for `-m slow` existing at all.

The `—` cells still need **training runs** — passing tests is not a measurement.
Do not quote a val loss for any of them until the acceptance gates below have run.

- **Ariadne** learns genuine per-token depth allocation (depth std ≈ 0.70;
  `corr(depth, difficulty) ≈ +0.12` — real but weak at this scale).
- **Mnemosyne** memory helps: predicting a segment with the compressed gist of
  the previous 128 tokens beats predicting it without, by ~0.39 nats.
- **Echo — "can it be forced into fewer loops?"** This is the question the whole
  thread started from, and it now has a measurement rather than an argument.
  `scripts/echo_sweep.py` trains two otherwise-identical Labyrinths differing
  only in `--echo-weight`, on a synthetic corpus, 300 steps, R=4, same seed and
  batch order in both arms:

  | loops | mean Δ (n=3 seeds) | per-seed deltas |
  |------:|-------------------:|:----------------|
  | **1** | **−0.1205** | −0.2258, −0.0613, −0.0744 |
  | 2 | −0.0125 | −0.0179, −0.0032, −0.0164 |
  | 3 | −0.0046 | −0.0059, −0.0009, −0.0071 |
  | 4 *(training depth)* | −0.0033 | −0.0058, −0.0005, −0.0038 |

  **The shallow-end claim holds.** All three seeds improve at every depth, and
  the loop-1 effect is an order of magnitude larger than the rest — which is
  exactly the shape the hypothesis predicted.

  **Why this table is n=3 and not n=1.** The first run used seed 0 alone and
  showed loop-1 loss *halving* (0.4385 → 0.2128). That was an artifact: seed 0's
  Echo-off baseline was unusually bad (0.4385, against 0.2501 and 0.2695 for the
  other two seeds), so the single-seed number overstated the effect by roughly
  2×. The real mean improvement is −0.12, not −0.23. One seed reported noise as
  signal, in the same document that warns against doing exactly that.

  The deep end did not degrade, contrary to what this section used to predict —
  but at 0.003–0.005 nats those deltas are near-negligible in absolute terms,
  consistent in sign rather than large. Toy scale, synthetic corpus, 300 steps.

### Acceptance gates (how the `—` rows get filled)

```bash
python train.py --model labyrinth --steps 3000 --mixer moirai   # Moirai row
python scripts/naiads_eval.py --data ./data --n-banks 4         # Naiads vs Mnemosyne
python scripts/echo_sweep.py --data ./data                      # Echo loop sweep
python scripts/proteus_probe.py                                 # Proteus stability + adaptation
python scripts/proteus_probe.py --self-referential              # the full SRWM
```

**Flagship — `DaedalusFull` on Rust.** The fully integrated model (1.66M params,
RoPE + MoE + injection + interleaved memory + variable-loop recurrence), trained
on ~11M tokens of Rust (ripgrep, tokio, serde, clap, bat) on a single T4:
reaches **0.88 val loss (1.26 bits/byte)** in ~13 min, still descending. All 8
experts stay balanced under recurrence + interleaving; the test-time depth dial
survives (coherent generations at `r=3` and `r=5`). It generates Rust-textured
output — lifetimes, macros, `impl` blocks, byte strings — but not yet correct
code, exactly as expected at this size. *(The 1.26 bits/byte is not comparable
to the Python numbers above: Rust from a few repos is more repetitive, the model
is larger, and the context is longer.)*

**Honest scope:** at this size, expert and depth specialization is *structural*
(whitespace, case, punctuation), not *semantic*. Semantic specialization needs
scale. This repo is for understanding the mechanisms and as a base to scale up.

## Install

```bash
pip install torch
git clone <your-fork-url> && cd daedalus
```

## Quickstart

```python
import torch
from daedalus import Labyrinth, ByteTokenizer

tok = ByteTokenizer()
model = Labyrinth(vocab_size=256, n_embd=128, core_layers=3, n_loops=4, block_size=128)

ids = torch.tensor([tok.encode("def add(a, b):")])
logits, _ = model(ids)                    # (1, T, 256)
logits, _ = model(ids, n_loops=8)         # think deeper at inference (train variable-loops first)
```

Prepare data and train:

```bash
# option A: local source files (e.g. the Python stdlib)
python data.py --source /usr/lib/python3.12 --out ./data

# option B: fetch a Rust corpus by cloning GitHub repos (needs internet + git)
python scripts/fetch_rust.py --out ./data --ext rs

# train (checkpoint-and-resume; long runs can span multiple sessions)
python train.py --model labyrinth --steps 3000 --variable-loops
python train.py --model moe       --steps 3000
python train.py --model adaptive  --n-embd 512 --core-layers 3 --steps 40000 --resume

# the newer axes (all default to off, so existing recipes are unchanged)
python train.py --model labyrinth --steps 3000 --mixer moirai        # fast-weight core
python train.py --model labyrinth --steps 3000 --echo-weight 0.5     # loop distillation
python train.py --model full      --steps 3000 --n-mem-banks 4       # mixture-of-memories
python train.py --model proteus   --steps 3000                       # self-modifying weights
python train.py --model proteus   --steps 3000 --self-referential    # full SRWM (unstable)
```

Generate from a checkpoint:

```bash
python generate.py --checkpoint checkpoint.pt --model adaptive \
    --n-embd 512 --core-layers 3 --prompt "fn " --rep-pen 1.4
```

Architecture flags must match the run that produced the checkpoint, or
`load_state_dict` will reject it — that includes `--mixer`, `--n-mem-banks` and
`--self-referential`, which change the parameter shapes. (`--echo-weight` is a
training-only loss term and adds no parameters, so generation never needs it.)

Run the test suite (the isolation checks that validate every component):

```bash
pip install pytest && pytest -q          # fast isolation tests
pytest -q -m slow                        # plus the training-based checks (minutes)
```

The `slow` marker covers the checks that need actual training to mean anything —
Echo's shallow-end claim and Proteus's long-run weight-norm stability. They are
excluded by default (see `pytest.ini`) so the fast suite stays fast.

What each new test file is actually asserting:

| File | The claim it defends |
|------|----------------------|
| `test_moirai.py` | the scan matches the recurrence re-derived longhand from the module's own projections; gradients reach both gates; state stays bounded over 256 tokens; the default mixer is still softmax |
| `test_naiads.py` | unselected banks are **bit-identical** after forward *and* after backward (`torch.equal`, not `allclose`); bank balance obeys the same bounds as expert balance; `n_banks=1` is still plain Mnemosyne with zero aux |
| `test_echo.py` | the distillation term is zero when passes agree, positive when they disagree, and the teacher receives **no gradient**; `--echo-weight 0` reproduces plain CE exactly |
| `test_proteus.py` | the self-written matrix is non-zero and input-dependent; `\|\|W\|\|` stays finite within a sequence and across 300 training steps; the main model line is untouched |

⚠️ As noted in the results table, **none of these have been executed yet.**

## Scaling notes (honest)

A ~37M-param `DaedalusFullAdaptive` trained on ~117M tokens of Rust reaches a low
byte-level loss quickly (~0.46 bits/byte) — but early generation collapses into
whitespace. Two reasons, both worth knowing:

1. **Loss ≠ capability.** Deeply-nested code is dominated by indentation, so a
   model can drive loss down by mastering whitespace long before it learns real
   structure. Watch *generation*, not just the loss curve.
2. **Redundant data inflates the number.** Scraped repos share boilerplate,
   generated code, and near-duplicate files, so low loss partly reflects how
   predictable the data is.

Coherent code needs (a) much more training (this is <1 epoch), (b) more/cleaner
data (the full Stack, deduped), and (c) scale. See the roadmap.

**Proteus is expected to be the shaky one, and that is the point.** A weight
matrix that writes its own updates has a known failure mode: it teaches itself to
write ever harder and `||W||` runs away, often while the loss still looks healthy.
Two guards are on by default (L2-normalised queries/keys, and Moirai's erase gate
initialised near 1.0), but they are guards, not proofs. Watch
`DaedalusProteus.weight_norms()` over training rather than the loss curve — the
same lesson as the whitespace collapse above, in a different disguise. If the
fully self-referential mode (`--self-referential`) diverges where the single-level
one does not, that is a legitimate result to report, not a bug to hide.

## Roadmap

- [x] RoPE positions (`daedalus/rope.py`)
- [x] Input injection into the recurrent core (Huginn-style)
- [x] Integrated `DaedalusFull` + first Rust training run
- [x] Fuse adaptive halting (Ariadne) into `DaedalusFull` (`DaedalusFullAdaptive`)
- [x] Gated fast-weight mixer (`daedalus/moirai.py`, `--mixer moirai`) — *code done, run pending*
- [x] Mixture-of-memories (`daedalus/naiads.py`, `--n-mem-banks`) — *code done, run pending*
- [x] Loop self-distillation (`daedalus/echo.py`, `--echo-weight`) — *code done, run pending*
- [x] Self-referential weights (`daedalus/proteus.py`, `--model proteus`) — *code done, run pending*
- [ ] Chunk-wise parallel form of the Moirai scan (the `for t` loop is v1)
- [ ] DeepSeek-style auxiliary-loss-free load balancing
- [ ] `transformers`-compatible model class (for LoRA / vLLM ecosystem)
- [ ] Scale up the compute ladder (100M → 1B) and release weights
- [ ] Plan → execute flow (Metis → Talos) and constitution verifier (Oracle)

## License

MIT — see [LICENSE](LICENSE). Contributions and forks welcome.
