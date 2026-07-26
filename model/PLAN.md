# Retrospective and recovery plan

Written 2026-07-26, after a working session that produced a large amount of code
and left the project less verified than it started.

> **Progress — 2026-07-26, later the same day.**
> Stage 0 **done**. Stage 1 **substantially done**.
>
> | | At the time of writing | Now |
> |---|---|---|
> | Fast tests | 5 files uncollectable, 19 ACP failures | **168 passed** |
> | Slow tests | never executed | **4 passed** |
> | Real bugs found and fixed | — | **5** |
>
> The five bugs are listed in §1.8. Two of them — the Echo KL scale error and
> the `echo_step` depth confound — were invisible to every unit test and only
> surfaced when the training-based sweep was finally run. That is the clearest
> possible evidence for P2, and it happened on this project's own code.
>
> Stage 3 **done for the local arm** — the with/without eval reran at n=5 on
> `gemma4:e4b` with zero provider errors and every case paired. `repo_specific`
> +0.59, `general` control −0.08 (reproducing the n=1 figure exactly), fabrication
> down from 15/15 to 6/15. The hosted arm is still n=1, so the "same delta across
> a 15× size gap" claim has been demoted to an observation.
>
> Stage 2 **substantially done**. The Rust harness's four distinctive pieces are
> now in the Python one, built in dependency order rather than the order §3
> listed — a verifier needs something to verify, so the jail and the executor
> came first:
>
> | Module | Tests | What it carries over |
> |---|---:|---|
> | `workspace.py` | 16 | path jail, write staging |
> | `tools.py` | 24 | argv-not-shell execution, allowlist, fail-closed parsing |
> | `ariadne.py` | 13 | hard ceiling, escalating pressure, the β=0.01 lesson |
> | `talos.py` | 13 | the loop; completion decided by the verifier |
> | `oracle.py` | 15 | tiered ladder, fail fast, judgement gated last |
> | ACP execute mode | 10 | tool calls as editor actions, staged by default |
>
> Mnemosyne was **not** ported, per §3 — Argus already does that job and has an
> eval behind it.
>
> Still open: archiving `daedalus-harness`, Metis (planner), Lethe (bounded
> context), and rerunning the hosted eval arm.
>
> **Two things worth noting about how Stage 3 actually got unblocked.** The
> provider quota in §1.9 was never the real obstacle — a local Gemma was
> reachable the whole time at `192.168.4.103:11434`, and a previous session had
> already used it. I probed `localhost`, found nothing, and went looking for
> Open WebUI credentials that were never needed. That is P1 again: the survey
> was of the wrong machine.
>
> And running locally is what produced the clean result. Every Groq attempt
> lost cases to quota or truncation, and those losses correlated with condition.
> The local run lost nothing, so the n is real rather than whatever survived.

This is not a list of bugs. It is a list of the *practices* that produced them,
each one traced to evidence, followed by an ordered plan to get back to a state
where the numbers in a README can be trusted.

---

## Part 1 — What the results actually say

Seven findings, all reproducible today.

### 1.1 The ACP server cannot complete a handshake

```
19 failed, 100 passed
```

Every failure is in `tests/test_acp.py`, every one is `KeyError: 'result'`, and
all of them die at the same call — `session/new`:

```
File "harness/argus.py", line 640, in save
TypeError: keys must be str, int, float, bool or None, not tuple
```

`FileRecord` has three `Counter` fields — `idents`, `defs`, `path_terms`
([argus.py:104-112](harness/argus.py:104)). `save()` rescues only `idents`
([argus.py:630](harness/argus.py:630)).

`dataclasses.asdict` rebuilds a dict subclass by handing its constructor a
generator of `(key, value)` tuples. `dict` accepts that; `Counter` does not —
it counts the tuples as *elements*, producing `{("router", 3): 1}`. JSON refuses
tuple keys. It only triggers when a Counter is non-empty, which is why an empty
fixture would not catch it.

**Everything downstream of ACP is blocked by this.** Zed, JetBrains, and any
future Lapce integration all enter through `session/new`.

### 1.2 The architecture tests have still never run

```
ERROR tests/test_components.py
ERROR tests/test_echo.py
ERROR tests/test_moirai.py
ERROR tests/test_naiads.py
ERROR tests/test_proteus.py
Interrupted: 5 errors during collection
```

No torch on this machine, so they cannot even be collected. The README has said
"none of these have been executed yet" since before this session started. It is
still true.

Four mechanisms — Moirai, Naiads, Echo, Proteus — remain code-reviewed but
unexecuted, while roughly 7,000 lines of *new* unverified surface was built on
top of them.

### 1.3 A stale index silently degrades retrieval

`daedalus-harness/.argus/index.json`, 253KB, written 12:15. Its records contain:

```
path, sha, language, n_lines, exact, symbols, imports, idents, parse_error
```

No `defs`. No `path_terms`. It predates the 14:13 change that added them.

`save()` writes `"version": 1`. `load()` never reads it
([argus.py:643](harness/argus.py:643)). So this file loads without complaint,
defaults fill in empty Counters, and ranking runs with two of its three scoring
signals blank — no error, no warning.

`defs` exists specifically because flat term frequency "ranked the user of a
symbol above its definition." That correction is currently switched off wherever
a pre-14:13 index exists.

**A crash gets noticed. This does not.**

### 1.4 The same retrieval system was built twice

| | `harness/argus.py` | `daedalus-harness/src/mnemosyne.rs` |
|---|---|---|
| Algorithm | BM25 over identifiers | BM25 over symbol chunks |
| Lines | 562 | ~430 |
| Eval behind it | yes | none |

The second was written without checking whether the first existed — despite
project notes naming `harness/` and "ACP seam first" explicitly. The
BM25-over-embeddings rationale was then presented as a fresh recommendation,
when `harness/README.md` already stated it under "Known limits."

### 1.5 Two harnesses, neither complete

| | Python `harness/` | Rust `daedalus-harness` |
|---|---|---|
| Editor seam | **ACP** — Zed, JetBrains, any client | VS Code extension only |
| Engine slot | 9 providers, local weights, retrieval-only | Anthropic + Ollama |
| Evaluation | **eval.py, evalset.py, measured deltas** | none |
| Retrieval | Argus | Mnemosyne (duplicate) |
| Verification | planned | **Oracle, 4 tiers, real cargo** |
| Safety | — | **path jail, argv-not-shell, allowlist** |
| Halting | — | **Ariadne budget** |
| Diff review | — | **per-hunk accept/reject** |

Complementary rather than redundant, which is worse than either — it means
neither one is a finished system, and the split is invisible from inside either
repository.

### 1.6 The results table contradicts its own methodology note

`harness/README.md` says:

> Use `--repeat 3` or more. A single sample per condition reports noise as signal.

The headline table immediately above it is `n=1`.

| Row | Value | Does it survive n=1? |
|---|---|---|
| `repo_specific` +0.65 / +0.64 | large, n=10 cases | probably |
| `general` −0.08 / −0.17 | the **control** | no — indistinguishable from noise |
| `negative` 0.00 → 0.33 | n=3 | no — that is *one case* flipping |

The `general` row is load-bearing: the retrieval gate's justification rests on it
being "a small but real measured penalty at both scales." At n=1 that is not yet
measured.

The single most interesting claim in the document — *the retrieval delta is
constant across a 15× model size gap*, which is the actual argument for the
engine slot being real — also rests on one sample per condition.

### 1.7 Green test suites over broken systems

Both sides show the same shape:

- **Python**: 100 unit tests green while the ACP server could not complete a
  handshake. The tests that failed were precisely the ones driving real pipes.
- **Rust**: 130 tests green, `MockEngine` throughout, **zero live engine calls
  ever made**. The Anthropic and Ollama backends have never exchanged a packet.

A suite that is green on a system that does not work is not measuring the system.

### 1.8 What the first real run actually found

Five bugs, all fixed the same day the tests were first executed.

| # | Bug | Where | Effect | Could a unit test have caught it? |
|---|---|---|---|---|
| 1 | `load_balance_loss` result not unpacked | `full.py` ×2 | Both flagship models raised on every forward | Yes — and one did, once it could be collected |
| 2 | `targets.view()` on a non-contiguous tensor | 6 call sites | `RuntimeError` on strided targets | Yes |
| 3 | `echo_loss` KL scaled by `T` | `echo.py` | `batchmean` divides by `shape[0]`; on `(B,T,V)` the term was **64× too large** and collapsed training | **No** |
| 4 | `echo_step` moved the CE depth | `echo.py` | `--echo-weight` changed two things, so the sweep was not an ablation | **No** |
| 5 | `Argus.save()` dropped two Counters | `harness/argus.py` | `TypeError`; ACP `session/new` failed | Yes — `test_acp.py` caught it |

**Bugs 3 and 4 are the important ones.** Both `echo_loss` unit tests pass with
the 64× scale error in place, because "zero when they agree" and "positive when
they disagree" are scale-invariant. No amount of unit testing would have found
it. Only the training-based sweep did — the one marked `slow` and excluded by
default.

Echo's central claim, once those were fixed (n=3 seeds):

| loops | mean Δ | per-seed deltas |
|------:|-------:|:----------------|
| **1** | **−0.1205** | −0.2258, −0.0613, −0.0744 |
| 2 | −0.0125 | −0.0179, −0.0032, −0.0164 |
| 3 | −0.0046 | −0.0059, −0.0009, −0.0071 |
| 4 *(training depth)* | −0.0033 | −0.0058, −0.0005, −0.0038 |

All three seeds improve at every depth, and loop 1 improves by an order of
magnitude more than the rest. The claim holds.

**And P3 caught me writing it up.** The first version of this table was seed 0
alone and reported the loop-1 loss as *halving*. Seed 0's Echo-off baseline was
an outlier (0.4385 against 0.2501 and 0.2695), so the single-seed figure
overstated the effect by about 2×. That number sat in the README for an hour,
in a document whose own Part 2 names "claims ahead of sample size" as a root
cause. The rule in Part 4 — *no number without its n, and no n below 5* — is not
a rule anyone follows by intending to.

### 1.9 The eval quota lesson

The first attempt at the with/without answer eval ran against a *reasoning*
model. It spent 7,000–8,800 characters per call inside `<think>`, exhausted
`max_tokens` before answering, and burned the entire 100,000 token/day provider
quota producing **zero usable answers**.

Worse, the failures were **biased**: the `raw` condition errored far more often
than `harness`, because a model with no context reasons longer. Since provider
errors are excluded from scoring, that exclusion correlates with the condition
being measured — which invalidates the comparison rather than merely thinning it.

Two changes follow:

- Pick a non-reasoning model for eval runs, or set `--max-tokens` well above the
  thinking budget and verify on one case before spending the quota.
- **Errors must be reported per condition.** An error count that is lopsided
  across arms is a confound, not noise, and the current summary does not make
  that visible.

---

## Part 2 — The practices that produced this

Six root causes. Each maps to findings above.

| # | Practice | Evidence | Consequence |
|---|---|---|---|
| **P1** | Build before surveying | 1.4, 1.5 | Duplicated retrieval; two half-systems |
| **P2** | Mock-only verification | 1.1, 1.7 | Suites green on broken systems |
| **P3** | Claims ahead of sample size | 1.6 | Headline numbers not defensible |
| **P4** | No schema discipline | 1.3 | Silent quality regression |
| **P5** | Breadth before depth | 1.2 | Oldest verification debt never paid |
| **P6** | Seam chosen by momentum | 1.5 | Built for the editor you dislike |

**P1 — Build before surveying.** No inventory of existing work preceded new
work, in a project that already spanned two repositories.

**P2 — Mock-only verification.** Mocks were treated as sufficient rather than as
a fast inner loop. The one integration-level suite (`test_acp.py`) is the only
thing that caught a total failure.

**P3 — Claims ahead of sample size.** The methodology note and the results table
were written by the same hand and disagree. Discipline was documented, not
applied.

**P4 — No schema discipline.** A version field was written and never enforced,
so a format change degraded behaviour silently instead of failing loudly.

**P5 — Breadth before depth.** Every session added surface. None retired debt.
The four unrun mechanisms were the top of the list at the start of the session
and are still there.

**P6 — Seam chosen by momentum.** A VS Code extension, a webview panel, an
NDJSON protocol, and a fork scaffold were built while a working ACP server —
the seam that reaches Zed, JetBrains and Lapce — already existed and was broken.

---

## Part 3 — Recovery plan

Ordered. **Each stage gates the next.** No new capability is added until the
stage before it is verified.

### Stage 0 — Stop the bleeding (hours)

| Task | Detail |
|---|---|
| 0.1 | Fix `save()` — add `defs` and `path_terms` to the Counter override |
| 0.2 | Fix `load()` — restore both as `Counter`, and **return `False` when `version != 1`** so a stale index rebuilds instead of degrading |
| 0.3 | Delete `daedalus-harness/.argus/` |
| 0.4 | Delete `daedalus-harness/fork/` — the fork question is deferred, not open |
| 0.5 | Add a regression test: save → load → assert all three Counters survive non-empty |

**Gate:** `pytest tests/ -q` shows 0 failures among the collectable files.

### Stage 1 — Make the suite mean something (half a day)

| Task | Detail |
|---|---|
| 1.1 | Install torch. Run the five collection-error files. |
| 1.2 | Record what Moirai, Naiads, Echo and Proteus actually do when executed — pass or fail. Update the README's four `—` rows with results or with the failure. |
| 1.3 | Add one live-engine smoke test, `--engine ollama`, marked `slow` and skipped by default. One real request, asserting a non-empty reply. |

**Gate:** no file in `tests/` is unrunnable, and at least one test path touches a
real model.

**Why 1.2 first:** this is the oldest debt in the project and the only part that
is genuinely novel research. It has been deferred through every session.

### Stage 2 — Collapse to one main line (a day)

Python `harness/` becomes the main line. Three reasons, none of them sunk cost:

1. **ACP reaches every editor you care about.** A VS Code extension binds you to
   the one you dislike.
2. **The engine slot must eventually hold your own model**, which is PyTorch.
   Python loads it in-process; Rust needs candle, ONNX, or a subprocess —
   friction on the one swap the architecture exists for.
3. **The evals live there.** Measured deltas beat unmeasured features.

| Task | Detail |
|---|---|
| 2.1 | Port **Oracle** — tiered ladder, fail-fast, model judgement last and only when deterministic tiers pass |
| 2.2 | Port **the safety model** — path jail, argv-not-shell execution, program allowlist. Required before Talos can write anything. |
| 2.3 | Port **Ariadne** — hard ceiling, escalating pressure past target, carrying the β=0.01 collapse lesson |
| 2.4 | Port **per-hunk review** — accept/reject before anything reaches disk |
| 2.5 | Archive `daedalus-harness` with a README pointing at `harness/`. Do not maintain both. |

Mnemosyne is **not** ported. Argus already does that job and has an eval behind
it.

**Gate:** the Python harness can plan, execute, verify and halt — and refuses to
write outside its workspace.

### Stage 3 — Make the numbers defensible (a day, mostly waiting)

| Task | Detail |
|---|---|
| 3.1 | Re-run `--mode answer --repeat 5`, both models. Replace the n=1 table. |
| 3.2 | State `n` in the table itself, not only in prose below it. |
| 3.3 | Report the `general` control with its spread. If the −0.08/−0.17 penalty does not survive n=5, **say so and re-argue the gate on cost alone.** |
| 3.4 | Hold out 6–8 fresh gate cases never used for tuning. Report in-sample and held-out separately. |
| 3.5 | Drop "95%" from headline framing until 3.4 exists. |

**Gate:** every number in `harness/README.md` states its `n`, and no number
tuned on a set is reported as a measurement of that set.

### Stage 4 — Editor integration, once (half a day)

| Task | Detail |
|---|---|
| 4.1 | Verify ACP end-to-end in Zed with a real provider |
| 4.2 | Document the exact `settings.json` that works, with the absolute interpreter path |
| 4.3 | Re-open the fork question **only** if ACP proves insufficient — with the specific insufficiency named |

---

## Part 4 — Working rules

The plan above fixes the current state. These stop it recurring.

1. **Survey before building.** Both repositories, every time. If a component
   exists, extend it or replace it deliberately — never in parallel.
2. **One main line.** Two implementations of the same thing is a bug, not
   optionality.
3. **A test is green only if it exercises the real path.** Mocks are the inner
   loop, not the verdict. Every subsystem needs at least one test that touches
   the real interface.
4. **No number without its `n`, and no `n` below 5** for anything reported as a
   result.
5. **Enforce every version field you write.** A schema that fails loudly beats
   one that degrades quietly.
6. **Pay verification debt before adding surface.** If something is marked "code
   done, run pending," nothing new gets built on top of it.
7. **Choose the seam from the requirement.** "Which editors must this reach"
   comes before "what shall I build."

---

## What this costs

Stages 0–3 are roughly **three days**. None of it is new capability — it is
turning existing work into work that can be trusted.

The alternative is a project with two harnesses, four unexecuted mechanisms, and
a results table that its own methodology note tells you not to believe. That is
a worse position than the project was in this morning, and it got there by
building faster.
