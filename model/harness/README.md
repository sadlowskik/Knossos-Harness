# The Daedalus harness

Everything around the engine slot. The `daedalus/` package is the architecture —
torch-only, meant to be trained from. This package is the system that *uses* a
model, and it does not care which one.

| Module | What it is |
|---|---|
| `argus.py` | Repo-wide index and retrieval — which parts of the codebase belong in the context window |
| `acp.py` | [Agent Client Protocol](https://agentclientprotocol.com) server, so editors can spawn Daedalus as an agent |
| `jsonrpc.py` | Bidirectional JSON-RPC 2.0 over newline-delimited stdio |
| `engine.py` | The swappable engine slot — retrieval-only, any OpenAI-compatible endpoint, or local weights |
| `gate.py` | Decides whether repository context belongs in the prompt at all |
| `eval.py` / `evalset.py` | Measures what the harness is worth: the same model with and without it |
| `workspace.py` | The path jail and write-staging boundary — everything an executor may touch |
| `tools.py` | What an executor may *do*, and how a text-only engine asks for it |
| `ariadne.py` | Halting policy: a hard ceiling, and escalating pressure before it |
| `talos.py` | The executor loop — completion decided by the verifier, not the engine |
| `oracle.py` | Tiered verification: fail fast, model judgement last of all |

Planned: Metis (planner), Lethe (bounded context).

## Execute mode

By default the agent answers questions. `--execute` lets it carry out tasks —
reading, editing and running commands.

```bash
python -m harness --engine api --provider openwebui --execute
python -m harness --engine api --provider openwebui --execute --write
```

**Both safety defaults are opt-out, not opt-in.** Execute mode is off unless
asked for, because a retrieval agent cannot damage a workspace and an executor
can. And within it, edits are **staged rather than written** unless you pass
`--write` — so the default failure mode is a change you have to approve, not one
you have to notice.

`--execute` with the default `retrieval` engine is refused outright.
`RetrievalOnlyEngine` never emits a tool call, so an executor built on it would
spend its entire budget doing nothing and look broken rather than misconfigured.

Tool calls surface to the editor as real ACP `tool_call` updates with status, not
as prose — so a refused command renders as a **failed action** rather than
disappearing into a transcript. Verification gets its own step in the activity
list.

`stopReason` tells the truth: only a verified run reports `end_turn`. A task that
ran out of budget or failed verification says so, rather than letting the editor
render it as complete.

A second prompt in the same session continues the conversation and keeps any
staged edits, rather than starting over.

| Flag | Default | Meaning |
|---|---|---|
| `--execute` | off | carry out tasks rather than only answering |
| `--write` | off | write directly instead of staging for review |
| `--max-steps` | 12 | Ariadne's hard ceiling |
| `--target-steps` | 6 | where budget pressure begins |

## The execution side

Until now the harness only *read* — Argus retrieves, the engine answers. Writing
needs a boundary, and these three are it.

**`workspace.py` — the jail.** Every path resolves through it, and anything
landing outside the root is refused: `../` traversal, absolute paths, symlinks
pointing out of the tree. In `dry_run` mode nothing reaches disk; edits are
staged in memory and *reads consult the staging area first*, so a previewed
multi-step change is what would actually have happened rather than a guess about
it.

**`tools.py` — and why prompted JSON.** Most providers offer native tool calling,
and using it would be more reliable. It would also put tool calling inside the
engine slot — and the slot has to hold a scaled-up Daedalus core one day, which
will emit bytes and nothing else. An executor that only works with providers
implementing OpenAI's function-calling shape is one the from-scratch engine can
never fill. So tools are described in the prompt, the engine emits fenced JSON,
and the harness parses it. `parse_calls` fails closed: anything it cannot read as
a call stays prose, so a malformed reply costs a turn instead of firing the wrong
action.

`run_command` uses **no shell**. Commands are split into program plus argument
vector and executed directly, so `&&`, `|`, `;` and backticks arrive as literal
arguments. The program must also be on an allowlist, and `git` is limited to
read-only subcommands. Removing shell semantics makes "the model appended
`&& rm -rf`" a non-event rather than something to pattern-match for.

**`talos.py` — and the one rule that shapes it.** *Completion is decided by the
verifier, not by the engine.* An engine that stops calling tools is making a
**request** for verification, not announcing success; only a passing verdict
produces `DONE`. Without that, "finished" means "the model felt finished", which
is the exact claim this harness exists to stop trusting. There is a test that
runs an engine insisting it is done three times over and asserts the run does
*not* succeed.

Oracle does not exist yet, so Talos takes a `Verifier` — a callable returning a
`Verdict`. Oracle will satisfy it without the loop changing. The default is
called `accept_everything` and says so in its own summary, because a run with no
verifier has no check on correctness and that should be visible rather than
comfortable.

One honest cost: `Engine.generate` takes two strings and has no message history,
so the conversation is **re-rendered into the prompt every turn**. That is
quadratic in tokens over a long run — the price of keeping the engine interface
narrow enough for a from-scratch core to fill. Lethe is the intended fix; until
then the step ceiling bounds the growth, which is a blunt instrument rather than
a solution.

**`ariadne.py` — the halting policy, carrying a measured lesson.** Not PonderNet:
none of that math transfers to an agent loop. What transfers is the failure mode
`daedalus/ariadne.py` actually recorded — at β=0.01 the halting distribution
collapsed to maximum depth (7.5 of 8 steps, no adaptivity), at β=0.1 it settled
near 5. Without explicit *increasing* pressure to stop, an adaptive-compute
system spends its whole budget on every input regardless of difficulty. So:
a hard ceiling, escalating prompt pressure past a target, and verification
passing as the deterministic stop signal.

## Run it

```bash
python -m harness --engine retrieval
```

It speaks JSON-RPC on stdin/stdout and logs to stderr, so it looks like it hangs
when run by hand — that is correct. An editor drives it.

## Register it with Zed

In `settings.json`:

```json
{
  "agent_servers": {
    "Daedalus": {
      "type": "custom",
      "command": "C:/Users/you/AppData/Local/Python/pythoncore-3.14-64/python.exe",
      "args": ["-m", "harness", "--engine", "api", "--provider", "groq"],
      "env": { "PYTHONPATH": "/absolute/path/to/Recurring Transformer Model" }
    }
  }
}
```

Use the **absolute** path to the interpreter. `"command": "python"` only works if
`python` resolves on the PATH the editor inherits, which on Windows it often does
not — the Store app-execution alias is not always on PATH, and Zed does not
launch through your shell profile.

The API key is read from the environment, so it need not appear here. Set it once
at user scope and restart the editor so it inherits the value.

`PYTHONPATH` is needed because Zed spawns the agent with the *workspace* as cwd,
which is not necessarily this repo. Drop it once the package is installed.

Same protocol works in JetBrains IDEs and any other ACP client.

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--engine` | `retrieval` | `retrieval` (no model), `api` (hosted or local endpoint), `transformers` (local weights) |
| `--provider` | `groq` | `api` engine: which preset endpoint |
| `--model` | provider default | Model id |
| `--base-url` | provider default | Override the endpoint entirely |
| `--list-models` | — | Ask the provider what it serves, then exit |
| `--adapter` | none | `transformers` engine: PEFT/LoRA directory |
| `--budget` | `8000` | Retrieval budget, in characters |
| `--hops` | `1` | How far to expand along import edges |

## Using a hosted model

`OpenAICompatEngine` speaks the OpenAI `/chat/completions` shape, which nearly
every provider exposes — so switching provider is a flag, not a code change.
It is stdlib-only (`urllib`), because the harness has no third-party
dependencies and this was not a good enough reason to start.

```bash
export GROQ_API_KEY=...            # never passed as an argument, never logged
python -m harness --engine api --provider groq
```

| Preset | Endpoint | Key variable |
|---|---|---|
| `groq` | `api.groq.com` | `GROQ_API_KEY` |
| `nvidia` | `integrate.api.nvidia.com` | `NVIDIA_API_KEY` |
| `gemini` | `generativelanguage.googleapis.com` | `GEMINI_API_KEY` |
| `cerebras` | `api.cerebras.ai` | `CEREBRAS_API_KEY` |
| `openrouter` | `openrouter.ai` | `OPENROUTER_API_KEY` |
| `mistral` | `api.mistral.ai` | `MISTRAL_API_KEY` |
| `together` | `api.together.xyz` | `TOGETHER_API_KEY` |
| `ollama` | `localhost:11434` | none |
| `local` | `localhost:8000` | none — llama.cpp / vLLM / LM Studio |

Base URLs are stable; **model ids drift constantly**. When one 404s:

```bash
python -m harness --list-models --provider groq
```

Reasoning models (Qwen3, DeepSeek-R1, gpt-oss) spend part of the token budget
inside `<think>`, which is why `max_tokens` defaults to 4096 — at 1024 the whole
budget can go to thinking and the turn ends truncated with an empty reply.
Thinking is separated out and sent as `agent_thought_chunk`, so editors render it
as collapsed reasoning rather than as the answer.

Free tiers generally reserve the right to train on your prompts. Fine for
testing against an open-source repo; use `ollama` or `local` for anything else.

## One turn

```
session/new       cold-index the workspace with Argus
session/prompt    rescan (incremental) → retrieve → engine → stream back
                  ├── session/update  tool_call         "Searching the repository"
                  ├── session/update  tool_call_update   completed + file locations
                  ├── session/update  agent_message_chunk (×N)
                  └── { "stopReason": "end_turn" }
```

Retrieval is emitted as a visible tool call rather than hidden, so the locations
Argus chose become clickable references in the editor. When the agent answers
from the wrong file, you can see that it did, and why.

## Filling the engine slot

One method, streaming, cancellable:

```python
class MyEngine:
    name = "my-engine"

    def generate(self, prompt, context, cancelled):
        yield "some text"
```

`context` is an `argus.Context` — a `str` subclass, so it behaves like the
rendered text everywhere, with the structured `.hits` still attached for engines
that want to report on the retrieval rather than reason over it.

`cancelled()` is polled between chunks; return promptly when it goes True.

## Known limits

- **Rust symbols are approximate.** Python goes through `ast` and is exact. Rust
  goes through a declaration scanner that resolves no types, no generics, and no
  cross-file references. `FileRecord.exact` is `False` for Rust and says so.
  Upgrade path: tree-sitter standalone, or rust-analyzer inside an editor.
- **One turn at a time.** `session/prompt` occupies the worker thread, so two
  sessions cannot prompt concurrently. `session/cancel` is exempt — it runs on
  the reader thread, or it could never interrupt the turn it targets.
- **`session/load` is not implemented**, and `loadSession` is advertised `false`.
- **`TransformersEngine` is unexercised by the test suite** — no torch on the
  dev machine. Verify it on a GPU box before trusting it.
- **Ranking is multi-field BM25 over identifiers, with no semantic model.** A
  question sharing no identifiers with the code it is about will still miss.

## Ranking

Three fields, scored separately with their own idf and summed (a simplified
BM25F), because they are three different strengths of evidence:

| field | weight | evidence |
|---|---|---|
| `idents` | 1.0 | what the file *says*. Weak — anything can mention anything |
| `defs` | 2.5 | what the file *defines*. Strong — `moe.py` defines `Router` |
| `path_terms` | 2.0 | what the file is *called*. "moe" means `moe.py` |

Files whose job is to discuss code — tests, fixtures, eval sets — are scaled by
`META_PENALTY` (0.6). They stay findable; they just stop outranking the code they
quote.

Each version failed differently, which is why the sequence is worth keeping:

1. **tf-idf** — a long file repeating "file" and "defines" outranked the short one
   that defined the thing asked about. Fixed by BM25 term saturation.
2. **single-field BM25** — still ranked a symbol's *user* above its *definition*:
   `naiads.py` imports `Router`, `moe.py` defines it, and `naiads.py` won for
   being shorter and saying the word more often.
3. **multi-field** — current.

Measured on the eval set, before → after:

| | single-field | multi-field |
|---|---|---|
| recall | 10/10 | 10/10 |
| in top 3 | 8/10 | **10/10** |
| mean rank | 2.1 | **1.2** |

Eight of ten cases now rank first. `moe-router`, the case that produced wrong
answers on *both* evaluated models, went from rank 5 to rank 1.

Weights are tuned in-sample against these ten cases. Re-measure with
`python -m harness.eval --mode retrieval` after changing them.

## Results

`gemma4:e4b` (8B 4-bit, local), **5 samples per case per condition**, 190 calls,
**zero provider errors — every case paired**:

| kind | raw | harness | delta |
|---|---:|---:|---:|
| repo_specific (n=10/10) | 0.23 | **0.82** | **+0.59** |
| general (n=6/6) — *control* | 1.00 | 0.92 | −0.08 |
| negative (n=3/3) | 0.00 | **0.60** | **+0.60** |

The control behaves: `general` moves −0.08, not up. A lift there would mean the
grader was rewarding something other than retrieval and would invalidate the
`repo_specific` number too.

**What changed against the earlier n=1 run**, which is the argument for `--repeat`:

| kind | n=1 | n=5 | |
|---|---:|---:|---|
| repo_specific | +0.65 | +0.59 | holds, slightly smaller |
| general | −0.08 | −0.08 | **reproduced exactly** |
| negative | +0.33 | +0.60 | n=1 **understated** it by half |

The `general` control reproducing to two decimals across a fivefold increase in
samples is the strongest single result here — it is what lets the retrieval gate
be argued on a measured penalty rather than a guessed one.

### Fabrication

Counted directly, as known-wrong citations across all 5 samples of the 3 absent
features:

| condition | fabrications |
|---|---:|
| raw | **15 / 15** |
| harness | **6 / 15** |

A 60% reduction, not a fix. The failure is concentrated: `no-database` fabricates
in 5/5 raw *and* 5/5 harness — retrieval does not help when the model is
determined to invent a database layer. `no-auth` and `no-retries` are corrected
almost completely (5→0 and 5→1).

### Cross-scale claim: not currently supported at this n

An earlier n=1 run put `gpt-oss-120b` at +0.64 on `repo_specific` against
gemma's +0.65, and this file claimed "the retrieval delta is the same across a
15× size gap." Only the gemma arm has been rerun at n=5. Until the hosted arm is
repeated, that claim rests on one measurement at n=5 and one at n=1, which is
not enough to assert. It is a promising observation, not a result.

Do **not** read absolute scores across models either: `gpt-oss-120b` ran at
`max_tokens=1000` against ~1685 characters of reasoning and answered more
tersely (614 vs 1292 chars), and a keyword grader rewards verbosity.

## The retrieval gate

Injecting is the default; skipping must be argued for, because a wrong skip loses
a repo answer while a wrong inject loses a general one.

**Honest note on why this exists.** An earlier measurement showed injecting repo
excerpts into a general question dropping `gemma4:e4b` from 1.00 to 0.00, and the
gate was built on that. Once Ollama's `num_ctx` truncation was fixed, the real
effect turned out to be **−0.08**. A second claim — that small models suffer more
from irrelevant context — is also unsupported here: the 120B model degraded *more*
(−0.17) than the local 8B. The gate still earns its place, on cost (~2000 tokens
and a chunk of latency saved per general question) and on a small but real
measured penalty at both scales. It is not the "irrelevant context wrecks small
models" story the first draft of this file told.

```bash
python -m harness.eval --mode gate       # measure it; no model, no network
python -m harness --no-gate              # disable, always inject
```

Currently **18/19 (95%)** on the labelled eval set, against 68% for injecting
unconditionally — 13/13 repo questions still injected, 5/6 general ones skipped.

Signals, each named in the decision so a wrong call can be argued with:

| signal | means |
|---|---|
| anchor | "which file", "in this codebase" — asks about *here* |
| filename | `moe.py` appears in the question |
| distinctive | a word naming something defined in ≤3 files. `Mnemosyne` means this repo; `forward` means nothing |
| concentration | retrieval spiked on one file rather than smearing over ten |
| generality | "in general", "conceptually" — asks about the idea |

Generality wins unless an anchor, a filename, or a proper name overrides it.
Additive weights alone got this wrong: an incidental match (`post` → `_post`)
cancelled an explicit "in general" and dragged the score back over the line.

**The 95% is in-sample.** The thresholds were tuned against these same 19 cases,
so treat it as "the rule is consistent with the examples", not as generalisation.
`gradclip-general` ("why does gradient clipping help stabilise training?") is kept
in the set precisely because it fails — no marker, no framing, no symbol.

## Evaluation

```bash
python -m harness.eval --mode retrieval              # no model, no network
python -m harness.eval --mode answer --repeat 3      # raw vs harness
```

Two evaluations, separate because they fail for different reasons.

**Retrieval** asks whether Argus surfaced the expected files. Deterministic and
free, so run it after any ranking change — a regression shows up here first.

**Answer** runs each case twice, `raw` (question alone) and `harness` (question
plus retrieved context), and reports the delta. Cases are tagged:

| kind | what it measures |
|---|---|
| `repo_specific` | Facts existing only in this codebase. The harness should win big |
| `general` | Pretraining knowledge. A **control** — the harness should change nothing |
| `negative` | Things absent from the repo. Pass = saying so. Measures fabrication |

The control group is the load-bearing part. A lift on `general` would mean the
grader is rewarding something other than retrieval, and would invalidate the
`repo_specific` result too. Read the by-kind block, never the totals.

Limits worth knowing before trusting a number:

- The grader matches keywords and citations; it does not judge meaning.
  `expect_terms` are necessary, not sufficient.
- Absence detection uses a phrase list, so a model declining in unanticipated
  wording scores as fabrication. The negative score is a **lower bound** on
  honesty.
- Provider errors (429, unreachable) are excluded from scoring, never counted as
  wrong answers — otherwise a rate limit makes the harness look worse the more
  you test it.
- Use `--repeat 3` or more. A single sample per condition reports noise as
  signal.

## Tests

```bash
python -m pytest tests/test_argus.py tests/test_acp.py tests/test_engine.py -q
```

No torch required — the harness does not depend on the model. The ACP tests
drive the agent over real pipes as an editor would, rather than calling handlers
directly, because framing, threading, and mid-turn cancellation are where this
actually breaks.
