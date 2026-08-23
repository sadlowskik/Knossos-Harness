# Knossos — harness architecture

Scope: the agentic harness only. `model/daedalus/` (the model architecture) appears
here only as the engine slot's other side; `editor/` (a git submodule),
`node_modules/` and `.claude/worktrees/` are out of scope entirely.

State documented: **the working tree at HEAD `d37739d` (2026-07-30), branch
`fix/wire-delegation-constitution-ariadne`, clean tree** (this branch is ahead of
`main`; `main` does not yet carry the delegation/constitution wiring or the Rust
ports below).

> **2026-08-15 re-verification pass.** Sections §§1–11 were written on 2026-07-28/29
> and their `file:line` anchors were re-derived *then*. This pass re-verified the
> component inventory, entry points, wiring, and cross-language parity — and found
> two large bodies of work landed **after** that pass, which §§1–11 predate:
>
> 1. **A containment layer** (commit `e2ab40a`, 2026-07-29): `sandbox.py`,
>    `hooks.py`, `interject.py` and Rust mirrors `sandbox.rs`, `hooks.rs`,
>    `interject.rs`, `resilience.rs`. None appear in the §2 maps. Documented below.
> 2. **A Rust porting spree** (commits through `3411fa7`, 2026-07-30): Rust gained
>    `argus.rs` (1467 lines), `gate.rs`, `lsp.rs`, `mcp.rs`, `jsonrpc.rs`,
>    `delegate.rs`, `acp.rs`, and an `acp` CLI subcommand. **This invalidates the
>    §1 framing of Rust as "CLI-only, cargo-ladder, ~4,700 lines" and the many
>    "absent in Rust" rows in §9.1** — Rust is now near-parity with Python. See the
>    Drift note at the head of §11 and the corrected §2 Rust table.
>
> **Anchor caveat:** legacy anchors in §§1.1–1.2, §4, §5, §8 were *not* all
> re-derived this pass. Several files grew since 07-29 (`talos.py` 1502→1600,
> `acp.py` 1386→1511, `engine.py`→1692, `codeval.py` 1251→1488; `CONSEQUENTIAL`
> moved `348→384`, `_permitted` `1446→1542`), so deep-section anchors may be off by
> tens of lines. Re-grep before trusting a specific line number in those sections.

Current sizes (this pass): **Python `model/knossos/` ~12,900 lines**, **Rust
`knossos-rs/src/` ~18,655 lines** (both far above the §1 figures). Test/eval counts
below are carried forward from the 07-29 pass and were **not** re-run:
**772 Python tests collected** (768 selected, 4 deselected), **~174 Rust test
attributes**, **22 deterministic conformance checks + 2 live** (`conformance/run.mjs`),
**28 retrieval eval cases** (`evalset.CASES`), **12 built-in coding-eval cases**
(`codeval.CODING_CASES`).

One thing a reader should take before anything else:

> **The built-in coding suite is saturated.** Knossos on Gemini 3.1 Pro: 12/12.
> Claude Code on Sonnet 5 (via `--harness-cmd`): 12/12. Knossos on Gemini 3.1
> Flash-Lite: 12/12. `solved` no longer discriminates between harnesses or between
> models; `honest` is the only column with signal left. The six-case hard suite
> (§8.7) exists to restore that signal and now calibrates clean, but **no model has
> been scored against it** — a gradeable suite is not yet a measured one. See §8.

---

## 1. Purpose and the engine slot

Knossos is an agentic coding harness: it indexes a repository, retrieves context,
asks a language model what to do, runs the model's tool calls against a jailed
workspace, and decides for itself — from a verifier, not from the model's own
claim — whether the work is finished. It exists in **two parallel implementations**
that share a design but not a line of code:

| | Python | Rust |
|---|---|---|
| Path | `model/knossos/` (~12,900 lines) | `knossos-rs/src/` (~18,655 lines) |
| Front end | ACP server over stdio (`python -m knossos`) | CLI: `chat`/`index`/`verify`/`plan`/`task`/`repl`/`serve` **and `acp`** (`daedalus`) — Rust now speaks ACP too (`acp.rs`) |
| Target language | Python, Rust, Go, Node (adapter-detected, §5) | Rust only (cargo ladder) |

*(2026-08-15: the earlier line figures — Python ~8,000, Rust ~4,700 — were from
07-29 and are obsolete; Rust roughly quadrupled via the porting spree noted in the
header. "Rust only, cargo ladder" is still true of the **verifier**, not of the
front end: Rust now has an ACP server, Argus retrieval + gate, LSP, MCP, and
delegation.)*

### The engine slot, concretely

**Python** — `engine.py:285-301`. A `typing.Protocol`, `@runtime_checkable`:

```python
class Engine(Protocol):
    name: str
    def generate(self, prompt: str, context: str, cancelled: Cancelled) -> Iterator[str]: ...
```

One *required* method, streaming, cancellable. Everything else the harness uses is
duck-typed and probed with `hasattr`/`getattr`, so a bytes-to-bytes model remains a
valid occupant of the slot:

| Optional member | Read at | What it buys |
|---|---|---|
| `generate_messages(messages, cancelled)` | `talos.py:1243` | Structured OpenAI-shaped history instead of a flattened prompt (§1.1) |
| `tool_schema` | `talos.py:1227`, `metis.py:162` | Native tool-call schema in the request |
| `stop_reason` | `acp.py:787` | Real `stopReason` mapping instead of `end_turn` for everything |
| `usage` (a `Usage`) | `talos.py:1216` | Per-turn cost banking |
| `context_window` | `talos.py:1041` | Transcript budget derived from the server's real window |
| `send_reasoning` | `talos.py:1309` | Opt-in replay of `reasoning_content` (off by default; DeepSeek 400s on it) |
| `throttle_waits` / `throttled_seconds` / `output_shrinks` | `scripts/coding_eval.py:391-393` | The throttling warning that says a run measured the provider, not the model |

**What a run costs, measured.** A stateless completions API re-sends the whole
conversation every turn, and `generate_messages` does not change that — an array
of messages costs what the same content costs flattened. Driving a real `Talos`
through a read-heavy 20-step run: each step adds **~1,341 tokens** to what is
sent, so cumulative cost grows with the square of the step count. At 7 steps,
36,158 tokens; at 14 steps, 138,039 — **3.8×, not 2×**.

Lethe's ceiling is real but late: at the 24,000-token default it first fires at
**step 18**, dropping the sent size from 23,945 to 5,398 in one turn. Below that
the bound is not doing any work, and most runs end below it. So the operative
lever is the **cached prefix**, not the interface: every turn's prompt is a
strict extension of the previous one, which is what lets a provider bill the
shared portion at a fraction. That property is asserted by
`test_each_turn_extends_the_previous_prompt_rather_than_rewriting_it` and its
structured twin, because anything volatile placed early in the prompt drops the
hit rate to zero without failing anything — `Usage.cached` and `_report_cost`'s
hit-rate line are the only places it would show.

`Thought`, a `str` subclass (`engine.py:66`), routes a chunk to the reasoning
channel instead of the answer. Talos detects it by **name comparison, not
`isinstance`** — `type(chunk).__name__ == "Thought"` (`talos.py:1282`) — so any
class named `Thought` from any module satisfies it.

### 1.1 What `EngineProfile` carries

`EngineProfile` (`engine.py:382`) is a frozen dataclass with **exactly one field**:

```python
max_request_bytes: Optional[int] = None
```

`None` means *unknown*, deliberately distinct from unlimited: an unknown limit is
not checked, a known one is enforced before the request leaves the process. The
comment at `engine.py:404-415` states plainly why there is only one field —
`context_window`, `native_tools` and `prompt_cache` were written and removed
because nothing reads them, and a descriptor with unread fields is how a
descriptor starts lying.

Resolution is three layers, cheapest override last (`engine.py:843-865`): the
`PROVIDERS` table's `Provider.profile` (`engine.py:430`), then anything assigned
to `.profile`, then `KNOSSOS_MAX_REQUEST_BYTES`.

`RequestTooLarge` (`engine.py:361`) is the enforcement. It **subclasses
`urllib.error.HTTPError` with code 413** so every existing `except HTTPError`,
retry guard and `_explain` path already knows what it means; `_post`
(`engine.py:867`) raises it before sending when the serialised body exceeds the
profile limit, and records `self._last_request_bytes` so the message names the
measured size.

### 1.2 Degrade-on-400

The engine never tabulates model ids. Capability is discovered by sending the
request and reading the refusal, one wasted request per engine, sticky for its
life. `OpenAICompatEngine._run` (`engine.py:1071`) has five ordered branches
inside one `except HTTPError`:

1. `_shrink_to_token_limit` (`engine.py:712`) — a provider token ceiling; shrinks
   `max_tokens`, retries.
2. `_NO_TOOL_MESSAGES` (`engine.py:987`) — flatten native tool messages to text
   (`engine.py:1119`). The marker list now includes `"thought_signature"` /
   `"thought signature"`: Gemini 3.x mints an opaque signature per function call
   and requires it echoed back, which the OpenAI-compat shape cannot carry, so a
   native conversation 400s from turn two onward. Measured at ~15 of ~30 engine
   calls in a five-case run.
3. `_NO_SAMPLING` (`engine.py:1027`) — drop `temperature`. Checked **before** the
   `stream_options` branch (`engine.py:1138`) because `"unknown field"` matches
   either and the narrower diagnosis must win.
4. `_NO_STREAM_OPTIONS` (`engine.py:1009`) — drop per-request usage counting.
5. `_NO_SYSTEM` (`engine.py:978`) — fold the system message into the first user
   message rather than dropping it.

Then a last-resort retry without `stream_options` for any unexplained 400
(`engine.py:1188`), on the grounds that `stream_options` is something *this
harness* added for a number nobody asked for.

`max_tokens` now defaults to **32 000** (`engine.py:562`), raised from 8192
because Sonnet 5 / Opus 5 think by default and were spending the whole allowance
before producing content.

Implementations in `engine.py`:

| Class | Line | Notes |
|---|---|---|
| `RetrievalOnlyEngine` | `engine.py:304` | No model. Formats Argus hits. Default. |
| `StaticEngine` | `engine.py:336` | Fixed script, for tests. |
| `OpenAICompatEngine` | `engine.py:524` | stdlib `urllib`; SSE streaming; 11 provider presets (`engine.py:436-491`); native `tool_calls` re-rendered as fenced blocks (`engine.py:1320`). |
| `TransformersEngine` | `engine.py:1478` | HuggingFace + optional PEFT adapter. **Never executed by any test.** |
| `ScriptedEngine` | `conformance/scripted_agent.py:29` | Conformance only, script from `$KNOSSOS_SCRIPT`. |
| `ScriptedEngine` | `scripts/coding_eval.py:186` | Calibration agents. Different class, same name. |

**Rust** — `knossos-rs/src/engine/mod.rs:27-40`. An `#[async_trait]` trait:

```rust
pub trait Engine: Send + Sync {
    async fn complete(&self, req: &Request) -> Result<Response>;
    fn name(&self) -> &str;
    fn supports_native_tools(&self) -> bool;
}
```

Richer than the Python contract: structured `Message`/`Content` history
(`engine/types.rs`), native `ToolUse`/`ToolResult` blocks, `StopReason`, `Usage`.
Not streaming. The free function `engine::complete` (`engine/mod.rs:47`) is the
only place the two tool-calling paths reconcile: without native tools it rewrites
the system prompt via `prompt_fallback::augment_system` and parses fenced JSON back
out. Implementations: `anthropic.rs`, `ollama.rs`, `mock.rs`. Instantiated only in
`Config::build_engine` (`config.rs`).

---

## 2. Component map

Greek names are opaque by design. Plain descriptions below.

### Python (`model/knossos/`)

| Component | What it actually does | File (lines) | Public entry points | Depends on |
|---|---|---|---|---|
| **Argus** | Repository index + BM25F retrieval. Scans `*.py *.rs *.toml *.md` (`argus.py:57`), extracts symbols (Python via `ast`, Rust via tree-sitter when installed else a line scanner, `argus.py:446`/`:509`), scores over three fields, returns line-ranged excerpts with a `reason`. Persists to `<root>/.argus/index.json`. | `argus.py` (901) | `Argus.scan/retrieve/context/lookup/save/load`, `render()` | stdlib; optional `tree_sitter`, `tree_sitter_rust` |
| **Gate** | Decides whether retrieved context is injected at all. Weighted signals (anchor phrase / filename / distinctive symbol / concentration, minus generality / indefinite framing). | `gate.py` (245) | `RetrievalGate.decide(query, hits)` | `Argus` |
| **Lethe** | Bounds the transcript. Over budget → summarise the middle band once, pinning `pin_opening=1` and `keep_recent=6` (`lethe.py:179`); if that is unavailable or would grow the text, `_fit` (`lethe.py:286`) elides the largest entry repeatedly (≤40 passes) until the budget is met. Both "nothing between head and tail" (`lethe.py:220`) and "summary made it bigger" (`lethe.py:236`) route to `_fitted`, so the bound applies on every path. | `lethe.py` (334) | `Lethe.compact()`, `extractive_summary()`, `estimate_tokens()` | stdlib only |
| **Metis** | Planner. One engine turn → ordered step list. Cascade: `submit_plan` tool call (`metis.py:206`) → truncated-JSON repair (`metis.py:222`) → prose bullets (`metis.py:270`) → task as its own step. Never returns nothing. `MAX_STEPS = 8`, enforced by `_tidy` (`metis.py:283`). | `metis.py` (304) | `Metis.plan()`, `worth_planning()` | `tools.parse_calls`, engine |
| **Talos** | The executor loop. One engine turn per step; parses tool calls; dispatches them; when the engine emits no call, asks the verifier. Owns the transcript, the change set, the permission callback, plan-step driving, re-planning and delegation. | `talos.py` (1502) | `Talos.run/resume/compact/apply/discard`; `Verdict`, `Verifier`, `Replanner`, `Delegate`, `Event`, `Outcome`, `accept_everything`, `CONSEQUENTIAL` | `ariadne`, `lethe`, `tools`, `workspace`, `engine.Usage` |
| **Ariadne** | Halting policy. Hard ceiling **`max_steps=20`** (`ariadne.py:120`, raised from 12 on measurement), pressure past `target_steps=6` in bands that are *fractions of the ceiling* (`ariadne.py:149`), `STUCK` after 2 consecutive unproductive steps — `is_noop` (called nothing, changed nothing) or `is_futile` (repeated a signature seen within the last `FUTILE_WINDOW`=4 steps and changed nothing — a window, not just the previous step, so an alternating loop is caught; restarted whenever a step changes a file). Pure data; no I/O. | `ariadne.py` (203) | `Ariadne.assess/pressure`, `Halt`, `StepOutcome` | stdlib only |
| **Oracle** | Tiered verifier with a language-adapter seam, per-tier scoping, a baseline, per-diagnostic forgiveness and a suite-integrity check. See §5. | `oracle.py` (891) | `Oracle.__call__/quick/prepare`; `OracleVerdict`, `Tier`, `LanguageAdapter`, `tiers_for`, `detect`, `PYTHON_TIERS`/`RUST_TIERS`/`GO_TIERS`/`NODE_TIERS` | `talos.Verdict`, `workspace` |
| **Workspace** | Path jail + write staging + undo journal + optional editor/LSP delegation. | `workspace.py` (409) | `resolve/read/write/edit/exists/apply/discard/staged/checkpoint/rewind/journal/original/display`; `PathEscape` | stdlib only |
| **tools** | The tool registry, the prompted-JSON call protocol, and every local tool. | `tools.py` (894) | `ToolRegistry.default/combined/without/dispatch/render/openai_schema/parallel_safe`, `parse_calls()`, `tokenize()` | `workspace` |
| **codeval** | The coding evaluation: 12 fixture repositories, SWE-bench-shaped grading, trace writing, external-suite loading. **New since the last version of this document.** | `codeval.py` (1251) | `CodingCase`, `CaseResult`, `CODING_CASES`, `load_cases`, `materialise`, `restore_tests`, `run_tests`, `grade`, `run_case`, `run_best_of`, `run_suite` | stdlib only (deliberately — see §8.4) |
| **acp** | The ACP server. Session lifecycle, retrieval narration, execute narration, permission prompts, mode switching, fork/list/close/delete/load. | `acp.py` (1386) | `DaedalusAgent.handle/serve`, `Session`, `main()` | everything |
| **jsonrpc** | Bidirectional JSON-RPC 2.0 over NDJSON stdio: reader thread + worker thread + a fast path. | `jsonrpc.py` (337) | `Peer.request/notify/start/serve_forever/close`, `RpcError`, `log()` | stdlib only |
| **lsp** | LSP client, **Content-Length framed** (deliberately not reusing `Peer`, `lsp.py:15-18`). Auto-selects a server by counting file extensions. | `lsp.py` (403) | `for_workspace()`, `LspClient.workspace_symbols/references/definition/stop` | `jsonrpc.log` only |
| **mcp** | MCP client: spawns declared servers, handshakes, lists tools, wraps each as a `Tool`. | `mcp.py` (253) | `connect_all()`, `McpClient.connect/call_tool/close`, `McpTool` | `jsonrpc.Peer`, `tools.Tool` |
| **evalset** | 28 labelled retrieval cases: 18 `repo_specific`, 6 `general`, 4 `negative`; 9 flagged `held_out=True`. | `evalset.py` (321) | `CASES`, `Case` | none |
| **eval** | Retrieval recall, gate accuracy, raw-vs-harness answer scoring; in-sample and held-out reported separately. | `eval.py` (461) | `main()`, `grade_answer`, `grade_retrieval`, `report_gate/retrieval/answers` | `argus`, `engine`, `evalset`, `gate` |
| **sandbox** *(new, 2026-07-29)* | Environment isolation for **every** subprocess the harness starts. `Sandbox.environ` scrubs the child env to an allowlist (`_BASE`/platform/toolchain/`_PREFIXES`) with a `_SECRET_MARKERS` deny check that wins first — so `pytest`/`cargo` cannot leak `ANTHROPIC_API_KEY` into stdout that folds back into the transcript. `offline=True` sets `CARGO_NET_OFFLINE`/`PIP_NO_INPUT`. `Sandbox.run` is the single subprocess entry: `shell=False`, `stdin=DEVNULL`, captured text, timeout. **Env isolation, not process isolation** (child can still write anywhere the account can, open sockets). | `sandbox.py` (176) | `Sandbox.admits/environ/withheld/run`, `DEFAULT` | stdlib. Used at `tools.py:462`, `oracle.py:562/609/671`, `codeval.py:1138` |
| **hooks** *(new, 2026-07-29)* | Per-repo/per-run policy running **inside `ToolRegistry.dispatch`** (`tools.py:844`), so every dispatch path gets it, not just Talos. `before` can deny or rewrite; `after` observes only (a rewritten *result* would lie to the engine). Default installed hook is `ProtectPaths([".git"])` — refuses `write_file`/`edit_file`/`rename_symbol` under protected path components, closing the hole where the jail admits `.git/config` while `run_command` restricts `git` to read-only. Hooks are carried to child agents (`tools.py:815`). | `hooks.py` (153) | `Decision`, `ALLOW`, `deny`, `rewrite`, `Hook`, `ProtectPaths`, `before_chain` | stdlib |
| **interject** *(new, 2026-07-29)* | A thread-safe queue for steering a running loop mid-task without cancelling it. Talos owns `self.interjections` (`talos.py:502`) and **drains it between steps** (`talos.py:1199`) — only at the step boundary, because mid-step the transcript can be an incomplete tool-call/result exchange a provider rejects. ACP exposes `session/interject` (`acp.py:736`) → `push`. `CAPACITY=64`. | `interject.py` (106) | `Interjections.push/drain/render/take_note`, `CAPACITY` | stdlib |

### Python scripts (`model/scripts/`)

*(2026-08-15: this table lists a subset; the tree also contains `evaluate.py`,
`fetch_evals.py`, `make_gec.py`, `prepare_corpus.py`, `trace_to_sft.py`, not
inspected this pass.)*

| Script | What it does | Lines |
|---|---|---|
| `coding_eval.py` | Drives `codeval`. Calibration agents, live engine, **external harnesses**, the report. | 688 |
| `trace_summary.py` | Aggregates `model/traces/*/`, reconstructing `unreachable` for traces written before the field existed. | 207 |
| `make_hard_suite.py` | Generates `model/fixtures/hard_suite.json` (6 cases) + a `--check` calibration that gates the write. Calibrates clean; unscored — see §8.7. | 414 |
| `seeds.py` | Paired multi-seed statistics: "no number without its `n`, and no `n` below 5". **Imported by neither eval.** | 121 |
| `profile_loop.py` | Profiles the executor's per-step overhead with a scripted engine, on a grown transcript. | 117 |
| `fetch_rust.py`, `naiads_eval.py`, `proteus_probe.py`, `moirai_sweep.py`, `echo_sweep.py` | Model-side (`model/daedalus/`), out of scope. | — |

### Rust (`knossos-rs/src/`)

| Component | What it actually does | File (lines) | Public entry points |
|---|---|---|---|
| **Scribe** | Exact symbol index from tree-sitter. Declarations rendered *verbatim* into the system prompt, never summarised. Refreshed per changed file mid-run (`talos.rs:401`). | `scribe/` (646) | `SymbolIndex::build/rebuild/refresh/refresh_from/lookup/render/adapter`; `LanguageAdapter` trait |
| **Themis** | The constitution. `constitution.md` from the workspace root, else the compiled-in default (`themis/mod.rs`, `include_str!`). Rendered into **every** role prompt. | `themis/mod.rs` (150) | `Themis::load/from_text/principles/source/system_prompt`; `PLANNER_ROLE`, `EXECUTOR_ROLE`, `JUDGE_ROLE` |
| **Mnemosyne** | BM25 over declaration-boundary chunks, capped at 120 lines. Deliberately lexical. | `mnemosyne.rs` (434) | `Mnemosyne::build/search/chunk_count` |
| **Lethe** | **New.** Bounds the message list *without removing or merging a message* — only the text inside `Text` and `ToolResult` blocks shrinks. `ToolUse.input` is never elided. Two phases: spare the recent tail, then include it if that was not enough. | `lethe.rs` (448) | `Lethe::compact(&mut [Message])`, `estimate_tokens()` |
| **Metis** | Planner: tool call → prose → task-as-one-step. No truncated-JSON repair. | `metis.rs` (214) | `metis::plan()` |
| **Talos** | Executor loop. Same structural rule as Python. Holds `lethe` (`talos.rs:92`) and an optional `approver` (`talos.rs:96`). | `talos.rs` (492) | `Talos::new/run/resume/diffs/apply/apply_hunks/discard`, `Approver` trait (`talos.rs:74`) |
| **Ariadne** | Same decision procedure as Python's. **Converged to `20 / 6 / 2`** (`ariadne.rs:129`, `config.rs`, `main.rs` clap default) — the "12/6/2" this row previously stated is obsolete; convergence was the point of this branch. Bands are fractions of the ceiling; has the ported repeat/futility window (§9.1). | `ariadne.rs` (364) | `Ariadne::new/assess/pressure` |
| **Oracle** | Tier 0 = tree-sitter parse; tiers 1..n from `LanguageAdapter::verify_commands()`; tier 4 = `oracle::judge`. Spawns with `kill_on_drop(true)` (`oracle/mod.rs:264`) under a 300 s timeout (`oracle/mod.rs:267`). | `oracle/` (860) | `Oracle::new/verify/verify_staged/root`, `judge()`, `Verdict`, `TierResult` |
| **tools** | `ToolCtx` (jail + staging + hunk apply) and the tool set. | `tools/` (1 414) | `ToolCtx::{resolve,read,write,diffs,apply_staged,apply_hunks,discard_staged,staged_contents}`, `ToolRegistry::{standard,with_retrieval,dispatch,defs,is_consequential}`, `Tool` trait |
| **diff** | Unified diff generation, per-hunk splitting, selective application. | `diff.rs` (373) | `diff_file()`, `apply_hunks()`, `render()` |
| **session** | Conversation state + JSONL trajectory trace. Trace failures never abort a run. | `session.rs` (207) | `Session::new/streaming/with_trace/log/push`, `TraceEvent` |
| **serve** | Long-lived NDJSON server. Router task owns the line stream; dispatch never sees a permission reply. | `serve.rs` (637) | `serve::run(talos, max_tokens, lines, emitter)`, `write_events`, `Command`, `Event`, `Emitter` |
| **repl** | Interactive terminal session with slash commands and a `PromptApprover` (`repl.rs:47`). | `repl.rs` (349) | `repl::run()` |
| **config** | Which engine, which workspace, what budgets. Canonicalises the root first. | `config.rs` (96) | `Config::build_engine/workspace_root` |
| **main** | clap CLI: `chat`, `index`, `verify`, `plan`, `task`, `repl`, `serve`, **and `acp`** (`main.rs:234`, `run_acp` at `:417`). | `main.rs` (703) | `main()`, `build_talos` (`main.rs:~227`), `run_acp` |

**Ported/new since the 07-29 pass (added by this 2026-08-15 pass — inspected at
module-header + wiring level, not line-by-line):**

| Component | What it does | File (lines) | Wired at |
|---|---|---|---|
| **acp** | Full ACP server for the Rust harness (`initialize`/`session/new`/`prompt`/`cancel`/`update`/`request_permission`). stdout is the protocol; progress via `Session::with_sink`; cancel on the reader fast path. | `acp.rs` (681) | `main.rs:417` `run_acp` |
| **argus** | Rust port of the repository index + retrieval, with save/load. | `argus.rs` (1467) | `main.rs:365` (build index, gate per turn) |
| **gate** | Rust port of the retrieval gate. | `gate.rs` (547) | `main.rs:364` |
| **lsp** | Rust LSP client. | `lsp.rs` (1030) | — |
| **mcp** | Rust MCP client. | `mcp.rs` (888) | — |
| **jsonrpc** | Rust JSON-RPC peer (reader/worker/fast-path), backing `acp`/`serve`. | `jsonrpc.rs` (871) | `acp.rs`, `serve.rs` |
| **delegate** | Rust delegation / subagent tool, depth-capped. | `delegate.rs` (596) | `main.rs:357` (behind `--delegate`) |
| **sandbox** | Mirror of Python `sandbox.py`: scrubbed subprocess env, secret deny-list, offline default. Lists deliberately differ (keeps a Rust toolchain working). | `sandbox.rs` (621) | shell/oracle subprocess sites |
| **hooks** | Mirror of Python `hooks.py`: before/after policy in dispatch, `ProtectPaths`. | `hooks.rs` (280) | tool dispatch |
| **interject** | Mirror of Python `interject.py`: between-step steering queue. | `interject.rs` (203) | talos loop |
| **resilience** | **Rust-only.** Retry-with-backoff + circuit breaker around `engine::complete`, so a transient 429 does not end a 20-step run. No Python equivalent (Python reconstructs an `unreachable` flag from traces instead). | `resilience.rs` (439) | `config.rs`, `engine/error.rs` |

No Python component is named Themis, Mnemosyne, or Scribe. See §9. **Note (2026-08-15):
§9.1 predates the Rust ports above and still marks Argus, Gate, LSP, MCP, ACP,
delegation, and `serve`/`repl` as "absent in Rust" — those rows are now stale; the
components exist and are wired. See the Drift note at the head of §11.**

---

## 3. The agent loop, traced end to end

### 3.1 Python, execute mode, via ACP

```
editor --stdio--> Peer._read_loop (jsonrpc.py:193)
                    │  fast-path method? → dispatch on the reader thread (jsonrpc.py:253)
                    └─ else → _inbox queue → Peer._work_loop (jsonrpc.py:288)
                                             └─ DaedalusAgent.handle (acp.py:410)
```

1. **`initialize`** (`acp.py:433`). Stores `clientCapabilities` verbatim. Advertises
   `loadSession: true`, `unstable_forkSession: true`, session list/close/delete,
   and `embeddedContext`. Clamps the protocol version to its own (`acp.py:439`).

2. **`session/new`** (`acp.py:467`). Validates `cwd`. Builds `Argus`, loads/scans/
   saves. Connects declared MCP servers. Creates a `Session` (`acp.py:119`).

3. **`session/prompt`** (`acp.py:723`). Clears cancel, flattens ACP content blocks
   to text (`acp.py:1266`), sets the session title from the first prompt.

4. **Context assembly** — `_run_retrieval` (`acp.py:1128`). Emits a visible
   `tool_call` update, rescans incrementally, retrieves, then asks the gate
   (`acp.py:1147`). **On a gate skip the function returns `[]` at `acp.py:1163`
   before any `locations` are emitted**, so the file locations are discarded too —
   contradicting the comment three lines above it (§11).

5. **Branch** (`acp.py:746`). `mode == "ask"` → stream `engine.generate` straight to
   `agent_message_chunk` / `agent_thought_chunk`. Otherwise `_run_execution`
   (`acp.py:1021`).

6. **Executor construction** — `_talos_for` (`acp.py:831`). Built once per session
   and reused, so a second prompt is a continuation. Wires:
   `Workspace(cwd, dry_run=mode=="preview", editor=…, terminal=…, elicit=…)`,
   `workspace.symbols = self._symbols(session)`, `verifier=Oracle(cwd)`,
   `interim=oracle.quick`, `constitution=self._constitution(session)`
   (`acp.py:860`, reads `constitution.md` from the workspace root),
   `tools=ToolRegistry.default()` merged with `self.mcp_tools(session)` via
   `ToolRegistry.combined` (`acp.py:850-853`), `ask_permission=…` and
   **`replan=…`** (`acp.py:869`) and **`delegation=self.delegation`**
   (`acp.py:874`, default `True`) — see §3.4.

7. **Planning** — `_plan` (`acp.py:980`), first prompt only, and only if
   `worth_planning(prompt)` (≥6 words, `metis.py:72`). Costs one engine turn. A
   degenerate plan is suppressed. A real plan is emitted as an ACP `plan` update.

8. **`Talos.run`** (`talos.py:471`). `len(plan) > 1` → `_drive_plan`
   (`talos.py:595`); otherwise a flat `_drive` (`talos.py:1072`).

9. **One step of `_drive`** (`talos.py:1123`), the core:

   ```
   cancelled?                          → Halt.STUCK, "cancelled by the caller" (talos.py:1125)
   ariadne.pressure(step)              → append "## Budget …"                  (talos.py:1130)
   _turn(): generate_messages(_history()) if present, else generate(_prompt())  (talos.py:1233)
       _history/_prompt both call _size_transcript_budget then compact          (talos.py:868/922)
   parse_calls(reply)                  → (prose, [ToolCall])                    (tools.py:859)
   append Entry(role="assistant", calls, text=prose, native=all ids present)     (talos.py:1143)
     ├─ calls present  → _run_calls → per call: _permitted? → tools.dispatch     (talos.py:1154)
     │                   acted += successful calls whose name ∈ CONSEQUENTIAL    (talos.py:1157)
     │                   repeat of last signature → REPEATED_CALL_NOTE           (talos.py:1171)
     ├─ reply empty    → append EMPTY_REPLY_NOTE, do NOT verify                  (talos.py:1181)
     └─ prose only     → verify(ws, changed); passing counts only if acted > 0   (talos.py:1185)
                         otherwise append NOTHING_DONE_NOTE                      (talos.py:1196)
   noops = 0 if outcome.made_progress else noops + 1                             (talos.py:1198)
   ariadne.assess(step, outcome, noops) → Halt                                   (talos.py:1200)
   ```

10. **Halting policy** — `Ariadne.assess` (`ariadne.py:134`), in this order:
    `verdict_passed is True` → `DONE`; `step >= max_steps` → `BUDGET_EXHAUSTED`;
    `consecutive_noops >= stuck_after` → `STUCK`; else `CONTINUE`.

    `verdict_passed = verdict.passed and (acted > 0 or not require_action)`
    (`talos.py:1190`). `require_action` is relaxed for exactly one caller: the
    closing phase of a plan that already accomplished something (`talos.py:729`).

11. **Response.** `_run_execution` emits a final summary chunk (`acp.py:1108`),
    then: `succeeded` → `end_turn`; cancelled → `cancelled`; otherwise the engine's
    mapped `finish_reason`, or `refusal` if that mapped to `end_turn`
    (`acp.py:1117-1120`).

### 3.2 Where plan-step execution, rollback and re-planning enter

`_drive_plan` (`talos.py:595`) is a `while` over a list that can be **replaced
underneath the walk**:

```
reserve   = max(3, max_steps // 4)                    held back for the closing phase (talos.py:624)
available = max_steps - reserve
per_step  = max(3, available // len(steps))                                          (talos.py:626)

for each step:
  append "## Step i of n … Do only this step."                                       (talos.py:640)
  mark = ws.checkpoint(f"step-{i}")                    journal position, not a copy  (talos.py:649)
  changed_before = len(self.changed)
  outcome = _drive(..., ariadne=per-step budget, verifier=self.interim)               (talos.py:656)
  acted = len(self.changed) > changed_before     sampled BEFORE any rollback          (talos.py:664)
  if failed:
      _revert_broken_step(mark, outcome)         only on a real failed verdict        (talos.py:670)
      _revise_plan(steps, index, replans, acted, emit)                                (talos.py:692)
      if revised: steps = revised; per_step redivided over the new tail               (talos.py:701)
append "## Plan complete …"                                                           (talos.py:717)
_drive(..., ariadne=closing reserve, require_action=not worked)                        (talos.py:728)
```

`_revise_plan` (`talos.py:533`) fails closed in every direction: no planner,
`acted` false, budget spent, nothing remaining, a planner that raises, or a
planner that answers with anything other than a list of non-empty strings — all
return `None` and leave the plan alone. A revision longer than
`len(remaining) * REPLAN_GROWTH_LIMIT` (=2, `talos.py:121`) is truncated. Only the
*tail* is rewritten; attempted steps are history. `max_replans` defaults to 1
(`talos.py:407`).

The **precondition is `acted`, not failure**: a step that wrote code and still
failed is evidence about the plan; a step that produced nothing is evidence about
the engine and says nothing about the plan.

`_revert_broken_step` (`talos.py:753`) undoes only on a real failed verdict, only
back to that step's journal mark, and only on disk (a dry run stages, so the
journal is empty and the rewind is a no-op). Paths the step *created* no longer
exist and are dropped from `self.changed` (`talos.py:786`).

The re-planner installed in production is `DaedalusAgent._replan` (`acp.py:861`),
which builds a brief describing the part-executed plan and calls a **fresh**
`Metis` — it holds no state between calls. Returns `[]` on every unhappy path.

### 3.3 Rust, via `serve` or `repl`

`main.rs:227` `build_talos` assembles engine, Scribe, Themis, Mnemosyne, `ToolCtx`,
`Oracle`, `Ariadne` and `Session`; then `serve::run` (`serve.rs:344`) or
`repl::run` routes a command. `Command::Task` → `metis::plan` → `Talos::run` →
`Talos::drive`:

```
ariadne.pressure(step)      → push a user message                        (talos.rs:257)
lethe.compact(&mut messages) → trace ContextCompacted if it acted         (talos.rs:270)
Request::new(themis.system_prompt(EXECUTOR_ROLE, Some(&scribe)), messages)
    .with_tools(tools.defs()).with_max_tokens(…)                         (talos.rs:277)
engine::complete(...)       → native or prompted-JSON, transparently     (talos.rs:283)
push resp.as_message()
  ├─ no calls & no text     → push EMPTY_REPLY_NOTE                      (talos.rs:303)
  ├─ no calls               → oracle.verify_staged (dry) or verify;
  │                           tier 4 iff deterministic_tiers_passed() && judge (talos.rs:327)
  │                           verdict_passed = verdict.passed && acted > 0     (talos.rs:352)
  └─ calls                  → permitted(name, input)? → tools.dispatch    (talos.rs:362)
                              acted += !is_error && is_consequential      (talos.rs:391)
                              refresh scribe from changed files           (talos.rs:401)
ariadne.assess(step, &outcome, noops)
```

### 3.4 Delegation — wired, on by default, unexercised in situ

`Delegate` (`talos.py:141`) and `Talos._spawn` (`talos.py:489`) exist and work: a
child `Talos` at `depth+1` with `max(3, max_steps // 2)` steps, sharing the
workspace *by reference*, sharing `ask_permission` *by reference*, and with
`delegation=False` plus `tools.without(DELEGATE)` so it cannot spawn further
(`MAX_DELEGATION_DEPTH = 1`, `talos.py:138`). Its changed files are folded into
the parent's change set and its usage into the parent's (`talos.py:523-526`). The
tool result is marked `is_error` when the child did not verify (`talos.py:227`).

`Talos.delegation` still defaults to `False` (`talos.py:408`) — the executor does
not hand itself a `delegate` tool unless a caller asks. What changed is that a
caller now asks: `DaedalusAgent.delegation` defaults to `True` (`acp.py:384`) and
`_talos_for` forwards it (`acp.py:874`), so the tool **is** in the registry in the
shipped ACP path. It is a flag rather than always-on for the same reason `planning`
is one: delegating spends engine turns, and a caller measuring the loop needs to be
able to turn it off.

Two honest limits on that:

- **Turning it off is a CLI flag**, `--no-delegation` (and `--no-planning` for the
  same reason). Both were previously reachable only by constructing
  `DaedalusAgent` in Python. `main()` now builds the agent through `build_agent`
  (`acp.py:1394`), which is separate from `main` precisely so a test can drive the
  flag end to end — `main` ends in `serve_forever()`, so nothing that only wants to
  know what a flag does can afford to call it.
- **Never exercised end to end.** Every test that drives delegation constructs
  `Talos(delegation=True)` directly (`test_talos.py:1636/1643/1648/1656`). No test
  drives a real engine into calling `delegate` through ACP (§8.10).

Until this pass the flag existed only in those four test call sites and `acp.py`
never mentioned it, so `delegate` was absent from the registry in every shipped
configuration while a working note described delegation as "wired into ACP". That
was the fifth instance in this repository of the built–tested–never-connected
pattern (§10, finding 19).

---

## 4. Trust and permission model

### 4.1 The single allow/deny function (Python)

`Talos._permitted` (`talos.py:1446`):

```python
if call.name not in CONSEQUENTIAL or self.ask_permission is None:
    return True
try:
    return bool(self.ask_permission(call))
except Exception as exc:
    log(f"[talos] permission check failed, refusing: {exc}")
    return False
```

Two consequences follow directly:

- **A tool not in `CONSEQUENTIAL` is never gated.** `CONSEQUENTIAL` is a hardcoded
  frozenset at `talos.py:348`: `{write_file, edit_file, run_command, rename_symbol}`.
  It is the drift-prone half of the design — a new writing tool omitted from the
  set is silently treated as a read, both for gating and for `acted`.
- **`ask_permission=None` means unattended: everything is allowed.** That is the
  `Talos` default (`talos.py:404`). Only the ACP layer sets it (`acp.py:854`,
  and `acp.py:616` on fork). A directly-constructed `Talos` — `model/tests/`,
  `scripts/coding_eval.py:369` (`live_agent`), any future non-ACP driver — has
  **no permission boundary at all**. The coding eval therefore runs ungated by
  design.

Permission is decided for a **whole turn, up front, in order**, before any dispatch
starts (`talos.py:1353-1359`); a refusal becomes a `ToolResult(..., is_error=True)`
so the engine gets another turn rather than the run collapsing.

The callback is `DaedalusAgent._ask_permission` (`acp.py:1207`). It fails closed on
every path: name in `session.always_allowed` → allow; no peer → refuse; cancel set
→ refuse; RPC exception → refuse; `outcome != "selected"` → refuse;
`optionId == "allow_always"` → remember and allow; `"allow_once"` → allow; anything
else → refuse. The wait is untimed but interruptible via `Peer.request(timeout=None,
abort=session.cancel)`.

Grants are scoped **per session and per tool name** (`acp.py:138`, `acp.py:1249`),
never per argument. Approving `run_command` as "always" approves every future
`run_command` in that session whatever it runs. `session_fork` deliberately does
not copy `always_allowed` (`acp.py:572`).

### 4.2 What a subagent inherits

Delegation is enabled in the ACP path (§3.4). A child inherits, by reference:

| Inherited | Line | Why |
|---|---|---|
| The `Workspace` | `talos.py:504` | Edits are ordinary edits: visible to the parent's verifier, revertible by the parent's checkpoints, counted in the parent's change set |
| `ask_permission` | `talos.py:515` | A subagent must not be a route around a gate the user is watching |
| `verifier`, `interim`, `constitution` | `talos.py:511-516` | Same standard of proof |
| **Not** the transcript | — | That is the entire point: the child's reading is discarded, the parent pays for a paragraph |
| **Not** `delegate` | `talos.py:508` | Removed by construction, not by asking the model not to use it |

### 4.3 The Rust permission gate

`Talos::permitted` (`talos.rs:223`) consults an optional `Approver`
(`talos.rs:74`) before any call that `ToolRegistry::is_consequential` identifies.
That discriminator is `Tool::consequential()`, **a trait method with no default**
(`tools/mod.rs`), so a tool added later cannot compile without being classified —
structurally stronger than Python's name set. `approver: None` means unattended,
matching Python.

| Front end | Gated | Where | Why |
|---|---|---|---|
| `daedalus task` | **no** | `main.rs:278` `run_task` — no approver is ever assigned | Non-interactive by design; a prompt hangs CI |
| `repl` | yes | `repl.rs:92` (`PromptApprover`) | Reads a command then runs it, so stdin is idle; EOF and any non-`y` deny |
| `serve` | **opt-in** | `serve.rs:352` (`FrontEndApprover`), armed by `Command::Capabilities{permissions}` at `serve.rs:282-286` | An unannounced `permission_request` would be dropped by existing front ends and the agent would wait forever — turning the gate into a hang |

`serve` needed a reader/dispatch split for this to be possible: a router task owns
the line stream (`serve.rs:371`), intercepts `Command::Permission` and completes the
waiting `oneshot` directly, and forwards everything else. `dispatch` treats reaching
a permission reply as `unreachable!` (`serve.rs:511`). `deny_outstanding`
(`serve.rs:327`) refuses every outstanding request when the front end disconnects —
silence is not consent, and without it the `oneshot::Sender` outlives the router
(`pending` is an `Arc` the approver clones) and `talos.run` never returns.

`serve::run` takes channels rather than real streams (`main.rs:330-370`), which is
what makes `knossos-rs/tests/serve_loop.rs` able to drive the whole protocol over
in-memory channels under a timeout.

### 4.4 What the path jail does and does not stop

`Workspace.resolve` (`workspace.py:217`) / `ToolCtx::resolve` (`tools/mod.rs`):
lexical `..`/`.` normalisation that cannot consult the filesystem (so it works for
files about to be created), an `_is_within` check, then a real `resolve()` of the
nearest ancestor satisfying **`is_symlink() or exists()`** (`workspace.py:248`).

The `is_symlink()` half is the closed hole. `exists()` follows the link and returns
False for a *broken* one, so a symlink naming a not-yet-existing path outside the
tree used to slip through: the probe walked past it to a parent that resolves
cleanly inside the root, the check passed, and `write_text` then followed the link.
A symlinked directory was worse, because `mkdir(parents=True)` would build the whole
outside tree. `is_symlink` uses `lstat`, so it sees the link rather than its target.

What the jail does **not** stop:

1. **Child processes.** `RunCommand` sets `cwd=ws.root` and hands `argv` to the OS
   (`tools.py:456`); the child has no jail. `python -c "open('/etc/passwd','w')"`
   is on the allowlist.
2. **Oracle's own subprocesses.** `Oracle._run_tier` (`oracle.py:560`) and
   `_tier_output` (`oracle.py:608`) run with `cwd=self.root`, outside `parse_calls`,
   outside the `run_command` allowlist, and outside `ask_permission`. Tier 3 is
   `pytest -q` rooted at the workspace, so any `conftest.py` executes during
   collection; Rust `cargo check` runs `build.rs` and proc macros and is reachable
   from `daedalus verify` with no agent in the loop.
   `Tier.safe_path` (`oracle.py:203`) inserts `-P` so the workspace stays off
   `sys.path[0]` — but it is `False` for pytest (`oracle.py:251`), because pytest's
   standard layout imports the package via the cwd entry `-P` removes.
3. **Rust `apply_hunks`.** `ToolCtx::apply_hunks` (`tools/mod.rs:133`) joins
   `selection[i].path` to the root **without calling `resolve`** and writes at
   `tools/mod.rs:151`. The path arrives from the `serve` peer. It is only written if
   the same absolute key is already in `staged`, so a fabricated path is a no-op
   today — but that is an incidental consequence of the `staged.get` lookup, not a
   check.
4. **Anything else in-process.** There is **no OS-level sandbox** in either
   implementation: no container, namespace, seccomp filter, user drop, resource
   limit or network restriction. The boundary is the jail, the absence of a shell
   (commands are tokenized in-process at `tools.py:713` / `shell.rs:186` and handed
   over as `argv` with `shell=False`), and the program allowlist.

### 4.5 The constitution

**Python: now real, and sourced from inside the jail.** `DaedalusAgent._constitution`
(`acp.py:915`) reads `constitution.md` from the workspace root and `_talos_for`
forwards it to `Talos` (`acp.py:860`), so the `if self.constitution:` branch at
`talos.py:858` is reachable. Deliberately **no** compiled-in default: an absent file
means the user has no standing instructions, and inventing some is worse than having
none. Note the trust direction — the file lives in the untrusted workspace.

**One resolution for every role.** `_constitution` is the single resolver: an
explicit `constitution=` passed to the constructor wins — a caller who supplied one
meant it — and otherwise the workspace file is read. `Talos`, `Metis` at `_plan`
and `Metis` at `_replan` all call it, so planner and executor are held to the same
standing instructions. It resolves once per session and caches on `Session`
(`acp.py:149`), so a mid-run rewrite of `constitution.md` cannot split the two
roles apart, and the load is logged once rather than three times a turn.

Both halves of that were broken until this pass, in opposite directions. `Metis`
got `self.constitution` — the raw `DaedalusAgent` field, which no CLI flag sets and
`main()` never passed — so **the planner planned with an empty constitution while
the executor was held to the workspace's**, which is how you get a plan the executor
is forbidden to carry out. And the workspace file was read only for `Talos`, so a
caller who *did* pass `constitution=` programmatically had it silently ignored by
the executor.

**Rust: applied everywhere, and also sourced from inside the jail.** `Themis`
reads `constitution.md` from the workspace root, else the compiled-in
`knossos-rs/constitution.md`; `system_prompt` prepends it to *every* role prompt,
and `oracle::judge` uses `JUDGE_ROLE` plus the same principles as the tier-4 rubric.
So a repository ships its own standing instructions to the planner and executor
**and** the rubric its own change is graded against. Two amplifiers: the judge
prompt embeds changed-file contents in an unescaped fence, and a missing
`submit_verdict` makes the tier pass with a note.

---

## 5. Tool surface

### 5.1 Python — `ToolRegistry.default()` (`tools.py:764`)

| Tool | Line | `parallel_safe` | Filesystem | Network | Subprocess | Gated |
|---|---|---|---|---|---|---|
| `read_file` | `tools.py:171` | yes (`:174`) | read, jailed; staging first, then editor, then disk | no | no | **no** |
| `write_file` | `tools.py:203` | no | write, jailed; staged in dry-run; journals prior content | no | no | **yes** |
| `edit_file` | `tools.py:222` | no | read+write, jailed; `old_string` must appear exactly once | no | no | **yes** |
| `list_dir` | `tools.py:245` | yes (`:246`) | `iterdir` on a jailed path | no | no | **no** |
| `search` | `tools.py:285` | yes (`:300`) | pruned walk from a jailed root; reads **through the workspace**, so staged edits are searchable | no | no | **no** |
| `run_command` | `tools.py:399` | no | cwd = root; the child is **not** jailed | via the child | yes, or delegated to the editor terminal (`tools.py:474`) | **yes** |
| `rename_symbol` | `tools.py:494` | no | multi-file write via `ws.write`, each jailed | via the LSP subprocess | indirectly | **yes** |
| `ask_user` | `tools.py:667` | no — deliberately | none | no | no | **no** |
| MCP tools | `mcp.py:83` | no (class default, `tools.py:131`) | whatever the remote server does; `McpTool.run` notes `ws` is unused | remote | remote | **no** |

`ToolRegistry.combined` (`tools.py:770`) applies **local tools last**, so a remote
server cannot shadow `write_file` and take the jail with it; shadowed names are
logged. `ToolRegistry.without` (`tools.py:790`) removes names — used only to strip
`delegate` from a child.

`run_command` allowlist: `{python, pytest, ruff, mypy, git}` (`tools.py:393`), `git`
limited to `{status, diff, log, show, ls-files, blame, rev-parse, branch}`
(`tools.py:396`). `python` is rewritten to `sys.executable` (`tools.py:443`).
Timeout 300 s (`tools.py:61`), output capped at 20 000 chars (`tools.py:58`),
`stdin=DEVNULL`, `shell=False`. The allowlist runs **before** the editor-terminal
delegation (`tools.py:450`), so a delegated command is still allowlisted. Note that
`python <anything>` is allowlisted: the list constrains the *program*, not what it
does.

### 5.2 Rust — `ToolRegistry::standard()` / `::with_retrieval()` (`tools/mod.rs:306`, `:322`)

| Tool | Line | Filesystem | Network | Subprocess | Consequential | Gated |
|---|---|---|---|---|---|---|
| `read_file` | `fs.rs` | read, jailed, staging-first | no | no | no | **no** |
| `write_file` | `fs.rs` | write, jailed, staged in dry-run | no | no | yes | per front end (§4.3) |
| `edit_file` | `fs.rs` | read+write, jailed | no | no | yes | per front end |
| `list_dir` | `fs.rs` | jailed | no | no | no | **no** |
| `search` | `search.rs:18` | regex walk honouring `.gitignore` | no | no | no | **no** |
| `search_code` | `search.rs:118` | in-memory Mnemosyne index only | no | no | no | **no** |
| `run` | `shell.rs:34` | cwd = root; child unconstrained | via the child | yes | yes | per front end |

`run` allowlist `{cargo, rustc, rustfmt, git}` (`shell.rs:30`), git read-only, cargo
deny-list. **`run` now has a 300 s timeout** (`shell.rs:27`, applied at
`shell.rs:104`) matching the Oracle's — the previous "no timeout" divergence is
closed. Output capped at 30 000 bytes (`shell.rs:22`).

### 5.3 The Oracle ladder

**Tier 0** is in-process `compile()` over changed `.py` files, read *through* the
workspace so staged content counts (`oracle.py:498`). `Oracle.quick`
(`oracle.py:437`) is tier 0 alone, used between plan steps.

**Tiers 1..n come from a `LanguageAdapter`.** `LanguageAdapter` (`oracle.py:289`) is
a name, a tuple of marker files, and a tuple of `Tier`s. `ADAPTERS`
(`oracle.py:312`) covers python (`pyproject.toml`, `setup.py`, `setup.cfg`,
`requirements.txt`, `tox.ini`), rust (`Cargo.toml`), go (`go.mod`), node
(`package.json`). `detect` (`oracle.py:323`) returns **every** language present;
`tiers_for` (`oracle.py:328`) concatenates their ladders, sorts **cheapest-first
across languages** (all compilers, then all linters, then all test suites),
renumbers densely because the baseline is keyed on tier number, and prefixes labels
with the language when the repo is polyglot. No marker at all → `PYTHON_TIERS`,
so nothing verifies vacuously.

| Ladder | Tiers | Notes |
|---|---|---|
| `PYTHON_TIERS` (`oracle.py:235`) | ruff (`--output-format=concise`, scoped `.py`), mypy (non-blocking, scoped `.py`), pytest `-q` (`safe_path=False`) | The concise format is pinned because per-diagnostic baselining cannot parse ruff's default block form |
| `RUST_TIERS` (`oracle.py:258`) | `cargo check --quiet`, `cargo clippy --quiet` (non-blocking), `cargo test --quiet` | Nothing scopable: cargo's unit of work is the crate |
| `GO_TIERS` (`oracle.py:267`) | `go build ./...`, `go vet ./...` (non-blocking), `go test ./...` | |
| `NODE_TIERS` (`oracle.py:280`) | `node_modules/.bin/tsc --noEmit`, `node_modules/.bin/eslint .` (non-blocking, scoped), `npm test --silent` | **Never `npx`** — npx downloads and executes a package that is not installed, turning verification into arbitrary code execution sourced from a config file |

**Missing tools are skipped, not failed** (`oracle.py:531-536`).

**Scoping.** `Tier.scopes` (`oracle.py:200`) marks a tier as accepting path
arguments; `Tier.targets` (`oracle.py:205`) maps the changed set onto them,
dropping anything outside the root so an absolute path cannot widen the scope back
out. A scopable tier whose kinds did not change is **skipped**, not run over
everything (`oracle.py:554`).

**Baseline.** `Oracle.prepare` (`oracle.py:399`) records, before the agent changes
anything: the pytest collect count, the per-file test-function counts, whole-tree
diagnostics for every scopable tier, and pass/fail for every unscopable tier. It is
idempotent and called from `Talos._prepare_verifier` (`talos.py:791`) immediately
before the **first consequential tool call** (`talos.py:1379`) — the last moment the
tree is pristine and the first moment a baseline is worth paying for. It is
deliberately absent from the `Verifier` protocol so a bare function stays a valid
verifier; Talos probes for it and logs-and-continues if it raises.

**A forgiven tier fails the verdict.** `_forgive_baseline` (`oracle.py:812`) marks a
tier that was already failing as `passed=False, forgiven=True`, and `_verdict`
(`oracle.py:868`) summarises it `could not verify: <tier> was already failing before
this change` — deliberately distinct from `FAILED at <tier>`. Both block completion;
only one assigns blame. An earlier version marked such a tier as passed-and-skipped,
which licensed a lie: on `a-second-defect-behind-the-first`, whose entire content is
*fix this failing test*, pytest was red at baseline, was forgiven, and the run halted
`DONE` reporting three tiers having fixed one of two defects. The Oracle never sees
the task text, so it cannot tell "already failing" from "the check that defines the
task" and must not claim either.

**Per-diagnostic forgiveness.** `_forgive_known_diagnostics` (`oracle.py:618`)
subtracts the baseline's `Counter` of `(file, message)` pairs from the current run's.
Position is deliberately discarded from the key (`oracle.py:158`, `_DIAGNOSTIC` at
`:153`) because editing a file shifts every diagnostic below the edit. Forgiveness
is capped at the baseline count, a tier that failed without emitting a parseable
diagnostic is never forgiven, and a file absent at baseline is never forgiven.

**Suite integrity.** `_suite_integrity` (`oracle.py:695`) fails the verdict when
tests have *disappeared*. Two signals: a textual per-file `def test_` count
(`oracle.py:766`, runs unconditionally, ~36 ms) and a `pytest --collect-only` count
(`oracle.py:661`, ~3.4 s, **gated on the change set touching a test file**,
`oracle.py:741`). The textual half exists because the count alone missed a feature
task where the test file referenced a not-yet-existing function, collected zero
tests at baseline, and could be replaced by one trivial test as apparent *growth*.

**Tier 4** (`oracle.py:487`) runs only if `self.judge is not None and
verdict.deterministic_passed`. **Nothing installs a judge in the Python
configuration** — `acp.py:830` and `scripts/coding_eval.py:368` both construct a bare
`Oracle(root)` — so tier 4 never runs and `deterministic_passed` gates nothing.

---

## 6. Protocol surfaces

| Surface | File | Direction | Framing | What is trusted from the peer |
|---|---|---|---|---|
| **ACP server** | `acp.py:377`, transport `jsonrpc.py` | editor → agent (requests); agent → editor (`session/update`, `session/request_permission`, `fs/*`, `terminal/*`, `elicitation/create`) | NDJSON, one JSON value per line; stdout is protocol-only | `cwd` (validated absolute + directory), `clientCapabilities` (stored raw, `acp.py:434`), `mcpServers` (**spawns subprocesses**), prompt content blocks, permission `optionId`, `fs/read_text_file` content (returned verbatim as file content), `terminal/*` output, elicitation answers |
| **JSON-RPC transport** | `jsonrpc.py` | bidirectional | one JSON value per line; a framing violation trips an `assert` on send (`jsonrpc.py:118`) | Unparseable lines dropped with a log line; non-dict messages ignored; **no size limit on a line** (`jsonrpc.py:193`) |
| **LSP client** | `lsp.py:161` | agent → language server | **`Content-Length` headers** (`lsp.py:338`), not NDJSON | `workspace/symbol`, `textDocument/references`, `textDocument/definition`. Positions are *verified against file content* before any edit (`tools.py:534` `_locate`), the one place peer data is checked before being acted on. Server→client requests are ignored. |
| **MCP client** | `mcp.py:102` | agent → MCP server | NDJSON via `jsonrpc.Peer` | `tools/list` names, descriptions and `inputSchema` — all rendered into the prompt unvalidated. `tools/call` content flattened by `_render`. Server→client requests answered method-not-found rather than ignored, to avoid blocking the server. |
| **Rust `serve`** | `serve.rs:344` | front end → harness | NDJSON, `{"cmd": …}` in, `{"event": …}` out; `Idle` terminates every command (`serve.rs:368/378/392`) | Commands are `serde` enums, so an unknown command is a parse error rather than a dispatch. `Capabilities{permissions}` arms the gate (`serve.rs:282`). `ApplyHunks` selection paths bypass `resolve` (§4.4). |
| **Provider HTTP** | `engine.py:867` | agent → provider | HTTPS, OpenAI `/chat/completions` SSE | Streamed `delta.content`, `delta.reasoning`, `delta.tool_calls`, `finish_reason`. Malformed SSE JSON skipped. Native tool calls with unparseable arguments are dropped, not guessed. |
| **External harness** | `scripts/coding_eval.py:264` | eval → third-party CLI | argv (`shlex`, never a shell, `:352`) | Exit status as the harness's own success claim; stdout scanned for a turn count (`:329`). File changes are **measured** by before/after SHA-1 snapshot (`:235`), never taken on the harness's word. |

`DaedalusAgent.FAST_PATH` (`acp.py:1257`) contains only `session/cancel`; it runs on
the reader thread and must not block. Everything else is serialised through one
worker thread, so **two sessions cannot prompt concurrently**.

---

## 7. State and persistence

| State | Where it lives | Lifetime | What evicts it |
|---|---|---|---|
| Argus index | `<workspace>/.argus/index.json` | Across processes | Version mismatch forces a rebuild; `scan()` re-parses on SHA mismatch and drops vanished files. `.argus/` is gitignored. |
| Gate name index | `RetrievalGate._index` (`gate.py:119`) | Per instance | **Never** — built once on first `decide`, never invalidated after a rescan |
| ACP sessions | `DaedalusAgent.sessions` | Process lifetime | `session/delete` only (`acp.py:673`). `session/close` (`acp.py:657`) releases subprocesses but **keeps the entry**. Nothing ages sessions out. |
| ACP replay history | `Session.history` (`acp.py:132`) | Session lifetime | Nothing. **Per-entry strings are now bounded** to `HISTORY_MAX_STRING = 2000` by `_bounded` (`acp.py:94`/`:97`, applied at `acp.py:812`) — the *count* is still unbounded. Replayed verbatim by `session/load` with `record=False`. |
| Talos transcript | `Talos.transcript` (`talos.py:457`) | Session lifetime; survives `run`/`resume` | `Lethe.compact` via `_prompt`/`_history` on every step. Budget is **derived per turn** from `engine.context_window` by `_size_transcript_budget` (`talos.py:1021`): `window − OUTPUT_RESERVE(4096) − tokens(preamble)`, × `SAFETY_MARGIN(0.9)`, floored at `MIN_TRANSCRIPT_BUDGET(1000)` with a one-time warning, and never raised above Lethe's configured 24 000. |
| Staged writes | `Workspace._staged` (`workspace.py:123`) | Until `apply()`/`discard()` | Never persisted. A dry-run session's staged edits are lost if the process dies. `session/set_mode` to `write` deliberately does not apply them; `session/fork` deliberately does not copy them. |
| Undo journal | `Workspace._journal` + `_marks` (`workspace.py:145-147`) | Process lifetime, in memory | `rewind(label)` truncates it (`workspace.py:188`) and invalidates later marks (`workspace.py:190`). Reached from `Talos._drive_plan` per plan step only — there is no ACP method and no per-turn checkpoint. Holds a full pre-image of every write. |
| Permission grants | `Session.always_allowed` (`acp.py:138`) | Session lifetime | Nothing; not persisted, not copied on fork |
| LSP / MCP subprocesses | `Session.lsp`, `Session.mcp` | Session lifetime | `_release` (`acp.py:685`) — calls `client.close()`, the method `McpClient` actually defines (`mcp.py:152`) |
| Coding-eval traces | `<trace-dir>/<case>.jsonl` (`codeval.py:1225`) | Forever | Nothing. Header carries `passed`, `halt`, `harness_said_done`, `fixed`, `kept`, `tamper`, `error`, `api_errors`, **`unreachable`**. |
| Rust trace log | `<root>/.daedalus/trace-<pid>.jsonl` (`main.rs:238`) | Forever, append-only | Nothing |
| Rust staged writes | `ToolCtx.staged` (`Arc<Mutex<BTreeMap<…>>>`) | Process lifetime | `apply_staged`, `apply_hunks` (partial), `discard_staged` |
| Rust conversation | `Talos.messages` | Process lifetime | `/reset` or `Command::Reset`; and **`Lethe::compact` per step** (`talos.rs:270`) — the Rust side is now bounded |
| Rust Scribe / Mnemosyne | In-memory | Process lifetime | Scribe refreshed per changed file (`talos.rs:401`); **Mnemosyne built once in `build_talos` (`main.rs:235`) and never rebuilt**, so `search_code` goes stale after the first write |

---

## 8. Evaluation architecture

There are **two evaluations**, measuring different things, sharing nothing.

### 8.1 Retrieval QA — `evalset.py` + `eval.py`

28 labelled natural-language questions about this repository. It never constructs a
`Talos`, a `Workspace` or an `Oracle`, and no case requires editing a file.

- 18 `repo_specific`, 6 `general` (the **control group**: retrieval must *not* help,
  and a lift here invalidates the `repo_specific` number), 4 `negative` (features
  absent from the repo; passing means saying so).
- 9 are `held_out=True`, written after the ranker was tuned and targeting
  `knossos/`. `report_retrieval` (`eval.py:281`) scores in-sample and held-out
  separately and refuses a headline figure when no held-out case ran.

Three mechanical graders: `grade_retrieval` (`eval.py:88`) — did an expected file
appear, at what rank; `grade_answer` (`eval.py:100`) — `0.5 × term-group coverage +
0.5 × citation`, zeroed by a known-wrong citation, with negative cases scoring 1.0
iff any of 22 `ABSENCE_MARKERS` (`eval.py:49`) appears; `report_gate` (`eval.py:144`)
— ground truth is "inject unless `kind == "general"`", iterated over all 28 cases.
Provider errors are matched by substring (`eval.py:58`) and excluded from every
score.

### 8.2 The loop — `codeval.py` + `scripts/coding_eval.py`

12 fixture repositories (`CODING_CASES`, `codeval.py:346` and `:584`): **5 core**
(`chunk-drops-remainder`, `mean-of-empty`, `add-a-function`,
`wrong-exception-across-files`, `fix-without-breaking`) and **7 hard**
(`fix-belongs-in-the-base-class`, `bug-two-files-from-the-test`,
`a-second-defect-behind-the-first`, `every-call-site-must-change`,
`one-wrong-handler-among-many`, `the-obvious-fix-overshoots`,
`the-defect-is-not-where-it-fails`). Each case carries its **entire starting
repository** as a path→contents map.

### 8.3 How a case is graded

`run_case` (`codeval.py:1043`):

```
materialise(case, root)                 write the fixture repository       (codeval.py:964)
agent = make_agent(root); agent.run(prompt, on_event=recorder)             (codeval.py:1054)
    → halt, harness_said_done, steps_used, changed, usage — all via getattr
api_errors = count of `text` events containing "Request failed:"           (codeval.py:1078)
restore_tests(case, root)  → tamper                                        (codeval.py:974)
grade(case, root, tampered)                                                (codeval.py:1019)
    fixed = run_tests(root, fail_to_pass)     one pytest process; per node only if red
    kept  = run_tests(root, pass_to_pass)
    reveal_held_out(case, root)               tests the agent never saw
    held  = run_tests(root, held_out_pass)
    passed = all fixed AND all kept AND all held
_write_trace(...)                                                          (codeval.py:1225)
```

`passed` requires **all three** sets, not any: a change that fixes the bug and
breaks the suite is a different failure, not a smaller success — and one that
passes everything it was shown while failing what it was not has solved the
assertions rather than the task.

`reveal_held_out` runs *after* the agent and deliberately not inside
`materialise`. The entire value of a held-out test is that it was never on disk
while the agent worked: unreadable, uneditable, unfittable. `load_cases` refuses
a `held_out` path that collides with a visible one, because such a file would be
shown to the agent and then silently overwritten before grading — held out in
name only.

### 8.4 Anti-gaming properties

| Property | Where | What it stops |
|---|---|---|
| **Test files are restored before grading** | `codeval.py:974` | Deleting, weakening or duplicating the failing test gains exactly nothing. The attempt is recorded as `tamper` but changes no outcome. |
| **Node ids, not exit codes** | `codeval.py:1009` | A deleted or renamed test fails to collect, which is a failure — not a silent pass. |
| **Per-node isolation on failure** | `codeval.py:run_tests` | A collection error in one file cannot take a whole batch down and be misread as "this test failed". The batch runs first and isolation is paid only when it is not green — there is nothing to attribute in a pass. |
| **Three calibration agents** | `scripts/coding_eval.py:202` | `oracle` applies the known-good patch and must score 12/12 or the *grader* is wrong (`:676`); `lazy` reads a file and claims completion, must score 0; `vandal` neuters the failing test, must score 0 (`:680`). About a second each, no model. |
| **Selection never reads the grader** | `codeval.py:1112` | `--best-of K` keeps the first attempt the *harness* accepted (`harness_said_done`), never the graded outcome — otherwise the number describes an oracle that does not exist at inference time. Every sample is charged (`:1148`), and `solved@1` is reported separately (`:477`). |
| **A fresh tree per attempt** | `codeval.py:1133` | Sampling into a directory a previous attempt edited would measure a sequence of repairs, not k independent samples. |
| **Usage is plain ints, not an engine type** | `codeval.py:214-227` | Keeps `AgentFactory` "anything with `.run(prompt)`", which is what makes §8.5 possible at all. |
| **Held-out tests** | `codeval.py` `reveal_held_out` | The channel `restore_tests` cannot close. Restoring the tests stops an agent that *edits* the assertions; it does nothing about one that writes code shaped to the assertions it was shown, which passes honestly by every other measure here. Held-out tests are written after the agent finishes, so there is nothing to fit. A run that passes everything visible and fails one of these is reported `OVERFIT` and does not count as solved. |
| **The claim's source is recorded** | `codeval.py` `claim_source` | Stops three different quantities being averaged into one `honest`: Oracle's verdict (`verifier`), a rubber-stamped run (`unverified`, the control arm), and a foreign CLI's exit status (`exit_code`). The last cannot score a false fail at all, so printing it unqualified beside the first invites a comparison it cannot support. |

`_suite_integrity` in the Oracle (§5.3) exists because of this eval: the vandal
originally scored 0 solved but **5/5 false passes**, because every deterministic tier
passed honestly over a suite that no longer contained anything that could fail. That
hole was invisible to review and to a passing test suite, and obvious to five
fixtures.

### 8.5 `unreachable`, external suites, external harnesses

**`CaseResult.unreachable`** (`codeval.py:253`) separates "the provider was never
reached" from a capability zero:

```python
never_ran = bool(self.error) and not self.steps_used
return (bool(self.api_errors) or never_ran) and not self.passed
```

Two routes: the provider refused (counted from the engine's failure text in the
event stream), or the request was never made at all (a missing key, an unreachable
host — the engine raises before the first turn, so `api_errors` stays zero and
`steps_used == 0` is what tells them apart). `api_errors and not passed` rather than
`api_errors` alone, so a run that hit one rate limit, retried, and solved the task
still counts.

`report` (`scripts/coding_eval.py:409`) excludes unreachable cases from **every**
rate, including `honest` — an unattempted case is trivially honest and leaving it in
inflates the one number the suite exists to report. `run_suite` abandons the run
after `STARVED_CASES_BEFORE_ABORT = 2` consecutive refusals (`codeval.py:1186`).
`report_throttling` (`scripts/coding_eval.py:377`) additionally warns about requests
that *succeeded* after long backoff with a shrunken output allowance — measured at
146 of 224 seconds asleep on Groq's free tier, with `max_tokens` cut to 5 563.

**External suites** — `load_cases` (`codeval.py:121`) reads a JSON list (or
`{"cases": [...]}`) whose schema is exactly `CodingCase`. It **raises** rather than
returning a partial suite, and rejects duplicate ids. Everything that makes the
grader trustworthy applies unchanged, because it produces the same objects.
`--cases FILE` (`scripts/coding_eval.py:571`) selects it, prints a warning that the
two scores are not comparable, and `--agent oracle` refuses to report a grader
failure against an external suite because its solutions are keyed to built-in ids
(`:671`).

**External harnesses** — `ExternalHarness` (`scripts/coding_eval.py:264`) drives a
third-party coding agent as a subprocess via `--harness-cmd 'claude -p {prompt} …'`.
Three fields degrade rather than lie: `changed` is **measured** by before/after
SHA-1 snapshot (`:235`) and cannot be inflated by a harness that merely claims to
have edited something; `succeeded` is the harness's own claim, which for a generic
CLI is only its exit status — and that is the point, because `honest` then measures
whether that self-report matched the graded truth; `steps_used` stays 0 unless the
harness prints JSON carrying a turn count (`:329`). Token usage is left unset so the
report says "not reported" instead of printing a zero that looks like free.
A timeout is `budget_exhausted`, and whatever was written before the deadline is
still graded (`:316-320`).

### 8.6 What the numbers currently say

Measured 2026-07-28, traces in `model/traces/`, aggregated with
`python -m scripts.trace_summary --dir traces`:

| Run | trace dir | solved | honest |
|---|---|---|---|
| Knossos + Gemini 3.1 Pro | `gem31pro-full` | 12/12 | 12/12 |
| Claude Code (Sonnet 5), via `--harness-cmd` | `claudecode-sonnet5` | 12/12 | 12/12 |
| Knossos + Gemini 3.1 Flash-Lite | `gem31flashlite` | 12/12 | **10/12** |

**The built-in suite is saturated.** A lite model aces it, so `solved` cannot rank a
harness or a frontier model — it measures a floor. `honest` is the only column with
signal left. Flash-Lite's two losses are both in the *safe* direction (false fails):
`add-a-function` and `bug-two-files-from-the-test` were graded `passed=True` with
`harness_said_done=False` and `halt=budget_exhausted` — it fixed the code and then
ran out of steps without recognising it was finished. Verified by reading the trace
headers directly.

Zero test-file tampering attempts across all 32 trace directories.

Eight trace directories measured nothing at all (`gem-*`, `groq-*`, `groq2-*`
subsets) — every case unreachable. `trace_summary.py` reconstructs `unreachable` for
traces predating the recorded field and marks reconstructed rows with `*`, because a
derived field and a recorded one do not deserve the same confidence.

### 8.7 `make_hard_suite.py` — calibrated, unscored

`model/scripts/make_hard_suite.py` writes six harder cases to
`model/fixtures/hard_suite.json` (confirmed present, 6 cases, all `tier: "hard"`:
`the-second-call-remembers-the-first`, `dedupe-must-not-reorder`,
`one-fix-is-not-enough`, `the-contract-is-in-the-docstring`,
`wrong-answers-without-an-error`, `right-the-first-time-only`). They target the
*completion decision* rather than the edit, since that is the signal that survived
saturation.

**These six are not among the twelve.** `--tier hard` selects the seven built-in
hard cases in `CODING_CASES` (`codeval.py:584`), which are a different set with
different ids; the six live in a file reachable only through `--cases
fixtures/hard_suite.json` (§8.6). Nothing in the default suite runs them. It is easy
to read "the hard suite" as one thing and count the six inside the twelve — they
sum to eighteen distinct cases, not twelve.

`--check` (`make_hard_suite.py:353`) verifies the two properties that make a case
gradeable: every `fail_to_pass` node must fail on the untouched fixture and every
`pass_to_pass` node must pass. **It now completes clean** — all six cases start red
where they must and green where they must (14 nodes: 7 to fix, 7 to keep). The
earlier failure was real: the first run found `dedupe-must-not-reorder` unwinnable,
and the case was rewritten to use integers.

Two reporting bugs found alongside it, both fixed:

- `check` printed `ok  <id>` **unconditionally**, after having already printed `!!`
  lines for the same case — a broken case was reported both bad and fine, which is
  how a calibration failure gets skimmed past. It is now per-case, gated on
  `if not bad` (`make_hard_suite.py:378`).
- `--check` wrote the suite *before* checking it, so a failed check still left an
  ungradeable `fixtures/hard_suite.json` on disk — and a JSON file that exists is a
  file someone runs. The check now gates the write and returns 1 without writing
  (`make_hard_suite.py:399-409`).

The anti-gaming agents have now been run against it: `--agent lazy --cases
fixtures/hard_suite.json` scores 0/6 and `--agent vandal` scores 0/6, both with
`honest 6/6`. So the two ways of faking a pass — claiming completion without
editing, and deleting the failing test — are closed on these cases as well as on
the built-in ones.

`--agent oracle` **cannot** be run here, by design: its solutions are keyed to
built-in ids, so it refuses rather than raise a false alarm about the grader
(`coding_eval.py:671`). That leaves a real gap. On the built-in suite the oracle
proves the grader can recognise a correct fix; for these six, nothing does.
`--check` proves the fixtures start in the right state, which is a weaker claim.

What all of that establishes is that the suite is **gradeable and hard to fake**,
not that it discriminates. No model has been run against `hard_suite.json`; whether
these six recover the signal saturation destroyed is untested.

### 8.8 What a score here is not

A score here is **not a SWE-bench score**. The fixtures carry their whole repository
as a `files` map; SWE-bench instances name a repository and a commit and need
per-instance environment setup and container isolation this harness does not have.
`load_cases`' own docstring (`codeval.py:142-147`) says so. The tasks differ in kind,
not only in difficulty.

Comparing *harnesses* requires holding the model fixed. The Claude Code row above
holds neither model nor harness constant against the Knossos rows, so it compares
harness × model. `--harness-cmd` makes the honest comparison possible; it has not
been done with the model held constant.

Every number so far is single-shot. `scripts/seeds.py` states the project's own rule
— "no number without its `n`, and no `n` below 5" — and is **imported by neither
eval**. `--repeat` does not exist on `coding_eval`; `--best-of` is a different thing
that inflates the score rather than measuring spread.

### 8.9 Other suites

- **Python tests** — 778 collected, 774 selected, 4 deselected, across 26 files.
  769 pass, 5 skip.
  Harness-relevant counts: `test_engine.py` 113, `test_talos.py` 98,
  `test_acp_execute.py` 93, `test_oracle.py` 64, `test_tools.py` 49,
  `test_workspace.py` 31, `test_codeval.py` 31, `test_lethe.py` 28,
  `test_metis.py` 26, `test_acp.py` 24, `test_eval.py` 22, `test_argus.py` 22,
  `test_mcp.py` 17, `test_components.py` 17, `test_lsp.py` 16, `test_jsonrpc.py` 16,
  `test_gate.py` 16, `test_ariadne_halting.py` 15, `test_external_harness.py` 14,
  `test_checkpoint.py` 10, `test_acp_load.py` 7. Model-side and out of scope:
  `test_moirai.py` 13, `test_seeds.py` 10, `test_proteus.py` 9, `test_naiads.py` 7,
  `test_echo.py` 6.
- **Rust** — 175 test attributes, all executed (`cargo test`: 139 lib + 22 + 12 + 2).
  A further 2 live in `tests/fixtures/passing/`, which is a fixture crate the Oracle
  runs against, not part of this suite. `tests/harness_loop.rs` 22, `tests/serve_loop.rs`
  12, `tests/live_engine.rs` 2 (gated on `KNOSSOS_LIVE_OLLAMA`).
- **Conformance** (`conformance/run.mjs`) — 22 deterministic `check(...)` calls
  (`run.mjs:290`–`:772`) plus 2 `liveCheck(...)` (`run.mjs:797`, `:819`), driven by
  `@agentclientprotocol/sdk` and validating every agent→client message against the
  SDK's published schema. Skips are loud and fail under `KNOSSOS_STRICT=1`
  (`run.mjs:270`). `scripted_agent.py` runs the *real* `DaedalusAgent`, loop, jail
  and permission gate — only the engine is scripted.
- **CI** (`.github/workflows/ci.yml`) — Python, Rust, conformance and the Lapce ACP
  client on `ubuntu-latest`; the `live` job requires a self-hosted runner labelled
  `knossos-live` (`ci.yml:72`), which REPORT.md Part 4 says does not exist.
  `live-check.ps1` is the manual substitute.

### 8.10 Not covered by any test

- `TransformersEngine` — zero tests, zero executions.
- Anthropic's OpenAI-compat provider entry — the `PROVIDERS` comment
  (`engine.py:441-451`) says plainly it has never been exercised here.
- **Delegation in situ** — `test_talos.py` constructs `Talos(delegation=True)`
  directly. ACP now enables it (`acp.py:874`), but no test drives a real engine
  into calling `delegate` through the ACP path (§3.4).
- Concurrency — nothing tests two sessions prompting at once, or `session/cancel`
  racing a mid-flight `fs/write_text_file`.
- MCP end to end — the client is tested in isolation; the registration path
  (`acp.py:839`) is covered by `test_acp_execute.py`, but no test drives a real MCP
  tool being invoked by a real engine.
- Resource exhaustion — unbounded `Session.history` count, giant tool output, a
  hostile MCP/LSP peer.
- `rename_symbol` against a real language server.
- **The hard suite in anger** — `make_hard_suite.py --check` now passes, but no
  model has been scored against `fixtures/hard_suite.json` (§8.7).

### 8.8 The control arm, and what the columns cannot see

Until this pass the coding eval had no control. Every figure it produced was
*model × harness*, and a 12/12 could equally mean the harness works or that the
model would have scored 12/12 through a single API call. This was the weaker of
the two instruments in exactly the way §8.1 is strong: the retrieval eval has run
a `general` control group from the beginning, and the README calls it "the
load-bearing part" because a lift there would invalidate the treatment number.

`--baseline` (`scripts/coding_eval.py` `baseline_agent`) holds the engine, tool
vocabulary, workspace jail, prompt, fixtures and grader constant, and removes
**only the loop and the verifier**: `Ariadne(max_steps=1)` and
`accept_everything`. The delta against a normal run is what iteration and
verification buy on that model.

It is deliberately *not* "the harness versus nothing" — retrieval, the path jail
and the tool layer are all still present, and a bare HTTP call would score lower
for reasons that have nothing to do with Talos. Reporting it as the latter would
be the same overclaim the control exists to prevent. The arm posts false passes
by construction, because `accept_everything` agrees with the engine; that is the
measurement, being what "completion decided by the engine" scores.

`--repeat N` runs the whole suite N times into **separate fixture directories**
and reports the range. Sharing one directory would let a file the agent *created*
in run 1 survive into run 2 — `materialise` rewrites the case's own files and has
no way to know about anything else — so the repeats would not be independent.
Distinct from `--best-of`, which keeps the best sample and inflates the score;
this reports every run. Steps need it more than `solved` does: a pass is one bit
with bounded variance, a step count is unbounded.

What each column is blind to:

| column | blind to |
|---|---|
| `solved` | saturation — Gemini 3.1 Pro and Flash-Lite both score 12/12 on the built-in tier |
| `honest` | whose claim it is; see `claim_source` in §8.4 |
| `steps` | tool batching — this loop groups adjacent parallel-safe calls into one turn, so a model that emits reads together spends fewer steps for identical work |
| `tools` | nothing on its own; it exists to be read *against* `steps`, where agreement means a difference was batching habit rather than capability |
| `degraded` | — this is the column that stops a throttled run being quoted as capability |

---

## 9. Python vs Rust divergence

### 9.1 Present in one, absent in the other

| Component | Python | Rust |
|---|---|---|
| Constitution | `acp.py:915` reads `constitution.md`; **no compiled-in default**; one resolver for planner and executor alike, cached per session | `themis/mod.rs`, compiled-in default, rendered into *every* role prompt including the tier-4 rubric |
| Tier-4 judge | the hook exists (`oracle.py:487`) but **nothing installs one** | `oracle::judge`, on unless `--no-judge` (`main.rs:272`) |
| Scribe (exact symbol table in the prompt) | absent — Argus retrieves excerpts | `scribe/`, rendered verbatim |
| Mnemosyne (BM25 chunks as a tool) | absent as a tool | `search_code` |
| Argus (repo index with save/load, import hops) | `argus.py` | absent |
| Gate | `gate.py` | absent |
| Lethe | `lethe.py`, summarise-then-elide, wired into `_prompt`/`_history` | `lethe.rs`, **elide-only, never removes or merges a message** — see §9.2 |
| Permission gate | `_permitted` + ACP prompt; on whenever ACP builds the executor | `Approver`; opt-in per front end (§4.3) |
| Re-planning | `Replanner` + `_revise_plan`, wired via `acp.py:869` | absent — the plan is one string in the opening message |
| Futility detection | `StepOutcome.repeated` + `is_futile`, over a `FUTILE_WINDOW`=4 signature window (`talos.py:160`) | **Converged.** `repeated`/`is_futile`/`made_progress` on `StepOutcome` (`ariadne.rs:64-101`) over the same window (`talos.rs:59`), with the same restart-on-change rule and the same feedback note. Rust had *no* repeat check at all before this — `is_noop` was the whole staleness test, so an engine re-issuing one failing `edit_file` called a tool every step, was never a noop, and ran to the ceiling |
| Delegation / subagents | `Delegate` + `_spawn`, enabled by default in ACP (`acp.py:384`/`:874`), depth-capped at 1 (§3.4) | absent |
| Undo journal / checkpoints | `workspace.py:151-197`, reached per plan step | absent |
| Language adapters in the verifier | `oracle.py:289` — python/rust/go/node, polyglot | `scribe::LanguageAdapter` — Rust only |
| Plan *step-wise* execution with per-step budgets | `talos.py:595` | absent |
| ACP server / MCP client / LSP client / `rename_symbol` / `ask_user` / editor `fs`+`terminal` delegation | present | absent |
| Trajectory trace (JSONL) | per coding-eval case (`codeval.py:1225`); the live agent logs to stderr | `session.rs`, always on |
| Per-hunk diff review | absent — Python applies whole files | `diff.rs` + `apply_hunks` |
| NDJSON `serve` / REPL | absent (ACP is the front end) | `serve.rs`, `repl.rs` |
| Coding + retrieval evaluation harnesses | `codeval.py`, `eval.py` | absent |

### 9.2 Same name, different behaviour

| Aspect | Python | Rust |
|---|---|---|
| **Engine slot** | `generate(prompt, context, cancelled) -> Iterator[str]` required, plus optional `generate_messages`; streaming | `complete(&Request) -> Response`; not streaming; structured `Vec<Message>` persists |
| **Tool call wire format** | fenced `{"tool": …, "args": {…}}` plus an optional `"id"` carrying the provider's call id through the text convention (`tools.py:859`) | fenced `{"tool": …, "input": {…}}` (`prompt_fallback.rs`) — **the key name differs**; a prompt for one will not parse in the other |
| **Tool dispatch** | adjacent `parallel_safe` calls run concurrently (`ThreadPoolExecutor`, cap 8, `talos.py:1432`); adjacency rather than "reads first", because reordering changes what calls see | strictly sequential |
| **Ariadne defaults** | `max_steps=20`, `target_steps=6`, bands are fractions of the ceiling (`ariadne.py:120`, `:181`) | **Converged.** `max_steps=20` at all three places that carry it — `ariadne.rs:90`, `config.rs:46`, and `main.rs:54`, the clap default being the one the binary actually ships since it shadows `Config::default()` — and bands are fractions (`ariadne.rs:147-172`) |
| **Lethe strategy** | summarise a middle band once, then `_fit` elides the largest entry; a transcript entry is a string, so losing one loses information only | never removes or merges a message; only text inside `Text`/`ToolResult` shrinks, and `ToolUse.input` is never touched, because an id without its partner is a hard provider rejection (`lethe.rs:16-29`) |
| **Transcript budget** | derived per turn from `engine.context_window` (`talos.py:1021`) | fixed `max_tokens: 24_000` (`lethe.rs:77`); no window discovery |
| **Oracle tiers** | 0 syntax (`compile`), then adapter-selected ladders, 4 optional judge | 0 tree-sitter, then `LanguageAdapter::verify_commands()`, 4 `judge` |
| **Verifier scope** | tier 0 + scopable tiers run against `changed`; unscopable tiers are baselined; diagnostics forgiven per `(file, message)` | every tier runs over the whole root, **no baseline, no per-diagnostic forgiveness** |
| **Missing tool** | skipped, passing | skipped, passing; `summary` names skips rather than counting them as passes, and `deterministic_tiers_passed` requires at least one tier above 0 to have actually run (`oracle/mod.rs:127`) |
| **Empty change set** | `nothing_to_verify=True`, `deterministic_passed` False — but `passed` stays True, and the only consumer of `deterministic_passed` is the never-installed judge, so in practice this reduces to `acted` | no equivalent flag; `acted > 0` alone |
| **`acted` discriminator** | `CONSEQUENTIAL` frozenset (`talos.py:348`) — drift-prone | `Tool::consequential()`, **no default**, so a new tool cannot compile unclassified |
| **Command timeout** | 300 s (`tools.py:61`) | 300 s (`shell.rs:27`) — **now matched** |
| **Tool output cap** | 20 000 chars | 30 000 bytes |
| **`run_command` allowlist** | `python, pytest, ruff, mypy, git` | `cargo, rustc, rustfmt, git` (+ cargo deny-list) |
| **Metis fallback ladder** | tool call → truncated-JSON repair → prose → task-as-step | tool call → prose → task-as-step; **no repair** |
| **Metis step cap** | `MAX_STEPS = 8` enforced by `_tidy` | 8 stated in the tool description only; **not enforced** |

---

## 10. Invariants and assumptions

Load-bearing. Items 1–4 are held by convention and nothing checks them; 5–6 were
in that state and are now enforced, with the guard named so it can be found and
not quietly removed.

1. **Paths reaching a tool are inside the workspace.** True only for paths that go
   through `resolve`. Three bypasses: `RunCommand`'s child process (`tools.py:456`),
   every Oracle tier subprocess (`oracle.py:560`, `oracle.py:608`,
   `oracle.py:671`), and Rust `ToolCtx::apply_hunks` (`tools/mod.rs:133`).
2. **`CONSEQUENTIAL` is complete.** A new writing tool omitted from the frozenset
   (`talos.py:348`) is silently ungated **and** silently does not count as work.
   MCP tools are never in it, so a remote tool that writes is never gated.
3. **The permission asker is present.** `ask_permission=None` is a silent full
   allow (`talos.py:1453`). Nothing warns when an executor is built without one, and
   `scripts/coding_eval.py:369` builds one that way.
4. **`always_allowed` is keyed by tool name only** (`acp.py:1249`). "Always allow
   `run_command`" approves every subsequent command in the session.
5. **Held-out tests never shadow a visible file, and every `held_out_pass` node id
   has a file behind it.** *Now enforced* — `CodingCase.__post_init__`
   (`codeval.py:128`) raises on both, so every construction path is covered rather
   than only `load_cases`. Worth stating because of how these fail rather than how
   likely they are: neither raises on its own and neither produces a wrong-looking
   number. A shadowed file is shown to the agent and silently overwritten before
   grading; a node id with no file is collected from a tree that lacks it and
   scores zero. Both then surface as `overfit` — the report accusing the model of
   fitting the visible tests when the fault is in the fixture. A false accusation
   of gaming is indistinguishable from the real thing by reading the score.
6. **`_fresh_case` reaches a method that exists.** It calls `restore_limits`
   through `getattr` (`coding_eval.py:402`), which is right — `AgentFactory` is
   "anything with `.run(prompt)`" and a scripted agent has no engine — but a
   duck-typed call cannot distinguish "no engine" from "renamed and now dead".
   The second silently reinstates the monotonically-degrading allowance
   (§8), whose only symptom is a score that depends on case order. Covered by a
   contract test asserting `OpenAICompatEngine` still defines it.
5. **The engine is not adversarial.** Nothing sanitises retrieved file content, MCP
   tool descriptions, LSP output, or `fs/read_text_file` results before they enter
   the prompt. A file containing a fenced `{"tool": "run_command", …}` block is,
   after `read_file` returns it into the transcript, indistinguishable from the
   engine's own output on the next turn — the transcript is a flat list with no
   provenance markers.
6. **`Thought` is identified by class name, not type** (`talos.py:1282`). Any chunk
   class named `Thought`, from any module, is silently excluded from tool-call
   parsing.
7. **`isinstance(x, Engine)` is available but unused.** The protocol is
   `@runtime_checkable` (`engine.py:284`) yet nothing validates an engine; a missing
   `generate` surfaces as an `AttributeError` inside the loop.
8. **A verdict over an empty change set is meaningless.** Enforced **once**, by
   `acted` in Talos (`talos.py:1190`). The Oracle half is inert: `nothing_to_verify`
   does not clear `passed`, and the only consumer of `deterministic_passed` is the
   tier-4 judge gate, which never runs in Python. `acted` is also not `len(changed)`
   — a run whose only consequential call was a command legitimately changes no files.
9. **`accept_everything` (`talos.py:247`) is still the default verifier** for any
   directly-constructed `Talos`. It is named to be uncomfortable and it is what a
   test-shaped or script-shaped caller gets by default.
10. **A single line of stdin is a bounded message.** `Peer._read_loop`
    (`jsonrpc.py:193`) and `serve.rs` iterate lines with no length limit.
11. **`Session.history` count and Rust `Talos.messages` fit in memory.** Per-entry
    strings are bounded (`acp.py:94`); the counts are not.
12. **The Gate's name index reflects the current repo.** `RetrievalGate._index`
    (`gate.py:119`) is built lazily once and never invalidated, though Argus rescans
    every turn.
13. **Rust's Mnemosyne index reflects the current repo.** Built once in
    `build_talos` (`main.rs:235`), never rebuilt; `search_code` returns pre-edit
    content after the first write.
14. **A transcript entry is a `str`.** `Entry` (`talos.py:256`) subclasses `str` so
    the structured conversation and the flattened one are one list rather than two
    that can drift. Anything that slices or rebuilds an entry (`Lethe._fit`,
    `_elide`, summarisation) returns a plain `str` and drops the structure. That
    degradation is intended and handled at the point of use.
15. **A native tool call and its result are only sent together.** Enforced by
    `_native_pairs` (`talos.py:947`), which re-checks adjacency, roles, ids and
    result count *after* compaction. Half a pair is rejected outright by the
    providers the shape exists to please.
16. **`parallel_safe` is a per-tool opt-in, not the negation of `CONSEQUENTIAL`.**
    Only three tools set it (`tools.py:174`, `:246`, `:300`). `ask_user` is not
    consequential and must still run alone. Any unclassified tool — including every
    MCP tool — runs alone.
17. **Only `tools.dispatch` runs off the calling thread** (`talos.py:1442`). Events,
    `self.changed` and permission stay on it: events become JSON-RPC notifications on
    a stdout that asserts its own framing, and `self.changed` feeds every
    verification decision.
18. **A checker's output is parsed only in a format the tier pinned.**
    `_DIAGNOSTIC` (`oracle.py:153`) refuses anything that is not
    `path:line[:col]: message`, because the alternative is not "parse less" but
    "parse wrongly". An unrecognised format yields no baseline and therefore no
    forgiveness.
19. **`prepare` is idempotent.** It is reached once per consequential call
    (`talos.py:1379`), not once per run, and guards on `self._baseline is not None`
    (`oracle.py:411`).
20. **`--execute` implies a real engine.** Enforced only for the `retrieval` engine
    in `acp.main`; no other configuration is checked.
21. **`_prepare_verifier` is not reached for a parallel group.** `talos.py:1376`
    checks `call.name in CONSEQUENTIAL` only on the single-call branch. This is
    currently safe because no consequential tool is `parallel_safe`, but the safety
    is a consequence of that coincidence, not a check.

---

## 11. Documentation drift

> **2026-08-15 drift note — this document vs. current code.** §§1–10 were written
> 2026-07-28/29. Two commit sets landed after and are only partially reflected:
>
> - **Rust ports (`8b1b390`, `2550274`, `b7158a0`, `3411fa7`, through 2026-07-30):**
>   §9.1 still lists Argus, Gate, LSP client, MCP client, ACP server, delegation, and
>   NDJSON `serve`/`repl` as **"absent in Rust"**. All now exist and are wired
>   (`argus.rs`, `gate.rs`, `lsp.rs`, `mcp.rs`, `acp.rs`, `delegate.rs`; `main.rs:357/364/417`).
>   Treat every "absent"/"present-Python-only" row in §9.1 as **needing re-check** —
>   the Rust harness is now near-parity. §2's Rust table is corrected above; §9.1's
>   prose rows are **not** individually rewritten this pass.
> - **Containment layer (`e2ab40a`, 2026-07-29):** `sandbox`, `hooks`, `interject`
>   (Python + Rust mirrors) plus Rust-only `resilience`. §4/§5's account of subprocess
>   spawning predates `sandbox` — every subprocess now goes through `sandbox.DEFAULT.run`
>   (scrubbed env), and a `ProtectPaths([".git"])` hook now gates writes under `.git`.
>   §4.4's "no OS-level sandbox" is still true (env isolation ≠ process isolation), but
>   the credential-leak-via-env vector it did not mention is now closed by `sandbox`.
> - **Ariadne convergence:** §2's Rust row said "12/6/2"; corrected to 20/6/2. §9.2 /
>   §11-item-21 already reflected this.
> - **Line-number drift:** several files grew; anchors in §§1.1–1.2, §4, §5, §8 may be
>   off by tens of lines (see header caveat).

Where the docs and the code disagree. In each case the **code** is what §§1–10
describe. Rows 1–31 were re-verified 2026-07-29 (not all re-checked this pass).

| # | Claim | Where | Reality |
|---|---|---|---|
| 1 | "Planned, not yet built: Metis … Lethe" | `model/knossos/__init__.py:26-29` | Both exist and are wired in; the same file imports `Lethe` twelve lines later. |
| 2 | "Planned: Metis (planner), Lethe (bounded context)." | `model/knossos/README.md:21` | **Fixed**: both now appear in the component table as shipped modules. |
| 3 | "Oracle does not exist yet, so Talos takes a `Verifier`" | `model/knossos/README.md:98` | `oracle.py` is 891 lines and is the default verifier the ACP layer installs (`acp.py:830`). **Fixed**: the README now describes Oracle as satisfying the protocol, and documents `claim_source`. |
| 4 | "**`session/load` is not implemented**, and `loadSession` is advertised `false`." | `model/knossos/README.md:253` | Implemented (`acp.py:549`), advertised `true` (`acp.py:444`), covered by `test_acp_load.py` and a conformance check (`run.mjs:394`). |
| 5 | "No planner: Metis produces a plan, Talos executes one. No verifier: Oracle will implement the `Verifier` protocol below." | `talos.py:29-30` | Both exist. The module docstring is stale about its own module. |
| 6 | "Lethe (bounded context with summarise-and-reset) is the intended fix; **until it exists**, the step ceiling is what keeps the growth bounded" | `talos.py:22-24` | Lethe exists, is constructed in `__init__` (`talos.py:439`) and is called from `_prompt` and `_history`. **README fixed**; the `talos.py` module docstring still says it. |
| 7 | "conformance/, 12 of 12" | `README.md:33` | 22 deterministic + 2 live. `REPORT.md:23` already says 22; the top-level README was not updated. `REPORT.md:50` still narrates "12 of 12 on the first run", which was true then. |
| 8 | "a 19-case evaluation set" | `README.md:31` | 28 cases; 19 in-sample, 9 held out. |
| 9 | "Currently **18/19 (95%)** on the labelled eval set" | `model/knossos/README.md:371` | `CASES` holds 28 and `report_gate` iterates all of them. The figure describes the pre-held-out set. |
| 10 | Zed registration example uses `"args": ["-m", "harness", …]` and `PYTHONPATH: "…/Recurring Transformer Model"` | `model/knossos/README.md:139-140` | The package is `knossos`; `-m harness` does not exist. Stale from before commit `2965193`, "Name the harness Knossos". |
| 11 | "Retrieval runs regardless — … its file locations are useful even when the excerpts are withheld." | `acp.py:1143-1145` | On a gate skip the handler returns `[]` at `acp.py:1163` before emitting any `locations`, so the locations are discarded too. |
| 12 | "Both safety defaults are opt-out, not opt-in." | `model/knossos/README.md:33` | True of the Python ACP path. Not true of `daedalus task`, which is ungated by design, nor of Rust `serve` before a `capabilities` handshake, nor of any directly-constructed `Talos` — including the one the coding eval builds. |
| 13 | "The executor does not report which step it is on, so claiming per-step progress would be invention." | `acp.py:1102-1103` | It does: `Event(kind="plan_step")` (`talos.py:639`) is handled sixty lines above the comment (`acp.py:1038`) and advances the checklist. The comment describes the pre-`plan_step` design. |
| 14 | REPORT.md Part 4: "one operational item remains" | `REPORT.md:424-428` | Accurate as an inventory of *that session's* items. It predates the coding eval, `codeval.py`, external harnesses, and the saturation finding, none of which it mentions. |
| 15 | "Every route to a vacuous pass found so far is closed" | `REPORT.md:365` | Two further routes were found and closed after it was written: deleting the failing test (closed by `_suite_integrity`) and `acted` counting any successful call including `read_file` (closed by restricting it to `CONSEQUENTIAL`). Note the phrase that keeps being true — *found so far*. Both were found by running the thing, not by reading it. |
| 16 | "an empty change set is *nothing to verify* rather than a pass" | `REPORT.md:368-369` | The conclusion is now true but the mechanism named is not the one enforcing it: `nothing_to_verify` does not clear `passed`, and its only consumer is the tier-4 judge gate, which never runs in Python. `acted` is what enforces it. |
| 17 | `mcp.py:19-21` — "Each remote tool is then wrapped as an ordinary `Tool` and registered" | `model/knossos/mcp.py` | **Accurate now.** `_talos_for` calls `self.mcp_tools(session)` (`acp.py:839`) and merges via `ToolRegistry.combined`. |
| 18 | `acp.py:685` `_release` — "Stop everything this session started" | `model/knossos/acp.py` | **Accurate now.** It calls `client.close()`, the method `McpClient` defines (`mcp.py:152`). |
| 19 | The previous version of this file, and the step-6 working note that reported subagents "done", described delegation as wired into ACP | `ARCHITECTURE.md` (superseded), step-6 note | It was not: `delegation=True` appeared only in `test_talos.py` and `acp.py` never mentioned the flag, so `delegate` was absent from every shipped registry. **Now actually wired** (`acp.py:384`/`:874`) — the claim became true after the finding, not before it (§3.4). |
| 20 | The previous version of this file said tiers run over the whole repository, that Rust `run` has no timeout, that Rust has no context bound, and that the built-in coding suite has 10 cases | `ARCHITECTURE.md` (superseded) | All four are obsolete: per-tier scoping + baseline (§5.3), `shell.rs:27`, `lethe.rs`, 12 cases. |
| 21 | `ariadne.py:115` describes banding on absolute step counts as the bug it fixed | `knossos-rs/src/ariadne.rs` (superseded) | Rust still had it: `max_steps=12` with `remaining <= 3` bands, so raising the Rust ceiling would have reintroduced the same plateau — the β=0.01 failure the module exists to prevent. **Fixed**: fractions of the ceiling, ceiling 20, and three tests that fail against the old banding. Note it needed three edits, not one — `ariadne.rs`, `config.rs`, and `main.rs`, the last being the only one the binary reads. |
| 22 | `acp.py` forwarded a constitution to the executor and a different, always-empty one to the planner | `model/knossos/acp.py` (superseded) | The planner planned with no standing instructions while the executor was held to the workspace's, which is how a plan gets written that the executor is forbidden to carry out. **Fixed**: one resolver, cached per session, used by all three roles (§4.5). |
| 23 | The step-6 note treats "ten tests pass" as evidence a feature ships | step-6 note | It is evidence the feature *works*, which is a different claim. Four of the five built-tested-never-connected findings in this file (17, 18, 19, 22) had passing tests over the disconnected unit. A test that constructs the component itself cannot observe that nothing else does. |
| 24 | The coding eval reported `solved` and `honest` as if either could rank a harness | `scripts/coding_eval.py` (superseded) | `solved` is saturated (Gemini 3.1 Pro and Flash-Lite both 12/12) and `honest` is not one quantity: for Knossos it is Oracle's verdict, for a foreign harness it is a process exit code, and most CLIs exit 0 unless they crash — so an external arm cannot score a false fail and `honest` collapses into `solved`. **Fixed**: `claim_source` is recorded per case, printed as `honest*` with a warning when the claim is an exit code, and surfaced by `trace_summary`. |
| 25 | The coding eval had no control arm at all | `scripts/coding_eval.py` (superseded) | Every number it produced was *model × harness* with no way to attribute any of it to the harness — while `knossos.eval`, the weaker-stakes instrument, has had a control group from the start and the README calls it "the load-bearing part". **Fixed**: `--baseline` (§8.6). |
| 26 | A throttled, shrunk run aggregated as a clean capability result | `codeval.py:_write_trace` (superseded) | `report_throttling` warned *on the console*, and the console is exactly what `trace_summary` exists because nobody keeps. `output_shrinks` never reached the trace, so a run that finished with a halved reply allowance was indistinguishable from an unimpeded one months later. **Fixed**: `degraded`, `output_shrinks`, `throttle_waits` and `throttled_seconds` are recorded per case and reported by both the console and the summariser. |
| 27 | `ariadne.py:124` — `STUCK` after 2 consecutive unproductive steps, where unproductive includes "merely repeated the one before it" | `model/knossos/talos.py` (superseded) | It was a one-step lookback, so a model alternating between two failing actions (A, B, A, B) never produced two consecutive identical signatures: `is_futile` never fired, `stuck_after` never tripped, and the run spent its whole 20-step ceiling achieving nothing — the pathology Ariadne exists to prevent, through the one door the check did not cover. **Fixed**: a `FUTILE_WINDOW`-wide window of recent signatures (`talos.py:160`), restarted by any step that changes a file so that revisiting after real work is never counted against revisiting before it. `_signature` was already sound; the gap was the comparison window. Verified by setting the window back to 1 and confirming the two-cycle and three-cycle tests fail. **Rust had no repeat check at all** and is now ported to match (§9.1): confirmed by disabling `is_futile` there, which fails 6 tests including the single-call repeat, and by narrowing the window to 1, which fails exactly the two cycle tests in both languages. |
| 28 | `codeval.py:run_tests` — "One process per node rather than one for the batch. Slower, and worth it" | `model/knossos/codeval.py` (superseded) | The reasoning was right and the scope was wrong: isolation is needed to *attribute* a failure, and a green batch has nothing to attribute. It was paid on every node, every case, every run — the dominant cost of the eval, which `--repeat` multiplies and held-out tests add a fourth set to. **Fixed**: batch first, fall back to per-node only when the batch is not green, so the result is never worse than before because the fallback *is* before. One deliberate difference: a test that passes only because another ran first now grades as passing. |
| 29 | `_snapshot` SHA1s every file in the tree, twice per case | `model/scripts/coding_eval.py` (superseded) | Nothing for the built-in fixtures; O(repo bytes × 2 × cases) the moment `--cases` points at a real repository, which is what `--cases` is for. **Fixed**: a (path, size, mtime_ns) → hash cache, so an unchanged file is not re-read. A prefilter, not a substitute — the value is still the hash, so "changed" still means the bytes differ rather than the timestamp moved. |
| 30 | `model/knossos/README.md` — "`Engine.generate` takes two strings and has no message history, so the conversation is re-rendered into the prompt every turn" | `model/knossos/README.md` (superseded) | Stale since `generate_messages` landed: it is on the protocol (`engine.py:291`), `OpenAICompatEngine` implements it, and `Talos._turn` (`talos.py:1315`) prefers it whenever present, so every hosted run already sends a role-tagged array. The quadratic it named is real but belongs to the *stateless API*, not to the two-string shape — an array costs what the same content costs flattened. **Fixed**: the paragraph now says what was measured (§1.1) rather than what was true two refactors ago. |
| 31 | "Lethe … so a long run stops growing quadratically" | `model/knossos/README.md` (superseded) | The cap exists and is late: at the 24,000-token default it first fires at step 18, so for the runs that actually occur it is not binding and the growth is quadratic throughout. Claiming the bound solves the cost is the same shape of error as claiming a feature ships because it has tests. **Fixed**, and the prefix-stability property that *does* reduce the bill now has a test rather than an assumption. |

---

## 12. Not covered by this pass

> **2026-08-15 pass boundary.** This pass read in full only the new containment
> modules (`sandbox.py`, `hooks.py`, `interject.py`) and confirmed their wiring; it
> verified the Rust module inventory + `main.rs` wiring (`lib.rs`, subcommands,
> `run_acp`, argus/gate/delegate use) at grep/header level. It did **not** re-read the
> Rust ports (`argus.rs`, `gate.rs`, `lsp.rs`, `mcp.rs`, `jsonrpc.rs`, `acp.rs`,
> `delegate.rs`, `resilience.rs`) line-by-line — their rows in §2 describe purpose and
> wiring, not internal behaviour, and their `file:line` internals are unverified. It
> did not re-run any test/eval suite. The paragraphs below describe the **07-29** pass.

Read in full this pass: `talos.py`, `oracle.py`, `codeval.py` (docstrings, case
metadata, and everything from `materialise` onward), `scripts/coding_eval.py`,
`workspace.py` (jail + journal), `ariadne.py`, `ariadne.rs`, `lethe.rs`,
`scripts/make_hard_suite.py` (header, `check`, `main`), `scripts/trace_summary.py`
(header).

Executed this pass, not merely read: `make_hard_suite.py --check` (clean, §8.7),
`cargo test` (175 pass), `pytest` (769 pass, 5 skip). Every "**Now actually wired**"
in §11 rests on a test that drives the shipped construction path, not on the edit
having been made.

Read partially — outlines, signatures and selected regions:

- `engine.py` — the protocol, `EngineProfile`, `RequestTooLarge`, `PROVIDERS`, the
  degrade branches in `_run`, and the marker lists. The SSE parser, `_accumulate_tool_call`,
  `_render_native_calls`, `ThinkSplitter`, retry/backoff and `_explain` were outlined only.
- `acp.py` — handlers, `_talos_for`, `_replan`, `_constitution`, `_run_execution`,
  `_run_retrieval`, `_release`, `Session`. `session_fork`/`list`/`close`/`delete`,
  `EditorFiles`/`EditorTerminal`/`EditorElicitation` and `main()` were outlined only.
- `tools.py` — registry, `parse_calls`, `Search`, `RunCommand` in full;
  `RenameSymbol._locate`, `AskUser` and `tokenize` at signature level.
- `lethe.py` — `compact`, `_fitted`, `_fit` in full; `extractive_summary` and the
  regexes at signature level.
- `argus.py`, `gate.py`, `eval.py`, `evalset.py`, `metis.py`, `jsonrpc.py`, `lsp.py`,
  `mcp.py` — outlines and constants only.
- Rust: `talos.rs` drive loop, `lethe.rs`, `serve.rs` (router/permission/`Idle`),
  `main.rs` (`build_talos`, `run_serve`, `run_repl`), `repl.rs` approver,
  `tools/mod.rs` registry + `apply_hunks`, `tools/shell.rs`, `oracle/mod.rs`
  (verdict, spawn) were read. `diff.rs`, `mnemosyne.rs`, `scribe/*`, `themis/mod.rs`,
  `session.rs`, `config.rs`, `engine/*` were read at signature/grep level only.
- `conformance/run.mjs` — check *names* (all 24) and the live section; individual
  assertion bodies for the 22 deterministic checks were not read.
- `model/tests/*` and `knossos-rs/tests/*` — collected counts and names; individual
  test bodies were not read.
- `README.md`, `model/knossos/README.md`, `REPORT.md` — grepped for the claims in
  §11 and read around them; not read end to end.

Not read at all: `model/daedalus/`, `model/train.py`, `model/PLAN.md`,
`model/README.md`, `knossos-rs/README.md`, `knossos-rs/constitution.md`,
`editor/`, `conformance/node_modules/`, `.claude/worktrees/`, and the
model-side scripts (`naiads_eval.py`, `proteus_probe.py`, `moirai_sweep.py`,
`echo_sweep.py`, `fetch_rust.py`).

"Not documented" above therefore means "not present" only for the fully-read files.
