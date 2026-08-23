# Knossos

An agentic coding harness whose components are the **system-level form of the
mechanisms in the [Daedalus](https://github.com/korbinsadlowski/daedalus) model
architecture**.

The engine is a swappable slot. Everything around it — exact symbol memory, an
always-on constitution, tiered verification, an explicit halting policy — is
engine-agnostic by construction and survives an engine swap. That is the point:
the model architecture is a long research project, and the harness is the half
that can have real capability today, because it is not compute-bound.

```bash
daedalus task "add a triple() function next to double()" -w ./my-crate
daedalus acp                          # editor agent (ACP on stdio)
daedalus eval --cases suite.json      # fail_to_pass / pass_to_pass grader
```

`python -m knossos` is training/legacy. The product binary is `daedalus`.
With no API key, `daedalus acp` still starts (retrieval-only: search the
tree, no edits). Pass `--engine ollama` or `--engine cameo` for a local
model, or set `ANTHROPIC_API_KEY`. Editors that advertise `fs`, `terminal`,
or elicitation get those channels; everyone else stays on the workspace jail.

## The mapping

Each component is the system-level form of a tensor-level mechanism. Only
mappings that force a **design decision you would not otherwise make** were
kept:

| Component | Model-level origin | The decision it forces |
|---|---|---|
| **Scribe** | exact symbol table | identifiers, paths and signatures are injected verbatim — never summarized |
| **Mnemosyne** | lossy gist memory | bodies are retrieved on demand, so "where is X handled" is answerable |
| **Themis** | always-on shared expert | the constitution enters *every* call, planning included, not a review step at the end |
| **Ariadne** | PonderNet halting | explicit escalating stopping pressure plus a hard ceiling |
| **Oracle** | — | deterministic tiers before any model judgement |
| **Metis → Talos** | — | the plan is an artifact you can read and reject before a file is touched |

Mappings that would only rename a standard pattern were left out. "Labyrinth →
the agent loop" is just a loop; calling it Labyrinth adds naming, not design.

### Why these three are load-bearing

**Scribe — the two-tier memory split.** *Approximate the prose, but keep
anything a compiler cares about bit-exact.* Most agent frameworks summarize the
whole context window, identifiers included, and then the model invents a method
name that never existed. Scribe parses; it returns ground truth, not a best
guess.

It uses tree-sitter rather than `syn` or regex for one specific reason:
tree-sitter is **error-tolerant**. A file with a syntax error still yields a
tree where the broken region is an `ERROR` node and every valid sibling is
intact — so the index keeps answering "what exists here" while the agent is
mid-edit. `syn` returns `Err` for the whole file. There is a test that asserts
exactly this.

**Ariadne — a measured failure mode, not ported math.** In the research repo,
PonderNet halting at β=0.01 collapsed to maximum depth (7.5 of 8 steps, no
adaptivity at all); at β=0.1 it settled at a healthy ~5 steps. The lesson is
that without explicit, *increasing* pressure to stop, an adaptive-compute system
spends its whole budget on every input regardless of difficulty — precisely the
agent pathology of burning twelve iterations on a one-line fix.

So this is not PonderNet. It is the three things that lesson says you need: a
hard ceiling, escalating pressure past a target (stated in prompt text, since
there is no gradient here), and a strong deterministic stop signal.

**Mnemosyne — BM25, not embeddings, and that is a decision not a shortcut.** The
tensor-level version compresses with learned cross-attention, so the obvious
translation is a vector index. I deliberately did not build one. Code queries
are overwhelmingly *lexical* — a name, a call, an error string — and lexical
scoring is very hard to beat on those while being exact, explainable,
dependency-free and instant to rebuild. Embeddings earn their cost on
natural-language paraphrase, a small fraction of what an agent asks a codebase.
What carries over is the **role** — fuzzy recall over context too large to hold,
paired with Scribe's exactness — not the mechanism. A local vector index can
slot in behind `Mnemosyne::search` if lexical retrieval proves measurably
insufficient.

**Oracle — two rules.** Fail fast: the first failing tier returns immediately,
because there is no point running clippy on code that does not compile. And
model judgement is the *last* tier, reachable only when every deterministic tier
passes — "compiles, lints and tests pass, but is it good?" is the only question
a model adds that cargo cannot answer.

```
Tier 0  tree-sitter parse       free, in-process
Tier 1  cargo check             --message-format=json → structured diagnostics
Tier 2  cargo clippy
Tier 3  cargo test
Tier 4  model vs constitution   expensive, gated on all of the above
```

## Install

```bash
git clone <this-repo> && cd daedalus-harness && cargo build --release
```

## Usage

```bash
# exact symbol index — no engine required
daedalus index -w ./my-crate
daedalus index -w ./my-crate --lookup Adder

# verification ladder — no engine required
daedalus verify -w ./my-crate

# plan without executing
daedalus plan "add a --json flag" -w ./my-crate

# plan and execute, verifying as it goes
daedalus task "add a --json flag" -w ./my-crate --max-steps 12 --target-steps 6

# propose changes without writing them, and print unified diffs
daedalus task "add a --json flag" -w ./my-crate --dry-run

# interactive session that keeps context between turns
daedalus repl -w ./my-crate --dry-run

# Agent Client Protocol — this is what an editor spawns (no Python)
daedalus acp --engine cameo --model qwen2.5-0.5b
```

### In a code editor (ACP)

You do **not** need the Python agent. Any editor that speaks [ACP](https://agentclientprotocol.com) (Zed, JetBrains ACP, VS Code ACP, this repo's Lapce fork) can spawn the Rust binary on stdio:

```
command: /absolute/path/to/daedalus
args:    acp
env:     CAMEO_BASE_URL=http://127.0.0.1:9090/v1
         CAMEO_MODEL=qwen2.5-0.5b
         CAMEO_SERVE_KEY=…          # consumer /v1
         CAMEO_CONSOLE_KEY=…        # optional; loads a cold model
```

Implemented: `initialize`, `session/new`, `session/prompt`, `session/cancel`, `session/list`, `session/close`, `session/set_mode` (`ask`/`preview`/`write`), `session/interject`, plus `session/update` and `session/request_permission`. Assistant prose still arrives as one chunk until engines stream.

Zed-style agent server (shape; field names follow your editor's schema):

```json
{
  "name": "daedalus",
  "command": "/absolute/path/to/daedalus",
  "args": ["acp"],
  "env": { "CAMEO_BASE_URL": "http://127.0.0.1:9090/v1" }
}
```

### Dry run

`--dry-run` stages every edit in memory instead of writing it. Reads and later
edits see the staged content, so a multi-step change previews exactly what it
would have done rather than approximating it.

The honest limitation is built into the verdict: with nothing on disk, cargo
would compile the *old* source and report a pass about the wrong code. So the
ladder stops at tier 0, the verdict is flagged `dry_run`, and
`deterministic_tiers_passed()` returns false — which means **a dry run can never
reach tier 4**, and every message it produces says "preview, not verification".

### Interactive session

`daedalus repl` keeps the conversation, the symbol index and the staged changes
alive between turns, so you can redirect the agent without losing what it
already worked out.

```
/diff              show staged changes
/apply             write them to disk
/discard           throw them away
/verify            run the verification ladder now
/index [name]      symbol counts, or look one up
/plan <task>       plan without executing
/reset             clear the conversation, keep the workspace
/steps <n>         change the step ceiling
/quit
```

Each turn gets its own step budget, rather than one allowance draining across a
long session. An engine failure prints and returns you to the prompt — it does
not end the session, because staged work would go with it.

## Editor integration

A VS Code extension lives in [`editor/vscode`](editor/vscode). It is a thin
front end: it spawns this binary and uses VS Code for input prompts, an output
channel, diff highlighting and cancellation. No harness logic is reimplemented
there, so the editor cannot drift from the CLI.

```bash
cd editor/vscode && npm install && npm run compile
```

Then open that folder in VS Code and press F5. **Daedalus: Preview Task** is the
command worth reaching for first — it gives you the accept/reject step that
makes an agent safe to point at a real repository.

### Engines

```bash
export ANTHROPIC_API_KEY=...
daedalus task "..." --engine anthropic --model claude-opus-5

# or fully local
daedalus task "..." --engine ollama --model qwen3-coder:30b
```

Anthropic has native tool use. Ollama's varies by model, so engines declare
`supports_native_tools()` and the harness falls back to a prompted-JSON protocol
it parses itself. That shim is strictly less reliable than native tool calling;
it exists so that "swappable engine" is a fact rather than a claim.

## Safety

Two properties are structural rather than advisory:

- **Path jail.** Every filesystem tool resolves through `ToolCtx::resolve`,
  which rejects `../` traversal, absolute paths outside the root, and symlinks
  pointing out of the tree.
- **No shell.** Commands are split into program plus argument vector and
  executed directly. No shell interpreter is involved, so `&&`, `|`, `;` and
  backticks are inert — they arrive as literal arguments. On top of that the
  program must be on an allowlist (`cargo`, `rustc`, `rustfmt`, `git`), and
  `git` is restricted to read-only subcommands.

Both are covered by tests that attempt the escape.

## The trace

Every plan, step, tool call, Oracle verdict and halt decision is appended to a
JSONL trajectory log. That is not logging. Those traces are the SFT/RL data that
would eventually train the model architecture's own core — capturing them now
costs almost nothing and reconstructing them later from logs costs a great deal.
It is the bridge from the harness back to the research repo.

## Configuration

`constitution.md` in the workspace root overrides the built-in default. One
document serves both roles: it shapes what the agent does, and it is the rubric
Oracle's final tier judges against.

## Verification

```bash
cargo test
```

The whole suite runs with no API key, no network and no model server — via a
scripted `MockEngine` — while the tools, path jail, Scribe, Oracle and Ariadne
are all real. The end-to-end tests genuinely invoke `cargo check`, `clippy` and
`test` against fixture crates.

What the tests defend:

| Claim | Test |
|---|---|
| Scribe extracts symbols from syntactically broken files | `scribe::rust::still_extracts_symbols_from_a_file_with_a_syntax_error` |
| Oracle fails fast and never reaches tier 4 on broken code | `a_broken_edit_fails_verification_and_the_loop_keeps_going` |
| Tier 0 catches syntax errors without spawning cargo | `tier_zero_catches_a_syntax_error_before_cargo_runs` |
| The loop always terminates | `the_step_ceiling_forces_a_halt` |
| The path jail refuses a hostile tool call | `the_path_jail_survives_a_hostile_tool_call` |
| The allowlist refuses `rm` | `disallowed_shell_commands_are_refused` |
| The exact index stays exact as the agent edits | `scribe_tracks_edits_made_during_the_run` |
| A dry run writes nothing and still produces usable diffs | `a_dry_run_proposes_changes_without_writing_them` |
| A passing dry run cannot be mistaken for verification | `a_passing_dry_run_still_blocks_tier_four_and_says_why` |
| Staged edits are visible to later reads, so chains preview correctly | `staged_content_is_visible_to_later_reads_and_edits` |
| Resume keeps the conversation instead of restarting | `resume_continues_the_same_conversation` |
| Accepting one hunk takes that change and leaves the other | `diff::accepting_one_hunk_takes_only_that_change` |
| Accepting no hunks reproduces the original byte for byte | `diff::accepting_no_hunk_reproduces_the_original_exactly` |
| Retrieval answers a question Scribe structurally cannot | `mnemosyne::answers_a_question_about_where_something_lives` |
| An unrelated query returns nothing rather than noise | `mnemosyne::an_unrelated_query_returns_nothing_rather_than_noise` |

## Honest scope

**This is v1 and it is small.** What it does: plans a task, executes it with
real file edits and build commands, verifies with a tiered ladder, and stops on
evidence or on a budget. That is a working agent, not a frontier one.

**What is in the harness today.** Scribe indexes Rust, Python, Go and Node
(tree-sitter for Rust; scanners for the rest). Oracle runs the matching
ladder. Lethe shrinks oversized tool results in place. Failed hypotheses
land in `{workspace}/.knossos/episodes.jsonl` and are injected after compact
and on the next run. First `Stuck` redirects once; the second is an honest
halt. Product spawn is `daedalus acp` (stdio JSON-RPC). `session/load`
replays already-sent updates; it does not re-run tools.

**Not implemented, deliberately:**

- **Apollo** (routing to specialist sub-agents) — `delegate` already runs a
  child agent; a routing layer on top waits until the core loop is the only
  ACP server editors spawn.
- **Naiads** (per-task memory namespaces) — episode store is workspace-scoped,
  not per-task.
- **Echo** (trajectory distillation / SFT) — collection and deterministic
  SFT/DPO curation are implemented in `model/scripts/curate_traces.py` with
  v2 external labels, secret quarantine, task-level splits, and exact-prompt
  DPO pairing. The actual model-training job remains a separate pipeline.
- **Proteus** (self-modifying weights → self-editing prompts) — **excluded
  permanently, not deferred.** It has the same runaway failure mode as the
  tensor-level version, but at system level there is no `‖W‖` to watch; you lose
  the diagnostic that made it studiable. The research repo isolates Proteus so
  it cannot destabilize the main line, and that argument only gets stronger when
  the blast radius is a tool-using agent on a filesystem.
- **Tab completion.** Calling an API per keystroke is 300ms–2s where Cursor Tab
  is under 100ms, and it costs tokens continuously. The version worth having is
  a small FIM-trained model, which is a separate project.
- **Token-level streaming on every backend.** OpenAI-compat and Cameo stream
  SSE into `agent_message_chunk` / `agent_thought_chunk` as tokens arrive.
  Anthropic and Ollama still complete in one shot (they inherit the default).
  `MockEngine` does not stream, so the halt tests stay one-shot.
- **Inline gutter accept/reject.** Hunk-level review happens in the panel, not
  as decorations over your editor buffer.
- **Editor `fs/*` / `terminal/*`.** The path jail is the product. The agent
  reads and writes through its own tools, not through the client.

**Known limits.** Quality tracks the engine, not the harness: a weak local model
will look like a harness bug. Ollama tool-use reliability varies by model. The
step budget is a blunt instrument — `Stuck` detection is based on tool calls and
file changes, which is evidence, but coarse evidence.

## License

MIT.
