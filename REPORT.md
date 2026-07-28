# Cross-harness report

Written 2026-07-26, after running Knossos against three harnesses other than its
own test suite. It continues [`model/PLAN.md`](model/PLAN.md), which is the
retrospective this project already runs on, and uses the same rule: every claim
below is something that was executed, not reviewed.

The headline is that **the ACP seam is in better shape than the project claimed,
and execute mode is in worse shape.** Conformance against the protocol authors'
own client passed on the first run. The first live model to touch execute mode
found a chain of three defects that made a turn which did nothing report success.

---

## Part 0 — What was run

| Harness | What it is | At the start | Now |
|---|---|---|---|
| Python suite | `model/tests` | 338 passed | **511 passed** |
| Python slow suite | training-based, `-m slow` | never run this session | **4 passed** (5m 01s) |
| `knossos-rs` | the Rust harness | 130 passed, all mocked | **135 passed**, 2 live |
| **A.** `editor/lapce-acp` | the Lapce fork's ACP client, driving the real Python agent over pipes | **never executed** | **16 passed** |
| **B.** `conformance/` | `@agentclientprotocol/sdk` 1.3.0 + the published JSON Schema | did not exist | **22 passed** |
| **C.** live provider | `gemma4:e4b` via Ollama | did not exist | **2 passed** |

Harness A existed already and had **never been run**: `against_real_agent.rs` is
gated on `LAPCE_ACP_TEST_PYTHON` and `LAPCE_ACP_TEST_AGENT_CWD`, and without them
it prints `skipped` and returns green. Setting two environment variables was the
entire cost of executing it. It passed.

Harness B is new (`conformance/`). Harness C is PLAN.md's Stage 4.1, which had
been open since the plan was written.

### What conformance actually establishes

The distinction that matters, because the project already had two real
integration suites:

| Suite | Client written by | Can catch |
|---|---|---|
| `test_acp.py` | this project | framing, threading, handler bugs |
| `against_real_agent.rs` | this project | cross-language wire bugs |
| `conformance/` | **the protocol's authors** | **misreadings of the spec** |

The first two drive real pipes and do find real bugs. Neither can find a *shared*
misunderstanding: if agent and client agree on the wrong field name, they agree,
and both stay green. Harness B closes that, and additionally validates every
agent→client message against `schema/schema.json` as shipped inside the SDK.

**It passed 12 of 12 deterministic checks on the first run.** Protocol version,
`InitializeResponse`, `NewSessionResponse`, `PromptResponse`, every
`SessionNotification`, `RequestPermissionRequest`, absolute path locations,
`session/load` replay, and error handling for an unknown session are all
conformant. The README's claim that Knossos "runs in Zed and JetBrains today"
now has evidence behind it rather than an argument.

---

## Part 1 — What the live model found

One real model, one execute-mode prompt: *"Create a file called generated.py
containing an add(a, b) function."*

The turn reported **`stopReason: end_turn`** — success — and created nothing.

That is the exact failure the project's own test file says is impossible.
`test_acp_execute.py` opens with:

> 3. `stopReason` tells the truth: only a verified run says `end_turn`.

Three separate defects compose to produce it.

### 1.1 An empty reply was read as a claim of completion

`Talos._drive` branches on whether the engine's reply contained tool calls. No
calls meant "the engine believes it is finished. The verifier decides."

An empty reply has no tool calls, so it took that branch.

### 1.2 The verifier passes vacuously on an empty change set

`Oracle._tier0` iterates the changed files, finds none, and returns
`passed=True, detail="0 file(s) parse cleanly"`. In a dry run every other tier is
skipped by design. So the verdict passes.

Reproduced directly:

```
Oracle(changed=[]) passed = True | syntax only -- nothing is on disk, so the
linter, type checker and tests could not run
```

The load-bearing rule — *completion is decided by the verifier, not the engine* —
was technically satisfied the whole time. The verifier decided. It just cannot
distinguish "did the work correctly" from "did nothing at all", and every test
covering this installed a deliberately failing verifier instead of exercising the
real Oracle over an empty change set. `test_acp_execute.py` even says so in a
comment: *"Oracle over an empty change set passes, so install a failing
verifier."* The vacuity was known at the unit-test level and never followed
through to what it meant at the turn level.

### 1.3 Execute mode ignored `finish_reason`

The retrieval path maps the provider's `finish_reason` onto an ACP stop reason,
with a comment stating exactly why:

> Reporting a reply truncated at the token limit as "end_turn" would tell the
> editor the answer was complete when it was cut off mid-sentence.

`_run_execution` did not do this. It returned `end_turn if outcome.succeeded else
"refusal"` and never consulted the engine. Every unfinished execute run was
reported identically, whatever the cause.

### 1.4 Why the reply was empty: native tool calls are dropped

The root cause is more interesting than truncation. `gemma4:e4b` answered with
OpenAI's **native** `tool_calls` format:

```
engine finish_reason: tool_calls
```

`OpenAICompatEngine._stream` reads `delta.content` and `delta.reasoning`. It does
not read `delta.tool_calls`. Knossos asks for tool calls as a fenced ```json
block in the message content and never sends a `tools` parameter — but the model
volunteered the native format anyway, and the entire reply landed in a field
nothing reads.

Isolating the stream confirms the model was working correctly:

```
THOUGHT chars : 401
CONTENT chars : 135
'```json\n{\n  "tool": "write_file",\n  "args": {\n    "file": "generated.py", ...
```

Note also `"file"` where the schema says `"path"` — a second adherence gap that
would have failed the call had it ever been read.

### 1.5 What was fixed

| # | Fix | Where |
|---|---|---|
| 1 | An empty reply is a no-op step, not a completion claim. It never reaches the verifier, is explained to the user, and is fed back to the engine. | `talos.py` |
| 2 | Execute mode consults `finish_reason`, so truncation reports `max_tokens` rather than `refusal` or `end_turn`. | `acp.py` |
| 3 | Native `tool_calls` deltas are counted and logged, turning a silent empty reply into a named diagnosis. | `engine.py` |
| 4 | `tool_calls` added to the `finish_reason` map, so it stops falling through to `end_turn`. | `acp.py` |

Verified against the same live model. Before: `end_turn`, nothing written. After:

```
[engine] gemma4:e4b replied with N native tool-call delta(s), which Knossos does
not consume -- it expects a ```json block in the message content.
STOP REASON: refusal
```

13 regression tests were added across `test_talos.py`, `test_acp_execute.py`,
`test_engine.py` and `test_acp.py`. Suite: **338 → 361**.

Harness B's live execute check now fails deliberately, naming the cause. That
failure is the honest state of the system, and it should stay red until §2.1 is
resolved.

---

## Part 2 — What was found, and what was done about it

Every item in this section has since been fixed. The original finding is kept
above each fix, because the finding is the part worth remembering.

### 2.1 Execute mode could not act with a model that prefers native tool calls

**Found:** `OpenAICompatEngine._stream` read `delta.content` and `delta.reasoning`
and ignored `delta.tool_calls`, so a model answering in OpenAI's native format
produced an entirely empty reply.

**Fixed** by taking the largest of the three options — consuming them properly
rather than detecting and refusing:

- Streamed `tool_calls` fragments are accumulated by index (`id` and the function
  name arrive once; the arguments arrive split at arbitrary points) and
  normalised into the fenced-block convention the rest of the pipeline already
  parses. One convention reaches Talos however the model chose to answer.
- `ToolRegistry.openai_schema()` builds a real `tools` array, and Talos hands it
  to the engine before each run. This is what stops argument names being
  guesses — the `"file"` vs `"path"` slip in §1.4 came from a model that had
  only ever seen the schema described in prose.
- Unparseable arguments are dropped with a log line rather than guessed at,
  matching how `parse_calls` already fails closed.

**Verified live.** The same prompt that opened this report now runs to
completion: `gemma4:e4b` writes `generated.py`, writes a test for it, runs
pytest, and the turn ends `end_turn` — earned this time.

### 2.2 The Oracle passed on an empty change set

**Found:** `Oracle(ws, [])` returned `passed=True` ("0 file(s) parse cleanly"),
so any route to an empty change set produced a verified success.

**Fixed** in two layers, because the trap is that "nothing changed" is not
always failure — `run_command` tasks legitimately change no files:

- `OracleVerdict` gained `nothing_to_verify`, distinct from passing, and it
  blocks `deterministic_passed` so model judgement cannot be reached either.
- Talos counts tool calls that ran without erroring, and a passing verdict only
  produces `DONE` if at least one did. The discriminator is *did anything
  succeed*, not *did a file change*, so a read-only task still completes and a
  run whose every call was refused does not.

### 2.3 The agent ignored the client's filesystem capabilities

**Found:** `client_capabilities` was assigned at `acp.py:152` and read nowhere.
The agent read and wrote disk directly, so it reasoned about stale content when
a buffer had unsaved changes, and its writes never entered the editor's undo
stack.

**Fixed:** `Workspace` takes an optional `EditorFiles` delegate, and the ACP
layer installs one when the client advertises `fs.readTextFile` /
`fs.writeTextFile`. Reads prefer staged content, then the editor, then disk;
writes and undo-journal entries go through the editor when there is one. The
jail runs *first* either way — delegation changes where the bytes come from,
never which paths may be touched, and there is a test pinning that. A client
that advertises the capability and then fails falls back to disk rather than
losing the write.

### 2.4 Reasoning was streamed in one mode and discarded in the other

**Found:** retrieval mode sent `Thought` chunks as `agent_thought_chunk`;
execute mode dropped them, so the mode where the agent *acts* showed nothing
between tool calls.

**Fixed:** `Talos._stream` emits a `thought` event instead of discarding, and
the ACP layer puts it on the same channel retrieval mode already used. Reasoning
still never reaches `parse_calls`, which has its own test — a scratchpad that
mentions a tool call is not a request to run one.

### 2.5 Cross-harness tests defaulted to green when unconfigured

**Found:** Harness A had never run once, because two unset environment variables
made it print `skipped` and return `ok`.

**Fixed:** skips are announced loudly and become failures under
`LAPCE_ACP_TEST_STRICT` / `KNOSSOS_STRICT` / `KNOSSOS_LIVE_STRICT`. A root
`.github/workflows/ci.yml` runs all four suites and sets the cross-language job's
variables with STRICT on, so the same silence cannot recur there. The conformance
runner now prints an explicit "N check(s) DID NOT RUN and proved nothing" block
rather than a bare skip count.

### 2.6 The Rust harness had zero live engine calls

**Found:** 130 tests, all `MockEngine`. The Anthropic and Ollama backends had
never exchanged a packet — and the Python side had just demonstrated what that
hides.

**Fixed on both counts.** Checking §1.1 and §1.2 against the Rust loop found
**both defects present**: `calls.is_empty()` treated an empty reply as a
completion claim, and `tier0` passed vacuously over an empty change set (worse
there, since the cargo ladder then verifies the *repository* rather than the
change). Both carry the same fix as the Python side, with three regression
tests. `tests/live_engine.rs` adds the crate's first real engine calls — a plain
question and a request carrying the full tool schema, which is the shape the
executor actually sends and which no mocked test exercises.

Three existing tests had to be updated: they scripted exactly two engine turns
and depended on a *refused* call ending the run. That dependency was the bug.

### 2.7 The protocol had grown and Knossos had not

**Found:** ACP is still protocol version 1, so nothing was broken — but the SDK
declares 29 agent methods and 14 client methods, and Knossos implemented a small
subset. "Speaks ACP" had become a claim about a much larger surface than when it
was made.

**Fixed** for everything an editor user would notice:

- **`session/set_mode`** — `ask`, `preview`, `write`. These are the two flags the
  CLI already had (execute, and stage-vs-write) named and made switchable at
  runtime, which turns "restart the agent with a different flag" into a dropdown.
  The prompt path now branches on `session.mode` rather than the process-wide
  `self.execute`. Switching to `write` deliberately does **not** apply staged
  edits: changing a dropdown is not approval to write anything.
- **`session/fork`** — branches the conversation and the mode. It does not carry
  staged edits (two sessions holding proposals for the same file would race, and
  the result would depend on click order) and does not carry `always allow`
  grants, which were answered about a different branch. Advertised as
  `unstable_forkSession`, matching how the schema marks it.
- **`terminal/*`** — `run_command` now runs in the editor's terminal when the
  client advertises one, via create → wait_for_exit → output → release, with the
  release in a `finally` so a failed command does not leak a terminal. The
  allowlist runs *first*: which commands may run stays this harness's decision,
  and there is a test asserting a refused command never reaches the editor.

- **`session/list`** — every live session with a title and an ISO 8601
  `updatedAt`, newest first, filterable by `cwd`. The title is the *first*
  prompt and later ones do not rename it, because renaming a session
  mid-conversation loses the user's place in a picker. No pagination:
  `nextCursor` is omitted, which the spec defines as "no more results", so a
  paginating client terminates instead of looping.
- **`elicitation/*`** — the agent can now ask the user a structured question
  instead of guessing, via a new `ask_user` tool. It is deliberately *not*
  consequential: approving a dialog in order to see a dialog is not a flow.
  Declining is a real answer — the model is told to proceed on its own
  judgement and say what it assumed, rather than being handed invented input.
  With no capable client the tool returns that same guidance instead of
  blocking.

### The falsy-capability bug

Worth its own heading, because it is a whole class rather than one slip.

ACP spells "supported" two ways. The original capabilities are booleans —
`terminal: true`, `fs: {readTextFile: true}`. Everything added since is an
object where the **empty** object means yes: `elicitation: {form: {}}`, and
likewise `session`, `plan`, `nes`.

`{}` is falsy in Python. So the obvious check —

```python
if not caps.get("elicitation"):
    return None
```

— rejects precisely the clients that conform. It was written that way here
first. From inside the agent it is invisible: no error, no log, just a client
that appears never to have been asked. The conformance client caught it only
because it advertises the real shape rather than a convenient one.

Audited every capability read against the schema's declared type. `terminal`
and the two `fs` fields are genuine booleans, so those were correct; only
`elicitation` was wrong. All three now go through one helper:

```python
def supported(capabilities, *path) -> bool:
    """Presence, not truth: `{}` means yes."""
```

Thirteen parametrised cases pin both spellings, including `{}`, an explicit
`false`, `null`, and a wrong-shaped value. The point of the helper is not the
three current call sites — it is that `session`, `plan` and `nes` are the same
shape and are not read yet.

The conformance client also had to advertise `elicitation` and handle
`unstable_createElicitation` — the SDK prefixes unstable methods, and a
mis-named handler surfaces as `Method not found` from the client rather than as
anything wrong on the agent side.

---

## Part 2b — Where this is now better than an ordinary harness

Not feature parity with better-resourced tools — that race is unwinnable and
not worth entering. These are the places where the design does something the
mainstream ones do not.

**A plan is executed, not quoted.** Metis produces steps; each gets its *own*
step budget, so one thrashing step can no longer starve the rest — the usual
way a plan-shaped task fails. Between steps a cheap structural check runs
(tier 0 only, in-process, reading through the workspace so staged content
counts), which means a step that leaves the tree unparseable is reported inside
that step, while the engine still holds the context that produced it. The full
ladder is reserved for a closing phase that settles the *task*, because the plan
was only ever a guess about how to get there. The ACP checklist advances from
where execution actually is.

**"Done" is earned.** `end_turn` requires a real verification over a non-empty
change set in which at least one tool call succeeded. Every route to a vacuous
pass found so far is closed: an empty reply is not a completion claim, an empty
change set is *nothing to verify* rather than a pass, and a run in which nothing
succeeded cannot be ended by a verdict. That last one was reopened today by the
closing phase and closed again within the hour — by a test that already existed.

**Renames are exact.** `rename_symbol` resolves uses through the language
server, so it does not touch the comment, the string literal, the longer
identifier, or the unrelated local that share the spelling. It refuses when no
server is running rather than degrading to text substitution: a rename that
quietly becomes find-and-replace is worse than one that did not happen. Every
edit still goes through the jail and the permission gate, per file.

**Retrieval is measured, including where it is weak.** Held-out cases now exist
(§3.4, open since PLAN.md), and they say ranking generalises worse than recall:
mean rank 1.4 in-sample against 3.5 held-out. That is a published weakness in
the component this harness is built around, and the confound that could explain
it is named rather than glossed.

**It fails closed, everywhere it can.** No client, broken client, dismissed
dialog, cancelled turn, dead language server, unparseable tool arguments,
absent capability — each one refuses rather than proceeding on a guess.

**And it is verified against someone else's implementation.** 22 conformance
checks against the protocol authors' own client and published JSON Schema, plus
a Rust client driving the real agent over pipes.

What has *not* changed: an agent's usefulness is dominated by its model, and
this one's slot holds a small local model or someone else's API. None of the
above beats Claude Code at Claude Code's job. What it does is make a local,
private, auditable agent whose completion signal means something — and that
combination is genuinely not otherwise available.

## Part 3 — What this says about the practices

PLAN.md named six root causes. Two are now measurably better and one is not.

| | Practice | Status |
|---|---|---|
| **P2** | Mock-only verification | **Addressed.** Both implementations now have a live path, the Python side has a third-party conformance suite, and every cross-harness suite fails rather than skips under STRICT. §2.5 named a new variant worth keeping in mind: a real test that never runs is worse than a mock, because a mock at least executes. |
| **P3** | Claims ahead of sample size | **Held, with one debt incurred.** Every number here was executed. The live findings are n=1 and labelled as such — `gemma4:e4b` preferring native tool calls does not establish how common that is. But sending a `tools` schema changes what models receive, so the retrieval eval numbers now predate the code, which is Part 4 item 1. |
| **P5** | Breadth before depth | **Paid.** The session added `conformance/` before the Rust harness's live-engine debt, which inverted the rule. That debt is now settled, and settling it found the same two defects there — which is the argument for the rule, not against it. |

One new practice is worth naming, because it caused §1.2 and it is not on
PLAN.md's list:

**P7 — A test that documents a weakness in a comment has not addressed it.**
`test_acp_execute.py` contained the sentence *"Oracle over an empty change set
passes, so install a failing verifier."* That is a correct and precise
observation about a vacuous verifier, written down, and then worked around
locally instead of followed to its consequence. The comment made the test pass;
it did not make the system correct. The distance between "I noticed this while
writing a test" and "this is a defect in the product" was never travelled.

---

## Part 4 — What is left

§2.1–§2.7 are done. One operational item remains:

| # | Task | Why |
|---|---|---|
| 1 | Point a self-hosted runner at a model server and set `KNOSSOS_LIVE_STRICT` | The live tests exist now and pass, but in CI they still skip. A live test nobody runs is what §2.5 was about. |

### A correction

An earlier draft of this report listed a second item: re-run the retrieval eval,
on the grounds that §2.1 changed what the model receives and so the measured
deltas in `model/README.md` predated the code.

**That was wrong.** `eval.py` builds an engine directly and never constructs a
`Talos`, and `tool_schema` is only ever set by `Talos._advertise_tools`. The eval
path therefore sends exactly the request it always did. Checked by driving
`_collect` against a recording server and inspecting the payload — `tools` is
absent and `tool_schema` is `None` — and pinned by
`test_the_eval_path_sends_no_tool_schema`, so it cannot become true silently.

The numbers stand. Worth recording rather than quietly deleting: a claimed debt
that turns out not to exist is the same failure as an unclaimed one that does —
both are the document disagreeing with the code.

---

## Appendix — Reproducing

```bash
# Python
cd model && pytest -q && pytest -m slow -q

# Rust harness
cd knossos-rs && cargo test

# Harness A: the Lapce client against the real agent
cd editor
LAPCE_ACP_TEST_PYTHON=<python> LAPCE_ACP_TEST_AGENT_CWD=<repo>/model \
  cargo test -p lapce-acp --tests

# Harness B: the protocol authors' client and schema
cd conformance && npm install && npm test

# Harness C: with a live model
cd conformance
KNOSSOS_LIVE=1 OLLAMA_HOST=<host> KNOSSOS_LIVE_MODEL=gemma4:e4b npm test
```
