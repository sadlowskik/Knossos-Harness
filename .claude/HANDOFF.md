# Handoff — Knossos harness hardening + first real measurements

## 1. Goal
Harden the Knossos agentic harness (`model/knossos/`) and get a coding-eval number that can be
trusted. The session ran audits, fixed what they found, then took the project's first genuine
frontier and harness-vs-harness measurements.

## 2. State

**Verified green:** 763 Python tests, 172 Rust tests, `cargo clippy` clean.
Full Python suite takes ~4 min.

**Landed this session (all tested):**
- Eval integrity — `CaseResult.unreachable` separates "provider never reached" from a capability
  zero; recorded in the trace header; excluded from every rate in `report()`.
- `EngineProfile` + `RequestTooLarge` preflight; 413 now names the measured body size.
- Sampling-param degrade-on-400 (Sonnet 5 / Opus 4.7+ reject `temperature`).
- `max_tokens` default 8192 → 32000 (Sonnet 5 / Opus 5 think by default).
- Gemini `thought_signature` fallback — was 400ing ~half of all engine calls.
- Language adapters in Oracle (Rust/Go/Node/polyglot).
- Re-planning (`Replanner`) and subagents (`Delegate`), both wired into ACP.
- External-suite loader (`--cases`) and external-harness adapter (`--harness-cmd`).
- Rust: `kill_on_drop` on Oracle spawn; timeout on the shell tool.

**Measurements (in `model/traces/`):**

| Run | solved | honest |
|---|---|---|
| Knossos + Gemini 3.1 Pro | 12/12 | 12/12 |
| Knossos + Gemini 3.1 Flash-Lite | 12/12 | **10/12** |
| Claude Code (Sonnet 5), via `--harness-cmd` | 12/12 | 12/12 |

**The finding:** the built-in suite is **saturated** — a lite model aces it, so it cannot rank
harnesses or frontier models. `honest` is the only column with signal left (Flash-Lite lost 2 to
`budget_exhausted` after already fixing the code).

**In progress / not verified:** `model/scripts/make_hard_suite.py`. First `--check` run found
`dedupe-must-not-reorder` unwinnable (ordering tests marked `pass_to_pass` already failed). The case
was rewritten to use integers so the `list(set(...))` trap is deterministic — **the re-check was
interrupted and has never run.**

## 3. Key files

- `model/knossos/codeval.py:253` — `unreachable`; `:121` `load_cases`; `:1225` `_write_trace`.
- `model/scripts/coding_eval.py:264` — `ExternalHarness`; `:345` `external_agent`; `:235`
  `_snapshot` (before/after hashing); `:409` `report` (denominator excludes unreachable).
- `model/knossos/engine.py:382` — `EngineProfile`; `:361` `RequestTooLarge`; `:843` `profile`
  property; `:987` `_NO_TOOL_MESSAGES` (holds the `thought_signature` markers); `:1027`
  `_NO_SAMPLING`.
- `model/knossos/talos.py:141` — `Delegate`; `:489` `_spawn`; `:96` `Replanner`; `:533`
  `_revise_plan` (fires only when the failed step produced evidence).
- `model/knossos/oracle.py:289` — `LanguageAdapter`; `:328` `tiers_for`; `:258`/`:280` Rust/Node tiers.
- `model/scripts/trace_summary.py` — aggregates `traces/`, reconstructs `unreachable` for old traces.
- `model/scripts/make_hard_suite.py` — 6 harder cases + `--check` calibration. **Unverified.**
- `ARCHITECTURE.md` — written early in the session, now **substantially stale**.

## 4. Decisions (settled — do not relitigate)

- **Watch `honest`, not `solved`.** `solved` is saturated across models and harnesses.
- **No provider key substitutes for another.** Groq/Gemini/Google keys cannot serve Claude; that
  needs an `sk-ant-` API key. A **Claude Pro subscription is not API credit** — it powers the
  `claude` CLI, not `--provider anthropic`.
- **Never tabulate model IDs.** Detect capability by degrading on the 400 (the provider table's own
  comment says ids drift). This is why the sampling fix is a marker list, not a model list.
- **Antigravity cannot be benchmarked.** Installed at `%LOCALAPPDATA%\Programs\antigravity`, but
  GUI-only: `resources/bin` has just `language_server.exe` and `webm_encoder.exe`, no CLI shim.
  Same applies to Cursor/Windsurf without a separate headless binary.
- **No "Hermes" coding harness exists.** PyPI matches are unrelated or unknown provenance; nothing
  was installed. Hermes is Nous Research's *model* line.
- **aider is the right second harness** — holds the model constant (same Gemini key), so a
  difference is attributable to the harness. Claude Code vs Knossos confounds harness × model.
  Needs `pip install aider-chat`; user has not yet approved.

## 5. Next step

Run `cd model && "$PY" -m scripts.make_hard_suite --check`. All 6 cases must report `ok`. Then run
the suite against Flash-Lite (the only config with headroom):
`"$PY" -m scripts.coding_eval --cases fixtures/hard_suite.json --engine api --provider gemini --model gemini-3.1-flash-lite`

## 6. Gotchas

- `python` is not on PATH. Use `$env:PY` / `"$PY"` (see user CLAUDE.md).
- **`GEMINI_API_KEY` is set at Windows *User* scope only** and is NOT inherited by tool shells.
  Load it explicitly:
  `export GEMINI_API_KEY="$(powershell.exe -NoProfile -Command '[Environment]::GetEnvironmentVariable("GEMINI_API_KEY","User")' | tr -d '\r\n')"`
- `ANTHROPIC_API_KEY` is unset. `claude` CLI auth was revoked mid-session; user fixed it with
  `/login` (needs the slash — bare `login` is sent as a chat message). It works now.
- `claude login` is interactive and cannot be run from a tool shell.
- **The repo changed under the session while work was in progress** (user edited files in parallel).
  Re-read before assuming any earlier finding still holds — several audit findings were already
  stale when reported.
- `--repeat` does not exist on `coding_eval` (it's on `eval.py`). `--best-of` is different: it
  inflates the score rather than measuring spread. No repeated-runs mode exists yet.
- Every number so far is single-shot; `seeds.py` ("no number without its `n`") is not imported by
  either eval.
