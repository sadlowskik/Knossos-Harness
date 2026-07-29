# Handoff — Rust Oracle attribution complete; next is the remaining ports

## 1. Goal
Port Python's Oracle attribution machinery to Rust. **Done.** Both halves —
forgiveness of pre-existing diagnostics, and a suite-integrity check that catches an
agent making failing tests disappear by deleting them — are committed and verified.

## 2. State
**Committed and verified.** Working tree clean apart from this file.
- `bf2a7ad` Oracle baseline + diagnostic forgiveness
- `24a24c0` Fail verification when the suite loses tests
- `780da1d` Quiet three lints a newer toolchain started reporting

284 Rust tests green (230 lib + 33 harness_loop + 16 serve_loop + 3 sandbox_env + 2
live_engine), clippy `--all-targets -D warnings` clean, rustdoc clean. Python untouched.

The two failures the previous handoff warned about — `scribe_tracks_edits_made_during_the_run`
and `resume_continues_the_same_conversation` — are fixed. The cause was as suspected:
`baseline_tests` was being recorded before the `use_baseline` early return in `prepare`,
so loop tests that opted out of the baseline still got the integrity check and burned
extra steps failing it.

## 3. Key files
- `knossos-rs/src/oracle/mod.rs` — all of the attribution work. `Baseline`/`diagnostic_key`/
  `tally`, then `count_test_fns`/`suite_integrity` above `impl Baseline`, `prepare` in
  `impl Oracle`, integrity wired into `verify` right after tier 0, tests at the bottom.
- `knossos-rs/src/talos.rs` — top of `drive()` calls `oracle.prepare()`, skipped on dry run.
- `knossos-rs/tests/harness_loop.rs`, `tests/serve_loop.rs` — harnesses use
  `Oracle::new(..).without_baseline()`; reason commented at the call site.
- `model/knossos/oracle.py` — the original that was ported. Nothing left to take from it.

## 4. Decisions (settled — do not relitigate)
- **Rust is the survivor.** Python keeps only the eval (`eval.py`, `codeval.py`,
  `evalset.py`), which drives the Rust binary via `coding_eval.py --harness-cmd`.
- `forgiven` is a **third tier state**, not a pass. It neither blocks the ladder nor
  satisfies `deterministic_tiers_passed`.
- Diagnostics keyed on `(file, message)` — line/column deliberately dropped, because an
  edit shifts every diagnostic below it.
- Integrity judged on the **total** count, not per file, so moving a test between modules
  is not a deletion. Per-file detail still reported.
- Test functions counted from source with a regex, not `cargo test --list`: the listing
  needs a tree that compiles, and a broken build is when the check matters most.
- `without_baseline()` disables forgiveness **and** counting — one decision, matching
  Python, where `use_baseline=False` returns before recording counts.
- Loop tests opt out of the baseline: running the ladder per test cost +83s.
- `TraceEvent::Exchange` is off by default (`--collect-exchanges`) and never streamed.

## 5. Next step
Nothing is in flight. The remaining ports, in the order they were last discussed:
`lsp`, `mcp`, `argus`+`gate`. `acp` only if Zed support is wanted. Pick one deliberately
rather than by default — none is started, so none is half-finished.

## 6. Gotchas
- **Never round-trip a source file through PowerShell.** `Get-Content -Raw` +
  `Set-Content -Encoding utf8` added a BOM and mojibake'd every em-dash in
  `oracle/mod.rs`. A "repair" via Latin-1 then destroyed the characters outright (the
  corruption was CP1252). Recovered with `git checkout`. Use the Edit tool only.
- **`git commit -m` breaks on embedded double quotes** under PowerShell native-arg
  passing. Write the message to a file and use `git commit -F <file>`.
- **Do not move the fixture's tests out of `passing/src/lib.rs`.** Tried it; the scripted
  agents rewrite that file wholesale and each drops a *different* part of the API
  (`Adder`, `double`, everything), so no `tests/` file can reference the crate and keep
  compiling. Fixed the right way instead — see the `without_baseline` decision.
- The toolchain has moved since the last session; `manual_repeat_n` and
  `private_intra_doc_links` were new. Run clippy with `--all-targets`, not just the lib.
- `python` is not on PATH; use `$env:PY`. Python tests need
  `$env:PYTHONPATH='C:\Users\korbi\Downloads\daedalus\model'`.
- Homelab ollama: `$env:OLLAMA_HOST='192.168.4.103'`, model `qwen3.5:9b-32k`. The plain
  `qwen3.5:9b` tag has no `num_ctx` and defaults to 4096, silently truncating the system
  prompt from the left.
- Fireworks key is billing-suspended; the DashScope key was leaked in a screenshot and
  should be rotated. Local ollama is the free path for the eval.
- Both repos were reconciled into `C:\Users\korbi\Downloads\daedalus`.
  `Recurring Transformer Model` still exists with its work on branch
  `checkpoint/before-harness-merge`; its `main` is untouched because Kaggle clones it.
- Nothing is pushed. All work is local commits on `fix/wire-delegation-constitution-ariadne`.
