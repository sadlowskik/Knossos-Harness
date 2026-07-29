# Handoff — the Python ports are finished; nothing is in flight

## 1. Goal
Port the remaining Python modules to Rust: `lsp`, `mcp`, `argus`+`gate`, and the
`jsonrpc` peer both transports sit on. **Done.** The Oracle attribution work that
preceded them is also done and verified.

## 2. State
**Committed and verified.** Working tree clean apart from this file.
- `bf2a7ad` Oracle baseline + diagnostic forgiveness
- `24a24c0` Fail verification when the suite loses tests
- `780da1d` Quiet three lints a newer toolchain started reporting
- `2550274` Port the JSON-RPC peer and the MCP client to Rust
- `8b1b390` Port Argus, the retrieval gate and the LSP client to Rust

376 Rust tests green (322 lib + 33 harness_loop + 16 serve_loop + 3 sandbox_env + 2
live_engine), clippy `--all-targets -D warnings` clean, rustdoc clean. Python untouched.

The 92 new tests are 22 argus, 14 gate, 16 jsonrpc, 21 mcp, 19 lsp.

## 3. Key files
- `knossos-rs/src/argus.rs` — file-level BM25 over four fields, import graph,
  budget-packed retrieval. `Located::in_test` and `is_meta` are the test-detection
  signals `gate` reuses. `users_of` is the LSP-backed half of `importers_of`.
- `knossos-rs/src/gate.rs` — whether to inject at all. Weights and thresholds at the
  top of `impl RetrievalGate`.
- `knossos-rs/src/jsonrpc.rs` — the bidirectional peer. Split into `PeerHandle`
  (the conversation) and `Peer` (the threads); see the module docs for why.
- `knossos-rs/src/mcp.rs` — remote tools. `over()` is the testable seam; `connect()`
  spawns a subprocess on top of it.
- `knossos-rs/src/lsp.rs` — Content-Length framing, which is the whole reason it
  cannot reuse `jsonrpc::Peer`.
- `knossos-rs/src/oracle/mod.rs` — the attribution work from the previous session.
- `model/knossos/` — the originals. Nothing left to take from any of them except
  `acp.py`, and only if Zed support is wanted.

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
Nothing is in flight and nothing is half-finished. The ports are done; `acp.py` is
the only Python module left untranslated, and only matters if Zed support is wanted.

Two things are built but not yet wired into the loop, which is the obvious next
choice rather than a defect: nothing calls `RetrievalGate` or `Argus::retrieve` on
the way into a prompt, and nothing passes ACP's `mcpServers` to `mcp::connect_all`.
Both are deliberate — this was a port, not an integration — but until that wiring
exists the modules are dead weight at runtime, however well tested.

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
