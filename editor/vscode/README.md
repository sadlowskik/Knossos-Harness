# Daedalus Harness — VS Code extension

A thin front end for the `daedalus` binary. All the behaviour lives in the Rust
harness; this provides input prompts, an output channel, diff highlighting and
cancellation. Nothing here reimplements harness logic, so the editor cannot
drift from what the CLI does.

## Prerequisites

1. Build the harness:

   ```bash
   cargo build --release
   ```

2. Either put `target/release/daedalus` on your PATH, or set
   `daedalus.binaryPath` to its full path in VS Code settings.

3. Set `ANTHROPIC_API_KEY` in the environment VS Code inherits — or switch
   `daedalus.engine` to `ollama` for a fully local run.

## Install (development)

```bash
cd editor/vscode
npm install
npm run compile
```

Then open `editor/vscode` in VS Code and press **F5** to launch an Extension
Development Host with the extension loaded.

To install it permanently, package it with `npx vsce package` and run
**Extensions: Install from VSIX** from the command palette.

## Commands

| Command | What it does |
|---|---|
| **Daedalus: Run Task** | Plans and executes, editing files. Warns first, and offers to preview instead. |
| **Daedalus: Preview Task (dry run)** | Stages changes in memory and opens the unified diff in a tab. Nothing is written. |
| **Daedalus: Verify Workspace** | Runs Oracle's tiered ladder. |
| **Daedalus: Show Symbol Index** | Scribe's exact index in the output channel. |
| **Daedalus: Look Up Symbol** | Exact declarations for a name. Uses the editor selection if there is one, and appears in the right-click menu. |

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `daedalus.binaryPath` | `daedalus` | Path to the executable |
| `daedalus.engine` | `anthropic` | `anthropic` or `ollama` |
| `daedalus.model` | *(empty)* | Model id; empty uses the engine default |
| `daedalus.maxSteps` | `12` | Ariadne's hard ceiling |
| `daedalus.targetSteps` | `6` | Where budget pressure begins |
| `daedalus.judge` | `true` | Run Oracle tier 4 after the deterministic tiers pass |

## Scope

This is a command front end, not a chat panel. The interactive session lives in
the terminal (`daedalus repl`), where the slash commands — `/diff`, `/apply`,
`/verify` — are already the natural interface. Building a webview chat panel
that duplicates it would add surface area without adding capability.

Preview is the command worth reaching for first. It gives you the accept/reject
step that makes an agent safe to point at a real repository.
