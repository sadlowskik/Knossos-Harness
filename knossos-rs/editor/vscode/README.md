# Knossos Harness — VS Code extension

A **session panel** in the sidebar plus a few command-palette entries, both
driving the `knossos` binary. All the behaviour lives in the Rust harness;
nothing here reimplements it, so the editor cannot drift from what the CLI does.

## The session panel

Click the Knossos icon in the activity bar. The panel holds a conversation —
you describe a task, watch each step and tool call stream in, then review what
it wants to change.

By default it runs in **preview mode**: edits are staged in memory, never
written. You get a file list with `+`/`-` counts; clicking a filename opens
VS Code's own side-by-side diff, and **Accept all** / **Reject all** decide
what happens. That accept/reject step is what makes an agent safe to point at
a real repository.

The panel is backed by `knossos serve`, a long-lived NDJSON process, which is
what lets it keep context across turns instead of restarting each time.

| Button | Effect |
|---|---|
| **Send** | First message starts a task; later ones continue the same conversation |
| **Verify** | Runs Oracle's tiered ladder now |
| **Reset** | Clears the conversation, keeps the workspace |

Turn off `knossos.previewByDefault` if you would rather the panel write
directly to disk.

## Prerequisites

1. Build the harness:

   ```bash
   cargo build --release
   ```

2. Either put `target/release/knossos` on your PATH, or set
   `knossos.binaryPath` to its full path in VS Code settings.

3. Set `ANTHROPIC_API_KEY` in the environment VS Code inherits — or switch
   `knossos.engine` to `ollama` for a fully local run.

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
| **Knossos: Open Session Panel** | Focus the sidebar panel |
| **Knossos: Run Task** | Prompts for a task and runs it *in the panel*, so it joins the conversation |
| **Knossos: Preview Task (dry run)** | Stages changes in memory and opens the unified diff in a tab. Nothing is written. |
| **Knossos: Verify Workspace** | Runs Oracle's tiered ladder. |
| **Knossos: Show Symbol Index** | Scribe's exact index in the output channel. |
| **Knossos: Look Up Symbol** | Exact declarations for a name. Uses the editor selection if there is one, and appears in the right-click menu. |

## Settings

| Setting | Default | Meaning |
|---|---|---|
| `knossos.binaryPath` | `knossos` | Path to the executable |
| `knossos.engine` | `anthropic` | `anthropic` or `ollama` |
| `knossos.model` | *(empty)* | Model id; empty uses the engine default |
| `knossos.maxSteps` | `12` | Ariadne's hard ceiling |
| `knossos.targetSteps` | `6` | Where budget pressure begins |
| `knossos.judge` | `true` | Run Oracle tier 4 after the deterministic tiers pass |
| `knossos.previewByDefault` | `true` | Panel stages changes for review instead of writing them |

## Scope

**This is not Cursor.** There is no tab completion, no `@`-mention file picker,
no codebase embedding, and no inline per-hunk accept — accept and reject are
whole-changeset operations. What you get is a conversation, streamed progress,
real diffs, and an accept gate, over a harness whose behaviour you control
completely.

The terminal REPL (`knossos repl`) remains the fuller interface; it has
`/discard`, `/steps`, `/plan` and `/index`, which the panel does not surface.
