# Knossos

An engine-agnostic coding harness with exact symbol memory, retrieval, a
constitution, tiered verification, explicit halting, and the Field command
surface for operating many agents.

The premise is a **swappable engine slot**. Knossos is the system around that
slot; Cameo, Ollama, Anthropic, and OpenAI-compatible providers can all power
it. Daedalus is the model architecture that inspired several of Knossos's
mechanisms, not the name of the harness or its CLI.

```bash
knossos task "add a --json flag" -w ./my-project
knossos field
knossos acp
knossos eval --cases suite.json
```

## Layout

| Path | What it is |
|---|---|
| `knossos-rs/` | The Knossos product: Rust library, `knossos` CLI, ACP/serve protocols, and VS Code extension |
| `field/` | Knossos Field: the Roman multi-agent command surface, server, web client, policies, and release bundle |
| `model/` | Daedalus model research and the legacy Python Knossos reference implementation |
| `conformance/` | Knossos driven by the ACP authors' own client and schema |
| `fixtures/scratch-crate/` | A minimal cargo library, used as a target for exercising Knossos against real Rust |
| `editor/` | Your Lapce fork, as a submodule |

Run `knossos field` from a source checkout, or point a released binary at the
separately downloadable Field bundle with `knossos field --dir <path>`. The command
starts the loopback-only server and prints its one-time authenticated bootstrap URL.

## Naming

- **Knossos** is the harness, crate, CLI, editor integration, and release name.
- **Field** is Knossos's Roman command surface for multi-agent operations.
- **Cameo** is an optional local inference runtime and operating environment.
- **Daedalus** is the model/research lineage.

Two Knossos implementations remain in the history, but their roles are settled:

- **Rust** (`knossos-rs/`) is the product implementation and the one releases build.
- **Python** (`model/knossos/`) is retained for research, training, and historical
  cross-implementation checks.

## Adding the editor fork

The editor is deliberately **not** vendored here. Lapce is a large, actively
developed upstream, and copying its source in makes `git pull upstream main`
painful forever. Fork it, then link it:

```bash
# 1. Fork lapce/lapce on GitHub, then from this repository:
git submodule add https://github.com/<you>/lapce editor
cd editor
git remote add upstream https://github.com/lapce/lapce
git fetch upstream
```

Thereafter `git pull upstream main` inside `editor/` keeps the fork current,
and this repository records only which commit of it you are on.

Lapce is Apache-2.0, which is why it was chosen over Zed — Zed's editor is
GPL-3.0, so a fork of it would force the whole harness GPL.

## The mapping

Each harness component is the system-level form of a tensor-level mechanism in
the model:

| Component | Model-level origin |
|---|---|
| Scribe / Argus | exact symbol table, never summarized |
| Mnemosyne | gist memory — lossy, compressed recollection |
| Ariadne | adaptive halting: how much compute this turn deserves |
| Themis / Oracle | the always-on constitution, and verification against it |
| Metis → Talos | plan, then execute and run the tests |
| Lethe | bounded context with summarize-and-reset |

## Product status

Knossos is pre-v1. The Rust runtime and Field are the supported product direction;
the Python harness and Daedalus model remain research inputs. A release is qualified
only by a clean full Rust/Field suite, supported-platform packaging, live-engine
conformance, and retained recovery/evaluation evidence—not by historical test counts.

Current usage belongs in [`knossos-rs/README.md`](knossos-rs/README.md), Field's
operator and design documentation belongs in [`field/README.md`](field/README.md),
and retained experiment outputs live under `reports/`. When Knossos is bundled by
Cameo, the parent repository's `PRODUCTIZATION_PLAN.md` owns cross-product scope and
release gates.

## History

This repository is a consolidation of three that were developed separately.
Their histories are preserved via `git subtree`, so `git log` reaches back
through all of them.

The Daedalus model research repository remains at
[sadlowskik/Daedalus](https://github.com/sadlowskik/Daedalus).

## Licence

MIT © Korbin Sadlowski
