# Daedalus

A from-scratch coding model and the agentic harness built around it.

The premise is a **swappable engine slot**. The model architecture is a long
research project; the harness is the half that can have real capability today,
because it is not compute-bound. Everything around the slot — exact symbol
memory, retrieval, a constitution, tiered verification, an explicit halting
policy — is engine-agnostic by construction and survives an engine swap.

## Layout

| Path | What it is |
|---|---|
| `model/` | The architecture (`daedalus/`) and the Python harness (`harness/`) |
| `knossos-rs/` | The Rust harness — Metis, Themis, Mnemosyne, Scribe, a VS Code extension |
| `fixtures/scratch-crate/` | A minimal cargo library, used as a target for exercising the harness against real Rust |

Two harness implementations exist deliberately. They are kept side by side while
the question of which line continues is still open:

- **Python** (`model/knossos/`) speaks the [Agent Client Protocol](https://agentclientprotocol.com),
  so it runs in Zed and JetBrains today without forking anything. It is the one
  with measured results — retrieval ranking, a gate, and a 19-case evaluation set.
- **Rust** (`knossos-rs/`) is architecturally further along and could be linked
  directly into an editor fork rather than spawned as a subprocess.

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

## History

This repository is a consolidation of three that were developed separately.
Their histories are preserved via `git subtree`, so `git log` reaches back
through all of them.

The upstream model repository remains at
[sadlowskik/Daedalus](https://github.com/sadlowskik/Daedalus).

## Licence

MIT © Korbin Sadlowski
