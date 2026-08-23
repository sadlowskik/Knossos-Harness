# Conformance

Knossos driven by the Agent Client Protocol's **own** client, not by ours.

Every other test of the ACP seam in this repository is written by the same hand
as the agent:

| Suite | Client | Catches |
|---|---|---|
| `model/tests/test_acp.py` | `FakeClient`, real pipes | framing, threading, handler bugs |
| `editor/lapce-acp/tests/against_real_agent.rs` | the Lapce fork's client | cross-language wire bugs |
| **this** | `@agentclientprotocol/sdk` + published JSON Schema | **misreadings of the spec** |

The first two are real integration tests and they do find real bugs. Neither can
find a *shared* misunderstanding: if the agent and the client agree on the wrong
field name, they agree, and both suites stay green. That is the gap this closes.
Messages are additionally validated against `schema/schema.json`, which ships
inside the SDK — a field renamed upstream fails here first.

## Running it

```bash
cd conformance && npm install && npm test
```

The agent under test is **`daedalus acp`** (`knossos-rs/target/debug/daedalus`).
Build it first: `cargo build --manifest-path knossos-rs/Cargo.toml`.

`KNOSSOS_ACP=python` drives the legacy `python -m knossos` / `scripted_agent.py`
path. `KNOSSOS_ACP_BIN` overrides the Rust binary.

Pass a substring to run one check: `node run.mjs permission`.

## Live checks

Two checks need a real model and are skipped otherwise:

```bash
KNOSSOS_LIVE=1 OLLAMA_HOST=<host> KNOSSOS_LIVE_MODEL=gemma4:e4b npm test
```

They are skipped rather than mocked on purpose. A mocked "live" check would
report exactly the thing that has already been wrong twice in this project — a
green suite over a path no model has ever taken.

## The scripted agent

`scripted_agent.py` serves the real `DaedalusAgent`, the real `Talos` loop, the
real workspace jail and the real permission gate, with a fixed list of engine
replies. Only the model is faked, because a conformance suite whose result
depends on whether a model felt like calling a tool is measuring the model.

It reads its script from `KNOSSOS_SCRIPT` rather than adding a `--engine
scripted` flag: a documented way to feed the agent arbitrary tool calls from
outside is precisely what the permission gate exists to prevent.
