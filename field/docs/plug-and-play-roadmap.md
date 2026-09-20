# Field — Productization Roadmap

From today's state to a product a stranger can download, run, and **command** on their
own code with their own models — as a living **real-time strategy game** where the
codebase is the map and the agents are the units. Each phase lists the code changes, the
runtime impact, the security impact, and how each is mitigated.

Companion docs: [`cities-plan.md`](cities-plan.md) (the map substrate, built).
Last updated: **2026-09-20** (amendment below).

---

## What Field is (the identity)

Field is an **RTS for commanding a fleet of coding agents.** This is not a theme bolted
on — RTS *is* the correct interface for the problem: command many autonomous units, under
a resource constraint, toward objectives, in real time, with fog of war. That is exactly
what operating a swarm of agents on a codebase is.

**The one rule that makes it real and not a game skin: the world never invents.** Every
unit, structure, resource, and event on the map is a faithful projection of the canonical
event log ([`contracts/field-event-v1.schema.json`](../../contracts/field-event-v1.schema.json)).
Units really edit code; resources really are the token/$ budget; combat really is red/blue
security and failing tests. No hallucinated HUD, no fictional state. This is what makes it
both beautiful and trustworthy — and it is why a thread-and-diff tool structurally cannot
copy it: it requires owning the verified brain and the event stream it emits.

### RTS ↔ orchestration mapping (most of it already exists)
| RTS primitive | What it really is | Status |
|---|---|---|
| Map / territory | codebase (dirs = districts, files = buildings, imports = roads) | ✅ Cities |
| Fog of war | un-indexed / unexplored code | ✅ retrieval + Scout |
| Units | agents (archetype + loadout + status) | ◑ archetypes / Barracks |
| Production (Barracks) | recruit an agent | ◑ Phase C |
| **Resources** | token / $ budget | ✅ budget ledger |
| Orders / campaigns | tasks & objectives | ✅ CampaignDirector |
| Combat | red / blue security, bug-fighting | ✅ Raider / Defender / Arbiter |
| Rules of engagement | permissions / autonomy caps | ✅ permission gate |
| Real-time engine | live concurrent action | ✅ event-sourced WS |
| **Game state (the truth)** | the canonical event log | ✅ field-event-v1 |

Field is already **~70% an RTS engine**; the remaining work is *rendering* it as one and
finishing the plug-and-play substrate that makes the units real for a stranger.

## Where we do NOT compete

T3 Code and the broader category are funded, full-time teams **plus AI agents** (their own
insights list `claude` and `codex` as top committers) shipping enormous volume. **We do not
win on throughput and will not try.** We win on the axes volume cannot buy: a **verified
brain** (Knossos owns the agent loop; they rent theirs), **sovereign / local** models
(Cameo), a **native lightweight** shell, and the **RTS identity** nobody else is building.
Every decision below favors difference over output.

---

## 0. Where we are (baseline)

**Done / working:**
- Single-origin server (built SPA + API + WS on one origin), bootstrap-cookie auth,
  loopback bind, trusted-origin check.
- **Cities** (the map substrate): workspace = city, deploy/orders/chat-feed, level from the
  existing `workspaceMaturity`. Mid-session orders via `command('say')`. Server curl-verified.
- Rehearsal/simulation (synthetic cohorts, zero cost).
- Provider-agnostic endpoints (Anthropic via CLI creds, Cameo/OpenAI-compatible).
- A real **permission gate**: workspace-scope enforcement, sensitive-file policy,
  network-egress denial, per-session budget caps. Event log **redacts registered secrets**
  (`sanitizeEventData({secrets})` + `registerSecret`).
- **✅ Harness-adapter contract (landed 2026-09-19):** `HarnessAdapter` base + declarative
  manifest; `contracts/field-event-v1.schema.json` versioning the **14-kind** canonical
  vocabulary; both existing adapters (Claude Code, Knossos) refactored onto it and
  schema-tested. This is the foundation for pluggable agents/harnesses **and** the
  game-state source of truth the RTS renders from. Full Field suite green.

**Why it is not yet the product:**
- Ships with the **author's** `field.yaml` (personal repos + endpoints).
- Adding a **model** or **folder** still means hand-editing YAML; no in-app setup (Phase A/B in flight).
- No **first-run** guidance; no clean **default config**; no verified **download → run** for non-authors.
- The **RTS surface is not rendered yet** — the map exists as cities, but units/combat/resources are not a game view.
- No pluggable **new** harnesses beyond the two built-ins (the contract exists; ACP/Codex/Claude-direct adapters are not built).

---

## 1. Principles (hold across every phase)
- **Faithful world.** The map only ever renders real events (`field-event-v1`); never invent
  state. Every rendered element must trace to an event.
- **Ship no secrets, no author data.** A fresh download boots into an empty, safe default.
- **UI-only configuration.** Everything a downloader needs is reachable by button — never a
  file or a shell.
- **Secrets never touch git or the log.** User API keys go to a gitignored local store, are
  `registerSecret`'d for log redaction, and never enter `field.yaml` or the snapshot.
- **Safe by default.** `permission_mode: manual`, budget caps on, workspace scope enforced —
  the downloader opts *into* autonomy.
- **Additive & reversible.** New modules/routes/panels; existing behavior unchanged; each
  phase shippable and revertible on its own.

## 2. Definition of "plug and play"
A person who has never seen the repo can, from a downloaded artifact:
1. Start it (one command or launcher) and reach the UI.
2. Add a model (API key / Ollama / Cameo box), tested live, in the UI.
3. Open one of their own folders as a city (map region), in the UI.
4. Recruit a unit (archetype × power × loadout) and deploy it, in the UI.
5. **Command it on the map and watch it work — in real time** — without ever editing a file.
…with no secrets shipped, secrets stored safely, and safe execution defaults.

---

## Phase A — Power Sources (add a model; secrets-safe) — the gate  ·  ◑ in flight (~90%)

Nothing runs without a model, and this forces the secret design early.
*RTS lens: a power source is a unit's energy supply.*

**Code changes**
- **NEW `server/src/secrets.js`** (a.k.a. `keystore.js`) — a gitignored local secret store at
  `.field-state/secrets.json`. `get/set/delete/list(ids only)`. Values never serialized to snapshot or log.
- **NEW `server/src/endpoints-store.js`** — runtime endpoint descriptors (non-secret:
  `{id, name, kind, model, base_url, secretRef}`) persisted to `.field-state/endpoints.json`,
  **merged** with `field.yaml` endpoints at load. Keys live only as `secretRef` → secrets.js.
- **`server/src/api.js`** — gated routes: `GET/POST/DELETE /api/endpoints`,
  `POST /api/endpoints/test` (probe health). `kind ∈ {anthropic, openai-compatible/ollama, cameo}`.
- **`server/src/index.js`** — load endpoints = `field.yaml` ∪ endpoints-store; register all
  stored keys as secrets at boot.
- **Web** — **NEW `web/src/setup/PowerSources.jsx`**: add/test/list/remove a model; write-only key field.

**Runtime impact → mitigation**
| Impact | Mitigation |
|---|---|
| Health probes hit external hosts | Async, throttled, on-demand + on-add only; timeout-bounded; failures degrade one endpoint, never block boot. |
| Endpoint list in snapshot | Already present; add only non-secret fields. |
| Secret file I/O | Tiny JSON, read once at boot, written on change. |

**Security impact → mitigation**
| Threat | Mitigation |
|---|---|
| **Key leakage** | Keys only in `.field-state/secrets.json` (gitignored); `registerSecret` → redacted from event log; never in config, snapshot, or API responses (write-only field). |
| **SSRF** via user `base_url` | Validate scheme (http/https only); operator-trusted single-operator input; no server-side following of endpoint-returned redirects. |
| **Unauth key write** | New routes join the gated table (401 without cookie); same-origin check. |
| **Key at rest** | 0600 perms where supported; optional OS-keychain backend later. |

---

## Phase B — Folders → Cities (open any folder)  ·  ⬜
*RTS lens: claim new territory.*

**Code changes**
- **`server/src/api.js`** — `POST/DELETE /api/workspaces` (mount/unmount a path). Reuse
  `canonicalizeWorkspace`/`workspace-path.js`; assign a map region; persist to
  `.field-state/workspaces.json`, merged with `field.yaml`.
- **`server/src/index.js` / `config.js`** — load workspaces = `field.yaml` ∪ store; bounded fs watcher per path.
- **Web** — "Open folder" control (path entry; native picker where available) → new city on the map.

**Runtime → mitigation:** reuse ignore globs (`node_modules`, `.git`, `dist`, `target`); cap watched roots; debounce. O(1) mount stat.
**Security → mitigation:** `canonicalizeWorkspace` + sensitive-name policy; **confirm** on mounting a home/system root; agents bounded by the workspace-scope gate; gated route (loopback + cookie).

---

## Phase C — The Barracks (archetypes & agents)  ·  ⬜  — *this IS unit production*

**Code changes**
- **NEW `server/src/archetypes.js`** — role → archetype metadata (name, silhouette, flavor,
  default tool loadout): builder→Soldier, architect→Commander, verifier→Sentinel,
  explorer→Scout, red/blue/referee→Raider/Defender/Arbiter.
- **`server/src/api.js`** — `POST/DELETE /api/agents` (create `field/agents/<id>.md` via the
  existing `updateFrontmatterFile` pattern), extend `POST /api/agents/settings`.
- **Web** — **NEW `web/src/barracks/Barracks.jsx`**: recruit = archetype × power source ×
  loadout (tools, thinking, budget, emblem, name); edit/retire; deploy to a city.

**Runtime → mitigation:** small — roster in memory; md writes rare and tiny.
**Security → mitigation:** tool allowlist intersected with the role's allowlist (an agent can
never exceed its archetype's authority); writes confined to `field/agents/`; power is a
`secretRef`, never a key.

---

## Phase D — Guided first-run & clean defaults  ·  ⬜

**Code changes**
- **Ship a clean default `field/`** (roles/archetypes, constitutions, **no** author workspaces,
  endpoints, or secrets). Author's personal `field.yaml` stays out of the release.
- **`server/src/api.js`** — `GET /api/setup-state` (any endpoint? workspace? agent?) to drive the wizard.
- **Web** — **NEW `web/src/setup/Onboarding.jsx`**: empty-state wizard —
  *add a model → open a folder → recruit a unit → deploy → command* — each step deep-links to
  the Phase A/B/C panels; dismissible once complete.

**Runtime:** negligible. **Security:** first-run exposes nothing; ships with zero secrets.

---

## Phase E — Native lightweight shell (REVISED — supersedes "packaged Node web app")  ·  ⬜

*Was: verify the release zip → `npm ci` → `npm start`. Revised because the RTS needs a real,
fast, native surface — and because store access (Apple/Google) makes desktop **and** mobile reachable.*

**Code changes / work**
- **Tauri 2.x shell** (Rust backend) — **strangler-fig**: first wrap the current Node server as
  a bundled **sidecar**, reuse the built SPA as the webview frontend, swap browser+loopback
  bootstrap for Tauri IPC. No orchestration logic rewritten up front.
- **Installers** for Windows/macOS/Linux; later **iOS/Android** (Tauri 2 mobile), published
  under the org developer accounts.
- Keep the self-contained, no-secrets, checksummed **release artifact** discipline from the old
  Phase E (default `field/` config only; assert no `.field-state`, no keys, no author `field.yaml`).
- Launcher UX: print/expose the bootstrap URL; optionally auto-open.

**Runtime impact → mitigation**
| Impact | Mitigation |
|---|---|
| Native webview (not Chromium) | Lightweight by design — the whole point vs. Electron. |
| Sidecar ships a Node runtime (heavier) | Temporary; port field-core to Rust later to drop the sidecar (see "Later"). |
| Startup / bundle size | Lazy-load heavy panels; SPA already chunked. |

**Security impact → mitigation**
| Threat | Mitigation |
|---|---|
| Webview capability creep | Tauri allowlist/capability model restricts the webview to needed IPC only. |
| Accidental network exposure | Keep loopback semantics inside the shell; mobile reaches a remote Field over the operator's own private network (Headscale), not an exposed port. |
| Shipping a secret | Release build asserts no `.field-state`/keys/author config; checksum + provenance. |
| Supply-chain | `npm ci --omit=dev` from a locked manifest; publish checksums. |

**Migration risk → mitigation:** strangler-fig means a working native app ships over the
current core immediately; modules port to Rust incrementally behind stable interfaces;
feature-flagged; the green test suite gates every step.

---

## Phase F — Safety, cost & polish for non-experts  ·  continuous
- **Safe defaults** shipped: `permission_mode: manual`, conservative budget, workspace scope on.
- **Cost visibility**: the cost meter is the RTS **resource counter**; per-session/per-city budget UI.
- **Discoverability**: every action reachable by button; empty states everywhere.
- **Privacy/telemetry**: ship with none, or strictly opt-in and local-only.
- **Updates**: version check; `.field-state` migration policy.

---

## Phase G — Harness Adapters (pluggable agents/harnesses)  ·  foundation ✅ · breadth ⬜

The contract landed (baseline §0). This phase adds the actual pluggable agents and the UI to
manage them — the "implement your own agents/harnesses" capability.
*RTS lens: each harness is a different **house/faction** of units on the same map.*

**Code changes**
- **New adapters on the contract** (manifest + optional mapper each): **generic-ACP** (reuses
  Knossos's ACP conformance → *any* ACP agent plugs in free), **Claude-API-direct** (reuses the
  `knossos-rs` anthropic path), then **Codex / OpenCode / Cursor / Grok** CLI adapters.
- **Adapter conformance suite** — gates every adapter, including user-authored ones (mirrors the
  existing `conformance/` ACP suite).
- **User-defined adapters** — a manifest (+ optional module) dropped in a plugins dir; simple
  CLI/HTTP agents need only a manifest.
- **Web** — "add agent/harness" surface + adapter picker driven by `capabilities()`.
- **BYO-auth** — adapters use the user's own creds via the keystore; **no token reselling**.

**Runtime impact → mitigation:** each adapter spawns its own process → reuse the existing
process/registry management + budget caps; `capabilities()` bounds what a unit can do.
**Security impact → mitigation:** one **Field-level permission gate** over every adapter's
consequential actions (never trust an adapter's self-report for irreversible ops); conformance
gates the zoo; secrets via keystore only.

---

## Phase H — The RTS Surface (the world engine)  ·  ⬜  — *the identity, rendered*

Render the faithful world. A **new projection** over the canonical event log + a rendered,
commandable client. Additive; does not touch the brain.

**Code changes**
- **NEW world projection** (`server/src/store/` projection): map tiles from the repo tree +
  import graph; unit positions/actions from `session.*` events; resources from the budget
  ledger; structures/damage from `session.verification` events.
- **Rendered client** (canvas/WebGL in `web/`): map, units, selection, minimap, resource HUD,
  real-time updates over the existing WS (already coalesces 500-event bursts).
- **Command layer**: select units → issue orders (tasks) → path to target (file/module) → act;
  rally/patrol (continuous verification); attack-move (fix everything in an area). Maps onto
  CampaignDirector orders.
- **Mine existing art/mechanics**: the sibling `../RTS/` folder and prior `field/web` RTS work.

**Runtime impact → mitigation:** rendering a large repo can be heavy → level-of-detail,
virtualize off-screen tiles, read-only projection (no new authority), WS burst coalescing.
**Security impact → mitigation:** the surface is **read-only** over the event log and issues the
**same gated orders** as today — no new privileges. The **faithfulness rule** (every visual
traces to an event) means it cannot misrepresent what an agent actually did; a debug overlay can
reveal the event behind any element.

---

## Cross-cutting concerns
- **Faithfulness:** the world = a projection of `field-event-v1`. Never invent state.
- **Two "kind" axes, kept separate:** *harness kind* (`cli-stream-json`/`ndjson-serve`/`acp`/
  `http-api`/`mcp` — which agent runtime) vs *engine/endpoint kind* (`anthropic`/
  `openai-compatible`/`cameo` — which model). The manifest maps endpoint kind → harness adapter.
- **Secrets:** one store, gitignored, `registerSecret`-redacted, write-only in the API, never in
  config/snapshot.
- **Auth:** loopback + single-use bootstrap cookie locally; remote/mobile over the operator's own
  private network (Headscale), never an exposed port.
- **State layering:** `field/` (git, durable, non-secret) vs `.field-state/` (local, operational,
  secrets) — never cross them.
- **Reversibility:** each store merges *over* `field.yaml`; deleting `.field-state` returns a clean default.

## Sequencing & milestones (dependency order)
1. ✅ **Harness-adapter contract** (foundation for G and H).
2. **A — Power Sources** (in flight; the secret spine). ← finish next
3. **B — Folders → Cities** (open any folder).
4. **G (breadth) — first new adapters** (generic-ACP, Claude-direct) — makes "plug in any agent" real.
5. **C — Barracks** (unit production).
6. **D — Guided first-run + clean defaults** (ties A–C into a wizard).
7. **H — RTS Surface** (render the world; can start incrementally once units + resources are real).
8. **E — Native shell** (house the RTS; strangler-fig over the working core).
9. **F — Safety/cost/polish** (continuous; lock before public release).
10. **(Later)** Port field-core to Rust (drop the sidecar); **remote + mobile** over Headscale.

**Milestones**
- **Plug-and-play** (end of D + first adapters): a stranger downloads, adds a model, opens a
  folder, recruits and deploys a unit — entirely in the UI, no secret shipped or leaked.
- **The world comes alive** (H): the same actions rendered as a real-time RTS over faithful events.
- **Native & everywhere** (E + mobile): download the app and command your codebase from desktop
  and phone, over your own network — the thing the category can't match, on the axes we chose.


---

## Amendment 2026-09-20 — all-Rust runtime, ACP agents, reach from anywhere via Cameo

Canonical record: `PRODUCTIZATION_PLAN.md` in the Cameo repository (workstreams B, G, L
and the issue queue `KNS-*` / `FIELD-*`). This amendment changes three things in the
phases above and adds one; everything else stands.

### 1. "Port field-core to Rust" moves from *Later* to the spine (`KNS-RUST-001`)

The harness is all Rust: every process a user runs is the `knossos` binary. `knossos field`
today launches a Node sidecar; that sidecar is retired by a strangler port of
`server/src` into the crate behind the interfaces the SPA already speaks. The SPA and
`contracts/field-event-v1.schema.json` do not change, which is what keeps this a port and
not a rewrite of the product.

| Order | Moves | Parity oracle (existing tests, ported to Rust one by one) |
|---|---|---|
| 1 | event log (`node:sqlite` + JSONL backends) and the projection fold | `eventlog`, `replay`, `api-pagination`, `stress` (100,004 events) |
| 2 | API, bootstrap-cookie auth, origin/host/cookie/WebSocket gates, body caps | `security`, `body`, `browser-url`, `workspace-path`, `ws` |
| 3 | harness adapters: Knossos becomes in-process; ACP adapter; adapter contract | `adapter-contract`, `event-schema`, `acp-session`, `knossos-session`, `process-generation` |
| 4 | budget ledger, admission, registry, routines, campaign director, world projection | `budget-ledger`, `registry-resilience`, `routines`, `routine-queue-recovery`, `campaign`, `director`, `world` |

Rules: the Node server stays selectable by a flag until every ported test is green; each
step ships behind that flag; the `npm` dependency and `server/` are deleted only at full
parity. Phase E's "wrap the Node server as a Tauri sidecar" is therefore withdrawn: the
native shell, when built, wraps the Rust core directly.

**Impact → mitigation.** Large surface (~8k lines JS) → the test suite is the spec and is
ported first per step, never after. SQLite in Rust → `rusqlite` bundled (the crate already
compiles C for tree-sitter); the JSONL backend remains the honest fallback. Behaviour drift
→ both servers run the same suite until the flag flips.

### 2. Plug-and-play agents standardise on ACP (`FIELD-ACP-001`)

Phase G's first new adapter is **generic-ACP**. Knossos is an ACP agent; Claude Code, Codex,
Gemini CLI and a user's own agent that speaks ACP become units on the same map with no
adapter code, only a manifest. Manifest-only CLI/HTTP agents follow. Two non-negotiables:
every adapter, including user-authored ones, passes the adapter conformance suite; and one
Field-level permission gate decides every consequential action, never the adapter's own
report. Field's own terminal spawns a real shell outside that gate today; it goes under
Knossos policy or is removed, and Field launches every mission with persistence on
(`FIELD-POLICY-001`).

### 3. "Reach it from anywhere" is built on the Cameo origin (`FIELD-REMOTE-001`)

The Cameo box already serves HTTPS with a self-signed certificate and a console key.
`cameod` reverse-proxies `/field/` to `knossos field` on loopback, so Field shares that
certificate, key and origin, and the console links to it. Off the LAN, access is the
operator's **own private network**: Cameo ships `wireguard-tools`, and `cameo remote` mints
a peer configuration and a QR code for a phone; Headscale/Tailscale stay optional. Field
ships a PWA manifest so it installs on a phone like an app, and because the event log is
the session store, any device resumes the same operation. No public relay, no hosted
accounts, no token reselling: that is the line that keeps this "sovereign" and is why it
is not a T3 clone.

Installed by `cameo knossos install`; runs as a systemd unit under the operator account,
not the `cameo` daemon user, because units edit the operator's workspaces.

### 4. Retire Python from the product path (`KNS-PY-001`)

`model/` keeps Daedalus training and trace tooling under a research marker outside the CI
gate. The Rust evaluator and the frozen suite lock become the only grading truth; the
"faithful port of codeval.py" dependency is removed by making the Rust suite the reference.

### Revised sequencing

1. ✅ Harness-adapter contract.
2. Knossos CI truth (`KNS-CI-001`) and the runtime P0 (`KNS-VERIFY-001`): a mission may
   never report success on a reused quick verify.
3. **A — Power Sources** (finish), then **B — Folders → Cities**.
4. **Rust port step 1–2** (event log, projection, API, gates) — the SPA keeps working
   against either server.
5. **G breadth — generic-ACP adapter** + conformance; **C — Barracks**.
6. **Rust port step 3–4**; Node server deleted at parity; `KNS-PY-001` alongside.
7. **D — first-run wizard**; **Cameo origin + `cameo remote` + PWA**.
8. **H — RTS surface** (can start earlier on the web side: it is a projection over an
   unchanged event schema).
9. **E — native shell** over the Rust core (no sidecar).
10. **F** continuous.
