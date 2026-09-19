# Field — Plug-and-Play Roadmap

From today's state to a product a stranger can download, run, and use on their own code with
their own models — no YAML, no code, no shell surgery. Each phase lists the code changes, the
runtime impact, the security impact, and how each is mitigated.

Companion docs: [`cities-plan.md`](cities-plan.md) (Phase 0, built).

---

## 0. Where we are (baseline)

**Done / working:**
- Single-origin server on 7749 (built SPA + API + WS one origin), bootstrap-cookie auth,
  loopback bind, trusted-origin check.
- **Cities** (built today): workspace = city, deploy/orders/chat-feed, level from the
  existing `workspaceMaturity`. Mid-session orders via `command('say')`. Server curl-verified.
- Rehearsal/simulation (synthetic cohorts, zero cost).
- Provider-agnostic endpoints (Anthropic via CLI creds, Cameo/OpenAI-compatible).
- A real **permission gate**: workspace-scope enforcement, sensitive-file policy, network-egress
  denial, per-session budget caps. Event log already **redacts registered secrets**
  (`sanitizeEventData({secrets})` + `registerSecret`).

**Why it is not yet plug-and-play:**
- Ships with the **author's** `field.yaml` (personal repos + endpoints).
- Adding a **model** or **folder** means hand-editing YAML; no in-app setup.
- No way to paste an **API key** / add **Ollama** / add a **Cameo box** from the UI.
- No **first-run** guidance; empty board with no "start here."
- No clean **default config** and no verified **download → run** packaging story for non-authors.

---

## 1. Principles (hold across every phase)
- **Ship no secrets, no author data.** A fresh download boots into an empty, safe default.
- **UI-only configuration.** Everything a downloader needs is reachable by button — never a
  file or a shell (right-click is never the only path).
- **Secrets never touch git or the log.** User API keys go to a gitignored local store, are
  `registerSecret`'d for log redaction, and never enter `field.yaml` or the snapshot.
- **Safe by default.** `permission_mode: manual`, budget caps on, workspace scope enforced —
  the downloader has to *opt into* autonomy, not discover it by accident.
- **Additive & reversible.** New modules/routes/panels; existing behavior unchanged; each
  phase shippable and revertible on its own.

## 2. Definition of "plug and play"
A person who has never seen the repo can, from a downloaded artifact:
1. Start it (one command or launcher) and reach the UI.
2. Add a model (API key / Ollama / Cameo box), tested live, in the UI.
3. Open one of their own folders as a city, in the UI.
4. Recruit an agent (archetype × power × loadout) and deploy it, in the UI.
5. See it work and command it — without ever editing a file.
…with no secrets shipped, secrets stored safely, and safe execution defaults.

---

## Phase A — Power Sources (add a model; secrets-safe) — the gate

Nothing runs without a model, and this forces the secret design early.

**Code changes**
- **NEW `server/src/secrets.js`** — a gitignored local secret store at
  `.field-state/secrets.json` (dir already gitignored; created 0600 where the OS allows).
  `get/set/delete/list(ids only)`. Values never serialized to snapshot or log.
- **NEW `server/src/endpoints-store.js`** — runtime endpoint descriptors (non-secret:
  `{id, name, kind, model, base_url, secretRef}`) persisted to `.field-state/endpoints.json`,
  **merged** with `field.yaml` endpoints at load. Keys live only as `secretRef` → secrets.js.
- **`server/src/api.js`** — gated routes: `GET /api/endpoints`, `POST /api/endpoints`
  (add: kind ∈ {anthropic, openai-compatible/ollama, cameo}; store key via secrets.js;
  `registerSecret`), `POST /api/endpoints/test` (probe health), `DELETE /api/endpoints`.
- **`server/src/index.js`** — load endpoints = `field.yaml` ∪ endpoints-store; register all
  stored keys as secrets at boot.
- **Web** — **NEW `web/src/setup/PowerSources.jsx`**: add/test/list/remove a model; a key
  field that is write-only (shows `••••`, never re-fetched). Entry from a top-bar button.

**Runtime impact → mitigation**
| Impact | Mitigation |
|---|---|
| Health probes hit external hosts | Async, throttled, on-demand + on-add only; timeout-bounded; failures degrade one endpoint, never block boot. |
| Endpoint list in snapshot | Already present (`endpoints`); add only non-secret fields. |
| Secret file I/O | Tiny JSON, read once at boot, written on change. |

**Security impact → mitigation**
| Threat | Mitigation |
|---|---|
| **Key leakage** (log, snapshot, git, error text) | Keys only in `.field-state/secrets.json` (gitignored); `registerSecret` → redacted from event log via existing `sanitizeEventData`; never in `field.yaml`, snapshot, or API responses (write-only field). |
| **SSRF** via user `base_url` (server/harness calls it) | Validate scheme (http/https only); it is the operator's own single-operator choice on their machine; document that base URLs are trusted operator input; no server-side following of endpoint-returned redirects to new hosts. |
| **Unauth key write** | New routes join the gated table (401 without cookie); same-origin check. |
| **Key on disk at rest** | 0600 perms where supported; documented; optional OS-keychain backend later. |

---

## Phase B — Folders → Cities (open any folder)

**Code changes**
- **`server/src/api.js`** — `POST /api/workspaces` (mount a path), `DELETE /api/workspaces`.
  Reuse `canonicalizeWorkspace`/`workspace-path.js` for validation; assign a map region;
  persist to `.field-state/workspaces.json`, merged with `field.yaml` at load.
- **`server/src/index.js` / `config.js`** — load workspaces = `field.yaml` ∪ store; start a
  bounded fs watcher for each mounted path.
- **Web** — "Open folder" control (path entry; native picker where available) → new city
  appears on the map.

**Runtime impact → mitigation**
| Impact | Mitigation |
|---|---|
| fs watchers per workspace | Reuse existing ignore globs (`node_modules`, `.git`, `dist`, `target`); cap watched roots; debounce. |
| Mount stat check | O(1) stat; refuse unmountable paths (existing behavior). |

**Security impact → mitigation**
| Threat | Mitigation |
|---|---|
| **Path traversal / mounting system dirs** | `canonicalizeWorkspace` + sensitive-name policy; reject non-directories; **confirm** on mounting a home/system root; agents still bounded by the workspace-scope permission gate. |
| **Unauth mount** | Gated route; loopback + cookie. |

---

## Phase C — The Barracks (archetypes & agents)

**Code changes**
- **NEW `server/src/archetypes.js`** — role → archetype metadata (name, silhouette, flavor,
  default tool loadout): builder→Soldier, architect→Commander, verifier→Sentinel,
  explorer→Scout, red/blue/referee→Raider/Defender/Arbiter.
- **`server/src/api.js`** — `POST /api/agents` (create `field/agents/<id>.md` via the
  existing `updateFrontmatterFile` pattern), extend `POST /api/agents/settings`,
  `DELETE /api/agents`.
- **Web** — **NEW `web/src/barracks/Barracks.jsx`**: recruit = archetype × power source ×
  loadout (tools, thinking, budget, emblem, name); edit/retire; deploy to a city.

**Runtime impact → mitigation:** small — roster in memory; md writes are rare and tiny.

**Security impact → mitigation**
| Threat | Mitigation |
|---|---|
| **Tool-authority escalation** | Tool allowlist intersected with the role's allowlist (already enforced in `agents/settings`); an agent can never exceed its archetype's authority. |
| **Path safety writing agent files** | Writes confined to `field/agents/`; id sanitized; no secrets in agent files (power is a `secretRef`, not a key). |

---

## Phase D — Guided first-run & clean defaults

**Code changes**
- **Ship a clean default `field/`** (roles/archetypes, constitutions, **no** author workspaces,
  **no** endpoints, **no** secrets). Author's personal `field.yaml` stays out of the release.
- **`server/src/api.js`** — `GET /api/setup-state` (has any endpoint? any workspace? any
  agent?) to drive the wizard.
- **Web** — **NEW `web/src/setup/Onboarding.jsx`**: empty-state wizard —
  *add a model → open a folder → recruit an agent → deploy* — each step deep-links to
  Phase A/B/C panels; dismissible once complete.

**Runtime impact:** negligible. **Security:** first-run exposes nothing; ships with zero secrets.

---

## Phase E — Packaging & distribution (download → run)

**Code changes / work**
- Verify the release path in the README (`knossos-field-<version>.zip` → `npm ci --omit=dev`
  → `npm start`) is **self-contained**: built SPA + server + **default** field config + license
  + checksum + provenance. Serves **same-origin on one port** (downloaders never see the
  7748/7749 dev split).
- **Launcher UX**: `npm start` (or `knossos field`) prints the bootstrap URL prominently and
  **optionally auto-opens** the browser to it.
- **Cross-platform** smoke: Windows/macOS/Linux (win32 branches already exist for shell/terminal).
- **End-user README**: 3-step quickstart, safety notes, how to add a model/folder.

**Runtime impact → mitigation**
| Impact | Mitigation |
|---|---|
| Startup time / bundle size | Lazy-load heavy panels (Workspace, Barracks); the built SPA is already chunked. |
| Event-log growth over long use | Document `.field-state` reset; add a log-compaction/prune command later if needed. |

**Security impact → mitigation**
| Threat | Mitigation |
|---|---|
| **Accidental network exposure** | Keep loopback bind (127.0.0.1); README warns against port-forwarding; bootstrap token required even on localhost. |
| **Shipping a secret by mistake** | Release build asserts no `.field-state`, no keys, no author `field.yaml`; checksum + provenance already in the artifact. |
| **Supply-chain** | `npm ci --omit=dev` from a locked manifest; publish checksums (already done for Cameo ISOs). |

---

## Phase F — Safety, cost & polish for non-experts

- **Safe defaults** shipped: `permission_mode: manual`, conservative `budget_usd_per_session`,
  workspace scope on. The user opts into autonomy.
- **Cost visibility**: the existing cost meter is prominent; a per-session/per-city budget UI.
- **Discoverability**: every action reachable by button (no right-click-only paths — the exact
  gap we hit); empty states everywhere.
- **Privacy/telemetry**: ship with **none**, or strictly opt-in and local-only.
- **Updates**: version check; `.field-state` migration policy.

---

## Cross-cutting concerns
- **Secrets:** one store (`secrets.js`), gitignored, `registerSecret`-redacted, write-only in
  the API, never in config/snapshot. Every phase that adds a key routes through it.
- **Auth:** loopback + single-use bootstrap cookie stays; fine for local first-run.
- **State layering:** `field/` (git, durable, non-secret) vs `.field-state/` (local, operational,
  secrets) — never cross them.
- **Reversibility:** each store merges *over* `field.yaml`; deleting `.field-state` returns a
  clean default.

## Sequencing & milestones
1. **A — Power Sources** (gate; establishes the secret spine). ← recommended first
2. **B — Folders → Cities** (open any folder).
3. **C — Barracks** (archetypes/agents).
4. **D — Guided first-run + clean defaults** (ties A–C into a wizard).
5. **E — Packaging** (download → run, same-origin, no secrets shipped).
6. **F — Safety/cost/polish** (continuous; lock before public release).

**Milestone = plug-and-play:** end of Phase E, a stranger downloads, runs, adds a model, opens
a folder, recruits and deploys an agent — entirely in the UI, with no secret shipped and none
leaked. Phase F hardens it for a public launch.
