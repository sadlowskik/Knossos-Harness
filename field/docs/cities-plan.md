# Field Cities — Implementation Plan

Turn each workspace into a **city** you can grow: send agents to a city, watch the city
level up as real work lands, and command the city through a chat panel.

Status: proposal. Nothing here is built yet.

---

## 1. Mechanics (what the operator gets)

1. **Send an agent to a city.** Click a city on the Rome map → *Deploy agent here* → a real
   harness session starts, scoped to that workspace. No right-click required.
2. **The city levels up.** As real work lands in that workspace (sessions completed, work
   verified, commits), the city gains XP and ranks: `outpost → town → city → capital`. The
   citadel visibly grows.
3. **City-level chat.** Clicking a city opens a panel with the city's agent roster, a feed of
   what those agents are reporting, and a compose box to **send new orders to the city** —
   redirect it, add context, or kick off new work.

---

## 2. Design principles (constraints this plan holds to)

- **City = workspace.** A city is exactly one mounted workspace (`cfg.workspaces[i]`). No new
  identity space; a city id **is** a workspace id. This means all city inputs are validated
  against the existing workspace allowlist — never against arbitrary paths.
- **Derive, don't persist config.** `config.js` states operational state lives in the event
  log / DB, never written back to `field/`. City level/XP are **derived from the event log**,
  recomputed on load and folded forward incrementally — no new writes to `field.yaml`.
- **Additive and reversible.** Every change is a new file, a new route, or a new UI panel.
  No existing route, registry method, or event schema changes meaning. Revert = delete the
  new files and remove the new route keys.
- **Reuse the existing security spine.** New endpoints join the same gated route table
  (loopback bind + bootstrap-cookie auth + trusted-origin check), the same permission gate,
  the same role tool allowlists, and the same rate-limit pattern already used by
  `functions/api/hardware-reports.js`. No new auth path, no new CORS.

---

## 3. Data model

### City (a workspace) — no new stored entity
A city id is a workspace id. Everything else is **derived**.

### `CityState` (computed, not stored)
```
{
  id,            // workspace id
  name,          // workspace name
  tier,          // 'outpost' | 'town' | 'city' | 'capital'
  level,         // integer
  xp,            // integer, derived
  nextLevelXp,   // threshold for next tier
  agents: [ { sessionId, role, name, status } ],   // live sessions in this workspace
  feed:   [ { at, sessionId, kind, text } ],        // recent reports/orders, capped
  stats:  { sessionsCompleted, verified, commits }
}
```

### Leveling formula (pure function, bounded)
```
xp = 10 * sessionsCompleted
   + 25 * verifiedTasks
   +  5 * commitsInWorkspace
tier = capital (xp>=1000) | city (>=300) | town (>=75) | outpost (else)
```
All inputs are counts already tracked in `projection`; the function is O(1) over cached
counters (see §5). No floating point, no unbounded growth per event.

---

## 4. Code changes

### Server

**NEW `server/src/cities.js`**
- `deriveCityState(projection, cfg, workspaceId)` → `CityState`. Reads only in-memory
  projection (sessions, assignments, per-workspace counters) — never the raw event log at
  request time.
- `cityLevel(xp)` → `{ tier, level, nextLevelXp }`. Pure.
- `cityFeed(projection, workspaceId, limit=50)` → last N report/order entries for the city.

**`server/src/projection.js` (store/projection.js) — additive counters**
- In `apply(evt)`, when an event carries a `workspaceId`, increment per-workspace counters
  (`sessionsCompleted`, `verified`, `commits`) held in a new `projection.cities` Map.
  This makes XP O(1) at read time and correct after event-log replay (replay already runs
  through `apply`). No new event types required.

**`server/src/api.js` — new gated routes (join the existing `routes` table)**

The route table is **exact-match** (`routes['GET /api/…']`), and the codebase's convention
for parameters is **query string / body**, not path params (see `events` = `?from=&limit=`,
`trace` = `?subject=`). So cities follow that convention:
- `GET  /api/cities` → `[CityState]` for all workspaces (compact; no full feed).
- `GET  /api/city?id=<ws>` → one `CityState` **with** feed. `id` validated to be a known
  workspace id; unknown → `DomainError('not_found')`.
- `POST /api/city/deploy` `{ id, role?, endpoint?, thinking?, orders? }` → thin wrapper over
  the **existing** `registry.spawn`, forcing `workspaceId = id`. Reuses all existing spawn
  validation, budget, and session caps.
- `POST /api/city/orders` `{ id, text }` → resolve active sessions in the workspace →
  `registry.command('say', {sessionIds, text})`; if none active, `registry.spawn` with
  `text` as orders. Rate-limited and bounded.

**`server/src/registry.js` — no new method needed (dependency resolved)**
- Mid-session injection **is supported** and already surfaced. `HarnessSession.send(text)`
  (`session.js:147`) pushes a real user turn into the running Claude CLI over its
  `stream-json` stdin; `KnossosSession.send(text)` (`knossos-session.js:136`) does the same
  over the Knossos `task`/`resume` channel. The registry already exposes this as the
  **`say`** command (`registry.js:737`: `sessions.get(id).send(text)`), budget-gated via
  `assertBudgetAllowsWork`, and **`redirect`** re-targets sessions via `assign`.
- So **city orders reuse the existing command path**: resolve the city's **active** session
  ids, then call `registry.command('say', { sessionIds, text })`. If the city has **no**
  active session, fall back to `registry.spawn(...)` with `text` as orders. No new registry
  method, no new event type.

**`server/src/index.js` — snapshot**
- Include a **compact** per-city summary (`{id, tier, level, xp, agentCount}`) in the WS
  snapshot so the map can render level badges. The **full feed is fetched on demand** via
  `GET /api/cities/:id`, never broadcast, to keep snapshot size flat.

### Web

**NEW `web/src/city/CityPanel.jsx`** (lazy-loaded, like `WorkspaceMode`)
- Header: city name, tier, level, XP bar (`xp / nextLevelXp`).
- Roster: live agents in the city (role silhouette, name, status).
- Feed: recent reports/orders (from `GET /api/cities/:id`), capped, auto-scrolling.
- Compose box: textarea + Send → `api.cityOrder(id, text)`.
- `Deploy agent here` button → `api.deployToCity(id, {...})`. **No right-click.**

**`web/src/theater/TheaterMode.jsx`**
- On city/capital click, open `CityPanel` for that workspace (a new panel state next to the
  existing agent/region selection). Render a **level badge/tier ring** on each capital from
  the snapshot city summary.

**`web/src/net/client.js` — new methods**
```
cities:       () => request('GET', '/api/cities'),
city:         (id) => request('GET', `/api/city?id=${encodeURIComponent(id)}`),
cityOrder:    (id, text) => request('POST', '/api/city/orders', { id, text }),
deployToCity: (id, body) => request('POST', '/api/city/deploy', { id, ...body }),
```

**`web/src/state/store.js`** — hold `cities` summaries from the snapshot.

**`web/src/field/renderer.js` + `layout.js`** — draw the tier ring / growth on the citadel.

**`web/src/styles/app.css`** — city panel + level badge styles (namespaced `.city-*`).

### Config / migrations
- **None.** No `field.yaml` change, no migration, no write-back. Cities are derived.

---

## 5. Runtime impact & mitigation

| Change | Impact | Mitigation |
|---|---|---|
| Per-workspace counters in `projection.apply` | +1 Map lookup + increment per event | O(1); Map keyed by workspace id; bounded by workspace count (small). Counts are integers. |
| `GET /api/cities*` | Request-time work | Read cached counters only; never scan the raw event log at request time. |
| Feed aggregation | Could be O(events) if naive | Keep a **capped ring buffer** (last N per city) updated in `apply`; read is O(N), N≈50. |
| City summary in WS snapshot | Larger snapshot on every broadcast | Include only 5 small fields per city; full feed is REST-on-demand, never broadcast. |
| Level recompute after replay | Event-log replay cost | Folded into existing `apply` during the boot replay that already runs — no separate scan. |
| Deploy/orders spawning agents | Resource/cost blowup if abused | Enforce existing `max_concurrent_sessions`, `max_children_per_session`, `budget_usd_per_session`; dedupe; rate-limit orders (see security). |
| New web panel | Bundle size | **Lazy `import()`** the panel (matches `WorkspaceMode` chunking); ~small delta, not in initial load. |
| Feed auto-scroll / re-render | UI churn on busy city | Virtualize/cap feed to last N; throttle snapshot-driven re-renders. |

## 6. Security impact & mitigation

New surface = two write endpoints that can trigger **real agent execution and real
filesystem/command tool use**. Treated accordingly.

| Surface / threat | Mitigation |
|---|---|
| **Unauthorized access / CSRF** on new routes | Register in the **existing gated `routes` table** in `api.js` so they inherit loopback bind, bootstrap-cookie auth, and the trusted-origin check. No separate handler, no new CORS, no unauthenticated path. |
| **Arbitrary target injection** (`:id`) | `:id` must resolve to a known `cfg.workspaces` id or the route returns `not_found`. City ids are never paths; no path is accepted from the client, so no traversal. |
| **Path traversal on deploy** | Deploy reuses `canonicalizeWorkspace` / `workspace-path.js` validation via the existing `registry.spawn`/`assign`; it introduces **no new path input**. |
| **Order text abuse** (oversized, control chars, prompt injection) | Bound `text` (≤ 8 KB) and strip control chars using the existing helpers (`boundedString` pattern from `hardware-reports.js` / `body.js`). Prompt-injection risk is the operator injecting into *their own* agents on a single-operator loopback UI — no cross-tenant exposure — but text is still capped and never executed as a shell string. |
| **Execution amplification** (orders → real tools) | Orders/deploy route through the **existing permission gate** and **role tool allowlists**; they cannot widen an agent's authority or bypass `permission_mode` (`manual`/`acceptEdits`/`auto`). |
| **DoS via order/deploy spam** | Rate-limit both endpoints per connection using the **same D1-backed limiter pattern** as `hardware-reports.js` (`applyRateLimit`); enforce session/budget caps so a flood cannot spawn unbounded agents or cost. |
| **Feed data exposure** | Feed only aggregates transcripts the operator can already open; served only to the loopback-authed operator. No new data leaves the box; nothing added to logs the operator can't already see. |
| **Cross-session message injection** (`messageCity`) | Deliver only to sessions whose `workspaceId === :id`; validate membership before delivery; never accept a raw `sessionId` from the client for city orders. |
| **Leveling manipulation / overflow** | XP is a bounded integer function of counts; no client input feeds XP; recompute is deterministic from the event log. |

## 7. Blast radius & rollback

- **Additive only.** No existing route, event schema, or registry method changes semantics.
- **Feature-flag.** Gate the City panel behind a settings toggle (default on for the
  loopback operator, off-switch available), and an env `FIELD_CITIES=0` to disable the
  routes entirely. A bug in cities cannot break Field's core: the map, existing dispatch,
  and rehearsals are untouched.
- **No durable side effects.** Cities write no config and (option B) emit no new event
  types; worst case a bad deploy is a normal cancelable session.
- **Revert.** Delete `cities.js` + `CityPanel.jsx`, remove the four route keys and four
  client methods, drop the projection counters. Clean removal, no migration to undo.

## 8. Testing (mirror `server/test/*`)

- `cities.test.mjs`: XP/tier derivation from a synthetic event stream; feed capping;
  unknown-workspace → not_found; deploy forces the right `workspaceId`.
- Order endpoint: oversized/control-char rejection; rate-limit trips at the cap;
  membership check on delivery.
- Replay: counters reconstructed correctly after a 10k-event replay (extend the existing
  stress replay test).

## 9. Sequencing

- **Phase 1 — command loop (ship first):** city click → `CityPanel` with roster, feed,
  compose (orders), and Deploy button; server routes `GET /api/cities`, `GET /api/cities/:id`,
  `POST .../deploy`, `POST .../orders` (option B delivery); rate-limit + validation. This
  delivers "send agent to city + see chat + tell it stuff."
- **Phase 2 — leveling:** projection counters, XP/tier, snapshot summary, citadel growth
  visual, `cities.test.mjs`.
Live mid-session orders ship **in Phase 1** — the harness supports it today (see §10), so
there is no separate later phase for it.

## 10. Dependency check — RESOLVED ✅

Verified in code: `server/src/harness/session.js:147` (`send()` → real user turn into the
Claude CLI stream-json stdin) and `knossos-session.js:136` (Knossos `task`/`resume`), both
already surfaced through `registry.command('say' | 'redirect', …)` at `registry.js:737`,
budget-gated. **Mid-session injection is available now**, so city orders reach live agents
in Phase 1 with no new registry code and no protocol change.
