# Knossos Field

A real multi-agent operating environment, controlled with the interaction language of a
real-time strategy game.

Field is not a visualization of agents. Every unit on the canvas is a live harness
process (`claude` for Anthropic endpoints, Knossos for Cameo and other
OpenAI-compatible endpoints). Every region is a directory that exists on disk. Every route is a tool call that
actually happened. Right-clicking a folder and assigning four agents to it sends real
orders into four real harness sessions, and the cost meter at the bottom of the screen is
money that has actually been spent.

Nothing on the Field moves unless something real moved.

---

## Running it

```bash
cd field
npm install
```

Start the server (loads `field/`, replays the event log, opens the operator interface):

```bash
npm run server
```

Then either open the built interface at `http://127.0.0.1:7749`, or run the dev server
with hot reload:

```bash
npm run web
```

The dev interface is on `http://127.0.0.1:7748` and proxies `/api` and `/ws` to the
server. To rebuild the static interface the server serves itself:

```bash
npm run build
```

Tagged Knossos releases also publish a `knossos-field-<version>.zip`. Extract it,
run `npm ci --omit=dev`, then `npm start`; the archive contains the built interface,
server, default Field configuration, license, checksum, and provenance register.
If the Knossos binary is installed, `knossos field --dir <extracted-path>` launches
the same server and wires Field-managed sessions back to that exact binary.

Run the tests. They cover every campaign transition, replay paging, campaign gates,
red/blue/referee independence, scope policy, provider failure, budget stops, graph lifecycle,
the Knossos adapter, WebSocket burst/reconnect behavior, filesystem attribution, and a
100,000-event / 30-agent / 10,000-node stress replay:

```bash
npm test
```

**Requirements.** Node 20+ (Node 22+ uses the built-in SQLite; older versions fall back to
a JSONL log with identical semantics) and the `claude` CLI on `PATH` for Anthropic-backed
units. Cameo/OpenAI-compatible units use `knossos serve`; set `FIELD_KNOSSOS_BIN` when
`knossos` is not on `PATH` (development also discovers a sibling/submodule build). The server binds to
loopback only — it can read and write real files and start real processes, so it is an
operator-local interface, not a public API.

---

## The five modes

| Mode | Key | What it is |
|---|---|---|
| **Field** | `F` | Spatial command surface. Every running agent, workspace, website, and endpoint. |
| **Campaign** | `C` | Objectives, formations, red challenges, mitigations, referee gates, and promotion. |
| **Workspace** | `W` | Real editor, file tree, git diff, Markdown viewer, terminal, and browser panes. |
| **Routines** | `R` | Persistent and scheduled work — triggers, roles, endpoints, budgets, completion. |
| **Traces** | `T` | Replayable history of every message, tool call, approval, and routing decision. |

---

## Control reference

Atlas (default) is a parallel board: one column per live agent, grouped by workspace.
Rome is the operations map. Selection and command on the map follow RTS convention.

| Input | Action |
|---|---|
| Click agent | Select it |
| Drag on empty space | Marquee-select agents |
| Shift-click | Add / remove from selection |
| `Ctrl`+`1`–`9` | Assign selection to a control group |
| `1`–`9` | Recall control group |
| Right-click a target | Contextual orders for the current selection |
| `Alt`-drag / middle-drag | Pan the camera |
| Scroll | Zoom on the cursor |
| `Alt` (hold) | Reveal every agent's name |
| `F` | Frame the selection, or the whole field |
| `Ctrl`+`A` | Select all agents |
| `Esc` | Clear selection and close menus |
| Click the minimap | Jump the camera |

Right-clicking a **folder, file, workspace, mission, or website** offers *Assign N agents
here*, which opens the order form: endpoint (auto-routed by health, or a specific one),
thinking level (`low` / `medium` / `high` / `adaptive`), and free-text orders. Field
composes the target context into the orders itself — assigning agents to a Markdown file
tells them to read it and execute the plan it contains.

Selected agents can be paused, resumed, cancelled, verified, or escalated from the HUD.
Use `/?mode=campaigns` to open directly into Campaign command.

---

## Campaigns and red/blue orchestration

A campaign is a replayable operation, not a group of chat tabs. Blue builds or operates;
red tries to disprove readiness inside a declared environment scope; an independent referee
reproduces the evidence; purple can synthesize doctrine after the verdict. High and critical
findings block promotion until they are acknowledged, mitigated, retested, and independently
verified.

The enforced lifecycle is:

`draft -> mobilizing -> blue building -> red challenging -> contested -> blue mitigating -> red retesting -> referee review -> verified -> promoted`

Commands are idempotent. Findings require evidence and reproduction. Blue cannot clear its
own finding, red cannot promote its conclusion, and a referee cannot also serve on blue or
red in the same campaign. Promotion creates a checkpoint and stable capability; rollback
selects an earlier checkpoint without deleting the failed history.

Campaign formations are batch controls. One dispatch sends an objective-scoped order to all
live members of a team; each formation can be reinforced, and individual units can retreat.
Clicking a unit opens its prompt, transcript, workspace, current objective, and progress.
The History control replays the operation at any event through the same server reducer used
at boot, locks all mutation controls, and identifies the checkpoint baseline at that moment.

Live, unavailable, and explicitly external members are different states. External actors are
useful for deterministic simulations and manual evidence entry, but are never counted as
running processes or eligible for batch dispatch. Restarted harness sessions become
unavailable rather than lingering as ghost units.

Harness final messages can report evidence by ending with `FIELD_REPORT:` and one JSON
object. The director validates team, phase, scope, evidence, and references before accepting
it. Ordinary prose and unsentinelled JSON are ignored.

Read-only challenger and referee roles do not auto-approve shell use. The permission policy
hard-denies direct writes, path escape, production-readonly mutation, and mutating shell
patterns before operator review. Persisted events redact credential-shaped fields, recognized
credential strings, and the actual values of credential environment variables.

See [the full campaign implementation plan](docs/red-blue-orchestration-plan.md).

---

## What each thing on the canvas is

- **Unit markers** are live harness sessions. Silhouette is the role — diamond `architect`,
  square `builder`, triangle `scout`, hexagon `verifier`, circle `archivist`. Fill is state.
  The bars above each unit are real context usage and, when the agent has a plan, real
  progress.
- **Regions** are mounted workspaces, labelled with their real git branch and change count.
- **Folder chips** are directories agents have actually touched. Emphasis is heat — a decay
  function over real filesystem and tool events. Cold folders disappear; a busy region caps
  its chips and reports the rest as `+N quieter`, because a wall of chips is not information.
- **Routes** connect an agent to its work, to a website it is browsing, or to a child
  session it delegated to. They brighten when a real event arrives on that session and fade
  over about a second. There is no idle animation anywhere in the renderer.
- **The rail** is real endpoint health, with real probe latency and the number of live
  sessions on each endpoint. Several agents can share one endpoint while keeping isolated
  context.
- **An amber pulse** means a real process is blocked waiting for you. That is the one thing
  in the interface that animates on its own, because it is a live request.

---

## Persistence

Git and the database have different jobs, and the split is deliberate.

**Git** holds the durable, human-readable civilization in `field/` — agent definitions,
roles, constitutions, skills, mission templates, routines, Markdown memory, and workspace
configuration. It is meant to be read, reviewed, and diffed by people.

**The database** (`.field-state/`) holds live operational state as an **append-only event
log**. Sessions, tool calls, assignments, endpoint health, approvals, costs, context usage,
UI positions, and control groups are all events. Every live view is a fold over that log,
and Traces replays the same events through the same fold — so what you watch live and what
you replay later cannot drift apart.

Operational state is never `UPDATE`d. See [docs/data-model.md](docs/data-model.md).

---

## Layout

```
field/                 civilization config, human-readable, lives in Git
  field.yaml           workspaces, endpoints, websites, defaults
  constitutions/       rules every agent inherits
  roles/               behavior and real tool restrictions per role
  agents/              the roster
  missions/            mission templates with definitions of done
  routines/            scheduled and triggered work
  skills/  memory/     reusable procedures, durable facts

server/src/
  index.js             boot: load config, replay log, open the interface
  config.js            loads field/, composes system prompts
  glob.js              glob compilation for ignore lists and triggers
  store/               event log (SQLite or JSONL) and the projection fold
  harness/             Claude + Knossos sessions, tool translation, permission gate
  orchestration/       campaign state machine, director, typed temporal graph
  watch/               filesystem and git observation
  endpoints.js         endpoint probes
  routines.js          cron and filesystem triggers
  api.js  ws.js        REST surface and the operator socket

web/src/
  field/               canvas renderer, spatial layout, RTS input
  hud/                 selection HUD, context menu, endpoint rail, approvals
  workspace/           editor, tree, diff, transcript, terminal, browser
  campaigns/           strategic contest and promotion command surface
  routines/  traces/   the other two modes
```

See [docs/design.md](docs/design.md) for why it is built this way.
