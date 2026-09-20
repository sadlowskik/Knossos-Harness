# Field — design

## The problem

Operating one agent is a chat window. Operating twenty is a command problem, and chat is
the wrong instrument for it. You cannot see which agents are stuck, which are burning
budget, which are waiting on you, or which are working on the same file as each other.
Tabs do not scale, and a dashboard of cards tells you the count of things without telling
you the shape of the situation.

RTS interfaces solved this specific problem: many autonomous units, continuous partial
attention, and a need to reassign work in one gesture. Field borrows that grammar —
selection, control groups, right-click orders, formations, a minimap — and applies it to
real harness sessions.

The borrowing stops at the semantics. Rome is the visual language, not fabricated state:
the map is an operational surface, settlements are real workspaces and workfronts, routes
are real harness activity, and every senator is a live session. The art may be atmospheric;
the state it represents may not be invented.

## The single rule

**Everything displayed corresponds to a real object, and nothing moves unless something
real moved.**

This rule is what makes the interface worth trusting, so it is enforced in the code rather
than stated in the docs:

- The renderer has no idle animation. Route pulses are driven by event timestamps
  (`pulseOf`), and folder emphasis is a decay function over real event recency and volume
  (`heatOf`). When nothing happens, the canvas is static, and the render loop stops
  drawing entirely.
- The one self-animating element is the ring around an agent waiting for approval, because
  that *is* a live state: a real process is blocked on a human.
- Agent positions are derived from what they touched. An agent that reads
  `core/placement/src` attaches to that folder node in the Cameo region, because the
  harness reported a tool call with that path — not because Field decided to move it.
- Endpoint health is a real probe, and the code says exactly what each probe measures. An
  HTTP probe proves a server is listening; a reachability probe proves the route out is
  open and explicitly does *not* claim the credentials are valid. Overclaiming here would
  make the rail worse than no rail.

## Why the Field stays quiet

The first working build drew every folder that had seen activity. It filled both regions
with chips and was unreadable — technically accurate and operationally useless.

Density is not information. The Field now draws only folders above a heat threshold, caps
each region's chips, and reports the remainder as `+N quieter`. Text is minimal by default;
names appear on hover, selection, or held `Alt`. Detail is revealed by selection, not by
being permanently on screen. The default state of a healthy Field is close to still.

This is also why the folder chips sort by path rather than by heat. Sorting by heat would
pack the busiest work into the corner — and make chips jump every time a folder got busy.
A spatial interface is only readable if position is stable.

## The spatial model

- **Regions** are workspaces: real mounted directories, positioned by config and
  overridable by the operator (positions persist as `ui.position` events).
- **Folder chips** flow in a stable grid inside their region.
- **Agents** anchor to the folder they are working in, or to their region if they have no
  narrower focus, and form up in a wedge around that anchor. The wedge is what makes
  "these five belong to that" readable without a label.
- **Websites** sit outside the regions as external surfaces, connected by route lines to
  the agents actually browsing them.
- **Endpoints** are a screen-fixed rail rather than a spatial object, because an endpoint
  has no place in the workspace — it is infrastructure the whole field draws on.
- **Unassigned agents** muster in a staging area to the left. They are visibly not working.

Role is encoded as silhouette rather than color, because color is already carrying state,
and an operator needs to distinguish "what kind of unit" from "how is it doing" at a
glance.

## Where the harness meets the interface

Field drives the real `claude` CLI in duplex `stream-json` mode: one process per session,
with its own `--session-id`, model, effort level, mounted directories, and composed system
prompt. Everything the Field knows about an agent is parsed from that process's actual
output.

Three details are load-bearing:

**Pausing is real.** A paused agent is a terminated process whose session id is retained;
resuming re-attaches with `--resume`. The alternative — pretending to pause while the
process runs on — would let the interface lie about the most consequential control it has.

**Approval is a real gate.** Field ships a small MCP server exposing one tool, wired in
with `--permission-prompt-tool`. The harness calls it before a consequential action, and it
blocks — parking the request with the Field server until a human decides. If the Field
server is unreachable, it denies. A permission gate that fails open is not a permission
gate.

**Read-only roles are really read-only.** The first build declared `tools_allow` in the
role files and assumed that restricted the agent. It does not — `--allowedTools` widens
auto-approval, it does not remove a tool, and a scout promptly reached for PowerShell to do
a job its allow-list said it should do with Glob. Roles that declare `read_only` now emit
`--disallowedTools`, so the declaration in `field/roles/` is a constraint rather than a
description.

## Adaptive thinking

`adaptive` is not a passthrough — the harness has no such level. Field resolves it from the
real size of the real target: a small file gets `low`, a large tree gets `high`, and the
role's own floor is respected so a verifier never drops below its minimum. The resolved
level *and the reason* are recorded on the spawn event, so Traces can show why an agent was
thinking as hard as it was.

## Cost of being wrong

Two bugs in this build are worth recording, because both produced plausible-looking output
while being completely wrong.

The **ignore matcher** was written as a chain of `.replace()` calls — the obvious way. Each
replacement inserted regex syntax that later replacements then rewrote, so `(?:` became
`([^\\/]:` and every ignore pattern silently matched nothing. The visible symptom was a
cluttered Field. The real symptom was that the event store's own SQLite WAL writes were
being watched, so every appended event produced a filesystem event that appended another
event. The log was feeding itself. It is now a single tokenizing pass in `server/src/glob.js`
with a test covering both the ignore lists and the routine triggers.

The **endpoint probe** appended a health event on every tick regardless of whether anything
changed. In an append-only log that is permanent noise, so health is now recorded on change
plus a slow heartbeat.

Both bugs share a shape: something that looked alive but was not reporting reality. In an
interface whose entire claim is that everything shown is real, that is the failure mode to
hunt for.
