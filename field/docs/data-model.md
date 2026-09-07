# Field — data model

## The split

**Git holds the civilization. The database holds the operation.**

The test for which side something belongs on: *would a person want to read it, review it in
a pull request, and carry it to another machine?* If yes it is a file in `field/`. If it
only describes what is happening right now, it is an event.

| Git — `field/` | Database — `.field-state/` |
|---|---|
| agent definitions, roles, constitutions | running sessions and their state |
| skills, mission templates | tool calls, messages, delegation |
| routines (including their `enabled` default) | assignments and operator commands |
| Markdown memory | endpoint health, routing decisions |
| workspace and endpoint configuration | permissions requested and decided |
| | costs, context usage, browser sessions |
| | UI positions and control groups |

A routine's *definition* is a YAML file in Git. Whether it is currently switched on is an
event. That boundary keeps the repository meaningful without making it a runtime database.

## Event sourcing

The event log is the only source of truth for operational state. Nothing operational is
ever mutated in place.

```
events(seq INTEGER PK, ts INTEGER, kind TEXT, actor TEXT, subject TEXT, data TEXT)
```

Live state is a fold over that log, implemented once in
`server/src/store/projection.js`. On boot the server replays every event through the same
fold that processes live events, which means:

- the running state and a historical replay are produced by identical code and cannot
  disagree;
- Traces can reconstruct any session or assignment exactly as it unfolded, at any point in
  its history, by folding a prefix of its events;
- a crash loses no operational history.

`subject` is what a trace is keyed on — a session id, an assignment id, a workspace id.
`actor` is which agent caused it, when that is meaningful.

### Storage backend

Preferred backend is Node's built-in `node:sqlite` (WAL enabled). If unavailable, the same
interface is served by an append-only JSONL file. The event-sourced design is what makes
that fallback honest rather than a downgrade: the store only needs `append`, `read`, and
`bySubject`, so both backends implement the identical contract and the Field behaves the
same on either.

## Event vocabulary

**Sessions**

| Kind | Carries |
|---|---|
| `session.spawned` | agent, role, model, endpoint, resolved thinking + why, cwd, workspace, parent |
| `session.state` | `spawning` → `ready` → `thinking` → `working` → `idle`, plus `waiting_permission`, `paused`, `blocked`, `interrupted`, `error` |
| `session.message` | assistant and user turns |
| `session.thinking` | reasoning text |
| `session.tool_use` | tool, summary, and the workspace / folder / path it resolves to |
| `session.tool_result` | success or failure, with a preview |
| `session.usage` | input, output, cache tokens, context total, cost |
| `session.delegated` | parent → child, with the subagent type |
| `session.turn_complete`, `session.ended` | result, turns, duration, reason |

**Work and command**

`assignment.created` · `assignment.completed` · `assignment.cancelled` ·
`command.issued` (`pause`, `resume`, `cancel`, `redirect`, `reinforce`, `verify`,
`escalate`, `say`) · `work.verified` · `routine.triggered` · `routine.enabled`

**World**

`fs.changed` · `git.status` · `browser.navigated` · `browser.closed` ·
`endpoint.health` · `endpoint.routed` · `terminal.run`

**Operator**

`permission.requested` · `permission.decided` · `ui.position` · `ui.control_group`

`ui.position` and `ui.control_group` are read last-write-wins by the fold, but they are
still events — so a replay reproduces the operator's layout at that moment too.

**Campaigns and contests**

`campaign.created` · `campaign.phase_changed` · `campaign.paused` ·
`campaign.resumed` · `campaign.cancelled` · `campaign.checkpoint_created` ·
`campaign.promoted` · `campaign.rolled_back` · `objective.created` ·
`objective.assigned` · `objective.progress` · `objective.satisfied` ·
`objective.blocked` · `team.member_assigned` · `team.member_removed` ·
`team.orders_issued` · `team.handoff_started` · `team.handoff_completed` ·
`finding.reported` · `finding.acknowledged` · `finding.disputed` ·
`mitigation.proposed` · `mitigation.started` · `mitigation.ready` ·
`retest.completed` · `referee.review_started` · `referee.verdict`

Campaign state and its temporal typed graph are separate projections over this same stream.
The graph contains agents, teams, campaigns, objectives, workspaces, artifacts, endpoints,
findings, mitigations, verdicts, checkpoints, and capabilities with provenance-bearing
edges. Websocket snapshots are capped at 4,000 nodes and 8,000 edges; the canonical graph
remains reconstructible from the unbounded event log.

## Derived values that are deliberately *not* stored

Some things are computed at read time rather than persisted, because persisting them would
create a value that ages into a lie:

- **Folder heat** is computed in the client from `lastTs` and `hits`. Storing a heat number
  would mean writing an event every time heat decayed — activity generated by the clock
  rather than by the world.
- **Route pulses** come from event timestamps, not from a stored "is animating" flag.
- **Context percentage** is derived from real token counts against the model window.

## Restart behavior

Harness processes do not survive a server restart. On boot, any session the log still
believes is live is marked `interrupted` with the reason recorded — rather than being shown
as a ghost that appears to be working. This is itself an appended event, so the interruption
is part of the trace.
