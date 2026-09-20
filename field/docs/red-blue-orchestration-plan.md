# Field campaign architecture

This is a domain and invariant reference for Field's campaign layer. It is not
the product roadmap or a current release-readiness claim. Cross-product scope,
open work, and release gates belong to Cameo's `PRODUCTIZATION_PLAN.md` when
Field is bundled there.

## Outcome

Field becomes the harness-level command surface for long-horizon, many-agent work. The
operator should be able to answer four questions without opening a chat:

1. Is every required role staffed and actually running?
2. What objective is each team pursuing, and what is blocking it?
3. Which agents, tools, endpoints, repositories, and external sites are interacting?
4. Has the result survived an independent challenge and earned promotion into the stable
   civilization?

The RTS interaction language supplies selection, formations, cohorts, objectives, doctrine,
reinforcement, retreat, and replay. Red/blue orchestration supplies the evidence-bearing
workflow that prevents a large agent group from merely agreeing with itself.

This layer does not replace chat for precise one-agent work. It is for batches, parallel
workstreams, handoffs, degradation, adversarial review, and operations whose state must stay
legible over hours or days.

## System boundary

The implementation extends the `field/` application inside the Knossos repository.

- **Cameo** remains the compute and fleet substrate: model serving, endpoint discovery,
  health, placement, VRAM residency, and routing.
- **Knossos** remains the agent harness: turns, tools, context, delegation,
  verification, and session traces.
- **Field server** becomes the campaign control plane: canonical event log, campaign state
  machine, temporal graph, policy, team orchestration, checkpoints, and strategic API.
- **Field web** remains a projection and intent surface. It never invents work, moves an
  agent without an event, or decides that work is verified.

The current direct Claude process adapter stays usable. A Knossos adapter can implement the
same registry interface, so the campaign model is not coupled to one provider or CLI.

## Non-negotiable rules

1. **Events are truth.** Operational state is appended, never mutated in place.
2. **The graph is physics; the field is a view.** The temporal graph explains why objects
   are related. Screen position is a stable projection of those relationships.
3. **Display only real state.** An agent appears when a harness session exists. A route
   appears when an interaction occurred. Infrastructure materializes only after observed,
   sustained use.
4. **Verification is independent.** A builder cannot promote its own result. Red cannot
   close its own finding. Referees do not inherit either team's conclusion as fact.
5. **Red is scoped.** Adversarial activity defaults to snapshot, sandbox, or staging. Every
   destructive capability requires an explicit scope and permission gate.
6. **Strategic commands, not tool micromanagement.** The operator assigns outcomes,
   doctrine, budget, concurrency, and authority. Each harness owns its own tool loop.
7. **Replay must reproduce live state.** The same reducers process boot replay, historical
   replay, simulations, and live events.
8. **No final art lock before behavioral proof.** The first UI is an instrumented command
   surface. Final Roman/default/other themes follow usability and stress testing.

## Product model

### Campaign

A campaign is the top-level operation. It has:

- name and operator intent;
- target workspace, service, endpoint, site, or fleet;
- doctrine and risk profile;
- budget, concurrency, deadline, and permission scope;
- one or more objectives;
- blue, red, purple, and referee teams;
- findings, mitigations, verdicts, checkpoints, and promoted capabilities;
- a deterministic status derived from events.

Campaign lifecycle:

`draft -> mobilizing -> blue_building -> red_challenging -> contested -> blue_mitigating -> red_retesting -> referee_review -> verified -> promoted`

Terminal alternatives are `failed`, `cancelled`, and `rolled_back`. `paused` is an
operational flag, not a lifecycle phase, so resuming does not destroy the prior phase.

### Objectives

An objective is a testable outcome, not an agent prompt. Each objective carries:

- explicit statement;
- definition of done;
- target and dependencies;
- priority and risk;
- owner team and assigned sessions;
- progress evidence;
- status: `queued`, `active`, `blocked`, `satisfied`, or `failed`.

Agent prompts are derived from the objective, doctrine, target, and role. They are stored in
the trace, but the objective remains the stable unit the operator manages.

### Teams and roles

- **Blue** builds, operates, repairs, and defends.
- **Red** challenges security, reliability, correctness, UX, cost, data quality,
  prompt-injection resistance, and rollback behavior.
- **Purple** synthesizes cross-team lessons and proposes durable doctrine. Purple does not
  waive a retest or referee review.
- **Referee** reproduces evidence independently, resolves disputes, and decides whether the
  campaign can advance or promote.

Team membership is temporal. An agent can move between campaigns, but it cannot hold
conflicting roles in the same active contest. Every reassignment is an event.

### Findings, mitigations, and verdicts

A red finding is valid only when it includes:

- category and severity;
- claim and affected scope;
- evidence references;
- reproduction steps or an explicit reason reproduction is unavailable;
- confidence;
- author session and campaign/objective linkage.

Finding status:

`open -> acknowledged -> mitigating -> ready_for_retest -> confirmed | rejected | waived`

`waived` requires an operator or referee event with rationale. A blue agent cannot reject a
finding by assertion.

A mitigation references one or more findings, changed artifacts, verification evidence,
and its blue owner. Red retest records whether each finding is fixed, persists, regressed,
or cannot be reproduced. A referee verdict is one of:

- `verified`: the definition of done and mandatory challenge gates passed;
- `rejected`: evidence disproves readiness;
- `inconclusive`: more evidence or a new environment is required.

Promotion is a separate event after a verified verdict. This keeps “proved ready” distinct
from “made part of the stable baseline.”

## Canonical event vocabulary

Existing session, tool, filesystem, browser, endpoint, permission, and assignment events
remain unchanged. The campaign layer adds:

### Campaign and objective

- `campaign.created`
- `campaign.phase_changed`
- `campaign.paused`, `campaign.resumed`, `campaign.cancelled`
- `campaign.doctrine_changed`
- `campaign.checkpoint_created`, `campaign.rolled_back`
- `objective.created`, `objective.assigned`, `objective.progress`
- `objective.blocked`, `objective.satisfied`, `objective.failed`

### Team orchestration

- `team.member_assigned`, `team.member_removed`
- `team.orders_issued`
- `team.reinforced`, `team.retreated`
- `team.handoff_started`, `team.handoff_completed`, `team.handoff_failed`
- `agent.communication` for observed session-to-session communication

### Contest

- `finding.reported`, `finding.acknowledged`, `finding.disputed`
- `mitigation.proposed`, `mitigation.started`, `mitigation.ready`
- `retest.completed`
- `referee.review_started`, `referee.verdict`
- `campaign.promoted`

### World and graph

- `world.contact_observed`
- `world.site_established`, `world.site_dormant`, `world.site_removed`
- `capability.proposed`, `capability.promoted`, `capability.deprecated`
- `graph.edge_observed` for adapter events that do not map through an existing typed event

Every campaign event carries `campaignId`; objective/finding/mitigation identifiers are
stable UUIDs. `subject` is the most specific aggregate so traces can be retrieved without
scanning the full log. Cross-links remain in event data.

## State machine and gates

The director validates every transition. Invalid transitions fail before any event is
appended.

| From | To | Required evidence |
|---|---|---|
| draft | mobilizing | at least one objective and a blue team |
| mobilizing | blue_building | required blue sessions started or explicitly external |
| blue_building | red_challenging | blue readiness evidence for each required objective |
| red_challenging | contested | at least one open finding |
| red_challenging | referee_review | red completed with zero qualifying findings |
| contested | blue_mitigating | every qualifying finding acknowledged or disputed |
| blue_mitigating | red_retesting | each blocking finding has a ready mitigation |
| red_retesting | referee_review | every blocking finding confirmed fixed, rejected by evidence, or waived by authority |
| referee_review | verified | independent referee verdict `verified` |
| referee_review | blue_building/blue_mitigating | verdict `rejected` or `inconclusive` with next action |
| verified | promoted | operator promotion and checkpoint |

Critical and high findings block advancement by default. Doctrine can make lower severities
blocking, but cannot make critical findings non-blocking without an explicit signed waiver.

The director is idempotent for commands carrying the same `commandId`. Retries after a
network timeout must not create duplicate campaigns, findings, agents, or promotions.

## Temporal typed graph

The graph projection is derived from the event log and rebuilt on replay.

### Node types

`agent`, `team`, `campaign`, `objective`, `workspace`, `folder`, `file`, `website`,
`service`, `endpoint`, `tool`, `artifact`, `finding`, `mitigation`, `verdict`, `capability`,
`checkpoint`, and `memory`.

Every node stores:

- stable typed id;
- label and limited display metadata;
- `firstSeen`, `lastSeen`, observation count;
- lifecycle state;
- campaign and workspace scope when applicable;
- provenance event sequence numbers.

### Edge types

`member_of`, `assigned_to`, `working_on`, `located_at`, `depends_on`, `communicates_with`,
`visited`, `used`, `produced`, `challenged_by`, `found`, `mitigated_by`, `retested_by`,
`verified_by`, `promoted_into`, `hosted_by`, and `routed_through`.

Edges store first/last observation, count, direction, strength, active/dormant state, and
provenance. Repeated events strengthen an edge without creating duplicate screen lines.

### World materialization

Folders, sites, and infrastructure move through:

`unknown -> contact -> active_worksite -> established -> promoted -> dormant -> removed`

Rules:

- sessions render immediately because their process existence is authoritative;
- a folder/site enters `contact` on first real access;
- it becomes an `active_worksite` after repeated observations or sustained dwell;
- it becomes `established` after a configurable hit, duration, or produced-artifact gate;
- only `active_worksite`, `established`, and `promoted` nodes enter the normal field
  projection;
- cold nodes become `dormant` with hysteresis so the map does not flicker;
- removal is explicit or retention-policy driven, never a visual timeout that destroys
  history.

Stable spatial memory keys positions by typed node id. Community assignment changes only
when graph evidence exceeds a hysteresis threshold. Edge bundling and level-of-detail
prevent a 30-agent operation from turning into a hairball.

## Baseline civilization and checkpoints

The stable civilization is the set of promoted capability nodes at a checkpoint. It is not
a fake pre-rendered city claiming infrastructure exists.

A checkpoint stores:

- campaign/event head sequence;
- promoted capability ids and their verification evidence;
- graph node positions and community membership;
- relevant repository revision and endpoint topology;
- scene metadata and optional rendered baseline image;
- hit-test metadata linking visual landmarks back to real objects.

The browser draws the checkpoint baseline first, then overlays current live sessions,
routes, contested work, transient contacts, and degradation. A new baseline is generated
only after meaningful promotion, rollback, or operator request.

Rollback appends an event selecting an earlier checkpoint. It does not delete later events;
the audit trail must show what was rolled back and why.

## Strategic API

The server exposes outcome-level operations:

- `POST /api/campaigns/create`
- `POST /api/campaigns/action`
- `GET /api/campaigns/:id/trace`
- `POST /api/campaigns/simulate` in explicit simulation mode only

`campaigns/action` accepts a `commandId`, `campaignId`, `kind`, and kind-specific payload.
Initial actions:

- `mobilize`
- `advance`
- `assign_team`
- `issue_orders`
- `reinforce`
- `retreat`
- `pause` / `resume` / `cancel`
- `report_finding`
- `acknowledge_finding` / `dispute_finding`
- `propose_mitigation` / `start_mitigation` / `mark_mitigation_ready`
- `record_retest`
- `begin_referee_review` / `record_verdict`
- `checkpoint` / `promote` / `rollback`

The API returns the appended event sequence and the current campaign projection. Errors are
structured (`code`, `message`, `currentPhase`, `allowedActions`) so the UI can explain a
blocked command without guessing.

## Real harness orchestration

Mobilization resolves a roster from durable agent definitions and campaign requirements.
For each role it:

1. validates workspace mount, endpoint availability, budget, and permission scope;
2. selects a configured agent definition or reports an explicit staffing gap;
3. spawns a real isolated harness session through `Registry`;
4. attaches campaign, team, objective, doctrine, and scope metadata to `session.spawned`;
5. records team membership and assignment only after spawn succeeds;
6. compensates partial failure by marking the campaign degraded and exposing retry/retreat;
   it does not pretend the requested formation exists.

Orders are composed from role doctrine:

- blue receives the objective, definition of done, target, allowed scope, and evidence
  format;
- red receives the same claim plus attack categories, environment boundary, non-destructive
  rules, and the finding schema;
- referee receives blue evidence, red evidence, the original definition of done, and an
  instruction to reproduce independently;
- purple receives the completed trace only after the verdict, so synthesis cannot influence
  the referee.

Endpoint failure uses the existing registry rerouter. Campaign projection additionally
marks under-staffed roles, affected objectives, and the handoff state. A rerouted session
must restate its last durable checkpoint before continuing.

## RTS command surface

The first functional UI adds a **Campaign** mode alongside Field, Workspace, Routines, and
Traces. It intentionally reuses current tokens and avoids final art polish.

### Default view

- center: the real workspace/service topology and current objective fronts;
- formations: blue, red, referee, and purple sessions grouped by active objective;
- routes: work, communication, evidence, finding, mitigation, and verification;
- left strip: campaign phases and unresolved gates;
- right inspection: selected agent, team, objective, finding, or capability;
- bottom rail: endpoint health, team readiness, budget, blocking findings, and approvals;
- minimap: communities and hotspots, not every node.

The operator can tell at a glance which roles are present, which are missing, which agents
are collaborating, what each formation is attacking/defending, and why the campaign cannot
advance.

### Interaction grammar

- click or marquee selects agents;
- shift adds/removes, control groups work as they do today;
- clicking a team crest selects the cohort;
- right-click objective: assign, reinforce, challenge, verify, pause, or retreat based on
  current role and phase;
- drag an objective priority marker to reorder outcomes, not individual tool calls;
- double-click an agent opens its prompt, conversation, objective, progress, evidence, and
  permissions in Workspace mode;
- click a finding to show reproduction, linked mitigation, retest, and verdict;
- timeline scrub replays the entire campaign through the same projection reducer;
- theme setting supports `default`, `roman`, and future themes without changing semantics.

No permanent prose explains the metaphor. Labels reveal on hover/selection; blocking gates
and approval requests remain visible because they require operator action.

## Safety and authority

- Field remains loopback-only by default.
- Campaign APIs require operator authority when exposed beyond loopback.
- Red sessions default to read-only plus dedicated test tools.
- A campaign declares one of `snapshot`, `sandbox`, `staging`, or `production-readonly`.
- Production mutation is outside the default doctrine and requires an explicit operator
  approval event per escalation.
- Secrets are referenced by provider bindings and never copied into campaign events.
- Event payloads apply size limits and redact known credential fields before persistence.
- Prompt-injection tests use controlled fixtures and decoy secrets, never real credentials.
- Promotion requires an independent referee and a checkpoint.

## Implementation sequence

### Phase 1: domain foundation

1. Add pure campaign constants, validators, transitions, and error types.
2. Add campaign/objective/team/finding/mitigation/verdict/capability projection maps.
3. Add the temporal graph projection and world lifecycle reducer.
4. Include both projections in snapshots and replay.
5. Add reducer tests for every valid and invalid transition.

Exit: a synthetic event stream reconstructs the same campaign and graph after replay.

### Phase 2: strategic director and API

1. Add an idempotent `CampaignDirector` over the event emitter and registry boundary.
2. Implement creation, phase advancement, team assignment, findings, mitigations, retests,
   verdicts, checkpoints, promotion, and rollback.
3. Add structured API errors and campaign trace retrieval.
4. Add payload limits, identifier validation, and authority/scope checks.

Exit: API contract tests complete a campaign without spawning a real provider process.

### Phase 3: real-session wiring

1. Extend registry session metadata with campaign/team/objective and parent handoff data.
2. Implement mobilization and role-specific order composition.
3. Map session completion, errors, endpoint failure, delegation, and permission events into
   objective/team readiness.
4. Add partial-mobilization compensation and restart interruption handling.
5. Add a Knossos adapter contract while retaining the current CLI adapter.

Exit: one blue, one red, and one referee session can complete a real local campaign trace.

### Phase 4: campaign command surface

1. Add Campaign mode and strategic client calls.
2. Render team formations, objectives, contest/evidence routes, and phase/gate state.
3. Add selection inspection for prompts, transcript, objective, progress, finding, and
   evidence.
4. Add legal contextual actions and disabled-state reasons from the server.
5. Add campaign replay and checkpoint baseline overlay.

Exit: an operator can run the vertical slice without using curl or opening session chats.

### Phase 5: hardening and scale

1. Add deduplication, command idempotency, projection invariants, and log corruption checks.
2. Bound snapshot/event sizes and coalesce graph changes.
3. Add graph LOD, edge bundling, stable placement, and lifecycle hysteresis.
4. Add failure injection and deterministic simulations.
5. Profile a 30-agent / 10,000-node / 100,000-event replay and live burst.

Exit: all release gates below pass on hardware without renting a training GPU.

### Phase 6: final art direction

1. Record usability sessions from the working vertical slice and stress scenarios.
2. Compare default utilitarian, Roman command, and reduced-fantasy skins against the same
   semantics and hit targets.
3. Lock iconography, density, motion, terrain/baseline imagery, typography, and sound.
4. Keep a no-theme accessibility mode and reduced-motion behavior.

Exit: visual decisions respond to real operational density rather than another static mock.

## Simulation and stress matrix

All scenarios use deterministic ids/time and assert the final projection, graph, legal
actions, and replay equality.

1. **Release rollback race**: blue changes restart and rollback code; red finds a race;
   mitigation lands; red retest passes; referee promotes.
2. **Fleet degradation**: one provider fails mid-campaign; sessions reroute; one role cannot;
   campaign remains honest about reduced readiness.
3. **Corpus contamination**: red finds poisoned traces; blue quarantines; red retests lineage;
   referee checks clean rebuild evidence.
4. **Prompt injection**: website content attempts to redirect an agent and exfiltrate a decoy
   secret; permission gate and red finding record the boundary.
5. **Provider outage**: duplicate health events are coalesced; no reroute storm or duplicate
   agent formation occurs.
6. **False red finding**: blue disputes with evidence; referee reproduces and rejects the
   finding without allowing blue to self-clear it.
7. **Blue regression**: one mitigation breaks a previously passing objective; campaign
   returns to mitigation rather than advancing.
8. **Referee disagreement**: first verdict is inconclusive; a second independent referee is
   assigned; evidence histories remain distinct.
9. **Crash during handoff**: process dies after target accepts but before source confirms;
   idempotent recovery yields one membership and one active owner.
10. **Thirty-agent operation**: multiple objectives, shared endpoints, communication edges,
    permission queues, and reassignments remain legible and bounded.
11. **Checkpoint rollback**: promotion creates a baseline; later regression selects the prior
    checkpoint without deleting the failed history.
12. **Event replay**: live projection hash equals a clean replay hash at every checkpoint.

Additional edge cases:

- empty teams and impossible staffing;
- duplicate command ids and reordered client retries;
- unknown identifiers and stale phases;
- duplicate findings with related but non-identical evidence;
- mitigation linked to the wrong campaign;
- critical finding waiver without authority;
- verifier that is also a blue/red member;
- session completion with missing structured evidence;
- endpoint flapping and all-endpoints-down;
- budget exhaustion during red retest;
- event payload and trace size limits;
- torn JSONL tail and SQLite restart;
- graph node churn, cold-site dormancy, and reactivation;
- websocket reconnect during event burst;
- browser refresh during pending permission;
- campaign cancel while agents are spawning;
- rollback to a nonexistent or cross-campaign checkpoint.

## Release gates

The layer is ready for the final art-direction pass only when:

- every campaign transition has positive and negative tests;
- command retries are idempotent;
- restart/replay produces the same campaign and graph state;
- red cannot mutate outside declared scope in the tested adapters;
- blue cannot clear its own finding or promote its own result;
- referee independence is validated;
- partial spawn, endpoint loss, budget exhaustion, and cancelled campaigns show accurate
  staffing and objective state;
- a 30-agent synthetic campaign does not exceed the agreed snapshot/frame latency budget;
- a 100,000-event replay completes within the agreed boot budget;
- the release rollback vertical slice is operable entirely from the UI;
- no simulated object appears as live in production mode;
- logs contain no raw configured secrets;
- all existing Field tests and the web production build still pass.

## First vertical slice

**Harden the release gateway**

- Blue: architect, two builders, reliability operator.
- Red: security challenger, rollback challenger, correctness challenger.
- Referee: independent verifier.
- Objective: deploy/restart/rollback behavior is safe under concurrency and partial failure.
- Expected contest: red finds a restart/rollback race, blue mitigates it, red reproduces the
  original failure and confirms it is closed, referee independently runs the checks, then a
  capability and checkpoint are promoted.

This slice exercises every important semantic: staffing, teams, objectives, work topology,
findings, mitigation, retest, verification, promotion, baseline generation, replay, and
rollback. Art direction begins only after this path is mechanically trustworthy.

## Functional completion evidence (2026-08-26)

The pre-art-direction implementation now satisfies the release gates above:

- the automated suite covers 35 valid transitions and a negative case from every phase;
- duplicate commands, partial staffing, cancellation, cross-campaign references, handoff
  recovery, restart interruption, endpoint flapping, all-down routing, and budget stops pass;
- challenger/referee permission policy denies writes, path escape, mutating shell commands,
  and read-only environment mutation before operator approval;
- campaign history pages through an unbounded event log and reuses the boot projection;
- a 500-event WebSocket burst produces one coalesced snapshot, then reconnect restores the
  current snapshot and a pending approval;
- the deterministic scale fixture reaches 100,004 events, 30 agents, and 10,246 graph nodes;
  on the audited machine replay was about 0.6 seconds, bounded snapshot generation about
  23 ms, and Field layout about 7 ms;
- the release-gateway campaign was completed from the UI through challenge, mitigation,
  retest, independent verdict, checkpoint, and promotion, then survived a server restart;
- the Knossos `serve` binary completed a real ready/capabilities/shutdown handshake, and its
  71 focused harness/serve tests passed without a model call or rented GPU;
- the production web build passes, and a 1440×900 headless render confirms the functional
  surface does not overflow and distinguishes live, unavailable, and external formations.

Provider-backed model work remains an environment exercise for the first inference endpoint;
it is not required for the GPU-free orchestration implementation or the final art-direction
decision.
