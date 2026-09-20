import assert from 'node:assert/strict';
import { CampaignProjection } from '../src/orchestration/campaign-projection.js';
import { CampaignDirector } from '../src/orchestration/director.js';

function fixture({ concurrency = 2, spawn } = {}) {
  let seq = 0;
  let ids = 0;
  const events = [];
  const projection = new CampaignProjection();
  const emit = (kind, data, meta = {}) => {
    const event = { seq: ++seq, ts: 1_700_200_000_000 + seq, kind, data, ...meta };
    events.push(event); projection.apply(event); return event;
  };
  const commands = [];
  const registry = {
    spawn: spawn ?? ((input) => ({ id: `session-${++ids}`, role: input.team === 'red' ? 'challenger' : 'builder', state: 'working' })),
    command: (kind, payload) => { commands.push({ kind, payload }); return { ok: true }; },
    assign: ({ sessionIds }) => ({ assignmentId: `assignment-${++ids}`, skipped: sessionIds.filter((id) => id.startsWith('external')) }),
    info: () => null,
  };
  const director = new CampaignDirector({ projection, registry, emit, eventHead: () => seq, id: () => `id-${++ids}` });
  const created = director.create({
    commandId: 'create', campaignId: 'edge', name: 'Edge operation', intent: 'Exercise failure boundaries.',
    scope: 'sandbox', concurrency,
    objectives: [{ statement: 'Keep state honest', definitionOfDone: ['projection agrees'] }],
  });
  return { projection, director, events, commands, objectiveId: created.objectiveIds[0] };
}

// Empty and over-capacity rosters fail or surface explicit staffing gaps.
const empty = fixture();
assert.throws(() => empty.director.action({ commandId: 'empty', campaignId: 'edge', kind: 'mobilize', roster: [] }), /at least one blue/);

const capped = fixture({ concurrency: 1 });
const cappedResult = capped.director.action({
  commandId: 'mobilize', campaignId: 'edge', kind: 'mobilize',
  roster: [
    { team: 'blue', agentId: 'blue-agent', objectiveId: capped.objectiveId },
    { team: 'red', agentId: 'red-agent', objectiveId: capped.objectiveId },
  ],
});
assert.equal(cappedResult.gaps.length, 1);
assert.match(cappedResult.gaps[0].reason, /concurrency/);
assert.equal(capped.projection.campaigns.get('edge').phase, 'mobilizing');
assert.throws(() => capped.director.action({
  commandId: 'over-cap-reinforcement', campaignId: 'edge', kind: 'reinforce',
  team: 'blue', agentId: 'extra-blue', objectiveId: capped.objectiveId,
}), /concurrency limit is 1/);
assert.throws(() => capped.director.action({
  commandId: 'over-cap-external', campaignId: 'edge', kind: 'assign_team',
  team: 'red', sessionId: 'external-red', agentId: 'external-red',
  objectiveId: capped.objectiveId, external: true,
}), /concurrency limit is 1/);

const budgeted = fixture();
budgeted.projection.campaigns.get('edge').budgetUsd = 1.5;
budgeted.projection.apply({ seq: 100, ts: 100, kind: 'session.usage', data: { campaignId: 'edge', sessionId: 'cost-a', costUsd: 0.6 } });
budgeted.projection.apply({ seq: 101, ts: 101, kind: 'session.usage', data: { campaignId: 'edge', sessionId: 'cost-a', costUsd: 1.1 } });
budgeted.projection.apply({ seq: 102, ts: 102, kind: 'session.usage', data: { campaignId: 'edge', sessionId: 'cost-b', costUsd: 0.5 } });
const budgetCampaign = budgeted.projection.campaigns.get('edge');
assert.equal(budgetCampaign.costUsd, 1.6, 'campaign cost sums per-session cumulative deltas');
assert.equal(budgetCampaign.budgetExhausted, true);
assert.throws(() => budgeted.director.action({
  commandId: 'over-budget', campaignId: 'edge', kind: 'mobilize', roster: [],
}), /campaign budget exhausted/);

// If only a non-blue unit starts, compensation pauses it and the campaign fails honestly.
let spawnCount = 0;
const partial = fixture({ spawn: (input) => {
  spawnCount += 1;
  if (input.team === 'blue') throw new Error('blue endpoint unavailable');
  return { id: `partial-${spawnCount}`, role: input.team, state: 'working' };
} });
const partialResult = partial.director.action({
  commandId: 'partial', campaignId: 'edge', kind: 'mobilize',
  roster: [
    { team: 'blue', agentId: 'blue-agent', objectiveId: partial.objectiveId },
    { team: 'red', agentId: 'red-agent', objectiveId: partial.objectiveId },
  ],
});
assert.equal(partialResult.phase, 'failed');
assert.equal(partial.projection.campaigns.get('edge').phase, 'failed');
assert.deepEqual(partial.commands.at(-1), { kind: 'pause', payload: { sessionIds: ['partial-2'] } });

// Cancellation is durable even immediately after mobilization, and a retry is idempotent.
const cancelled = fixture();
cancelled.director.action({
  commandId: 'mobilize', campaignId: 'edge', kind: 'mobilize',
  roster: [{ team: 'blue', agentId: 'blue-agent', objectiveId: cancelled.objectiveId }],
});
const firstCancel = cancelled.director.action({ commandId: 'cancel', campaignId: 'edge', kind: 'cancel', reason: 'operator stop' });
const beforeRetry = cancelled.events.length;
const retryCancel = cancelled.director.action({ commandId: 'cancel', campaignId: 'edge', kind: 'cancel', reason: 'duplicate request' });
assert.equal(firstCancel.campaign.phase, 'cancelled');
assert.equal(retryCancel.duplicate, true);
assert.equal(cancelled.events.length, beforeRetry);

// Cross-campaign references and unauthorized critical waivers are refused.
const refs = fixture();
assert.throws(() => refs.director.action({ commandId: 'missing-mitigation', campaignId: 'edge', kind: 'start_mitigation', mitigationId: 'elsewhere' }), /does not belong/);
assert.throws(() => refs.director.action({ commandId: 'bad-waiver', campaignId: 'edge', kind: 'waive_finding', findingId: 'elsewhere', authority: 'blue', reason: 'trust me' }), /does not belong/);
assert.throws(() => refs.director.action({ commandId: 'bad-phase-rollback', campaignId: 'edge', kind: 'rollback', checkpointId: 'none', reason: 'bad' }), /requires a promoted campaign/);

// Duplicate handoff completion converges to one owner and replay is identical.
const handoffEvents = [
  { seq: 1, ts: 1, kind: 'campaign.created', data: { campaignId: 'h', name: 'handoff', intent: 'recover', scope: 'sandbox', concurrency: 2, budgetUsd: 2, doctrine: { blockingSeverity: 'high' } } },
  { seq: 2, ts: 2, kind: 'team.member_assigned', data: { campaignId: 'h', team: 'blue', sessionId: 'source', status: 'active' } },
  { seq: 3, ts: 3, kind: 'team.member_assigned', data: { campaignId: 'h', team: 'blue', sessionId: 'target', status: 'active' } },
  { seq: 4, ts: 4, kind: 'team.handoff_started', data: { campaignId: 'h', team: 'blue', fromSessionId: 'source', toSessionId: 'target' } },
  { seq: 5, ts: 5, kind: 'team.handoff_completed', data: { campaignId: 'h', team: 'blue', fromSessionId: 'source', toSessionId: 'target' } },
  { seq: 6, ts: 6, kind: 'team.handoff_completed', data: { campaignId: 'h', team: 'blue', fromSessionId: 'source', toSessionId: 'target' } },
];
const live = new CampaignProjection();
const replay = new CampaignProjection();
for (const event of handoffEvents) { live.apply(event); replay.apply(structuredClone(event)); }
const members = live.campaigns.get('h').teams.blue.members;
assert.equal(members.filter((member) => member.status === 'active').length, 1);
assert.equal(members.find((member) => member.sessionId === 'source').status, 'retired');
assert.deepEqual(live.snapshot(), replay.snapshot());

// A process restart must not leave ghost units looking active in a campaign formation.
const interrupted = new CampaignProjection();
for (const event of [
  { seq: 1, ts: 1, kind: 'campaign.created', data: { campaignId: 'restart', name: 'restart', intent: 'recover', scope: 'sandbox', concurrency: 1, budgetUsd: 2, doctrine: { blockingSeverity: 'high' } } },
  { seq: 2, ts: 2, kind: 'objective.created', data: { campaignId: 'restart', objectiveId: 'restart-o', statement: 'resume honestly', definitionOfDone: ['live owner'], priority: 1 } },
  { seq: 3, ts: 3, kind: 'team.member_assigned', data: { campaignId: 'restart', team: 'blue', sessionId: 'lost-session', objectiveId: 'restart-o', status: 'active' } },
  { seq: 4, ts: 4, kind: 'objective.assigned', data: { campaignId: 'restart', objectiveId: 'restart-o', team: 'blue', sessionIds: ['lost-session'] } },
  { seq: 5, ts: 5, kind: 'session.state', data: { sessionId: 'lost-session', state: 'interrupted', detail: 'field server restarted' } },
]) interrupted.apply(event);
const restartCampaign = interrupted.campaigns.get('restart');
assert.equal(restartCampaign.teams.blue.ready, false);
assert.equal(restartCampaign.teams.blue.members[0].status, 'unavailable');
assert.equal(interrupted.objectives.get('restart-o').status, 'blocked');
interrupted.apply({ seq: 6, ts: 6, kind: 'session.state', data: { sessionId: 'lost-session', state: 'working' } });
assert.equal(restartCampaign.teams.blue.members[0].status, 'active');
assert.equal(interrupted.objectives.get('restart-o').status, 'active');

console.log('edges: staffing, compensation, cancel retry, reference guards, handoff, and restart honesty passed');
