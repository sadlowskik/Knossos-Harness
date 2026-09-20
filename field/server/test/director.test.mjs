import assert from 'node:assert/strict';
import { CampaignProjection } from '../src/orchestration/campaign-projection.js';
import { CampaignDirector } from '../src/orchestration/director.js';

let seq = 0;
let idSeq = 0;
const projection = new CampaignProjection();
const stored = [];
const emit = (kind, data, meta = {}) => {
  const evt = { seq: ++seq, ts: 1_700_000_000_000 + seq, kind, data, ...meta };
  stored.push(evt);
  projection.apply(evt);
  return evt;
};

const sessions = new Map();
const calls = [];
const registry = {
  spawn(input) {
    const id = `session-${sessions.size + 1}`;
    const role = input.team === 'referee' ? 'verifier' : input.team === 'red' ? 'scout' : 'builder';
    const s = { id, agentId: input.agentId, role, state: 'working', ...input };
    sessions.set(id, s); calls.push(['spawn', input]); return s;
  },
  info(id) { return sessions.get(id) ?? null; },
  assign(input) { calls.push(['assign', input]); return { assignmentId: `assign-${calls.length}`, skipped: [] }; },
  command(kind, input) { calls.push([kind, input]); return { ok: true }; },
};

const director = new CampaignDirector({
  projection, registry, emit, eventHead: () => seq, id: () => `id-${++idSeq}`,
});

const created = director.create({
  commandId: 'cmd-create', campaignId: 'campaign-1', name: 'Release gateway',
  intent: 'Prove rollback under concurrency.', scope: 'sandbox',
  objectives: [{
    statement: 'Rollback remains safe during restart',
    definitionOfDone: ['race harness passes 1000 iterations'],
    target: { type: 'folder', id: 'cameod/src', workspaceId: 'cameo' },
  }],
});
assert.equal(created.campaign.phase, 'draft');
const objectiveId = created.objectiveIds[0];
projection.objectives.get(objectiveId).definitionOfDone.push('restart check passes');
assert.throws(() => director.action({
  commandId: 'cmd-short-evidence', campaignId: 'campaign-1', kind: 'satisfy_objective',
  objectiveId, evidence: ['only one proof'],
}), /2 completion criteria but only 1 evidence entries/);
projection.objectives.get(objectiveId).definitionOfDone.pop();
const duplicateCreate = director.create({
  commandId: 'cmd-create', name: 'ignored duplicate', intent: 'ignored',
  objectives: [{ statement: 'ignored', definitionOfDone: ['ignored'] }],
});
assert.equal(duplicateCreate.duplicate, true);
assert.equal(projection.campaigns.size, 1);

const mobilized = director.action({
  commandId: 'cmd-mobilize', campaignId: 'campaign-1', kind: 'mobilize',
  roster: [
    { team: 'blue', agentId: 'blue-agent', objectiveId },
    { team: 'red', agentId: 'red-agent', objectiveId },
    { team: 'referee', agentId: 'ref-agent', objectiveId },
  ],
});
assert.equal(mobilized.phase, 'mobilizing');
assert.equal(mobilized.spawned.length, 3);
const [blueId, redId, refereeId] = mobilized.spawned;
assert.match(calls.find((x) => x[0] === 'spawn')[1].orders, /Your team: BLUE/);

const act = (commandId, kind, rest = {}) => director.action({
  commandId, campaignId: 'campaign-1', kind, ...rest,
});

act('cmd-blue', 'advance', { to: 'blue_building' });
act('cmd-satisfy', 'satisfy_objective', { objectiveId, sessionId: blueId, evidence: ['stress: pass'] });
act('cmd-red', 'advance', { to: 'red_challenging' });
const reported = act('cmd-find', 'report_finding', {
  objectiveId, authorSessionId: redId, category: 'rollback', severity: 'high',
  claim: 'restart and rollback interleave', scope: 'release gateway',
  evidence: ['trace://41'], reproduction: ['start restart', 'issue rollback'], confidence: 0.95,
});
act('cmd-contested', 'advance', { to: 'contested' });
act('cmd-ack', 'acknowledge_finding', { findingId: reported.findingId, sessionId: blueId });
act('cmd-mitigating', 'advance', { to: 'blue_mitigating' });
const mitigation = act('cmd-mitigation', 'propose_mitigation', {
  findingIds: [reported.findingId], ownerSessionId: blueId,
  claim: 'serialize gateway state transitions', artifacts: ['cameod/src/app.rs'],
});
act('cmd-mitigation-start', 'start_mitigation', { mitigationId: mitigation.mitigationId, sessionId: blueId });
act('cmd-mitigation-ready', 'mark_mitigation_ready', {
  mitigationId: mitigation.mitigationId, sessionId: blueId, evidence: ['stress: 1000 pass'],
});
act('cmd-retest-phase', 'advance', { to: 'red_retesting' });
act('cmd-retest', 'record_retest', {
  findingId: reported.findingId, sessionId: redId, result: 'fixed', evidence: ['original repro: pass'],
});
act('cmd-review-phase', 'advance', { to: 'referee_review' });
act('cmd-review', 'begin_referee_review', { sessionId: refereeId });
act('cmd-verdict', 'record_verdict', {
  sessionId: refereeId, verdict: 'verified', evidence: ['independent harness: pass'],
  rationale: 'definition of done reproduced independently',
});
act('cmd-verified', 'advance', { to: 'verified' });
const checkpoint = act('cmd-checkpoint', 'checkpoint', { name: 'gateway verified', revision: 'abc1234' });
assert.throws(() => act('cmd-early-rollback', 'rollback', {
  checkpointId: checkpoint.checkpointId, reason: 'too early',
}), /requires a promoted campaign/);
act('cmd-promote', 'promote', {
  checkpointId: checkpoint.checkpointId,
  capabilities: [{ id: 'gateway-race-safe', name: 'Race-safe release gateway', evidence: ['verdict'] }],
});

const final = projection.campaignView(projection.campaigns.get('campaign-1'));
assert.equal(final.phase, 'promoted');
assert.equal(final.teams.blue.members.length, 1);
assert.equal(final.teams.red.members.length, 1);
assert.equal(final.teams.referee.members.length, 1);
assert.equal(final.findings[0].status, 'confirmed');
assert.equal(final.capabilities[0].id, 'gateway-race-safe');

const before = stored.length;
const duplicate = act('cmd-promote', 'promote', {});
assert.equal(duplicate.duplicate, true);
assert.equal(stored.length, before);

assert.throws(() => act('cmd-cross-campaign', 'assign_team', {
  team: 'referee', sessionId: blueId, objectiveId, external: true,
}), /cannot serve on both/);

act('cmd-rollback', 'rollback', { checkpointId: checkpoint.checkpointId, reason: 'injected post-promotion regression' });
assert.equal(projection.campaigns.get('campaign-1').phase, 'rolled_back');
assert.equal(projection.campaigns.get('campaign-1').rollbackCheckpointId, checkpoint.checkpointId);

console.log(`director: full red/blue/referee vertical slice passed (${stored.length} events)`);
