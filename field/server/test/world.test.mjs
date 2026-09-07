import assert from 'node:assert/strict';
import { Projection } from '../src/store/projection.js';
import { rehearsalSnapshot } from '../../web/src/theater/fieldPreferences.js';

const projection = new Projection({
  workspaces: [{ id: 'alpha', name: 'Alpha', mounted: true }],
  endpoints: [],
  websites: [],
});

projection.apply({
  seq: 1,
  ts: 1000,
  kind: 'world.capital_selected',
  subject: 'alpha',
  data: { workspaceId: 'alpha' },
});
projection.apply({
  seq: 2,
  ts: 1001,
  kind: 'world.territory_assigned',
  subject: 'workspace:alpha',
  data: {
    clusterKey: 'workspace:alpha',
    territoryId: 'italia',
    label: 'Alpha',
    kind: 'project',
    workspaceId: 'alpha',
  },
});

const snap = projection.snapshot(2000);
assert.equal(snap.world.capitalWorkspaceId, 'alpha');
assert.equal(snap.world.capitalSelectedAt, 1000);
assert.equal(snap.world.assignments['workspace:alpha'].territoryId, 'italia');
assert.equal(snap.world.revision, 2);

snap.world.assignments['workspace:alpha'].territoryId = 'mutated';
assert.equal(projection.world.assignments['workspace:alpha'].territoryId, 'italia');

projection.apply({ seq: 3, ts: 1002, kind: 'world.territory_released', data: { clusterKey: 'workspace:alpha' } });
assert.equal(projection.snapshot().world.assignments['workspace:alpha'], undefined);

projection.apply({
  seq: 4, ts: 1003, kind: 'session.spawned', source: 'observed', subject: 'real-session',
  data: { sessionId: 'real-session', workspaceId: 'alpha', role: 'builder' },
});
projection.apply({
  seq: 5, ts: 1004, kind: 'fs.changed', source: 'observed', subject: 'real-session',
  data: { sessionId: 'real-session', workspaceId: 'alpha', dir: 'src', path: 'src/real.js', change: 'change' },
});
projection.apply({
  seq: 6, ts: 1005, kind: 'simulation.started', source: 'synthetic', subject: 'demo-1',
  data: { simulationRunId: 'demo-1', simulated: true },
});
projection.apply({
  seq: 7, ts: 1006, kind: 'session.spawned', source: 'synthetic', subject: 'sim-session',
  data: { sessionId: 'sim-session', workspaceId: 'alpha', role: 'verifier', simulated: true, simulationRunId: 'demo-1' },
});
projection.apply({
  seq: 8, ts: 1007, kind: 'fs.changed', source: 'synthetic', subject: 'sim-session',
  data: { sessionId: 'sim-session', workspaceId: 'alpha', dir: 'demo', path: 'demo/fake.js', change: 'add', simulated: true },
});
let partitioned = projection.snapshot();
assert.deepEqual(partitioned.sessions.map((session) => session.id), ['real-session']);
assert.deepEqual(partitioned.files.map((file) => file.path), ['src/real.js']);
assert.equal(partitioned.workspaces[0].changeCount, 1);
assert.deepEqual(partitioned.rehearsal.sessions.map((session) => session.id), ['sim-session']);
assert.deepEqual(partitioned.rehearsal.files.map((file) => file.path), ['demo/fake.js']);
assert.equal(partitioned.rehearsal.workspaces[0].changeCount, 1);
const rehearsalView = rehearsalSnapshot(partitioned, { runId: 'demo-1' });
assert.deepEqual(rehearsalView.sessions.map((session) => session.id), ['sim-session']);
assert.deepEqual(rehearsalView.files.map((file) => file.path), ['demo/fake.js']);
assert.equal(rehearsalView.rehearsalMode, true);
assert.equal(rehearsalSnapshot(partitioned, null), partitioned, 'normal UI receives production state unchanged');

projection.apply({
  seq: 9, ts: 1008, kind: 'simulation.started', source: 'synthetic', subject: 'demo-2',
  data: { simulationRunId: 'demo-2', simulated: true },
});
partitioned = projection.snapshot();
assert.equal(partitioned.rehearsal.sessions.length, 0, 'a new rehearsal resets prior synthetic state');
assert.equal(partitioned.sessions.length, 1, 'resetting a rehearsal cannot alter production state');

projection.apply({ seq: 10, ts: 1010, kind: 'session.ended', source: 'observed', data: { sessionId: 'real-session', reason: 'completed' } });
assert.equal(projection.snapshot(2000).workspaces[0].maturity.score, 0, 'a done session cannot manufacture maturity');
projection.apply({
  seq: 11, ts: 1011, kind: 'campaign.created', source: 'observed',
  data: { campaignId: 'release', name: 'Release', intent: 'Ship', scope: 'sandbox', target: { type: 'workspace', id: 'alpha', workspaceId: 'alpha' }, doctrine: {}, concurrency: 1, budgetUsd: 1 },
});
projection.apply({
  seq: 12, ts: 1012, kind: 'objective.created', source: 'observed',
  data: { campaignId: 'release', objectiveId: 'criterion', statement: 'Pass release check', definitionOfDone: ['test passes'], required: true, target: { type: 'workspace', id: 'alpha', workspaceId: 'alpha' } },
});
projection.apply({
  seq: 13, ts: 1013, kind: 'objective.satisfied', source: 'observed',
  data: { campaignId: 'release', objectiveId: 'criterion', evidence: ['test://pass'], criteriaEvidence: [{ criterion: 'test passes', evidence: 'test://pass' }] },
});
let maturity = projection.snapshot(2000).workspaces[0].maturity;
assert.equal(maturity.complete, 100);
assert.equal(maturity.verified, 0, 'completion evidence is not an independent verdict');
assert.equal(maturity.persisted, 0, 'completion evidence is not a revision-bound checkpoint');
projection.apply({
  seq: 14, ts: 1014, kind: 'referee.verdict', source: 'observed',
  data: { campaignId: 'release', verdictId: 'verdict-1', sessionId: 'independent-referee', verdict: 'verified', evidence: ['replay://pass'], rationale: 'independent replay passed' },
});
projection.apply({ seq: 15, ts: 1015, kind: 'campaign.phase_changed', source: 'observed', data: { campaignId: 'release', from: 'referee_review', to: 'verified' } });
maturity = projection.snapshot(2000).workspaces[0].maturity;
assert.equal(maturity.verified, 100);
assert.equal(maturity.persisted, 0);
projection.apply({
  seq: 16, ts: 1016, kind: 'campaign.checkpoint_created', source: 'observed',
  data: { campaignId: 'release', checkpointId: 'checkpoint-1', revision: 'sha256:abcdef12', eventSeq: 15 },
});
maturity = projection.snapshot(2000).workspaces[0].maturity;
assert.equal(maturity.score, 100);
assert.equal(maturity.persisted, 100);
assert.ok(maturity.evidence.some((item) => item.type === 'verification' && item.verdictId === 'verdict-1'));
assert.ok(maturity.evidence.some((item) => item.type === 'persistence' && item.checkpointId === 'checkpoint-1'));

console.log('world: durable territory, synthetic partition, and evidence-derived maturity passed');
