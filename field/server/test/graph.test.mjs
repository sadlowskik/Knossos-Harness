import assert from 'node:assert/strict';
import { GraphProjection } from '../src/orchestration/graph-projection.js';

let seq = 0;
const T0 = 1_700_000_000_000;
const event = (kind, data, ts = T0 + seq * 1000) => ({ kind, data, ts, seq: ++seq });

const g = new GraphProjection();
g.apply(event('campaign.created', {
  campaignId: 'c1', name: 'Gateway', target: { workspaceId: 'cameo' },
}));
g.apply(event('session.spawned', {
  sessionId: 'blue-1', agentId: 'rhea', name: 'Rhea', role: 'builder',
  workspaceId: 'cameo', endpointId: 'local', model: 'ornith', campaignId: 'c1', team: 'blue',
}));
g.apply(event('team.member_assigned', {
  campaignId: 'c1', team: 'blue', sessionId: 'blue-1', agentId: 'rhea', role: 'builder',
}));

for (let i = 0; i < 3; i++) {
  g.apply(event('session.tool_use', {
    sessionId: 'blue-1', name: 'Read', workspaceId: 'cameo', dir: 'cameod/src', path: `cameod/src/f${i}.rs`,
  }, T0 + i * 31_000));
}

g.apply(event('finding.reported', {
  campaignId: 'c1', objectiveId: 'o1', findingId: 'f1', authorSessionId: 'red-1',
  claim: 'rollback race', severity: 'high',
}));
g.apply(event('mitigation.proposed', {
  campaignId: 'c1', mitigationId: 'm1', findingIds: ['f1'], claim: 'serialize rollback',
}));
g.apply(event('retest.completed', {
  campaignId: 'c1', findingId: 'f1', sessionId: 'red-1', result: 'fixed',
}));

let snap = g.snapshot(T0 + 70_000);
const folder = snap.nodes.find((n) => n.key === 'folder:cameo:cameod/src');
assert.equal(folder.state, 'active_worksite');
assert.ok(snap.edges.some((e) => e.type === 'working_on' && e.to === folder.key));
assert.ok(snap.edges.some((e) => e.type === 'mitigated_by'));
assert.ok(snap.edges.some((e) => e.type === 'retested_by'));

snap = g.snapshot(T0 + 33 * 60_000);
assert.equal(snap.nodes.find((n) => n.key === folder.key).state, 'dormant');
assert.ok(!snap.visibleNodeKeys.includes(folder.key));

g.apply(event('session.tool_use', {
  sessionId: 'blue-1', name: 'Read', workspaceId: 'cameo', dir: 'cameod/src', path: 'cameod/src/app.rs',
}, T0 + 34 * 60_000));
snap = g.snapshot(T0 + 34 * 60_000);
assert.ok(['active_worksite', 'established'].includes(snap.nodes.find((n) => n.key === folder.key).state));

console.log('graph: typed topology, lifecycle hysteresis, and contest edges passed');
