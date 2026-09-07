import assert from 'node:assert/strict';
import { performance } from 'node:perf_hooks';
import { Projection } from '../src/store/projection.js';
import { computeLayout } from '../../web/src/field/layout.js';

const cfg = {
  workspaces: [{ id: 'cameo', name: 'Cameo', path: '/cameo', mounted: true, region: { x: 0, y: 0, w: 900, h: 500 } }],
  endpoints: [
    { id: 'local-a', name: 'Local A', kind: 'openai-compatible', model: 'ornith', cost_per_mtok: { input: 0, output: 0 } },
    { id: 'cloud-b', name: 'Cloud B', kind: 'anthropic', model: 'frontier', cost_per_mtok: { input: 3, output: 15 } },
  ],
  websites: [],
};

const events = [];
let seq = 0;
const T0 = Date.now() - 120_000;
const push = (kind, data, subject = null) => events.push({
  seq: ++seq, ts: T0 + seq, kind, actor: data.sessionId ?? null, subject, data,
});

push('campaign.created', {
  campaignId: 'stress-campaign', commandId: 'stress-create', name: 'Thirty-agent operation',
  intent: 'Exercise field density and replay.', scope: 'sandbox', concurrency: 30,
  budgetUsd: 200, doctrine: { blockingSeverity: 'high', requireRed: true, requireReferee: true, redCategories: ['reliability'] },
  target: { type: 'workspace', id: 'cameo', workspaceId: 'cameo' }, objectives: [],
}, 'stress-campaign');
for (let o = 0; o < 6; o++) {
  push('objective.created', {
    campaignId: 'stress-campaign', objectiveId: `objective-${o}`,
    statement: `Objective ${o}`, definitionOfDone: [`check ${o}`], priority: o + 1,
    risk: 'medium', target: { type: 'workspace', id: 'cameo', workspaceId: 'cameo' },
  }, `objective-${o}`);
}

for (let i = 0; i < 30; i++) {
  const team = i < 18 ? 'blue' : i < 27 ? 'red' : i < 29 ? 'referee' : 'purple';
  const sid = `agent-${i}`;
  push('session.spawned', {
    sessionId: sid, agentId: sid, name: `Agent ${i}`, role: team === 'referee' ? 'verifier' : team === 'red' ? 'challenger' : 'builder',
    model: 'ornith', endpointId: i % 2 ? 'cloud-b' : 'local-a', thinking: 'medium',
    workspaceId: 'cameo', campaignId: 'stress-campaign', team, objectiveId: `objective-${i % 6}`,
  }, sid);
  push('team.member_assigned', {
    campaignId: 'stress-campaign', team, sessionId: sid, agentId: sid,
    role: team === 'referee' ? 'verifier' : team === 'red' ? 'challenger' : 'builder',
    objectiveId: `objective-${i % 6}`, status: 'active',
  }, 'stress-campaign');
  push('objective.assigned', {
    campaignId: 'stress-campaign', objectiveId: `objective-${i % 6}`, team, sessionIds: [sid],
  }, `objective-${i % 6}`);
  push('session.state', { sessionId: sid, campaignId: 'stress-campaign', state: 'working' }, sid);
}

push('campaign.phase_changed', { campaignId: 'stress-campaign', from: 'draft', to: 'mobilizing' }, 'stress-campaign');
push('campaign.phase_changed', { campaignId: 'stress-campaign', from: 'mobilizing', to: 'blue_building' }, 'stress-campaign');

// 100k total events: repeated work strengthens typed edges while 5,000 distinct files
// prove the websocket snapshot truncates without losing canonical graph state.
while (events.length < 100_000) {
  const i = events.length;
  const sid = `agent-${i % 30}`;
  if (i % 97 === 0) {
    push('agent.communication', {
      campaignId: 'stress-campaign', fromSessionId: sid, toSessionId: `agent-${(i + 7) % 30}`, channel: 'handoff',
    }, 'stress-campaign');
  } else if (i % 211 === 0) {
    push('session.usage', {
      sessionId: sid, campaignId: 'stress-campaign', inputTokens: i, outputTokens: i / 4,
      contextTokens: i % 180_000, costUsd: (i % 1000) / 1000,
    }, sid);
  } else {
    push('session.tool_use', {
      sessionId: sid, campaignId: 'stress-campaign', name: i % 5 === 0 ? 'Grep' : 'Read',
      workspaceId: 'cameo', dir: `zone-${i % 200}`, path: `zone-${i % 200}/file-${i % 10000}.rs`,
      summary: `inspect file ${i % 10000}`,
    }, sid);
  }
}

// Endpoint loss and an idempotent handoff are represented as ordinary facts at the tail.
push('endpoint.health', { endpointId: 'local-a', status: 'down', detail: 'injected outage' }, 'local-a');
push('endpoint.routed', { sessionId: 'agent-0', endpointId: 'cloud-b', model: 'frontier', reason: 'local-a went down' }, 'agent-0');
push('team.handoff_started', { campaignId: 'stress-campaign', team: 'blue', fromSessionId: 'agent-0', toSessionId: 'agent-1' }, 'stress-campaign');
push('team.handoff_completed', { campaignId: 'stress-campaign', team: 'blue', fromSessionId: 'agent-0', toSessionId: 'agent-1' }, 'stress-campaign');

function replay() {
  const p = new Projection(cfg);
  for (const event of events) p.apply(event);
  return p;
}

const start = performance.now();
const live = replay();
const firstMs = performance.now() - start;
const secondStart = performance.now();
const rebuilt = replay();
const secondMs = performance.now() - secondStart;

const snapshotStart = performance.now();
const a = live.snapshot();
const snapshotMs = performance.now() - snapshotStart;
const b = rebuilt.snapshot();
const layoutStart = performance.now();
const layout = computeLayout(a, a.positions, a.now, { missions: [] });
const layoutMs = performance.now() - layoutStart;
assert.equal(a.seq, events.at(-1).seq);
assert.equal(a.sessions.length, 30);
assert.equal(a.campaigns[0].teams.blue.members.filter((m) => m.status === 'active').length, 17);
assert.equal(a.endpoints.find((e) => e.id === 'local-a').status, 'down');
assert.equal(a.sessions.find((s) => s.id === 'agent-0').endpointId, 'cloud-b');
assert.equal(a.graph.totals.nodes, b.graph.totals.nodes);
assert.equal(a.graph.totals.edges, b.graph.totals.edges);
assert.deepEqual(a.campaigns, b.campaigns);
assert.equal(a.graph.nodes.length, 4000);
assert.equal(a.graph.truncated, true);
assert.ok(a.graph.edges.length <= 8000);
assert.ok(a.graph.totals.nodes >= 10_000, `expected 10k graph nodes, got ${a.graph.totals.nodes}`);
assert.equal(layout.agents.length, 30);
assert.ok(snapshotMs < 500, `bounded snapshot too slow: ${snapshotMs.toFixed(0)}ms`);
assert.ok(layoutMs < 50, `field layout too slow: ${layoutMs.toFixed(0)}ms`);
assert.ok(firstMs < 15_000 && secondMs < 15_000, `replay too slow: ${firstMs.toFixed(0)} / ${secondMs.toFixed(0)}ms`);

console.log(`stress: ${events.length.toLocaleString()} events, 30 agents, ${a.graph.totals.nodes} graph nodes; replay ${firstMs.toFixed(0)}ms/${secondMs.toFixed(0)}ms, snapshot ${snapshotMs.toFixed(1)}ms, layout ${layoutMs.toFixed(1)}ms`);
