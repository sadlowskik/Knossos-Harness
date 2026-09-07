import assert from 'node:assert/strict';
import { Registry } from '../src/harness/registry.js';

let seq = 0;
const events = [];
const emit = (kind, data, meta = {}) => {
  const event = { seq: ++seq, ts: Date.now(), kind, data, ...meta };
  events.push(event);
  return event;
};
const cfg = {
  defaults: { budget_usd_per_session: 1 },
  workspaces: [{ id: 'cameo', path: process.cwd(), mounted: true }],
  roles: [{ id: 'builder', default_endpoint: 'local', default_thinking: 'medium' }],
  agents: [{ id: 'builder-1', role: 'builder', endpoint: 'local', thinking: 'medium' }],
  endpoints: [
    { id: 'local', kind: 'openai-compatible', model: 'ornith' },
    { id: 'cloud', kind: 'openai-compatible', model: 'frontier' },
  ],
};

const registry = new Registry({
  cfg,
  emit,
  apiBase: 'http://127.0.0.1:1',
  permissionCapabilities: { mint: () => 'test-token', revoke() {} },
});
registry.endpointStatus.set('local', 'up');
registry.endpointStatus.set('cloud', 'up');

const counters = { pause: 0, resume: 0 };
const session = {
  id: 's1', agentId: 'builder-1', role: 'builder', state: 'working', workspaceId: 'cameo',
  endpointId: 'local', model: 'ornith', proc: {},
  pause() { counters.pause++; this.proc = null; },
  resume() { counters.resume++; this.proc = {}; },
};
registry.sessions.set('s1', session);
registry.meta.set('s1', { budgetUsd: 1, costUsd: 0, campaignId: 'c1', team: 'blue', objectiveId: 'o1' });

registry.onEndpointHealth('local', 'down');
assert.equal(session.endpointId, 'cloud');
assert.equal(counters.pause, 1);
assert.equal(counters.resume, 1);
assert.equal(events.filter((e) => e.kind === 'endpoint.routed').length, 1);

// Repeating the same down status cannot produce a reroute storm.
registry.onEndpointHealth('local', 'down');
assert.equal(events.filter((e) => e.kind === 'endpoint.routed').length, 1);

// When the remaining endpoint goes down, fail closed and expose a blocked session. Do not
// lie by "rerouting" it back onto the already-down preferred endpoint.
registry.onEndpointHealth('cloud', 'down');
assert.equal(counters.pause, 2);
assert.equal(counters.resume, 1);
assert.equal(events.at(-1).kind, 'session.state');
assert.equal(events.at(-1).data.state, 'blocked');
assert.match(events.at(-1).data.detail, /no alternative/);
assert.equal(registry.route('auto', 'builder').endpointId, null);

// Budget exhaustion pauses once the recorded cumulative cost crosses the configured cap.
session.proc = {};
registry.onSessionEvent(session, 'session.usage', { sessionId: 's1', costUsd: 1.25 });
assert.equal(events.some((e) => e.kind === 'session.state' && /budget exhausted/.test(e.data.detail)), true);
assert.equal(registry.meta.get('s1').costUsd, 1.25);

const secondSession = { id: 's2', agentId: 'builder-1' };
registry.meta.get('s1').assignmentId = 'assignment-success';
registry.meta.set('s2', { assignmentId: 'assignment-success' });
registry.assignments.set('assignment-success', {
  members: new Set(['s1', 's2']), outcomes: new Map(), settled: false,
});
registry.onSessionEvent(session, 'session.ended', { sessionId: 's1', reason: 'exit' });
assert.equal(events.filter((event) => event.kind === 'assignment.completed').length, 0);
registry.onSessionEvent(secondSession, 'session.ended', { sessionId: 's2', reason: 'exit' });
registry.onSessionEvent(secondSession, 'session.ended', { sessionId: 's2', reason: 'exit' });
assert.equal(events.filter((event) => event.kind === 'assignment.completed').length, 1, 'completion converges exactly once');

registry.meta.set('s3', { assignmentId: 'assignment-failure' });
registry.assignments.set('assignment-failure', {
  members: new Set(['s3']), outcomes: new Map(), settled: false,
});
registry.onSessionEvent({ id: 's3', agentId: 'builder-1' }, 'session.ended', { sessionId: 's3', reason: 'error' });
assert.equal(events.filter((event) => event.kind === 'assignment.failed').length, 1);

session.proc = {};
registry.meta.get('s1').campaignId = 'shared-budget';
registry.meta.get('s1').budgetUsd = 10;
registry.campaignPolicy = () => ({ id: 'shared-budget', costUsd: 2.1, budgetUsd: 2, budgetExhausted: true });
const pausesBeforeCampaignCap = counters.pause;
registry.onSessionEvent(session, 'session.usage', { sessionId: 's1', costUsd: 1.4 });
registry.onSessionEvent(session, 'session.usage', { sessionId: 's1', costUsd: 1.5 });
assert.equal(events.filter((event) => event.kind === 'campaign.budget_exhausted').length, 1);
assert.equal(counters.pause, pausesBeforeCampaignCap + 1, 'shared campaign cap pauses the cohort only once');

let completionPauses = 0;
const completionSession = { id: 's4', agentId: 'builder-1', pause() { completionPauses += 1; } };
registry.sessions.set('s4', completionSession);
registry.meta.set('s4', {
  budgetUsd: 10, costUsd: 0, outputTokens: 0, maxOutputTokens: 100,
  budgetStopped: false, campaignId: null,
});
registry.onSessionEvent(completionSession, 'session.usage', { sessionId: 's4', outputTokens: 100 });
registry.onSessionEvent(completionSession, 'session.usage', { sessionId: 's4', outputTokens: 120 });
assert.equal(completionPauses, 1);
assert.equal(events.filter((event) => event.kind === 'budget.exhausted' && event.data.sessionId === 's4').length, 1);

for (const kind of ['resume', 'escalate', 'say', 'redirect']) {
  assert.throws(() => registry.command(kind, { sessionIds: ['s1'] }), /exhausted/);
}
registry.onSessionEvent(session, 'session.usage', { sessionId: 's1', costUsd: 0.1 });
assert.ok(registry.meta.get('s1').costUsd >= 1, 'out-of-order usage never refunds spent cost');
assert.throws(() => registry.command('resume', { sessionIds: ['s1'] }), /exhausted/);
const deadlineSession = { ...session, id: 'deadline' };
registry.sessions.set('deadline', deadlineSession);
registry.meta.set('deadline', { deadlineAt: Date.now() - 1, budgetUsd: 10, costUsd: 0 });
assert.throws(() => registry.command('resume', { sessionIds: ['deadline'] }), /exhausted/);
console.log('registry: routing, budgets, resume bypass refusal, and exactly-once assignment settlement passed');

// Exercise the actual admission path; only the external child launch is stubbed.
const { KnossosSession } = await import('../src/harness/knossos-session.js');
const originalStart = KnossosSession.prototype.start;
let launched = 0;
const admission = new Registry({
  cfg: { ...cfg, constitutions: [], missions: [], roles: cfg.roles.map(r => ({ ...r, body: '' })) },
  emit, apiBase: 'http://127.0.0.1:1',
  campaignPolicy: () => ({ budgetUsd: 2.5, costUsd: 0 }),
});
try {
  KnossosSession.prototype.start = function () { launched++; };
  const first = admission.spawn({ agentId: 'builder-1', campaignId: 'limited', budgetUsd: 2 });
  const second = admission.spawn({ agentId: 'builder-1', campaignId: 'limited', budgetUsd: 2 });
  assert.equal(admission.meta.get(first.id).budgetUsd, 2);
  assert.equal(admission.meta.get(second.id).budgetUsd, 0.5);
  assert.throws(() => admission.spawn({ agentId: 'builder-1', campaignId: 'limited' }), /reserved/);
  assert.equal(launched, 2, 'denied work never starts a child');
  assert.throws(() => admission.command('verify', { sessionIds: [first.id], verifierAgentId: 'builder-1' }), /reserved/);
  admission.assignments.set('follow-up', { members: new Set([first.id]), outcomes: new Map(), settled: false });
  assert.throws(() => admission.command('reinforce', {
    assignmentId: 'follow-up', agentIds: ['builder-1'],
    target: { type: 'workspace', id: 'cameo', workspaceId: 'cameo' },
  }), /reserved/);
  assert.equal(launched, 2, 'verification and reinforcement cannot escape campaign reservations');
  assert.throws(() => admission.spawn({ agentId: 'builder-1', verifyFor: first.id, campaignId: 'bypass' }), /cannot differ/);
  admission.campaignPolicy = () => ({ budgetUsd: 10, costUsd: 0 });
  const reinforced = admission.spawn({ agentId: 'builder-1', assignmentId: 'follow-up' });
  assert.equal(admission.meta.get(reinforced.id).campaignId, 'limited');
  assert.ok(admission.assignments.get('follow-up').members.has(reinforced.id), 'reinforcement joins settlement membership');
  admission.campaignPolicy = () => ({ budgetUsd: 2.5, costUsd: 0 });
  admission.onSessionEvent(first, 'session.ended', { sessionId: first.id, reason: 'error' });
  assert.throws(() => admission.spawn({ agentId: 'builder-1', campaignId: 'limited' }), /reserved/, 'unbilled failure cannot refund the budget');
} finally {
  KnossosSession.prototype.start = originalStart;
  admission.shutdown();
}

const capacity = new Registry({ cfg: { ...cfg, defaults: { ...cfg.defaults, max_concurrent_sessions: 2 } }, emit,
  campaignPolicy: () => ({ concurrency: 1 }) });
capacity.sessions.set('running', { id: 'running', endpointId: 'local', proc: {} });
capacity.meta.set('running', { campaignId: 'one' });
assert.throws(() => capacity.assertCapacity({ endpointId: 'cloud', campaignId: 'one' }), /campaign session/);
capacity.assertCapacity({ sessionId: 'running', endpointId: 'local', campaignId: 'one' });
capacity.sessions.set('other', { id: 'other', endpointId: 'cloud', proc: {} });
assert.throws(() => capacity.spawn({ agentId: 'builder-1' }), /global session/);
let resumed = false;
capacity.sessions.set('paused', { id: 'paused', endpointId: 'local', proc: null, resume() { resumed = true; } });
assert.throws(() => capacity.command('resume', { sessionIds: ['paused'] }), /global session/);
assert.equal(resumed, false);
capacity.sessions.delete('other');
capacity.cfg = { ...capacity.cfg, endpoints: cfg.endpoints.map(e => ({ ...e, max_concurrent_sessions: 1 })) };
assert.throws(() => capacity.spawn({ agentId: 'builder-1' }), /endpoint session/);
capacity.sessions.get('running').proc = null;
capacity.command('resume', { sessionIds: ['paused'] });
assert.equal(resumed, true, 'drained processes release capacity');

// A terminated direct harness must not reuse its revoked permission capability.
const { HarnessSession } = await import('../src/harness/session.js');
const { createControlSecurity } = await import('../src/security.js');
const authority = createControlSecurity({ port: 7749 });
const registeredSecrets = [];
const direct = new Registry({
  cfg: { ...cfg, constitutions: [], missions: [], roles: cfg.roles.map(r => ({ ...r, body: '' })),
    endpoints: [{ id: 'local', kind: 'anthropic' }] },
  emit, apiBase: 'http://127.0.0.1:7749',
  permissionCapabilities: { mint: id => authority.mintHarnessToken(id), revoke: id => authority.revokeHarnessToken(id) },
  registerSecret: token => registeredSecrets.push(token),
});
const directStart = HarnessSession.prototype.start;
const authorized = token => authority.authorizeRequest({ method: 'POST', socket: { remoteAddress: '127.0.0.1' },
  headers: { host: '127.0.0.1:7749', 'content-type': 'application/json', authorization: `Bearer ${token}` } },
new URL('http://127.0.0.1:7749/api/internal/permission')).ok === true;
try {
  HarnessSession.prototype.start = function () { this.beforeStart?.(); return this; };
  const s = direct.spawn({ agentId: 'builder-1' });
  const token = () => JSON.parse(s.mcpConfig).mcpServers.field.env.FIELD_INTERNAL_TOKEN;
  const oldToken = token();
  assert.equal(authorized(oldToken), true);
  direct.onSessionEvent(s, 'session.ended', { sessionId: s.id, reason: 'exit' });
  assert.equal(authorized(oldToken), false);
  assert.equal(direct.sessionDeadlineTimers.has(s.id), false);
  s.resume();
  assert.equal(direct.sessionDeadlineTimers.has(s.id), true, 'resume rearms the original absolute deadline');
  assert.notEqual(token(), oldToken);
  assert.equal(authorized(token()), true);
  assert.equal(authorized(oldToken), false);
  assert.ok(registeredSecrets.includes(token()), 'replacement token enters log redaction');
} finally {
  HarnessSession.prototype.start = directStart;
  direct.shutdown();
}
