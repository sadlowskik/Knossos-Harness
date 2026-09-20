import assert from 'node:assert/strict';
import { BudgetLedger } from '../src/budget-ledger.js';

const ledger = new BudgetLedger();
const events = [];
const apply = (kind, data) => { const event = { kind, data }; events.push(event); ledger.apply(event); };
apply('budget.reserved', { sessionId: 'a', campaignId: 'campaign', limitUsd: 4 });
apply('budget.reserved', { sessionId: 'b', campaignId: 'campaign', limitUsd: 4 });
assert.equal(ledger.campaign('campaign', 0, 10).remainingUsd, 2);
assert.equal(ledger.campaign('campaign', 0, 10).unknownCostSessions, 2);
apply('session.usage', { sessionId: 'a', costUsd: 1 });
apply('session.usage', { sessionId: 'a', costUsd: 0.5 });
assert.equal(ledger.campaign('campaign', 1, 10).remainingUsd, 2);
apply('session.turn_complete', { sessionId: 'a' });
apply('session.ended', { sessionId: 'a' });
assert.equal(ledger.campaign('campaign', 1, 10).remainingUsd, 5);
apply('session.ended', { sessionId: 'b' });
assert.equal(ledger.campaign('campaign', 1, 10).remainingUsd, 5, 'missing telemetry is not refunded');
apply('budget.reactivated', { sessionId: 'a' });
assert.equal(ledger.campaign('campaign', 1, 10).remainingUsd, 2);
const replay = new BudgetLedger();
for (const event of events) { replay.apply(event); replay.apply(event); }
assert.deepEqual(replay.snapshot(), ledger.snapshot(), 'replay and duplicate delivery retain the same reservations');
const zero = new BudgetLedger();
zero.apply({ kind: 'budget.reserved', data: { sessionId: 'zero', limitUsd: 1 } });
zero.apply({ kind: 'session.usage', data: { sessionId: 'zero', costUsd: 0 } });
assert.equal(zero.snapshot()[0].costStatus, 'reported_zero');
zero.apply({ kind: 'session.usage', data: { sessionId: 'zero', costUsd: NaN } });
assert.equal(zero.snapshot()[0].spentUsd, 0);
console.log('budgets: concurrent reservation, unknown/zero cost, monotonic usage and replay passed');

// UI and campaign totals must agree with the replayed reservation ledger.
const { Projection } = await import('../src/store/projection.js');
const projection = new Projection({ endpoints: [], workspaces: [], websites: [] });
let seq = 0;
for (const value of [2, 1, 2, NaN, Infinity, -1, 3]) {
  projection.apply({ seq: ++seq, ts: seq, kind: 'session.usage', data: {
    sessionId: 'cumulative', costUsd: value, inputTokens: value,
    outputTokens: value, deltaInput: 99, deltaOutput: 99,
  } });
}
assert.equal(projection.totals.costUsd, 3);
assert.equal(projection.totals.inputTokens, 3);
assert.equal(projection.totals.outputTokens, 3);
assert.equal(projection.sessions.get('cumulative').costUsd, 3);
