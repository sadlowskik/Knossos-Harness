import assert from 'node:assert/strict';
import { HarnessSession } from '../src/harness/session.js';

const session = new HarnessSession({
  id: 'usage-1', agentId: 'agent', name: 'Agent', role: 'builder', model: 'model',
  endpointId: 'endpoint', cwd: process.cwd(), workspaceId: 'workspace',
  systemPrompt: '', workspaces: [], providerKind: 'anthropic',
});
const events = [];
session.on('event', (kind, data) => events.push({ kind, data }));

session.emitUsage({ input_tokens: 10, output_tokens: 5, cache_read_input_tokens: 2 });
session.handleResult({
  usage: { input_tokens: 10, output_tokens: 5, cache_read_input_tokens: 2 },
  total_cost_usd: 0.25, result: 'done', num_turns: 1, duration_ms: 10,
});
const usage = events.filter((event) => event.kind === 'session.usage');
assert.equal(usage.reduce((sum, event) => sum + (event.data.deltaOutput ?? 0), 0), 5, 'result usage cannot double-count streamed completion tokens');
assert.equal(usage.at(-1).data.costUsd, 0.25);

session.emitUsage({ input_tokens: 4, output_tokens: 7, cache_read_input_tokens: 0 });
const latest = events.filter((event) => event.kind === 'session.usage').at(-1).data;
assert.equal(latest.inputTokens, 14);
assert.equal(latest.outputTokens, 12);
assert.equal(latest.deltaOutput, 7);

console.log('session usage: cumulative totals and non-duplicated result accounting passed');
