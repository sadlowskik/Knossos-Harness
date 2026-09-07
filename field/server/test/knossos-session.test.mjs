import assert from 'node:assert/strict';
import { KnossosSession, resolveKnossosBinary } from '../src/harness/knossos-session.js';

const session = new KnossosSession({
  id: 'k1', agentId: 'ornith-1', name: 'Ornith', role: 'builder', model: 'ornith-35b-a3b',
  endpointId: 'cameod-local', cwd: process.cwd(), workspaceId: 'cameo',
  systemPrompt: 'constitution', env: {}, engine: 'cameo',
});
const events = [];
const writes = [];
session.on('event', (kind, data) => events.push([kind, data]));
session.proc = { stdin: { writable: true, write: (line) => writes.push(JSON.parse(line)) } };
session.pendingOrders = 'build the gateway';

session.handleLine(JSON.stringify({ event: 'ready', workspace: process.cwd(), engine: 'cameo', files: 10, symbols: 20 }));
assert.equal(session.state, 'thinking');
assert.deepEqual(writes[0], { cmd: 'capabilities', permissions: true });
assert.equal(writes[1].cmd, 'task');
assert.match(writes[1].text, /constitution/);
assert.match(writes[1].text, /build the gateway/);

session.handleLine(JSON.stringify({ event: 'plan', steps: ['inspect', 'edit', 'verify'] }));
session.handleLine(JSON.stringify({ event: 'verdict', passed: true, summary: 'green', tiers: [{ tier: 1, passed: true }] }));
session.handleLine(JSON.stringify({
  event: 'outcome', halt: 'done', succeeded: true, steps_used: 3,
  summary: 'complete\nFIELD_REPORT: {"kind":"objective_satisfied","evidence":["green"]}',
  changed: ['gateway.rs'], dry_run: false,
}));
session.handleLine(JSON.stringify({ event: 'permission_request', id: 7, tool: 'write', input: { path: 'x' } }));

assert.ok(events.some(([kind, data]) => kind === 'session.progress' && data.total === 3));
assert.ok(events.some(([kind, data]) => kind === 'session.verification' && data.passed));
assert.ok(events.some(([kind, data]) => kind === 'session.turn_complete' && /FIELD_REPORT/.test(data.result)));
assert.ok(events.some(([kind, data]) => kind === 'harness.permission_requested' && data.requestId === 7));

session.decidePermission(7, 'allow');
assert.deepEqual(writes.at(-1), { cmd: 'permission', id: 7, allow: true });
assert.ok(
  /(?:knossos|daedalus)(?:\.exe)?$/.test(resolveKnossosBinary()),
  'the adapter resolves the Knossos binary or the v0.1 compatibility name',
);

console.log('knossos: Cameo adapter event translation and permission handshake passed');
