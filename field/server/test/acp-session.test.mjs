// Conformance for the generic ACP adapter (AcpSession):
//   * it satisfies the formal HarnessAdapter contract (like adapter-contract.test.mjs),
//   * the manifest layer selects it via an explicit harness selector, additively,
//   * driven against a mock ACP agent over real stdio it completes the handshake, round-trips a
//     prompt, translates every ACP session/update into a canonical Field event, and completes a
//     permission handshake — and every event it emits validates against field-event-v1
//     (reusing the schema-validation approach from event-schema.test.mjs).
//
// Nothing here touches the network or a real external agent: the only process is the local Node
// fixture acp-mock-agent.mjs, spawned via an explicit launch spec.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { HarnessAdapter, HARNESS_KINDS, CANONICAL_EVENTS } from '../src/harness/adapter.js';
import { AcpSession } from '../src/harness/acp-session.js';
import { ADAPTER_MANIFESTS, resolveAdapterManifest } from '../src/harness/adapters.js';
import { HarnessSession } from '../src/harness/session.js';
import { KnossosSession } from '../src/harness/knossos-session.js';

const MOCK = fileURLToPath(new URL('./acp-mock-agent.mjs', import.meta.url));
const schema = JSON.parse(readFileSync(new URL('../../../contracts/field-event-v1.schema.json', import.meta.url), 'utf8'));

// --- Compact field-event-v1 validator (same subset of Draft 2020-12 as event-schema.test.mjs) ---
function checkType(value, spec, where, errors) {
  if (spec.enum && !spec.enum.includes(value)) errors.push(`${where}: ${JSON.stringify(value)} not in enum`);
  if (spec.type === 'string' && typeof value !== 'string') errors.push(`${where}: expected string`);
  if (spec.type === 'boolean' && typeof value !== 'boolean') errors.push(`${where}: expected boolean`);
  if (spec.type === 'string' && spec.minLength && typeof value === 'string' && value.length < spec.minLength) {
    errors.push(`${where}: shorter than minLength ${spec.minLength}`);
  }
}
function validate(event) {
  const errors = [];
  if (event === null || typeof event !== 'object' || Array.isArray(event)) return ['not an object'];
  for (const key of schema.required) if (!(key in event)) errors.push(`missing required top-level "${key}"`);
  checkType(event.kind, schema.properties.kind, 'kind', errors);
  if ('sessionId' in event) checkType(event.sessionId, schema.properties.sessionId, 'sessionId', errors);
  for (const clause of schema.allOf ?? []) {
    const cond = clause.if?.properties?.kind;
    const matches = cond?.const !== undefined ? event.kind === cond.const
      : Array.isArray(cond?.enum) ? cond.enum.includes(event.kind) : false;
    if (!matches) continue;
    for (const key of clause.then.required ?? []) if (!(key in event)) errors.push(`kind ${event.kind}: missing required "${key}"`);
    for (const [key, spec] of Object.entries(clause.then.properties ?? {})) {
      if (key in event) checkType(event[key], spec, `${event.kind}.${key}`, errors);
    }
  }
  return errors;
}

// ------------------------------------------------------------------ contract (no process)
{
  const s = new AcpSession({ id: 'contract', cwd: process.cwd() });
  assert.ok(s instanceof HarnessAdapter, 'AcpSession extends HarnessAdapter');
  for (const method of ['start', 'send', 'cancel', 'capabilities', 'stream']) {
    assert.equal(typeof s[method], 'function', `AcpSession implements ${method}()`);
  }
  const caps = s.capabilities();
  assert.equal(caps.kind, 'acp');
  assert.ok(HARNESS_KINDS.includes(caps.kind), 'acp is a known harness kind');
  assert.equal(caps.permissions, 'inline-handshake');
  assert.equal(typeof s.emitEvent, 'function', 'inherits canonical emitEvent');
}

// ------------------------------------------------------------------ manifest selection (additive)
{
  const acp = ADAPTER_MANIFESTS.find((m) => m.id === 'acp');
  assert.ok(acp, 'an acp manifest is registered');
  assert.equal(acp.kind, 'acp');
  assert.deepEqual(acp.endpointKinds, [], 'acp claims no endpoint kind (harness axis, not engine axis)');

  // Explicit harness selector routes to ACP, by manifest id or by kind.
  assert.equal(resolveAdapterManifest('anthropic', 'acp').Adapter, AcpSession, 'explicit acp selector wins over endpoint kind');
  assert.equal(resolveAdapterManifest('openai-compatible', 'acp').Adapter, AcpSession);
  assert.equal(resolveAdapterManifest(undefined, 'acp').Adapter, AcpSession);

  // Absent selector: byte-identical to the historical endpoint-kind mapping.
  assert.equal(resolveAdapterManifest('anthropic').Adapter, HarnessSession, 'default selection unchanged');
  assert.equal(resolveAdapterManifest('openai-compatible').Adapter, KnossosSession, 'default selection unchanged');
  assert.equal(resolveAdapterManifest(undefined).Adapter, HarnessSession, 'unknown still falls back to default');
  assert.notEqual(resolveAdapterManifest('anthropic').Adapter, AcpSession, 'acp is never auto-selected');
  // An unknown explicit selector falls through to endpoint-kind resolution.
  assert.equal(resolveAdapterManifest('anthropic', 'nonexistent-harness').Adapter, HarnessSession);
}

// ------------------------------------------------------------------ live run against the mock agent
const events = [];
const errors = [];
const session = new AcpSession({
  id: 'acp-live', agentId: 'a1', name: 'Ada', role: 'builder', model: 'mock-model',
  endpointId: 'acp-1', cwd: process.cwd(), workspaceId: 'ws', systemPrompt: 'BE CORRECT',
  command: process.execPath, args: [MOCK], providerKind: 'anthropic',
});
session.on('event', (kind, data) => {
  const event = { ...data, kind };
  const errs = validate(event);
  if (errs.length) errors.push(`invalid ${kind}: ${errs.join('; ')}`);
  events.push([kind, data]);
});

function waitFor(predicate, label, timeoutMs = 8000) {
  return new Promise((resolve, reject) => {
    const found = events.find(([k, d]) => predicate(k, d));
    if (found) return resolve(found);
    const timer = setTimeout(() => {
      session.off('event', onEvent);
      reject(new Error(`timed out waiting for ${label}`));
    }, timeoutMs);
    timer.unref?.();
    function onEvent(kind, data) {
      if (!predicate(kind, data)) return;
      clearTimeout(timer);
      session.off('event', onEvent);
      resolve([kind, data]);
    }
    session.on('event', onEvent);
  });
}

try {
  session.start('inspect the workspace');

  // Handshake: session/new -> ready + endpoint.routed.
  await waitFor((k) => k === 'session.state' && events.some(([kk, d]) => kk === 'session.state' && d.state === 'ready'), 'ready state');
  await waitFor((k) => k === 'endpoint.routed', 'endpoint.routed');
  assert.equal(session.acpSessionId, 'mock-s1', 'adapter captured the ACP-side session id');

  // First prompt round-trips and every ACP session/update is translated.
  await waitFor((k) => k === 'session.turn_complete', 'first turn_complete');
  assert.ok(events.some(([k, d]) => k === 'session.message' && d.role === 'user'), 'user turn recorded');
  assert.ok(events.some(([k, d]) => k === 'session.message' && d.role === 'assistant' && /here is what I found/.test(d.text)), 'agent_message_chunk -> session.message');
  assert.ok(events.some(([k, d]) => k === 'session.thinking' && /thinking/.test(d.text)), 'agent_thought_chunk -> session.thinking');
  assert.ok(events.some(([k, d]) => k === 'session.progress' && d.total === 2), 'plan -> session.progress');
  assert.ok(events.some(([k, d]) => k === 'session.tool_use' && d.toolId === 'call-read-1'), 'tool_call -> session.tool_use');
  assert.ok(events.some(([k, d]) => k === 'session.tool_result' && d.toolId === 'call-read-1' && d.ok === true), 'terminal tool_call -> session.tool_result');
  const firstDone = events.find(([k]) => k === 'session.turn_complete');
  assert.equal(firstDone[1].isError, false);
  assert.equal(firstDone[1].stopReason, 'end_turn');

  // Systemprompt is prepended to the first prompt only.
  assert.equal(session.hasTurn, true);

  // Second turn drives the permission handshake: agent asks, operator decides, turn finishes.
  const permCount = events.filter(([k]) => k === 'harness.permission_requested').length;
  session.send('please make the write that needs permission');
  const [, perm] = await waitFor((k) => k === 'harness.permission_requested', 'permission request');
  assert.ok(perm.requestId, 'permission carries a requestId');
  assert.equal(perm.toolName, 'write_file config.json', 'toolName from the ACP toolCall title');
  assert.ok(Array.isArray(perm.options) && perm.options.length === 2, 'permission options passed through');
  assert.equal(events.filter(([k]) => k === 'harness.permission_requested').length, permCount + 1);

  assert.equal(session.decidePermission(perm.requestId, 'allow'), true, 'decidePermission answers the agent');
  await waitFor((k, d) => k === 'session.tool_result' && d.toolId === 'call-danger' && d.ok === true, 'approved write result');
  await waitFor((k, d) => k === 'session.turn_complete' && events.filter(([kk]) => kk === 'session.turn_complete').length >= 2, 'second turn_complete');

  // cancel() tears the process down and ends the session.
  session.cancel();
  await waitFor((k, d) => k === 'session.ended' && d.reason === 'cancelled', 'session.ended');

  assert.equal(errors.length, 0, `every emitted event must validate against field-event-v1:\n${errors.join('\n')}`);

  // Sanity: the kinds this adapter is responsible for were all exercised and validated.
  const produced = new Set(events.map(([k]) => k));
  for (const kind of ['session.state', 'session.message', 'session.thinking', 'session.tool_use',
    'session.tool_result', 'session.progress', 'session.turn_complete', 'endpoint.routed',
    'harness.permission_requested', 'session.ended']) {
    assert.ok(produced.has(kind), `expected to have produced ${kind}`);
    assert.ok(CANONICAL_EVENTS.includes(kind), `${kind} is canonical`);
  }

  console.log(`acp: handshake, prompt round-trip, permission handshake and ${produced.size} canonical event kinds validate against field-event-v1`);
} finally {
  session.cancel();
}
