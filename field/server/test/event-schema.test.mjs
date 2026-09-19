// Conformance: every canonical event emitted by BOTH shipping adapters must validate against
// contracts/field-event-v1.schema.json. Rather than pull in a JSON-Schema library (the repo has
// none), this test interprets the exact subset of Draft 2020-12 the schema uses:
// top-level `required` + `properties.kind.enum`, and an `allOf` of { if:{kind}, then:{required, properties} }.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { CANONICAL_EVENTS } from '../src/harness/adapter.js';

process.env.FIELD_CLAUDE_BIN = process.execPath;
process.env.FIELD_KNOSSOS_BIN = process.execPath;
const { HarnessSession } = await import('../src/harness/session.js');
const { KnossosSession } = await import('../src/harness/knossos-session.js');

const schema = JSON.parse(readFileSync(new URL('../../../contracts/field-event-v1.schema.json', import.meta.url), 'utf8'));

// The schema's `kind` enum and the code's CANONICAL_EVENTS must agree exactly.
assert.deepEqual(
  [...schema.properties.kind.enum].sort(),
  [...CANONICAL_EVENTS].sort(),
  'schema kind enum and CANONICAL_EVENTS drifted apart',
);

function checkType(value, spec, where, errors) {
  if (spec.enum && !spec.enum.includes(value)) errors.push(`${where}: ${JSON.stringify(value)} not in enum`);
  if (spec.type === 'string' && typeof value !== 'string') errors.push(`${where}: expected string`);
  if (spec.type === 'boolean' && typeof value !== 'boolean') errors.push(`${where}: expected boolean`);
  if (spec.type === 'string' && spec.minLength && typeof value === 'string' && value.length < spec.minLength) {
    errors.push(`${where}: shorter than minLength ${spec.minLength}`);
  }
}

/** Validate one event object against the field-event-v1 schema. Returns a list of errors. */
function validate(event) {
  const errors = [];
  if (event === null || typeof event !== 'object' || Array.isArray(event)) return ['not an object'];
  for (const key of schema.required) {
    if (!(key in event)) errors.push(`missing required top-level "${key}"`);
  }
  checkType(event.kind, schema.properties.kind, 'kind', errors);
  if ('sessionId' in event) checkType(event.sessionId, schema.properties.sessionId, 'sessionId', errors);
  for (const clause of schema.allOf ?? []) {
    const cond = clause.if?.properties?.kind;
    const matches = cond?.const !== undefined ? event.kind === cond.const
      : Array.isArray(cond?.enum) ? cond.enum.includes(event.kind) : false;
    if (!matches) continue;
    for (const key of clause.then.required ?? []) {
      if (!(key in event)) errors.push(`kind ${event.kind}: missing required "${key}"`);
    }
    for (const [key, spec] of Object.entries(clause.then.properties ?? {})) {
      if (key in event) checkType(event[key], spec, `${event.kind}.${key}`, errors);
    }
  }
  return errors;
}

const seen = new Set();
function record(events, label) {
  for (const [kind, data] of events) {
    const event = { ...data, kind };
    const errors = validate(event);
    assert.equal(errors.length, 0, `${label} emitted invalid ${kind}: ${errors.join('; ')}`);
    seen.add(kind);
  }
}

// ---- Direct Claude adapter (cli-stream-json): drive it with real stream-json messages. ----
{
  const s = new HarnessSession({
    id: 'schema-claude', agentId: 'a', name: 'A', role: 'builder', model: 'claude',
    endpointId: 'anthropic-1', cwd: process.cwd(), workspaceId: 'ws', systemPrompt: '',
    workspaces: [], providerKind: 'anthropic',
  });
  const events = [];
  s.on('event', (kind, data) => events.push([kind, data]));
  s.handleLine(JSON.stringify({ type: 'system', subtype: 'init', model: 'claude' }));
  s.handleAssistant({
    content: [
      { type: 'text', text: 'working on it' },
      { type: 'thinking', thinking: 'consider the plan' },
      { type: 'tool_use', id: 't1', name: 'Read', input: { file_path: 'a.txt' } },
      { type: 'tool_use', id: 't2', name: 'WebFetch', input: { url: 'https://example.com/doc' } },
      { type: 'tool_use', id: 't3', name: 'Task', input: { subagent_type: 'Explore', description: 'scout' } },
      { type: 'tool_use', id: 't4', name: 'TodoWrite', input: { todos: [{ status: 'completed' }, { status: 'pending' }] } },
    ],
    usage: { input_tokens: 5, output_tokens: 3, cache_read_input_tokens: 1 },
  });
  s.handleToolResults({ content: [{ type: 'tool_result', tool_use_id: 't1', content: 'ok', is_error: false }] });
  s.handleResult({ usage: { input_tokens: 5, output_tokens: 3 }, total_cost_usd: 0.1, result: 'done', is_error: false, num_turns: 1, duration_ms: 12 });
  s.cancel();
  record(events, 'HarnessSession');
}

// ---- Knossos adapter (ndjson-serve): drive it with real serve protocol events. ----
{
  const s = new KnossosSession({
    id: 'schema-knossos', agentId: 'b', name: 'B', role: 'builder', model: 'ornith',
    endpointId: 'cameo-1', cwd: process.cwd(), workspaceId: 'ws', systemPrompt: 'c', engine: 'cameo',
  });
  const events = [];
  s.on('event', (kind, data) => events.push([kind, data]));
  s.proc = { stdin: { writable: true, write() {} }, kill() {} };
  s.pendingOrders = 'go';
  s.handleLine(JSON.stringify({ event: 'ready', engine: 'cameo' }));
  s.handleLine(JSON.stringify({ event: 'plan', steps: ['a', 'b'] }));
  s.handleLine(JSON.stringify({ event: 'verdict', passed: true, summary: 'green', tiers: [] }));
  s.handleLine(JSON.stringify({ event: 'permission_request', id: 3, tool: 'write', input: { path: 'x' } }));
  s.handleLine(JSON.stringify({ event: 'outcome', succeeded: true, summary: 'done', steps_used: 2, changed: [], dry_run: false }));
  s.handleLine(JSON.stringify({ event: 'error', message: 'boom' }));
  s.cancel();
  record(events, 'KnossosSession');
}

// Every canonical event kind must have been produced and validated by at least one adapter.
const missing = CANONICAL_EVENTS.filter((k) => !seen.has(k));
assert.equal(missing.length, 0, `canonical events never exercised by an adapter: ${missing.join(', ')}`);

console.log(`event schema: ${seen.size} canonical event kinds from both adapters validate against field-event-v1`);
