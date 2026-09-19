// Conformance: both shipping adapters satisfy the formal HarnessAdapter contract, and the
// manifest-driven selection reproduces the historical endpoint-kind -> adapter mapping exactly.
import assert from 'node:assert/strict';
import { HarnessAdapter, HARNESS_KINDS, CANONICAL_EVENTS } from '../src/harness/adapter.js';
import { HarnessSession } from '../src/harness/session.js';
import { KnossosSession } from '../src/harness/knossos-session.js';
import { ADAPTER_MANIFESTS, resolveAdapterManifest } from '../src/harness/adapters.js';

const CONTRACT_METHODS = ['start', 'send', 'cancel', 'capabilities', 'stream'];

for (const Adapter of [HarnessSession, KnossosSession]) {
  const s = new Adapter({ id: 'contract', cwd: process.cwd() });
  assert.ok(s instanceof HarnessAdapter, `${Adapter.name} extends HarnessAdapter`);
  for (const method of CONTRACT_METHODS) {
    assert.equal(typeof s[method], 'function', `${Adapter.name} implements ${method}()`);
  }
  const caps = s.capabilities();
  assert.ok(caps && typeof caps === 'object', `${Adapter.name}.capabilities() returns an object`);
  assert.ok(HARNESS_KINDS.includes(caps.kind), `${Adapter.name} declares a known harness kind (${caps.kind})`);
  // The base guards against a half-built adapter: an adapter that forgot start/send/cancel throws.
  assert.equal(typeof s.emitEvent, 'function', `${Adapter.name} inherits the canonical emitEvent`);
}

// The abstract base refuses to run unimplemented lifecycle methods.
const bare = new HarnessAdapter();
assert.throws(() => bare.start(), /must be implemented/);
assert.throws(() => bare.send(), /must be implemented/);
assert.throws(() => bare.cancel(), /must be implemented/);

// stream() yields canonical { kind, data } events over the existing emitter interface.
{
  const s = new KnossosSession({ id: 'stream', cwd: process.cwd(), endpointId: 'e', model: 'm', engine: 'cameo' });
  const controller = new AbortController();
  const iterator = s.stream({ signal: controller.signal });
  const first = iterator.next();
  s.emitEvent('session.state', { state: 'ready' });
  const { value } = await first;
  assert.equal(value.kind, 'session.state');
  assert.equal(value.data.state, 'ready');
  assert.equal(value.data.sessionId, 'stream', 'stream events carry the sessionId stamped by emitEvent');
  assert.ok(CANONICAL_EVENTS.includes(value.kind));
  controller.abort();
  await assert.rejects(iterator.next(), (err) => err.name === 'AbortError');
}

// Manifest selection: exactly reproduces `ep.kind === 'openai-compatible' ? Knossos : Harness`.
assert.equal(resolveAdapterManifest('openai-compatible').Adapter, KnossosSession);
assert.equal(resolveAdapterManifest('anthropic').Adapter, HarnessSession);
assert.equal(resolveAdapterManifest(undefined).Adapter, HarnessSession, 'unknown kind falls back to the default adapter');
assert.equal(resolveAdapterManifest('some-future-kind').Adapter, HarnessSession);

// Manifests are well-formed and their declared capabilities agree with the live adapter.
for (const m of ADAPTER_MANIFESTS) {
  assert.ok(HARNESS_KINDS.includes(m.kind), `${m.id} declares a known harness kind`);
  assert.ok(Array.isArray(m.endpointKinds), `${m.id} declares endpointKinds`);
  const instance = new m.Adapter({ id: `m-${m.id}`, cwd: process.cwd(), engine: 'cameo' });
  assert.equal(instance.capabilities().kind, m.kind, `${m.id} manifest kind matches its adapter capabilities`);
}
assert.equal(ADAPTER_MANIFESTS.filter((m) => m.default).length, 1, 'exactly one default adapter manifest');

console.log('adapter contract: both adapters satisfy HarnessAdapter, stream() and manifest selection conform');
