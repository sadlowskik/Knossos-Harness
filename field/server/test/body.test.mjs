import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { readJsonBody } from '../src/body.js';

class FixtureRequest extends EventEmitter {
  resume() { this.resumed = true; }
}

async function bodyFrom(chunks, options) {
  const req = new FixtureRequest();
  const result = readJsonBody(req, options);
  for (const chunk of chunks) req.emit('data', chunk);
  req.emit('end');
  return result;
}

assert.deepEqual(await bodyFrom([Buffer.from('{"ok":true}')]), { ok: true });
assert.deepEqual(await bodyFrom([]), {});

await assert.rejects(
  bodyFrom([Buffer.from('{nope}')]),
  (error) => error.status === 400 && error.code === 'invalid_json',
);

// Four emoji are 8 JS UTF-16 code units but 16 bytes. The wire-byte limit must win.
await assert.rejects(
  bodyFrom([Buffer.from('"😀😀😀😀"')], { maxBytes: 12 }),
  (error) => error.status === 413 && error.code === 'body_too_large',
);

const stalled = new FixtureRequest();
await assert.rejects(
  readJsonBody(stalled, { timeoutMs: 5 }),
  (error) => error.status === 408 && error.code === 'body_timeout',
);
assert.equal(stalled.resumed, true);

console.log('body: byte cap, JSON errors, empty bodies, and ingestion deadline passed');
