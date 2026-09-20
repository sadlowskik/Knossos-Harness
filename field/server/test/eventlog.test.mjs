import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { EventLog, EVENT_LOG_SCHEMA_VERSION } from '../src/store/db.js';

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'field-eventlog-'));
const configuredSecret = 'configured-secret-value-92';
let first = new EventLog(dir, { secrets: [configuredSecret] });
first.append('campaign.created', { campaignId: 'c1', name: 'restart' }, { subject: 'c1' });
first.append('campaign.phase_changed', { campaignId: 'c1', from: 'draft', to: 'mobilizing' }, { subject: 'c1' });
first.append('simulation.started', { simulated: true }, { subject: 'sim-1', simulated: true });
first.append('session.tool_use', {
  sessionId: 's1', inputTokens: 42,
  input: { api_key: 'must-not-persist', nested: { authorization: 'Bearer must-not-persist' } },
  preview: `tool accidentally echoed ${configuredSecret} and sk-examplecredential99`,
}, { subject: 's1' });
assert.equal(first.size, 4);
first.close();

const second = new EventLog(dir);
assert.equal(second.size, 4);
assert.deepEqual(second.health(), {
  backend: second.backend,
  schemaVersion: EVENT_LOG_SCHEMA_VERSION,
  integrity: 'ok',
  events: 4,
});
assert.deepEqual(second.read(0).map((e) => e.kind), ['campaign.created', 'campaign.phase_changed', 'simulation.started', 'session.tool_use']);
assert.deepEqual(second.read(0).map((e) => e.source), ['observed', 'observed', 'synthetic', 'observed']);
assert.equal(second.bySubject('c1').length, 2);
const safe = second.bySubject('s1')[0].data;
assert.equal(safe.inputTokens, 42, 'ordinary token counters are not credentials');
assert.equal(safe.input.api_key, '[REDACTED]');
assert.equal(safe.input.nested.authorization, '[REDACTED]');
assert.doesNotMatch(JSON.stringify(safe), /configured-secret-value-92|sk-examplecredential99/);
assert.match(safe.preview, /\[REDACTED\]/);
second.close();
fs.rmSync(dir, { recursive: true, force: true });

const jsonlDir = fs.mkdtempSync(path.join(os.tmpdir(), 'field-jsonl-'));
let jsonl = new EventLog(jsonlDir, { backend: 'jsonl' });
jsonl.append('fixture.one', { ok: true });
jsonl.append('fixture.two', { ok: true });
jsonl.close();
const jsonlPath = path.join(jsonlDir, 'events.jsonl');
fs.appendFileSync(jsonlPath, '{"seq":3');
jsonl = new EventLog(jsonlDir, { backend: 'jsonl' });
assert.equal(jsonl.size, 2, 'one torn tail record is ignored during recovery');
jsonl.close();

const committed = fs.readFileSync(jsonlPath, 'utf8').split('\n');
committed[0] = '{broken committed history';
fs.writeFileSync(jsonlPath, committed.join('\n'));
assert.throws(
  () => new EventLog(jsonlDir, { backend: 'jsonl' }),
  /malformed at line 1/,
  'malformed committed history must stop startup instead of silently losing state'
);
fs.rmSync(jsonlDir, { recursive: true, force: true });

console.log('eventlog: versioned durability, redaction, integrity, and tail recovery passed');
