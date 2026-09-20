import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { EventLog } from '../src/store/db.js';
import { createApi, parseEventPage } from '../src/api.js';

const pageUrl = (query) => new URL(`http://127.0.0.1/api/events?${query}`);
assert.deepEqual(parseEventPage(pageUrl('')), { from: 0, limit: 2000 });
assert.deepEqual(parseEventPage(pageUrl('from=4&limit=3')), { from: 4, limit: 3 });
for (const query of ['from=-1', 'from=1.5', 'from=nope', 'limit=0', 'limit=10001', 'limit=1.2', 'limit=nope']) {
  assert.throws(() => parseEventPage(pageUrl(query)), (error) => error.code === 'invalid_pagination');
}

for (const backend of ['jsonl', ...(process.versions.node >= '22.5.0' ? ['sqlite'] : [])]) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), `field-page-${backend}-`));
  const log = new EventLog(dir, { backend });
  for (let i = 0; i < 7; i += 1) log.append('session.message', { i }, { subject: 's1' });
  log.append('other.event', { i: 8 }, { subject: 'other' });
  assert.deepEqual(log.read(0, 3).map((e) => e.seq), [1, 2, 3]);
  assert.deepEqual(log.read(3, 3).map((e) => e.seq), [4, 5, 6]);
  assert.deepEqual(log.bySubject('s1', { fromSeq: 3, limit: 3 }).map((e) => e.data.i), [3, 4, 5]);
  assert.deepEqual(log.bySubject('s1', { fromSeq: 6, limit: 3 }).map((e) => e.data.i), [6]);
  assert.deepEqual(log.bySubject('missing', { fromSeq: 0, limit: 3 }), []);
  log.close();
  fs.rmSync(dir, { recursive: true, force: true });
}

const routeDir = fs.mkdtempSync(path.join(os.tmpdir(), 'field-api-routes-'));
const routeLog = new EventLog(routeDir, { backend: 'jsonl' });
for (let i = 0; i < 5; i += 1) routeLog.append('session.message', { i, campaignId: 'c1' }, { subject: 's1' });
routeLog.append('campaign.created', { campaignId: 'c1' }, { subject: 'c1' });
const routeApi = createApi({
  cfg: {}, projection: { campaigns: { campaigns: new Map([['c1', {
    objectiveIds: [], findingIds: [], mitigationIds: [], verdictIds: [], checkpointIds: [],
  }]]) } },
  registry: {}, director: {}, simulator: {}, routines: {}, log: routeLog, broadcast() {},
});
async function call(pathname) {
  let payload;
  const req = { method: 'GET', url: pathname };
  const res = { writeHead(_status, headers) { this.headers = headers; }, end(text) { payload = JSON.parse(text); } };
  assert.equal(await routeApi(req, res), true);
  return payload;
}
const eventPage = await call('/api/events?from=0&limit=2');
assert.deepEqual(eventPage.events.map((e) => e.seq), [1, 2]);
assert.equal(eventPage.nextFrom, 2);
const eventFinal = await call('/api/events?from=5&limit=2');
assert.equal(eventFinal.nextFrom, null);
const subjectPage = await call('/api/trace?subject=s1&from=0&limit=2');
assert.deepEqual(subjectPage.events.map((e) => e.data.i), [0, 1]);
const subjectSecond = await call(`/api/trace?subject=s1&from=${subjectPage.nextFrom}&limit=2`);
assert.deepEqual(subjectSecond.events.map((e) => e.data.i), [2, 3]);
const campaignPage = await call('/api/campaigns/trace?campaignId=c1&from=0&limit=2');
assert.equal(campaignPage.events.length, 2);
assert.equal(campaignPage.nextFrom, 2);
assert.deepEqual((await call('/api/events?from=0&limit=10001')).code, 'invalid_pagination');
routeLog.close();
fs.rmSync(routeDir, { recursive: true, force: true });

console.log('api pagination: bounded event and subject windows preserve cursor order');
