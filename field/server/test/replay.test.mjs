import assert from 'node:assert/strict';
import { readEventRange, replayInto } from '../src/store/replay.js';

const events = Array.from({ length: 25_017 }, (_, index) => ({
  seq: index + 1,
  ts: 1_700_000_000_000 + index,
  kind: 'fixture.tick',
  subject: 'paging',
  data: { index },
}));

const log = {
  read(from, limit) {
    return events.filter((event) => event.seq > from).slice(0, limit);
  },
};

const all = readEventRange(log, { pageSize: 997 });
assert.equal(all.length, events.length);
assert.equal(all.at(-1).seq, events.length);

const prefix = readEventRange(log, { to: 10_003, pageSize: 512 });
assert.equal(prefix.length, 10_003);
assert.equal(prefix.at(-1).seq, 10_003);

const seen = [];
const result = replayInto(log, { apply: (event) => seen.push(event.seq) }, { to: 20_001, pageSize: 333 });
assert.equal(result.count, 20_001);
assert.equal(result.lastEvent.seq, 20_001);
assert.deepEqual(seen.slice(-3), [19_999, 20_000, 20_001]);

console.log('replay: unbounded paging and exact historical ceilings passed');
