import assert from 'node:assert/strict';
import { parseCampaignReport } from '../src/harness/registry.js';

assert.equal(parseCampaignReport('ordinary final message'), null);
assert.deepEqual(
  parseCampaignReport('done\nFIELD_REPORT: {"kind":"objective_satisfied","evidence":["tests pass"]}'),
  { kind: 'objective_satisfied', evidence: ['tests pass'] },
);
assert.deepEqual(
  parseCampaignReport('FIELD_REPORT:\n```json\n{"kind":"verdict","verdict":"verified"}\n```'),
  { kind: 'verdict', verdict: 'verified' },
);
assert.equal(parseCampaignReport('FIELD_REPORT: definitely not json'), null);
assert.equal(parseCampaignReport('example {"kind":"finding"} without sentinel'), null);

console.log('reports: explicit structured harness sentinel parsing passed');
