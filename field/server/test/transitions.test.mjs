import assert from 'node:assert/strict';
import { CAMPAIGN_PHASES, TRANSITIONS, transitionOptions, validateTransition } from '../src/orchestration/model.js';

const member = { sessionId: 'blue-1', status: 'active' };
const base = (phase) => ({
  id: 'transition-campaign', phase, paused: false,
  doctrine: { blockingSeverity: 'high' },
  teams: {
    blue: { members: [member] }, red: { members: [] },
    referee: { members: [] }, purple: { members: [] },
  },
});

function contextFor(from, to) {
  const context = {
    objectives: [{ id: 'o1', required: true, status: 'satisfied' }],
    findings: [], verdicts: [], checkpoints: [], latestVerdict: null,
    checkpointId: null, checkpointRevision: null,
  };
  if (from === 'red_challenging' && to === 'contested') {
    context.findings = [{ id: 'f1', severity: 'high', status: 'open' }];
  }
  if (from === 'contested' && to === 'blue_mitigating') {
    context.findings = [{ id: 'f1', severity: 'high', status: 'acknowledged' }];
  }
  if (from === 'blue_mitigating' && to === 'red_retesting') {
    context.findings = [{ id: 'f1', severity: 'high', status: 'ready_for_retest' }];
  }
  if (from === 'red_retesting' && to === 'referee_review') {
    context.findings = [{ id: 'f1', severity: 'high', status: 'confirmed' }];
  }
  if (from === 'referee_review' && to === 'verified') context.latestVerdict = 'verified';
  if (from === 'verified' && to === 'promoted') {
    context.checkpointId = 'cp1';
    context.checkpointRevision = 'abcdef1';
  }
  return context;
}

let positive = 0;
let negative = 0;
for (const from of CAMPAIGN_PHASES) {
  const legalTargets = (TRANSITIONS[from] ?? []).filter((to) => to !== 'paused');
  for (const to of legalTargets) {
    assert.doesNotThrow(() => validateTransition(base(from), to, contextFor(from, to)), `${from} -> ${to}`);
    positive += 1;
  }
  const illegal = CAMPAIGN_PHASES.find((to) => to !== from && !legalTargets.includes(to));
  if (illegal) {
    assert.throws(() => validateTransition(base(from), illegal, contextFor(from, illegal)), /cannot advance/);
    negative += 1;
  }
}

const blocked = base('blue_building');
const options = transitionOptions(blocked, {
  objectives: [{ id: 'o1', required: true, status: 'active' }], findings: [],
});
assert.equal(options.find((option) => option.to === 'red_challenging').legal, false);
assert.match(options.find((option) => option.to === 'red_challenging').reason, /not ready/);

console.log(`transitions: ${positive} positive and ${negative} negative transition cases passed`);
