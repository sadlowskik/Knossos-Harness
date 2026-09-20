import assert from 'node:assert/strict';
import { CampaignProjection } from '../src/orchestration/campaign-projection.js';
import {
  DomainError, isBlockingFinding, legalActions, validateCampaignInput, validateFinding,
  validateTransition,
} from '../src/orchestration/model.js';

let seq = 0;
const event = (kind, data, ts = 1_700_000_000_000 + seq * 1000) => ({ kind, data, ts, seq: ++seq });

const valid = validateCampaignInput({
  name: 'Release gateway',
  intent: 'Prove restart and rollback safety.',
  scope: 'sandbox',
  objectives: [{ statement: 'Rollback is race-safe', definitionOfDone: ['stress check passes'] }],
});
assert.equal(valid.objectives.length, 1);
assert.throws(() => validateCampaignInput({ name: 'x', intent: 'y', objectives: [] }), DomainError);
assert.throws(() => validateFinding({ claim: 'race', severity: 'high', evidence: ['trace'] }), /reproduction/);

const p = new CampaignProjection();
const events = [
  event('campaign.created', { campaignId: 'c1', ...valid }),
  event('objective.created', {
    campaignId: 'c1', objectiveId: 'o1', statement: 'Rollback is race-safe',
    definitionOfDone: ['stress check passes'], priority: 1, risk: 'high', target: null,
  }),
  event('team.member_assigned', {
    campaignId: 'c1', team: 'blue', sessionId: 'blue-1', agentId: 'rhea', role: 'builder',
  }),
];
for (const e of events) p.apply(e);

let campaign = p.campaigns.get('c1');
let ctx = p.context('c1');
assert.doesNotThrow(() => validateTransition(campaign, 'mobilizing', ctx));
assert.ok(legalActions(campaign, ctx).includes('advance:mobilizing'));

p.apply(event('campaign.phase_changed', { campaignId: 'c1', from: 'draft', to: 'mobilizing' }));
p.apply(event('campaign.phase_changed', { campaignId: 'c1', from: 'mobilizing', to: 'blue_building' }));
campaign = p.campaigns.get('c1');
ctx = p.context('c1');
assert.throws(() => validateTransition(campaign, 'red_challenging', ctx), /not ready/);

p.apply(event('objective.satisfied', { campaignId: 'c1', objectiveId: 'o1', evidence: ['check: pass'] }));
assert.doesNotThrow(() => validateTransition(campaign, 'red_challenging', p.context('c1')));
p.apply(event('campaign.phase_changed', { campaignId: 'c1', from: 'blue_building', to: 'red_challenging' }));

const finding = validateFinding({
  claim: 'Restart can interleave with rollback.', severity: 'high', category: 'rollback',
  scope: 'release gateway', evidence: ['trace://race'], reproduction: ['start restart', 'issue rollback'],
});
p.apply(event('finding.reported', {
  campaignId: 'c1', objectiveId: 'o1', findingId: 'f1', authorSessionId: 'red-1', ...finding,
}));
campaign = p.campaigns.get('c1');
ctx = p.context('c1');
assert.ok(isBlockingFinding(ctx.findings[0], campaign.doctrine));
assert.doesNotThrow(() => validateTransition(campaign, 'contested', ctx));
p.apply(event('campaign.phase_changed', { campaignId: 'c1', from: 'red_challenging', to: 'contested' }));
assert.throws(() => validateTransition(campaign, 'blue_mitigating', p.context('c1')), /not acknowledged/);

p.apply(event('finding.acknowledged', { campaignId: 'c1', findingId: 'f1' }));
assert.doesNotThrow(() => validateTransition(campaign, 'blue_mitigating', p.context('c1')));
p.apply(event('campaign.phase_changed', { campaignId: 'c1', from: 'contested', to: 'blue_mitigating' }));
p.apply(event('mitigation.proposed', {
  campaignId: 'c1', mitigationId: 'm1', findingIds: ['f1'], ownerSessionId: 'blue-1',
  claim: 'serialize transitions', artifacts: ['gateway.rs'], evidence: [],
}));
p.apply(event('mitigation.started', { campaignId: 'c1', mitigationId: 'm1' }));
p.apply(event('mitigation.ready', { campaignId: 'c1', mitigationId: 'm1', evidence: ['stress: pass'] }));
assert.doesNotThrow(() => validateTransition(campaign, 'red_retesting', p.context('c1')));
p.apply(event('campaign.phase_changed', { campaignId: 'c1', from: 'blue_mitigating', to: 'red_retesting' }));
p.apply(event('retest.completed', {
  campaignId: 'c1', findingId: 'f1', sessionId: 'red-1', result: 'fixed', evidence: ['repro no longer fails'],
}));
assert.doesNotThrow(() => validateTransition(campaign, 'referee_review', p.context('c1')));
p.apply(event('campaign.phase_changed', { campaignId: 'c1', from: 'red_retesting', to: 'referee_review' }));
assert.throws(() => validateTransition(campaign, 'verified', p.context('c1')), /verdict/);
p.apply(event('referee.verdict', {
  campaignId: 'c1', verdictId: 'v1', sessionId: 'ref-1', verdict: 'verified',
  evidence: ['independent check'], rationale: 'definition of done passed',
}));
assert.doesNotThrow(() => validateTransition(campaign, 'verified', p.context('c1')));
p.apply(event('campaign.phase_changed', { campaignId: 'c1', from: 'referee_review', to: 'verified' }));
assert.throws(() => validateTransition(campaign, 'promoted', p.context('c1')), /checkpoint/);
p.apply(event('campaign.checkpoint_created', {
  campaignId: 'c1', checkpointId: 'cp1', name: 'verified gateway', eventSeq: seq,
  revision: 'abcdef1',
}));
assert.doesNotThrow(() => validateTransition(campaign, 'promoted', p.context('c1')));
p.apply(event('campaign.promoted', {
  campaignId: 'c1', checkpointId: 'cp1', capabilities: [{ id: 'cap:gateway', name: 'Race-safe release gateway' }],
}));

const view = p.campaignView(campaign);
assert.equal(view.phase, 'promoted');
assert.equal(view.findings[0].status, 'confirmed');
assert.equal(view.capabilities[0].status, 'promoted');

// Replay the same immutable stream into a clean projection and compare stable state.
const replay = new CampaignProjection();
for (const e of events) replay.apply(e);
// Earlier `events` intentionally contains only the creation prefix. This checks that a
// prefix replay is coherent rather than accidentally sharing the live projection maps.
assert.equal(replay.campaigns.get('c1').phase, 'draft');
assert.notEqual(replay.campaigns.get('c1'), p.campaigns.get('c1'));

console.log('campaign: state machine, evidence gates, and projection assertions passed');
