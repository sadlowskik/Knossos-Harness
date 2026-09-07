import assert from 'node:assert/strict';
import { CampaignProjection } from '../src/orchestration/campaign-projection.js';
import { CampaignDirector } from '../src/orchestration/director.js';

function environment(name) {
  let seq = 0;
  let ids = 0;
  let commands = 0;
  const projection = new CampaignProjection();
  const emit = (kind, data, meta = {}) => {
    const event = { seq: ++seq, ts: 1_700_100_000_000 + seq, kind, data, ...meta };
    projection.apply(event);
    return event;
  };
  const registry = {
    info: () => null,
    command: () => ({ ok: true }),
    assign: () => ({ assignmentId: `a-${++ids}`, skipped: [] }),
    spawn: () => { throw new Error('scenario uses external deterministic sessions'); },
  };
  const director = new CampaignDirector({
    projection, registry, emit, eventHead: () => seq, id: () => `${name}-id-${++ids}`,
  });
  const result = director.create({
    commandId: `${name}-create`, campaignId: name, name, intent: `Exercise ${name}`,
    scope: 'sandbox',
    objectives: [{ statement: `${name} objective`, definitionOfDone: ['evidence attached'] }],
  });
  const objectiveId = result.objectiveIds[0];
  const act = (kind, data = {}) => director.action({
    commandId: `${name}-cmd-${++commands}`, campaignId: name, kind, ...data,
  });
  for (const [team, sessionId, role] of [
    ['blue', `${name}-blue`, 'builder'],
    ['red', `${name}-red`, 'challenger'],
    ['referee', `${name}-ref`, 'verifier'],
  ]) act('assign_team', { team, sessionId, role, agentId: sessionId, objectiveId, external: true });
  act('advance', { to: 'mobilizing' });
  act('advance', { to: 'blue_building' });
  act('satisfy_objective', { objectiveId, sessionId: `${name}-blue`, evidence: ['blue check'] });
  act('advance', { to: 'red_challenging' });
  return { director, projection, act, objectiveId, name };
}

function finding(env, overrides = {}) {
  return env.act('report_finding', {
    objectiveId: env.objectiveId, authorSessionId: `${env.name}-red`,
    category: 'correctness', severity: 'high', claim: 'readiness claim fails',
    scope: 'sandbox fixture', evidence: ['trace://failure'], reproduction: ['run fixture'],
    confidence: 0.9, ...overrides,
  }).findingId;
}

function reachRetest(env, findingId, dispute = false) {
  env.act('advance', { to: 'contested' });
  env.act(dispute ? 'dispute_finding' : 'acknowledge_finding', {
    findingId, sessionId: `${env.name}-blue`, evidence: dispute ? ['counterexample'] : undefined,
  });
  env.act('advance', { to: 'blue_mitigating' });
  const mitigationId = env.act('propose_mitigation', {
    findingIds: [findingId], ownerSessionId: `${env.name}-blue`,
    claim: 'apply isolated mitigation', artifacts: ['fixture'],
  }).mitigationId;
  env.act('start_mitigation', { mitigationId, sessionId: `${env.name}-blue` });
  env.act('mark_mitigation_ready', { mitigationId, sessionId: `${env.name}-blue`, evidence: ['mitigation check'] });
  env.act('advance', { to: 'red_retesting' });
  return mitigationId;
}

// A disputed false finding still goes through an evidence-bearing retest. Blue cannot clear
// it by assertion; the red retest marks it rejected before referee review.
const falsePositive = environment('false-positive');
const falseFinding = finding(falsePositive);
reachRetest(falsePositive, falseFinding, true);
falsePositive.act('record_retest', {
  findingId: falseFinding, sessionId: 'false-positive-red', result: 'false_positive',
  evidence: ['reproduction used stale fixture; clean fixture passes'],
});
assert.equal(falsePositive.projection.findings.get(falseFinding).status, 'rejected');
falsePositive.act('advance', { to: 'referee_review' });

// A persisting finding invalidates the prior mitigation and returns to blue. The old ready
// mitigation cannot unlock another retest without new work.
const regression = environment('blue-regression');
const regressionFinding = finding(regression, { category: 'rollback' });
const failedMitigation = reachRetest(regression, regressionFinding);
regression.act('record_retest', {
  findingId: regressionFinding, sessionId: 'blue-regression-red', result: 'persists',
  evidence: ['original race still reproduces'],
});
assert.equal(regression.projection.findings.get(regressionFinding).status, 'acknowledged');
assert.equal(regression.projection.mitigations.get(failedMitigation).status, 'failed');
regression.act('advance', { to: 'blue_mitigating' });
assert.throws(() => regression.act('advance', { to: 'red_retesting' }), /no ready mitigation/);

// An inconclusive referee cannot unlock verification. The campaign can return to blue with
// the evidence history intact.
const disagreement = environment('referee-disagreement');
disagreement.act('advance', { to: 'referee_review' });
disagreement.act('begin_referee_review', { sessionId: 'referee-disagreement-ref' });
disagreement.act('record_verdict', {
  sessionId: 'referee-disagreement-ref', verdict: 'inconclusive',
  evidence: ['test environment lacks required kernel feature'], rationale: 'reproduction environment is insufficient',
});
assert.throws(() => disagreement.act('advance', { to: 'verified' }), /verified referee verdict/);
disagreement.act('advance', { to: 'blue_building', reason: 'collect evidence on target kernel' });
assert.equal(disagreement.projection.campaigns.get('referee-disagreement').phase, 'blue_building');

// Prompt-injection doctrine is explicit, scoped, and uses decoys rather than credentials.
const injection = environment('prompt-injection');
const campaign = injection.projection.campaigns.get('prompt-injection');
const orders = injection.director.teamOrders(campaign, injection.projection.objectives.get(injection.objectiveId), 'red');
assert.match(orders, /Environment scope: sandbox/);
assert.match(orders, /decoy credentials/);
const injectionFinding = finding(injection, {
  category: 'prompt_injection', claim: 'website instruction attempts to redirect the agent',
  evidence: ['decoy-secret access was denied'], reproduction: ['open controlled malicious page'],
});
assert.equal(injection.projection.findings.get(injectionFinding).category, 'prompt_injection');

console.log('scenarios: false finding, persisting regression, referee disagreement, and prompt injection passed');
