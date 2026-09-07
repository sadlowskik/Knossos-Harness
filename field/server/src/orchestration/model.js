export const CAMPAIGN_PHASES = Object.freeze([
  'draft',
  'mobilizing',
  'blue_building',
  'red_challenging',
  'contested',
  'blue_mitigating',
  'red_retesting',
  'referee_review',
  'verified',
  'promoted',
  'failed',
  'cancelled',
  'rolled_back',
]);

export const TEAM_KINDS = Object.freeze(['blue', 'red', 'purple', 'referee']);
export const SEVERITIES = Object.freeze(['info', 'low', 'medium', 'high', 'critical']);
export const FINDING_STATES = Object.freeze([
  'open', 'acknowledged', 'disputed', 'mitigating', 'ready_for_retest',
  'confirmed', 'rejected', 'waived',
]);
export const VERDICTS = Object.freeze(['verified', 'rejected', 'inconclusive']);
export const ENVIRONMENT_SCOPES = Object.freeze([
  'snapshot', 'sandbox', 'staging', 'production-readonly', 'production-approved',
]);

export const TRANSITIONS = Object.freeze({
  draft: ['mobilizing', 'cancelled'],
  mobilizing: ['blue_building', 'paused', 'failed', 'cancelled'],
  blue_building: ['red_challenging', 'paused', 'failed', 'cancelled'],
  red_challenging: ['contested', 'referee_review', 'paused', 'failed', 'cancelled'],
  contested: ['blue_mitigating', 'paused', 'failed', 'cancelled'],
  blue_mitigating: ['red_retesting', 'paused', 'failed', 'cancelled'],
  red_retesting: ['referee_review', 'blue_mitigating', 'paused', 'failed', 'cancelled'],
  referee_review: ['verified', 'blue_building', 'blue_mitigating', 'paused', 'failed', 'cancelled'],
  verified: ['promoted', 'blue_building', 'cancelled'],
  promoted: ['rolled_back'],
  failed: ['mobilizing', 'cancelled'],
  cancelled: [],
  rolled_back: ['blue_building', 'cancelled'],
});

// `paused` is represented by a campaign flag rather than a stored phase. It appears in
// TRANSITIONS only as an operator-facing legal action.
const PHASE_SET = new Set(CAMPAIGN_PHASES);
const TEAM_SET = new Set(TEAM_KINDS);
const SEVERITY_SET = new Set(SEVERITIES);
const SCOPE_SET = new Set(ENVIRONMENT_SCOPES);

export class DomainError extends Error {
  constructor(code, message, detail = {}) {
    super(message);
    this.name = 'DomainError';
    this.code = code;
    this.detail = detail;
  }
}

export function assertId(value, name = 'id') {
  if (typeof value !== 'string' || value.length < 1 || value.length > 160 || !/^[a-zA-Z0-9._:-]+$/.test(value)) {
    throw new DomainError('invalid_id', `${name} must be 1-160 URL-safe characters`, { name });
  }
  return value;
}

export function validateCampaignInput(input = {}) {
  const name = String(input.name ?? '').trim();
  const intent = String(input.intent ?? '').trim();
  if (!name || name.length > 160) {
    throw new DomainError('invalid_campaign', 'campaign name is required and must be at most 160 characters');
  }
  if (!intent || intent.length > 8000) {
    throw new DomainError('invalid_campaign', 'campaign intent is required and must be at most 8000 characters');
  }
  const scope = input.scope ?? 'sandbox';
  if (!SCOPE_SET.has(scope)) {
    throw new DomainError('invalid_scope', `unknown campaign scope: ${scope}`);
  }
  const concurrency = boundedInt(input.concurrency ?? 8, 1, 100, 'concurrency');
  const budgetUsd = boundedNumber(input.budgetUsd ?? 25, 0, 1_000_000, 'budgetUsd');
  const objectives = Array.isArray(input.objectives) ? input.objectives : [];
  if (!objectives.length) {
    throw new DomainError('missing_objectives', 'a campaign requires at least one objective');
  }
  return {
    name,
    intent,
    scope,
    concurrency,
    budgetUsd,
    doctrine: normalizeDoctrine(input.doctrine),
    target: normalizeTarget(input.target),
    objectives: objectives.map((objective, i) => validateObjective(objective, i)),
  };
}

export function validateObjective(input = {}, index = 0) {
  const statement = String(input.statement ?? input.name ?? '').trim();
  if (!statement || statement.length > 2000) {
    throw new DomainError('invalid_objective', `objective ${index + 1} needs a statement of at most 2000 characters`);
  }
  const done = Array.isArray(input.definitionOfDone)
    ? input.definitionOfDone.map((x) => String(x).trim()).filter(Boolean)
    : [];
  if (!done.length) {
    throw new DomainError('invalid_objective', `objective ${index + 1} needs a definition of done`);
  }
  return {
    statement,
    definitionOfDone: done.slice(0, 40),
    priority: boundedInt(input.priority ?? index + 1, 1, 10_000, 'priority'),
    risk: String(input.risk ?? 'medium'),
    target: normalizeTarget(input.target),
    dependencies: Array.isArray(input.dependencies) ? input.dependencies.map(String).slice(0, 100) : [],
  };
}

export function validateFinding(input = {}) {
  const severity = input.severity ?? 'medium';
  if (!SEVERITY_SET.has(severity)) {
    throw new DomainError('invalid_severity', `unknown finding severity: ${severity}`);
  }
  const claim = String(input.claim ?? '').trim();
  if (!claim || claim.length > 8000) {
    throw new DomainError('invalid_finding', 'finding claim is required and must be at most 8000 characters');
  }
  const evidence = Array.isArray(input.evidence) ? input.evidence.filter(Boolean).slice(0, 100) : [];
  const reproduction = Array.isArray(input.reproduction)
    ? input.reproduction.map(String).filter(Boolean).slice(0, 100)
    : [];
  const noReproductionReason = String(input.noReproductionReason ?? '').trim();
  if (!evidence.length) throw new DomainError('missing_evidence', 'a finding requires evidence');
  if (!reproduction.length && !noReproductionReason) {
    throw new DomainError('missing_reproduction', 'a finding requires reproduction steps or a reason they are unavailable');
  }
  return {
    severity,
    category: String(input.category ?? 'correctness').slice(0, 80),
    claim,
    scope: String(input.scope ?? '').slice(0, 1000),
    evidence,
    reproduction,
    noReproductionReason: noReproductionReason || null,
    confidence: boundedNumber(input.confidence ?? 0.8, 0, 1, 'confidence'),
  };
}

export function validateTransition(campaign, to, context = {}) {
  if (!campaign) throw new DomainError('not_found', 'campaign does not exist');
  if (!PHASE_SET.has(to) || to === 'paused') {
    throw new DomainError('invalid_phase', `unknown campaign phase: ${to}`);
  }
  const from = campaign.phase;
  const allowed = (TRANSITIONS[from] ?? []).filter((x) => x !== 'paused');
  if (!allowed.includes(to)) {
    throw new DomainError('invalid_transition', `cannot advance campaign from ${from} to ${to}`, {
      currentPhase: from, allowedPhases: allowed,
    });
  }

  const objectives = context.objectives ?? [];
  const findings = context.findings ?? [];
  const teams = campaign.teams ?? {};
  if ((from === 'draft' || from === 'failed') && to === 'mobilizing') {
    if (!objectives.length) throw gate('campaign has no objectives');
    if (!teams.blue?.members?.some((member) => ['active', 'external'].includes(member.status))) {
      throw gate('campaign has no active or explicitly external blue team members');
    }
  }
  if (from === 'blue_building' && to === 'red_challenging') {
    const missing = objectives.filter((x) => x.required !== false && x.status !== 'satisfied');
    if (missing.length) throw gate(`${missing.length} required objectives are not ready for challenge`);
  }
  if (from === 'red_challenging' && to === 'contested') {
    if (!findings.some((x) => isBlockingFinding(x, campaign.doctrine))) throw gate('no blocking red finding exists');
  }
  if (from === 'red_challenging' && to === 'referee_review') {
    if (findings.some((x) => isBlockingFinding(x, campaign.doctrine))) throw gate('blocking findings require mitigation first');
  }
  if (from === 'contested' && to === 'blue_mitigating') {
    const untriaged = findings.filter((x) => isBlockingFinding(x, campaign.doctrine) && x.status === 'open');
    if (untriaged.length) throw gate(`${untriaged.length} blocking findings are not acknowledged or disputed`);
  }
  if (from === 'blue_mitigating' && to === 'red_retesting') {
    const unready = findings.filter((x) => isBlockingFinding(x, campaign.doctrine)
      && !['ready_for_retest', 'rejected', 'waived'].includes(x.status));
    if (unready.length) throw gate(`${unready.length} blocking findings have no ready mitigation`);
  }
  if (from === 'red_retesting' && to === 'referee_review') {
    const unresolved = findings.filter((x) => isBlockingFinding(x, campaign.doctrine)
      && !['confirmed', 'rejected', 'waived'].includes(x.status));
    if (unresolved.length) throw gate(`${unresolved.length} blocking findings lack a completed retest`);
  }
  if (from === 'referee_review' && to === 'verified' && context.latestVerdict !== 'verified') {
    throw gate('an independent verified referee verdict is required');
  }
  if (from === 'verified' && to === 'promoted'
    && (!context.checkpointId || !isContentRevision(context.checkpointRevision))) {
    throw gate('promotion requires a revision-bound checkpoint');
  }
  return true;
}

export function isContentRevision(value) {
  return typeof value === 'string' && /^(?:sha256:)?[a-f0-9]{7,64}$/i.test(value.trim());
}

export function legalActions(campaign, context = {}) {
  if (!campaign) return [];
  const out = campaign.paused ? ['resume', 'cancel'] : ['pause', 'cancel'];
  for (const phase of (TRANSITIONS[campaign.phase] ?? []).filter((x) => x !== 'paused')) {
    try {
      validateTransition(campaign, phase, context);
      out.push(`advance:${phase}`);
    } catch { /* gate remains visible in detail, but action is not legal */ }
  }
  if (!campaign.paused && !['cancelled', 'promoted'].includes(campaign.phase)) {
    out.push('assign_team', 'issue_orders', 'reinforce', 'retreat');
  }
  return [...new Set(out)];
}

/** Every phase reachable from the current phase, including the reason a gate is closed. */
export function transitionOptions(campaign, context = {}) {
  if (!campaign) return [];
  return (TRANSITIONS[campaign.phase] ?? [])
    .filter((to) => to !== 'paused')
    .map((to) => {
      try {
        validateTransition(campaign, to, context);
        return { to, legal: true, reason: null };
      } catch (error) {
        return { to, legal: false, reason: error.message, code: error.code ?? 'gate_blocked' };
      }
    });
}

export function isBlockingFinding(finding, doctrine = {}) {
  if (!finding || ['confirmed', 'rejected', 'waived'].includes(finding.status)) return false;
  const threshold = doctrine.blockingSeverity ?? 'high';
  return SEVERITIES.indexOf(finding.severity) >= SEVERITIES.indexOf(threshold);
}

export function assertTeam(kind) {
  if (!TEAM_SET.has(kind)) throw new DomainError('invalid_team', `unknown team: ${kind}`);
  return kind;
}

export function normalizeDoctrine(input = {}) {
  const blockingSeverity = input?.blockingSeverity ?? 'high';
  if (!SEVERITY_SET.has(blockingSeverity)) {
    throw new DomainError('invalid_doctrine', `unknown blocking severity: ${blockingSeverity}`);
  }
  return {
    blockingSeverity,
    requireRed: input?.requireRed !== false,
    requireReferee: input?.requireReferee !== false,
    maxFindingAgeMs: boundedInt(input?.maxFindingAgeMs ?? 86_400_000, 1000, 31_536_000_000, 'maxFindingAgeMs'),
    redCategories: Array.isArray(input?.redCategories)
      ? input.redCategories.map(String).slice(0, 30)
      : ['security', 'reliability', 'correctness', 'rollback'],
  };
}

export function normalizeTarget(input) {
  if (!input) return null;
  return {
    type: String(input.type ?? 'workspace').slice(0, 40),
    id: String(input.id ?? '').slice(0, 1000),
    label: input.label == null ? null : String(input.label).slice(0, 300),
    workspaceId: input.workspaceId == null ? null : String(input.workspaceId).slice(0, 160),
  };
}

function gate(message) {
  return new DomainError('gate_blocked', message);
}

function boundedInt(value, min, max, name) {
  const n = Number(value);
  if (!Number.isInteger(n) || n < min || n > max) {
    throw new DomainError('invalid_number', `${name} must be an integer between ${min} and ${max}`);
  }
  return n;
}

function boundedNumber(value, min, max, name) {
  const n = Number(value);
  if (!Number.isFinite(n) || n < min || n > max) {
    throw new DomainError('invalid_number', `${name} must be between ${min} and ${max}`);
  }
  return n;
}
