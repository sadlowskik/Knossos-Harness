import { legalActions, transitionOptions } from './model.js';

export class CampaignProjection {
  constructor() { this.reset(); }

  reset() {
    this.campaigns = new Map();
    this.objectives = new Map();
    this.findings = new Map();
    this.mitigations = new Map();
    this.verdicts = new Map();
    this.checkpoints = new Map();
    this.capabilities = new Map();
    this.commandResults = new Map();
    this.sessionCosts = new Map();
  }

  apply(evt) {
    const d = evt.data ?? {};
    const fn = HANDLERS[evt.kind];
    if (fn) fn.call(this, d, evt);
    if (d.commandId) this.commandResults.set(d.commandId, { seq: evt.seq, campaignId: d.campaignId });
    const campaign = d.campaignId ? this.campaigns.get(d.campaignId) : null;
    if (campaign) {
      campaign.updatedAt = evt.ts;
      campaign.lastSeq = evt.seq;
    }
  }

  context(campaignId) {
    const campaign = this.campaigns.get(campaignId);
    if (!campaign) return null;
    const objectives = campaign.objectiveIds.map((id) => this.objectives.get(id)).filter(Boolean);
    const findings = campaign.findingIds.map((id) => this.findings.get(id)).filter(Boolean);
    const verdicts = campaign.verdictIds.map((id) => this.verdicts.get(id)).filter(Boolean);
    const checkpoints = campaign.checkpointIds.map((id) => this.checkpoints.get(id)).filter(Boolean);
    return {
      objectives,
      findings,
      verdicts,
      checkpoints,
      latestVerdict: verdicts.at(-1)?.verdict ?? null,
      checkpointId: checkpoints.at(-1)?.id ?? null,
      checkpointRevision: checkpoints.at(-1)?.revision ?? null,
    };
  }

  campaignView(campaign) {
    const ctx = this.context(campaign.id);
    return {
      ...campaign,
      objectives: ctx.objectives,
      findings: ctx.findings,
      mitigations: campaign.mitigationIds.map((id) => this.mitigations.get(id)).filter(Boolean),
      verdicts: ctx.verdicts,
      checkpoints: ctx.checkpoints,
      capabilities: campaign.capabilityIds.map((id) => this.capabilities.get(id)).filter(Boolean),
      legalActions: legalActions(campaign, ctx),
      transitionOptions: transitionOptions(campaign, ctx),
    };
  }

  snapshot() {
    return {
      campaigns: [...this.campaigns.values()].map((campaign) => this.campaignView(campaign)),
      capabilities: [...this.capabilities.values()],
      checkpoints: [...this.checkpoints.values()],
    };
  }
}

function emptyTeams() {
  return {
    blue: { kind: 'blue', members: [], ready: false },
    red: { kind: 'red', members: [], ready: false },
    purple: { kind: 'purple', members: [], ready: false },
    referee: { kind: 'referee', members: [], ready: false },
  };
}

const HANDLERS = {
  'campaign.created'(d, evt) {
    if (this.campaigns.has(d.campaignId)) return;
    this.campaigns.set(d.campaignId, {
      id: d.campaignId,
      name: d.name,
      intent: d.intent,
      scope: d.scope,
      concurrency: d.concurrency,
      budgetUsd: d.budgetUsd,
      doctrine: d.doctrine,
      target: d.target ?? null,
      phase: 'draft',
      paused: false,
      createdAt: evt.ts,
      updatedAt: evt.ts,
      lastSeq: evt.seq,
      teams: emptyTeams(),
      objectiveIds: [],
      findingIds: [],
      mitigationIds: [],
      verdictIds: [],
      checkpointIds: [],
      capabilityIds: [],
      staffingGaps: [],
      costUsd: 0,
      budgetExhausted: false,
      failure: null,
    });
  },

  'campaign.phase_changed'(d) {
    const c = this.campaigns.get(d.campaignId);
    if (c) { c.phase = d.to; c.phaseReason = d.reason ?? null; }
  },

  'campaign.paused'(d) { const c = this.campaigns.get(d.campaignId); if (c) c.paused = true; },
  'campaign.resumed'(d) { const c = this.campaigns.get(d.campaignId); if (c) c.paused = false; },
  'campaign.cancelled'(d) {
    const c = this.campaigns.get(d.campaignId);
    if (c) { c.phase = 'cancelled'; c.paused = false; c.cancelReason = d.reason ?? null; }
  },
  'campaign.failed'(d) {
    const c = this.campaigns.get(d.campaignId);
    if (c) { c.phase = 'failed'; c.failure = d.reason ?? 'campaign failed'; }
  },
  'campaign.doctrine_changed'(d) {
    const c = this.campaigns.get(d.campaignId);
    if (c) c.doctrine = { ...c.doctrine, ...d.doctrine };
  },

  'objective.created'(d, evt) {
    if (this.objectives.has(d.objectiveId)) return;
    this.objectives.set(d.objectiveId, {
      id: d.objectiveId, campaignId: d.campaignId, statement: d.statement,
      definitionOfDone: d.definitionOfDone ?? [], priority: d.priority,
      risk: d.risk, target: d.target ?? null, dependencies: d.dependencies ?? [],
      required: d.required !== false, status: 'queued', team: 'blue', sessionIds: [],
      progress: null, evidence: [], blockedReason: null, createdAt: evt.ts, updatedAt: evt.ts,
      criteriaEvidence: [],
    });
    const c = this.campaigns.get(d.campaignId);
    if (c && !c.objectiveIds.includes(d.objectiveId)) c.objectiveIds.push(d.objectiveId);
  },
  'objective.assigned'(d, evt) {
    const o = this.objectives.get(d.objectiveId);
    if (!o) return;
    o.team = d.team ?? o.team;
    o.sessionIds = [...new Set([...(o.sessionIds ?? []), ...(d.sessionIds ?? [])])];
    o.status = 'active'; o.updatedAt = evt.ts;
  },
  'objective.progress'(d, evt) {
    const o = this.objectives.get(d.objectiveId);
    if (!o) return;
    o.progress = d.progress ?? null;
    if (d.evidence) o.evidence.push(d.evidence);
    o.updatedAt = evt.ts;
  },
  'objective.blocked'(d, evt) {
    const o = this.objectives.get(d.objectiveId);
    if (o) { o.status = 'blocked'; o.blockedReason = d.reason; o.updatedAt = evt.ts; }
  },
  'objective.satisfied'(d, evt) {
    const o = this.objectives.get(d.objectiveId);
    if (o) {
      o.status = 'satisfied';
      o.evidence.push(...(d.evidence ?? []));
      o.criteriaEvidence = Array.isArray(d.criteriaEvidence)
        ? d.criteriaEvidence
        : o.definitionOfDone.map((criterion, index) => ({ criterion, evidence: d.evidence?.[index] })).filter((item) => item.evidence);
      o.updatedAt = evt.ts;
    }
  },
  'objective.failed'(d, evt) {
    const o = this.objectives.get(d.objectiveId);
    if (o) { o.status = 'failed'; o.blockedReason = d.reason; o.updatedAt = evt.ts; }
  },

  'team.member_assigned'(d) {
    const c = this.campaigns.get(d.campaignId);
    const team = c?.teams?.[d.team];
    if (!team) return;
    const prior = team.members.find((m) => m.sessionId === d.sessionId);
    if (!prior) team.members.push({
      sessionId: d.sessionId, agentId: d.agentId ?? null, role: d.role ?? null,
      objectiveId: d.objectiveId ?? null, status: d.status ?? 'active', external: d.external === true,
      assignedAt: d.assignedAt ?? null,
    });
    team.ready = team.members.some((m) => ['active', 'external'].includes(m.status));
    c.staffingGaps = c.staffingGaps.filter((g) => !(g.team === d.team && g.role === d.role));
  },
  'team.member_removed'(d) {
    const team = this.campaigns.get(d.campaignId)?.teams?.[d.team];
    if (!team) return;
    team.members = team.members.filter((m) => m.sessionId !== d.sessionId);
    team.ready = team.members.some((m) => ['active', 'external'].includes(m.status));
  },
  'team.member_state'(d) {
    const team = this.campaigns.get(d.campaignId)?.teams?.[d.team];
    const member = team?.members.find((item) => item.sessionId === d.sessionId);
    if (!member) return;
    member.status = d.status;
    member.sessionState = d.sessionState ?? member.sessionState ?? null;
    member.external = d.status === 'external' || member.external === true;
    team.ready = team.members.some((item) => ['active', 'external'].includes(item.status));
  },
  'team.staffing_gap'(d) {
    const c = this.campaigns.get(d.campaignId);
    if (c && !c.staffingGaps.some((g) => g.team === d.team && g.role === d.role)) {
      c.staffingGaps.push({ team: d.team, role: d.role, reason: d.reason });
    }
  },
  'team.handoff_started'(d) {
    const team = this.campaigns.get(d.campaignId)?.teams?.[d.team];
    const m = team?.members.find((x) => x.sessionId === d.fromSessionId);
    if (m) m.status = 'handing_off';
  },
  'team.handoff_completed'(d) {
    const team = this.campaigns.get(d.campaignId)?.teams?.[d.team];
    const from = team?.members.find((x) => x.sessionId === d.fromSessionId);
    const to = team?.members.find((x) => x.sessionId === d.toSessionId);
    if (from) from.status = 'retired';
    if (to) to.status = 'active';
  },

  'session.state'(d, evt) {
    synchronizeSessionState(this, d.sessionId, d.state, evt);
  },
  'session.ended'(d, evt) {
    synchronizeSessionState(this, d.sessionId, d.reason === 'error' ? 'error' : 'done', evt);
  },
  'session.usage'(d) {
    const c = d.campaignId ? this.campaigns.get(d.campaignId) : null;
    if (!c || !Number.isFinite(d.costUsd) || d.costUsd < 0) return;
    const key = `${c.id}:${d.sessionId}`;
    const prior = this.sessionCosts.get(key) ?? 0;
    const current = Math.max(prior, d.costUsd);
    this.sessionCosts.set(key, current);
    c.costUsd += current - prior;
    c.budgetExhausted = c.budgetUsd > 0 && c.costUsd >= c.budgetUsd;
  },

  'finding.reported'(d, evt) {
    if (this.findings.has(d.findingId)) return;
    this.findings.set(d.findingId, {
      id: d.findingId, campaignId: d.campaignId, objectiveId: d.objectiveId,
      authorSessionId: d.authorSessionId, category: d.category, severity: d.severity,
      claim: d.claim, scope: d.scope, evidence: d.evidence ?? [],
      reproduction: d.reproduction ?? [], noReproductionReason: d.noReproductionReason ?? null,
      confidence: d.confidence, status: 'open', mitigationIds: [], retests: [],
      createdAt: evt.ts, updatedAt: evt.ts,
    });
    const c = this.campaigns.get(d.campaignId);
    if (c && !c.findingIds.includes(d.findingId)) c.findingIds.push(d.findingId);
  },
  'finding.acknowledged'(d, evt) { setFindingState(this, d, 'acknowledged', evt); },
  'finding.disputed'(d, evt) { setFindingState(this, d, 'disputed', evt, { dispute: d.evidence }); },
  'finding.waived'(d, evt) { setFindingState(this, d, 'waived', evt, { waiver: d.reason, authority: d.authority }); },

  'mitigation.proposed'(d, evt) {
    if (this.mitigations.has(d.mitigationId)) return;
    const m = {
      id: d.mitigationId, campaignId: d.campaignId, findingIds: d.findingIds ?? [],
      ownerSessionId: d.ownerSessionId, claim: d.claim, artifacts: d.artifacts ?? [],
      evidence: d.evidence ?? [], status: 'proposed', createdAt: evt.ts, updatedAt: evt.ts,
    };
    this.mitigations.set(d.mitigationId, m);
    const c = this.campaigns.get(d.campaignId);
    if (c && !c.mitigationIds.includes(d.mitigationId)) c.mitigationIds.push(d.mitigationId);
    for (const id of m.findingIds) {
      const f = this.findings.get(id);
      if (f && !f.mitigationIds.includes(m.id)) f.mitigationIds.push(m.id);
    }
  },
  'mitigation.started'(d, evt) {
    const m = this.mitigations.get(d.mitigationId);
    if (m) { m.status = 'active'; m.updatedAt = evt.ts; }
    for (const id of m?.findingIds ?? []) setFindingState(this, { findingId: id }, 'mitigating', evt);
  },
  'mitigation.ready'(d, evt) {
    const m = this.mitigations.get(d.mitigationId);
    if (m) { m.status = 'ready'; m.evidence.push(...(d.evidence ?? [])); m.updatedAt = evt.ts; }
    for (const id of m?.findingIds ?? []) setFindingState(this, { findingId: id }, 'ready_for_retest', evt);
  },
  'retest.completed'(d, evt) {
    const f = this.findings.get(d.findingId);
    if (!f) return;
    f.retests.push({
      result: d.result, sessionId: d.sessionId, evidence: d.evidence ?? [], ts: evt.ts,
    });
    f.status = d.result === 'fixed'
      ? 'confirmed'
      : d.result === 'false_positive'
        ? 'rejected'
        : d.result === 'persists'
          ? 'acknowledged'
          : 'ready_for_retest';
    if (d.result === 'persists') {
      for (const mitigationId of f.mitigationIds) {
        const mitigation = this.mitigations.get(mitigationId);
        if (mitigation?.status === 'ready') {
          mitigation.status = 'failed';
          mitigation.updatedAt = evt.ts;
        }
      }
    }
    f.updatedAt = evt.ts;
  },

  'referee.review_started'(d, evt) {
    const c = this.campaigns.get(d.campaignId);
    if (c) c.review = { sessionId: d.sessionId, startedAt: evt.ts, status: 'active' };
  },
  'referee.verdict'(d, evt) {
    if (this.verdicts.has(d.verdictId)) return;
    const v = {
      id: d.verdictId, campaignId: d.campaignId, sessionId: d.sessionId,
      verdict: d.verdict, evidence: d.evidence ?? [], rationale: d.rationale,
      createdAt: evt.ts,
    };
    this.verdicts.set(v.id, v);
    const c = this.campaigns.get(d.campaignId);
    if (c) {
      c.verdictIds.push(v.id);
      c.review = { ...(c.review ?? {}), status: 'complete', verdictId: v.id };
    }
  },

  'campaign.checkpoint_created'(d, evt) {
    if (this.checkpoints.has(d.checkpointId)) return;
    const cp = {
      id: d.checkpointId, campaignId: d.campaignId, name: d.name,
      eventSeq: d.eventSeq ?? evt.seq, revision: d.revision ?? null,
      workspaceId: d.workspaceId ?? null, branch: d.branch ?? null,
      mode: d.mode ?? 'record_only',
      capabilityIds: d.capabilityIds ?? [], scene: d.scene ?? null,
      createdAt: evt.ts,
    };
    this.checkpoints.set(cp.id, cp);
    const c = this.campaigns.get(d.campaignId);
    if (c) c.checkpointIds.push(cp.id);
  },
  'campaign.promoted'(d, evt) {
    const c = this.campaigns.get(d.campaignId);
    if (c) { c.phase = 'promoted'; c.promotedAt = evt.ts; c.promotedCheckpointId = d.checkpointId; }
    for (const capability of d.capabilities ?? []) {
      if (this.capabilities.has(capability.id)) continue;
      this.capabilities.set(capability.id, {
        ...capability, campaignId: d.campaignId, checkpointId: d.checkpointId,
        status: 'promoted', promotedAt: evt.ts,
      });
      if (c) c.capabilityIds.push(capability.id);
    }
  },
  'campaign.rolled_back'(d, evt) {
    const c = this.campaigns.get(d.campaignId);
    if (c) {
      c.phase = 'rolled_back'; c.rollbackCheckpointId = d.checkpointId;
      c.rollbackReason = d.reason; c.rolledBackAt = evt.ts;
      c.rollbackMode = d.mode ?? 'record_only';
      c.rollbackWorkspaceChanged = d.workspaceChanged === true;
    }
  },
};

function setFindingState(projection, d, state, evt, extra = {}) {
  const f = projection.findings.get(d.findingId);
  if (f) { Object.assign(f, extra); f.status = state; f.updatedAt = evt.ts; }
}

const LIVE_SESSION_STATES = new Set(['spawning', 'ready', 'thinking', 'working', 'waiting_permission']);

function synchronizeSessionState(projection, sessionId, state, evt) {
  if (!sessionId) return;
  const memberStatus = LIVE_SESSION_STATES.has(state)
    ? 'active'
    : state === 'paused'
      ? 'paused'
      : 'unavailable';

  for (const campaign of projection.campaigns.values()) {
    let touched = false;
    for (const team of Object.values(campaign.teams)) {
      const member = team.members.find((item) => item.sessionId === sessionId);
      if (!member) continue;
      member.status = memberStatus;
      member.sessionState = state;
      team.ready = team.members.some((item) => ['active', 'external'].includes(item.status));
      touched = true;
    }
    if (!touched) continue;

    // Only blue availability can block build readiness. Red/referee outages are visible in
    // their formations and staffing gates without rewriting already-proven blue evidence.
    for (const objectiveId of campaign.objectiveIds) {
      const objective = projection.objectives.get(objectiveId);
      if (!objective || objective.status === 'satisfied') continue;
      const blueAvailable = campaign.teams.blue.members.some((member) =>
        member.objectiveId === objectiveId && ['active', 'external'].includes(member.status));
      if (!blueAvailable && ['queued', 'active'].includes(objective.status)) {
        objective.status = 'blocked';
        objective.blockedReason = 'no live blue session remains assigned';
        objective.updatedAt = evt.ts;
      } else if (blueAvailable && objective.status === 'blocked'
        && objective.blockedReason === 'no live blue session remains assigned') {
        objective.status = 'active';
        objective.blockedReason = null;
        objective.updatedAt = evt.ts;
      }
    }
    campaign.updatedAt = evt.ts;
    campaign.lastSeq = evt.seq;
  }
}
