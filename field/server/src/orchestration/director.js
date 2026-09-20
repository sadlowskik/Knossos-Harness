import { randomUUID } from 'node:crypto';
import {
  DomainError, VERDICTS, assertId, assertTeam, isContentRevision, normalizeDoctrine, validateCampaignInput,
  validateFinding, validateTransition,
} from './model.js';

const TERMINAL_SESSION_STATES = new Set(['done', 'cancelled', 'error', 'interrupted']);

/**
 * Strategic command boundary for campaigns.
 *
 * It emits facts into the canonical log and asks Registry to perform real harness work.
 * It never mutates a projection directly; the event subscription applies emitted facts,
 * which keeps live commands and replay on the same path.
 */
export class CampaignDirector {
  constructor({ projection, registry, emit, eventHead = () => 0, id = randomUUID }) {
    this.projection = projection;
    this.registry = registry;
    this.emit = emit;
    this.eventHead = eventHead;
    this.id = id;
  }

  create(input = {}) {
    const commandId = this.commandId(input.commandId);
    const prior = this.prior(commandId);
    if (prior) return prior;

    const spec = validateCampaignInput(input);
    const campaignId = input.campaignId ? assertId(input.campaignId, 'campaignId') : this.id();
    if (this.projection.campaigns.has(campaignId)) {
      throw new DomainError('already_exists', `campaign ${campaignId} already exists`);
    }

    this.append('campaign.created', { campaignId, commandId, ...spec }, campaignId);
    const objectiveIds = [];
    for (const objective of spec.objectives) {
      const objectiveId = this.id();
      objectiveIds.push(objectiveId);
      this.append('objective.created', {
        campaignId, objectiveId, commandId, ...objective,
      }, objectiveId);
    }
    return this.result(campaignId, commandId, { objectiveIds });
  }

  action(input = {}) {
    const commandId = this.commandId(input.commandId);
    const prior = this.prior(commandId);
    if (prior) return prior;
    const campaignId = assertId(input.campaignId, 'campaignId');
    const campaign = this.requireCampaign(campaignId);
    const kind = String(input.kind ?? '');

    if (campaign.paused && !['resume', 'cancel', 'rollback'].includes(kind)) {
      throw new DomainError('campaign_paused', 'campaign is paused', { currentPhase: campaign.phase });
    }
    if (campaign.budgetExhausted && !['pause', 'cancel', 'rollback'].includes(kind)) {
      throw new DomainError(
        'budget_exhausted',
        `campaign budget exhausted ($${campaign.costUsd.toFixed(2)} of $${campaign.budgetUsd.toFixed(2)})`
      );
    }

    let detail;
    switch (kind) {
      case 'mobilize': detail = this.mobilize(campaign, input, commandId); break;
      case 'advance': detail = this.advance(campaign, input, commandId); break;
      case 'assign_team': detail = this.assignTeam(campaign, input, commandId); break;
      case 'issue_orders': detail = this.issueOrders(campaign, input, commandId); break;
      case 'reinforce': detail = this.reinforce(campaign, input, commandId); break;
      case 'retreat': detail = this.retreat(campaign, input, commandId); break;
      case 'pause': detail = this.pause(campaign, input, commandId); break;
      case 'resume': detail = this.resume(campaign, input, commandId); break;
      case 'cancel': detail = this.cancel(campaign, input, commandId); break;
      case 'change_doctrine': detail = this.changeDoctrine(campaign, input, commandId); break;
      case 'objective_progress': detail = this.objectiveProgress(campaign, input, commandId); break;
      case 'satisfy_objective': detail = this.satisfyObjective(campaign, input, commandId); break;
      case 'block_objective': detail = this.blockObjective(campaign, input, commandId); break;
      case 'report_finding': detail = this.reportFinding(campaign, input, commandId); break;
      case 'acknowledge_finding': detail = this.findingState(campaign, input, commandId, 'acknowledged'); break;
      case 'dispute_finding': detail = this.findingState(campaign, input, commandId, 'disputed'); break;
      case 'waive_finding': detail = this.waiveFinding(campaign, input, commandId); break;
      case 'propose_mitigation': detail = this.proposeMitigation(campaign, input, commandId); break;
      case 'start_mitigation': detail = this.mitigationState(campaign, input, commandId, 'started'); break;
      case 'mark_mitigation_ready': detail = this.mitigationState(campaign, input, commandId, 'ready'); break;
      case 'record_retest': detail = this.recordRetest(campaign, input, commandId); break;
      case 'begin_referee_review': detail = this.beginReview(campaign, input, commandId); break;
      case 'record_verdict': detail = this.recordVerdict(campaign, input, commandId); break;
      case 'checkpoint': detail = this.checkpoint(campaign, input, commandId); break;
      case 'promote': detail = this.promote(campaign, input, commandId); break;
      case 'rollback': detail = this.rollback(campaign, input, commandId); break;
      default: throw new DomainError('unknown_action', `unknown campaign action: ${kind}`);
    }
    return this.result(campaignId, commandId, detail);
  }

  /** Convert a validated, explicit final-message sentinel into ordinary campaign actions. */
  ingestSessionReport(envelope) {
    const base = {
      commandId: `report:${envelope.sessionId}:${envelope.eventSeq ?? this.id()}`,
      campaignId: envelope.campaignId,
      objectiveId: envelope.objectiveId,
      sessionId: envelope.sessionId,
    };
    const report = envelope.report ?? {};
    switch (report.kind) {
      case 'objective_progress':
        return this.action({ ...base, kind: 'objective_progress', progress: report.progress, evidence: report.evidence });
      case 'objective_satisfied':
        if (envelope.team !== 'blue') throw new DomainError('forbidden', 'only blue can submit build-readiness evidence');
        return this.action({ ...base, kind: 'satisfy_objective', evidence: report.evidence });
      case 'finding':
        if (envelope.team !== 'red') throw new DomainError('forbidden', 'only red can submit a finding report');
        return this.action({
          ...base, kind: 'report_finding', authorSessionId: envelope.sessionId,
          category: report.category, severity: report.severity, claim: report.claim,
          scope: report.scope, evidence: report.evidence, reproduction: report.reproduction,
          noReproductionReason: report.noReproductionReason, confidence: report.confidence,
        });
      case 'mitigation_ready':
        if (envelope.team !== 'blue') throw new DomainError('forbidden', 'only blue can submit mitigation readiness');
        return this.action({
          ...base, kind: 'mark_mitigation_ready', mitigationId: report.mitigationId,
          evidence: report.evidence,
        });
      case 'retest':
        if (envelope.team !== 'red') throw new DomainError('forbidden', 'only red can submit a retest');
        return this.action({
          ...base, kind: 'record_retest', findingId: report.findingId,
          result: report.result, evidence: report.evidence,
        });
      case 'verdict':
        if (envelope.team !== 'referee') throw new DomainError('forbidden', 'only a referee can submit a verdict');
        return this.action({
          ...base, kind: 'record_verdict', verdict: report.verdict,
          evidence: report.evidence, rationale: report.rationale,
        });
      default:
        throw new DomainError('invalid_report', `unknown FIELD_REPORT kind: ${report.kind ?? '(missing)'}`);
    }
  }

  mobilize(campaign, input, commandId) {
    if (campaign.phase !== 'draft' && campaign.phase !== 'failed') {
      throw new DomainError('invalid_transition', `cannot mobilize from ${campaign.phase}`);
    }
    const roster = Array.isArray(input.roster) ? input.roster : [];
    if (!roster.length && !campaign.teams.blue.members.length) {
      throw new DomainError('empty_roster', 'mobilization needs at least one blue roster entry');
    }

    const spawned = [];
    const gaps = [];
    const remaining = Math.max(0, campaign.concurrency - this.liveMemberIds(campaign).length);
    const admitted = roster.slice(0, remaining);
    for (const entry of roster.slice(remaining)) {
      gaps.push({ team: entry.team, role: entry.role ?? 'unspecified', agentId: entry.agentId, reason: 'campaign concurrency limit' });
    }
    for (const entry of admitted) {
      const team = assertTeam(entry.team);
      const objective = this.requireObjective(campaign, entry.objectiveId ?? campaign.objectiveIds[0]);
      try {
        const session = this.registry.spawn({
          agentId: assertId(entry.agentId, 'agentId'),
          workspaceId: entry.workspaceId ?? objective.target?.workspaceId ?? campaign.target?.workspaceId,
          missionId: entry.missionId,
          orders: this.teamOrders(campaign, objective, team, entry.orders),
          endpointId: entry.endpointId,
          thinking: entry.thinking,
          target: objective.target ?? campaign.target,
          campaignId: campaign.id,
          team,
          objectiveId: objective.id,
          doctrine: campaign.doctrine,
          environmentScope: campaign.scope,
        });
        this.assertIndependentMembership(campaign, team, session.id);
        this.append('team.member_assigned', {
          campaignId: campaign.id, commandId, team, sessionId: session.id,
          agentId: entry.agentId, role: entry.role ?? session.role ?? null,
          objectiveId: objective.id, status: 'active', assignedAt: Date.now(),
        }, campaign.id);
        this.append('objective.assigned', {
          campaignId: campaign.id, commandId, objectiveId: objective.id,
          team, sessionIds: [session.id],
        }, objective.id);
        spawned.push(session.id);
      } catch (error) {
        const gap = { team, role: entry.role ?? 'unspecified', agentId: entry.agentId, reason: error.message };
        gaps.push(gap);
        this.append('team.staffing_gap', { campaignId: campaign.id, commandId, ...gap }, campaign.id);
      }
    }

    const context = this.projection.context(campaign.id);
    const hasBlue = campaign.teams.blue.members.some((member) => member.status === 'active');
    if (!hasBlue) {
      if (spawned.length) this.registry.command('pause', { sessionIds: spawned });
      this.append('campaign.failed', {
        campaignId: campaign.id, commandId,
        reason: 'mobilization produced no active blue team; spawned non-blue sessions were paused',
      }, campaign.id);
      return { spawned, gaps, phase: 'failed' };
    }
    validateTransition(campaign, 'mobilizing', context);
    this.append('campaign.phase_changed', {
      campaignId: campaign.id, commandId, from: campaign.phase, to: 'mobilizing',
      reason: gaps.length ? `mobilized with ${gaps.length} staffing gaps` : 'roster mobilized',
    }, campaign.id);
    return { spawned, gaps, phase: 'mobilizing' };
  }

  advance(campaign, input, commandId) {
    const to = String(input.to ?? '');
    const context = this.projection.context(campaign.id);
    validateTransition(campaign, to, context);
    this.append('campaign.phase_changed', {
      campaignId: campaign.id, commandId, from: campaign.phase, to,
      reason: String(input.reason ?? 'operator advanced campaign').slice(0, 2000),
    }, campaign.id);
    return { from: campaign.phase, to };
  }

  assignTeam(campaign, input, commandId) {
    const team = assertTeam(input.team);
    const sessionId = assertId(input.sessionId, 'sessionId');
    this.assertIndependentMembership(campaign, team, sessionId);
    const alreadyAssigned = Object.values(campaign.teams)
      .some((formation) => formation.members.some((member) => member.sessionId === sessionId && member.status !== 'retired'));
    if (!alreadyAssigned) this.assertCampaignCapacity(campaign);
    const objective = this.requireObjective(campaign, input.objectiveId ?? campaign.objectiveIds[0]);
    const session = this.registry.info?.(sessionId);
    if (!session && !input.external) throw new DomainError('session_not_found', `session ${sessionId} is not live`);
    this.append('team.member_assigned', {
      campaignId: campaign.id, commandId, team, sessionId,
      agentId: input.agentId ?? session?.agentId ?? null,
      role: input.role ?? session?.role ?? null, objectiveId: objective.id,
      status: session
        ? (TERMINAL_SESSION_STATES.has(session.state) ? 'unavailable' : 'active')
        : 'external',
      external: !session,
      assignedAt: Date.now(),
    }, campaign.id);
    this.append('objective.assigned', {
      campaignId: campaign.id, commandId, objectiveId: objective.id, team, sessionIds: [sessionId],
    }, objective.id);
    return { team, sessionId, objectiveId: objective.id };
  }

  issueOrders(campaign, input, commandId) {
    const team = assertTeam(input.team);
    const objective = this.requireObjective(campaign, input.objectiveId ?? campaign.objectiveIds[0]);
    const members = campaign.teams[team].members.filter((m) => m.status === 'active');
    const requested = input.sessionIds?.length ? new Set(input.sessionIds) : null;
    const sessionIds = members.map((m) => m.sessionId).filter((id) => !requested || requested.has(id));
    if (!sessionIds.length) throw new DomainError('empty_team', `team ${team} has no active selected members`);
    const orders = this.teamOrders(campaign, objective, team, input.orders);
    const target = objective.target ?? campaign.target ?? { type: 'workspace', id: objective.id };
    const assignment = this.registry.assign({ sessionIds, target, orders, endpointId: input.endpointId, thinking: input.thinking });
    this.append('team.orders_issued', {
      campaignId: campaign.id, commandId, team, objectiveId: objective.id,
      sessionIds, assignmentId: assignment.assignmentId, orders,
    }, campaign.id);
    return { team, sessionIds, assignmentId: assignment.assignmentId, skipped: assignment.skipped ?? [] };
  }

  reinforce(campaign, input, commandId) {
    return this.mobilizeIntoExisting(campaign, input, commandId, 'team.reinforced');
  }

  mobilizeIntoExisting(campaign, input, commandId, eventKind) {
    this.assertCampaignCapacity(campaign);
    const team = assertTeam(input.team);
    const objective = this.requireObjective(campaign, input.objectiveId ?? campaign.objectiveIds[0]);
    const session = this.registry.spawn({
      agentId: assertId(input.agentId, 'agentId'),
      workspaceId: input.workspaceId ?? objective.target?.workspaceId ?? campaign.target?.workspaceId,
      orders: this.teamOrders(campaign, objective, team, input.orders),
      endpointId: input.endpointId, thinking: input.thinking,
      target: objective.target ?? campaign.target,
      campaignId: campaign.id, team, objectiveId: objective.id,
      doctrine: campaign.doctrine, environmentScope: campaign.scope,
    });
    this.assertIndependentMembership(campaign, team, session.id);
    this.append('team.member_assigned', {
      campaignId: campaign.id, commandId, team, sessionId: session.id,
      agentId: input.agentId, role: input.role ?? session.role ?? null,
      objectiveId: objective.id, status: 'active', assignedAt: Date.now(),
    }, campaign.id);
    this.append('objective.assigned', {
      campaignId: campaign.id, commandId, objectiveId: objective.id, team, sessionIds: [session.id],
    }, objective.id);
    this.append(eventKind, { campaignId: campaign.id, commandId, team, sessionIds: [session.id] }, campaign.id);
    return { team, sessionId: session.id };
  }

  retreat(campaign, input, commandId) {
    const team = assertTeam(input.team);
    const member = campaign.teams[team].members.find((m) => m.sessionId === input.sessionId);
    if (!member) throw new DomainError('not_a_member', `${input.sessionId} is not on team ${team}`);
    this.registry.command('pause', { sessionIds: [member.sessionId] });
    this.append('team.member_removed', {
      campaignId: campaign.id, commandId, team, sessionId: member.sessionId,
      reason: String(input.reason ?? 'operator retreat').slice(0, 2000),
    }, campaign.id);
    this.append('team.retreated', { campaignId: campaign.id, commandId, team, sessionIds: [member.sessionId] }, campaign.id);
    return { team, sessionId: member.sessionId };
  }

  pause(campaign, input, commandId) {
    if (campaign.paused) return { paused: true };
    const ids = this.liveMemberIds(campaign);
    if (ids.length) this.registry.command('pause', { sessionIds: ids });
    this.append('campaign.paused', { campaignId: campaign.id, commandId, reason: input.reason ?? null }, campaign.id);
    return { paused: true, sessionIds: ids };
  }

  resume(campaign, input, commandId) {
    if (!campaign.paused) throw new DomainError('not_paused', 'campaign is not paused');
    const ids = this.liveMemberIds(campaign);
    if (ids.length) this.registry.command('resume', { sessionIds: ids, orders: input.orders });
    this.append('campaign.resumed', { campaignId: campaign.id, commandId }, campaign.id);
    return { resumed: true, sessionIds: ids };
  }

  cancel(campaign, input, commandId) {
    if (campaign.phase === 'cancelled') return { cancelled: true };
    const ids = this.liveMemberIds(campaign);
    if (ids.length) this.registry.command('cancel', { sessionIds: ids });
    this.append('campaign.cancelled', {
      campaignId: campaign.id, commandId, reason: String(input.reason ?? 'operator cancelled').slice(0, 2000),
    }, campaign.id);
    return { cancelled: true, sessionIds: ids };
  }

  changeDoctrine(campaign, input, commandId) {
    const doctrine = normalizeDoctrine({ ...campaign.doctrine, ...(input.doctrine ?? {}) });
    this.append('campaign.doctrine_changed', { campaignId: campaign.id, commandId, doctrine }, campaign.id);
    return { doctrine };
  }

  objectiveProgress(campaign, input, commandId) {
    const objective = this.requireObjective(campaign, input.objectiveId);
    this.append('objective.progress', {
      campaignId: campaign.id, commandId, objectiveId: objective.id,
      progress: input.progress ?? null, evidence: input.evidence ?? null,
    }, objective.id);
    return { objectiveId: objective.id };
  }

  satisfyObjective(campaign, input, commandId) {
    const objective = this.requireObjective(campaign, input.objectiveId);
    const evidence = list(input.evidence);
    if (!evidence.length) throw new DomainError('missing_evidence', 'satisfying an objective requires evidence');
    if (evidence.length < objective.definitionOfDone.length) {
      throw new DomainError(
        'missing_evidence',
        `objective has ${objective.definitionOfDone.length} completion criteria but only ${evidence.length} evidence entries`
      );
    }
    const criteriaEvidence = objective.definitionOfDone.map((criterion, index) => ({
      criterion,
      evidence: evidence[index],
    }));
    this.append('objective.satisfied', {
      campaignId: campaign.id, commandId, objectiveId: objective.id, evidence, criteriaEvidence,
      sessionId: input.sessionId ?? null,
    }, objective.id);
    return { objectiveId: objective.id, status: 'satisfied' };
  }

  blockObjective(campaign, input, commandId) {
    const objective = this.requireObjective(campaign, input.objectiveId);
    this.append('objective.blocked', {
      campaignId: campaign.id, commandId, objectiveId: objective.id,
      reason: String(input.reason ?? 'blocked').slice(0, 4000),
    }, objective.id);
    return { objectiveId: objective.id, status: 'blocked' };
  }

  reportFinding(campaign, input, commandId) {
    if (!['red_challenging', 'red_retesting'].includes(campaign.phase)) {
      throw new DomainError('wrong_phase', `findings cannot be reported during ${campaign.phase}`);
    }
    const objective = this.requireObjective(campaign, input.objectiveId);
    this.assertMember(campaign, 'red', input.authorSessionId);
    const finding = validateFinding(input);
    const findingId = input.findingId ? assertId(input.findingId, 'findingId') : this.id();
    if (this.projection.findings.has(findingId)) throw new DomainError('already_exists', `finding ${findingId} exists`);
    this.append('finding.reported', {
      campaignId: campaign.id, commandId, findingId, objectiveId: objective.id,
      authorSessionId: input.authorSessionId, ...finding,
    }, findingId);
    return { findingId };
  }

  findingState(campaign, input, commandId, state) {
    const finding = this.requireFinding(campaign, input.findingId);
    const kind = state === 'disputed' ? 'finding.disputed' : 'finding.acknowledged';
    if (state === 'disputed' && !list(input.evidence).length) {
      throw new DomainError('missing_evidence', 'disputing a finding requires evidence');
    }
    this.append(kind, {
      campaignId: campaign.id, commandId, findingId: finding.id,
      sessionId: input.sessionId ?? null, evidence: list(input.evidence),
    }, finding.id);
    return { findingId: finding.id, status: state };
  }

  waiveFinding(campaign, input, commandId) {
    const finding = this.requireFinding(campaign, input.findingId);
    if (!['operator', 'referee'].includes(input.authority)) {
      throw new DomainError('forbidden', 'only an operator or referee can waive a finding');
    }
    const reason = String(input.reason ?? '').trim();
    if (!reason) throw new DomainError('missing_reason', 'waiving a finding requires a reason');
    this.append('finding.waived', {
      campaignId: campaign.id, commandId, findingId: finding.id,
      authority: input.authority, sessionId: input.sessionId ?? null, reason,
    }, finding.id);
    return { findingId: finding.id, status: 'waived' };
  }

  proposeMitigation(campaign, input, commandId) {
    const findingIds = list(input.findingIds);
    if (!findingIds.length) throw new DomainError('missing_findings', 'a mitigation must reference findings');
    for (const id of findingIds) this.requireFinding(campaign, id);
    if (input.ownerSessionId) this.assertMember(campaign, 'blue', input.ownerSessionId);
    const claim = String(input.claim ?? '').trim();
    if (!claim) throw new DomainError('invalid_mitigation', 'mitigation claim is required');
    const mitigationId = input.mitigationId ? assertId(input.mitigationId, 'mitigationId') : this.id();
    this.append('mitigation.proposed', {
      campaignId: campaign.id, commandId, mitigationId, findingIds,
      ownerSessionId: input.ownerSessionId ?? null, claim,
      artifacts: list(input.artifacts), evidence: list(input.evidence),
    }, mitigationId);
    return { mitigationId };
  }

  mitigationState(campaign, input, commandId, state) {
    const mitigation = this.requireMitigation(campaign, input.mitigationId);
    if (state === 'ready' && !list(input.evidence).length && !(mitigation.evidence?.length)) {
      throw new DomainError('missing_evidence', 'a ready mitigation requires evidence');
    }
    const kind = state === 'started' ? 'mitigation.started' : 'mitigation.ready';
    this.append(kind, {
      campaignId: campaign.id, commandId, mitigationId: mitigation.id,
      sessionId: input.sessionId ?? null, evidence: list(input.evidence),
    }, mitigation.id);
    return { mitigationId: mitigation.id, status: state === 'started' ? 'active' : 'ready' };
  }

  recordRetest(campaign, input, commandId) {
    const finding = this.requireFinding(campaign, input.findingId);
    this.assertMember(campaign, 'red', input.sessionId);
    if (!['fixed', 'persists', 'false_positive', 'inconclusive'].includes(input.result)) {
      throw new DomainError('invalid_retest', `unknown retest result: ${input.result}`);
    }
    const evidence = list(input.evidence);
    if (!evidence.length) throw new DomainError('missing_evidence', 'a retest requires evidence');
    this.append('retest.completed', {
      campaignId: campaign.id, commandId, findingId: finding.id,
      sessionId: input.sessionId, result: input.result, evidence,
    }, finding.id);
    return { findingId: finding.id, result: input.result };
  }

  beginReview(campaign, input, commandId) {
    if (campaign.phase !== 'referee_review') {
      throw new DomainError('wrong_phase', `referee review cannot begin during ${campaign.phase}`);
    }
    this.assertMember(campaign, 'referee', input.sessionId);
    this.append('referee.review_started', {
      campaignId: campaign.id, commandId, sessionId: input.sessionId,
    }, campaign.id);
    return { sessionId: input.sessionId };
  }

  recordVerdict(campaign, input, commandId) {
    if (campaign.phase !== 'referee_review') {
      throw new DomainError('wrong_phase', `verdict cannot be recorded during ${campaign.phase}`);
    }
    this.assertMember(campaign, 'referee', input.sessionId);
    this.assertIndependentMembership(campaign, 'referee', input.sessionId);
    if (campaign.review?.status !== 'active' || campaign.review.sessionId !== input.sessionId) {
      throw new DomainError('review_not_started', 'this referee must begin an independent review before recording a verdict');
    }
    if (!VERDICTS.includes(input.verdict)) throw new DomainError('invalid_verdict', `unknown verdict: ${input.verdict}`);
    const evidence = list(input.evidence);
    if (!evidence.length) throw new DomainError('missing_evidence', 'a referee verdict requires evidence');
    const rationale = String(input.rationale ?? '').trim();
    if (!rationale) throw new DomainError('missing_rationale', 'a referee verdict requires rationale');
    const verdictId = this.id();
    this.append('referee.verdict', {
      campaignId: campaign.id, commandId, verdictId, sessionId: input.sessionId,
      verdict: input.verdict, evidence, rationale,
    }, verdictId);
    return { verdictId, verdict: input.verdict };
  }

  checkpoint(campaign, input, commandId) {
    if (!['verified', 'promoted'].includes(campaign.phase)) {
      throw new DomainError('wrong_phase', `checkpoint cannot be promoted from ${campaign.phase}`);
    }
    const checkpointId = input.checkpointId ? assertId(input.checkpointId, 'checkpointId') : this.id();
    this.append('campaign.checkpoint_created', {
      campaignId: campaign.id, commandId, checkpointId,
      name: String(input.name ?? `${campaign.name} checkpoint`).slice(0, 200),
      eventSeq: this.eventHead(), revision: input.revision ?? null,
      workspaceId: input.workspaceId ?? campaign.target?.workspaceId ?? null,
      branch: input.branch ?? null, mode: input.checkpointMode ?? 'record_only',
      capabilityIds: list(input.capabilityIds), scene: input.scene ?? null,
    }, checkpointId);
    return { checkpointId };
  }

  promote(campaign, input, commandId) {
    const context = this.projection.context(campaign.id);
    const checkpointId = input.checkpointId ?? context.checkpointId;
    validateTransition(campaign, 'promoted', { ...context, checkpointId });
    const checkpoint = this.projection.checkpoints.get(checkpointId);
    if (!checkpoint || checkpoint.campaignId !== campaign.id) {
      throw new DomainError('invalid_checkpoint', 'checkpoint does not belong to this campaign');
    }
    if (!isContentRevision(checkpoint.revision)) {
      throw new DomainError('invalid_checkpoint', 'promotion requires a checkpoint bound to a Git or SHA-256 content revision');
    }
    const capabilities = (input.capabilities ?? []).map((capability) => ({
      id: assertId(capability.id ?? this.id(), 'capabilityId'),
      name: String(capability.name ?? capability.id ?? 'capability').slice(0, 300),
      evidence: list(capability.evidence), target: capability.target ?? campaign.target,
    }));
    if (!capabilities.length) throw new DomainError('missing_capabilities', 'promotion requires at least one capability');
    this.append('campaign.promoted', {
      campaignId: campaign.id, commandId, checkpointId, capabilities,
    }, campaign.id);
    return { checkpointId, capabilityIds: capabilities.map((x) => x.id) };
  }

  rollback(campaign, input, commandId) {
    if (campaign.phase !== 'promoted') {
      throw new DomainError('wrong_phase', `rollback requires a promoted campaign, not ${campaign.phase}`);
    }
    const checkpointId = assertId(input.checkpointId, 'checkpointId');
    const checkpoint = this.projection.checkpoints.get(checkpointId);
    if (!checkpoint || checkpoint.campaignId !== campaign.id) {
      throw new DomainError('invalid_checkpoint', 'rollback checkpoint does not belong to this campaign');
    }
    const reason = String(input.reason ?? '').trim();
    if (!reason) throw new DomainError('missing_reason', 'rollback requires a reason');
    this.append('campaign.rolled_back', {
      campaignId: campaign.id, commandId, checkpointId, reason,
      mode: 'record_only', workspaceChanged: false,
    }, campaign.id);
    return { checkpointId, rolledBack: true, mode: 'record_only', workspaceChanged: false };
  }

  teamOrders(campaign, objective, team, extra) {
    const lines = [
      `Campaign: ${campaign.name}`,
      `Intent: ${campaign.intent}`,
      `Your team: ${team.toUpperCase()}`,
      `Environment scope: ${campaign.scope}`,
      `Objective: ${objective.statement}`,
      'Definition of done:',
      ...objective.definitionOfDone.map((x) => `- ${x}`),
    ];
    if (team === 'blue') lines.push('Build or operate the result and attach concrete evidence for every completion claim.');
    if (team === 'red') {
      lines.push(
        `Challenge categories: ${campaign.doctrine.redCategories.join(', ')}.`,
        'Do not repair what you find. Report claim, severity, scope, evidence, reproduction, and confidence.',
        'Stay inside the declared environment scope and use decoy credentials for injection tests.',
      );
    }
    if (team === 'referee') {
      lines.push('Reproduce the definition of done independently. Neither blue nor red summaries are facts until you verify them.');
    }
    if (team === 'purple') lines.push('Synthesize lessons only after the referee verdict. You cannot waive retest or verification.');
    lines.push(
      '',
      'Structured campaign reporting:',
      'End the final message with `FIELD_REPORT:` followed by exactly one compact JSON object.',
    );
    if (team === 'blue') lines.push(
      'Use {"kind":"objective_satisfied","evidence":[...]} only when every definition-of-done item has concrete evidence.',
      'For partial work use {"kind":"objective_progress","progress":{"done":N,"total":N},"evidence":"..."}.',
    );
    if (team === 'red') lines.push(
      'Use {"kind":"finding","category":"...","severity":"high","claim":"...","scope":"...","evidence":[...],"reproduction":[...],"confidence":0.9}.',
      'During retest use {"kind":"retest","findingId":"...","result":"fixed|persists|false_positive|inconclusive","evidence":[...]}.',
    );
    if (team === 'referee') lines.push(
      'Use {"kind":"verdict","verdict":"verified|rejected|inconclusive","evidence":[...],"rationale":"..."}.',
    );
    if (extra) lines.push('', String(extra).slice(0, 20_000));
    return lines.join('\n');
  }

  append(kind, data, subject) {
    return this.emit(kind, data, { subject });
  }

  commandId(value) {
    return value ? assertId(value, 'commandId') : this.id();
  }

  prior(commandId) {
    const prior = this.projection.commandResults.get(commandId);
    if (!prior) return null;
    return this.result(prior.campaignId, commandId, { duplicate: true, originalSeq: prior.seq });
  }

  result(campaignId, commandId, detail = {}) {
    const campaign = this.projection.campaigns.get(campaignId);
    return {
      ok: true, commandId, campaignId, head: this.eventHead(),
      campaign: campaign ? this.projection.campaignView(campaign) : null,
      ...detail,
    };
  }

  requireCampaign(id) {
    const c = this.projection.campaigns.get(id);
    if (!c) throw new DomainError('not_found', `campaign ${id} does not exist`);
    return c;
  }

  requireObjective(campaign, id) {
    const o = this.projection.objectives.get(id);
    if (!o || o.campaignId !== campaign.id) throw new DomainError('not_found', `objective ${id} does not belong to campaign ${campaign.id}`);
    return o;
  }

  requireFinding(campaign, id) {
    const f = this.projection.findings.get(id);
    if (!f || f.campaignId !== campaign.id) throw new DomainError('not_found', `finding ${id} does not belong to campaign ${campaign.id}`);
    return f;
  }

  requireMitigation(campaign, id) {
    const m = this.projection.mitigations.get(id);
    if (!m || m.campaignId !== campaign.id) throw new DomainError('not_found', `mitigation ${id} does not belong to campaign ${campaign.id}`);
    return m;
  }

  assertMember(campaign, team, sessionId) {
    const member = campaign.teams[team]?.members.find((m) => m.sessionId === sessionId && m.status !== 'retired');
    if (!member) throw new DomainError('forbidden', `session ${sessionId} is not an active ${team} team member`);
    return member;
  }

  assertIndependentMembership(campaign, team, sessionId) {
    const conflicts = team === 'referee' ? ['blue', 'red'] : ['referee'];
    for (const other of conflicts) {
      if (campaign.teams[other].members.some((m) => m.sessionId === sessionId && m.status !== 'retired')) {
        throw new DomainError('role_conflict', `${sessionId} cannot serve on both ${team} and ${other}`);
      }
    }
  }

  liveMemberIds(campaign) {
    return [...new Set(Object.values(campaign.teams)
      .flatMap((team) => team.members)
      .filter((m) => m.status === 'active')
      .map((m) => m.sessionId))];
  }

  assertCampaignCapacity(campaign) {
    const active = this.liveMemberIds(campaign).length;
    if (active >= campaign.concurrency) {
      throw new DomainError(
        'campaign_capacity',
        `campaign concurrency limit is ${campaign.concurrency}; ${active} sessions are already active`
      );
    }
  }
}


function list(value) {
  return Array.isArray(value) ? value.filter((x) => x != null && x !== '').slice(0, 200) : [];
}
