// Live operational state as a fold over the event log.
// Rebuilt by replaying every event on boot, so the running state and a historical
// replay are produced by exactly the same code path.

import { CampaignProjection } from '../orchestration/campaign-projection.js';
import { GraphProjection } from '../orchestration/graph-projection.js';
import { BudgetLedger } from '../budget-ledger.js';

const MAX_CONTEXT_TOKENS = 200_000;
const ACTIVITY_WINDOW_MS = 60 * 60 * 1000;
const ACTIVITY_EVENTS_FOR_FULL_SIGNAL = 20;

export class Projection {
  constructor(cfg, { partitionSynthetic = true } = {}) {
    this.cfg = cfg;
    this.partitionSynthetic = partitionSynthetic;
    this.campaigns = new CampaignProjection();
    this.graph = new GraphProjection();
    this.synthetic = null;
    this.reset();
    if (partitionSynthetic) this.synthetic = new Projection(cfg, { partitionSynthetic: false });
  }

  reset() {
    this.budgets = new BudgetLedger();
    this.campaigns.reset();
    this.graph.reset();
    this.sessions = new Map();
    this.folders = new Map();
    this.files = new Map();
    this.websites = new Map();
    this.endpoints = new Map();
    this.assignments = new Map();
    this.permissions = new Map();
    this.workspaces = new Map();
    this.controlGroups = {};
    this.positions = new Map();
    this.routines = new Map((this.cfg.routines ?? []).map((routine) => [routine.id, {
      id: routine.id, enabled: !!routine.enabled, lastRunAt: null,
      lastOutcome: null, activeRunId: null, activeRunIds: [], currentOwner: null, queuedRuns: [], cooldownUntil: 0, history: [],
    }]));
    this.world = {
      capitalWorkspaceId: null,
      capitalSelectedAt: null,
      assignments: {},
      revision: 0,
    };
    this.totals = { costUsd: 0, inputTokens: 0, outputTokens: 0, sessionsSpawned: 0 };
    this.lastSeq = 0;
    if (this.synthetic) this.synthetic.reset();

    for (const w of this.cfg.workspaces) {
      this.workspaces.set(w.id, {
        id: w.id, name: w.name, path: w.path, mounted: w.mounted,
        region: w.region ?? { x: 0, y: 0, w: 700, h: 440 },
        git: null, changeCount: 0, lastTs: 0, activityEvents: [],
      });
    }
    for (const e of this.cfg.endpoints) {
      this.endpoints.set(e.id, {
        id: e.id, name: e.name, kind: e.kind, model: e.model,
        baseUrl: e.base_url ?? null,
        costPerMtok: e.cost_per_mtok ?? { input: 0, output: 0 },
        status: 'unknown', latencyMs: null, lastCheck: 0, detail: null, failures: 0,
      });
    }
    for (const s of this.cfg.websites) {
      this.websites.set(s.domain, {
        domain: s.domain, label: s.label ?? s.domain,
        sessions: new Set(), lastUrl: null, lastTs: 0, hits: 0,
      });
    }
  }

  apply(evt) {
    this.lastSeq = Math.max(this.lastSeq, evt.seq);
    const synthetic = evt.source === 'synthetic' || evt.data?.simulated === true;
    if (this.partitionSynthetic && synthetic) {
      if (evt.kind === 'simulation.started') this.synthetic.reset();
      this.synthetic.apply({ ...evt, source: 'synthetic' });
      return;
    }
    this.campaigns.apply(evt);
    this.budgets.apply(evt);
    this.graph.apply(evt);
    const d = evt.data ?? {};
    const fn = HANDLERS[evt.kind];
    if (fn) fn.call(this, d, evt);
    const s = d.sessionId ? this.sessions.get(d.sessionId) : null;
    if (s) s.lastEventTs = evt.ts;
  }

  session(id) {
    if (!this.sessions.has(id)) {
      this.sessions.set(id, {
        id, agentId: null, name: id.slice(0, 6), role: 'builder', model: null,
        endpointId: null, thinking: 'medium', cwd: null, workspaceId: null,
        state: 'spawning', parentId: null, children: [], depth: 0,
        target: null, assignmentId: null, lastTool: null, toolCount: 0, editCount: 0,
        tokens: { input: 0, output: 0, cacheRead: 0 }, contextPct: 0, costUsd: 0,
        startedAt: 0, endedAt: null, lastEventTs: 0, messageCount: 0,
        verified: 'unverified', browser: null, error: null, progress: null,
        campaignId: null, team: null, objectiveId: null,
        touched: [],
      });
    }
    return this.sessions.get(id);
  }

  folderKey(workspaceId, dir) { return workspaceId + ':' + dir; }

  touchFolder(workspaceId, dir, ts, sessionId) {
    const key = this.folderKey(workspaceId, dir);
    if (!this.folders.has(key)) {
      this.folders.set(key, { key, workspaceId, dir, hits: 0, lastTs: 0, agents: new Set() });
    }
    const f = this.folders.get(key);
    f.hits += 1;
    f.lastTs = Math.max(f.lastTs, ts);
    if (sessionId) f.agents.add(sessionId);
    return f;
  }

  snapshot(now = Date.now()) {
    const campaign = this.campaigns.snapshot();
    const workspaces = [...this.workspaces.values()].map((workspace) => {
      const { activityEvents, ...view } = workspace;
      return {
        ...view,
        maturity: workspaceMaturity(workspace, campaign.campaigns, now),
      };
    });
    const state = {
      seq: this.lastSeq,
      now,
      sessions: [...this.sessions.values()].map((s) => ({ ...s, touched: s.touched.slice(-40) })),
      workspaces,
      folders: [...this.folders.values()].map((f) => ({ ...f, agents: [...f.agents] })),
      files: [...this.files.values()],
      websites: [...this.websites.values()].map((s) => ({ ...s, sessions: [...s.sessions] })),
      endpoints: [...this.endpoints.values()],
      assignments: [...this.assignments.values()],
      permissions: [...this.permissions.values()].filter((p) => p.status === 'pending'),
      controlGroups: this.controlGroups,
      positions: Object.fromEntries(this.positions),
      world: structuredClone(this.world),
      totals: this.totals,
      budgetReservations: this.budgets.snapshot(),
      campaigns: campaign.campaigns,
      capabilities: campaign.capabilities,
      checkpoints: campaign.checkpoints,
      routines: [...this.routines.values()].map((routine) => ({
        ...routine, history: routine.history.slice(-50),
      })),
      // Websocket snapshots are bounded; the complete graph remains reconstructible from
      // the event log and can be paged by a future graph-inspection route.
      graph: this.graph.snapshot(now, { maxNodes: 4000, maxEdges: 8000 }),
    };
    if (this.synthetic) state.rehearsal = this.synthetic.snapshot(now);
    return state;
  }
}

const HANDLERS = {
  'session.spawned'(d, evt) {
    const s = this.session(d.sessionId);
    Object.assign(s, {
      agentId: d.agentId, name: d.name ?? s.name, role: d.role, model: d.model,
      endpointId: d.endpointId, thinking: d.thinking, cwd: d.cwd,
      workspaceId: d.workspaceId, parentId: d.parentSessionId ?? null,
      state: 'spawning', startedAt: evt.ts, target: d.target ?? null,
      assignmentId: d.assignmentId ?? null,
      campaignId: d.campaignId ?? null, team: d.team ?? null,
      objectiveId: d.objectiveId ?? null,
      simulated: !!d.simulated, simulationRunId: d.simulationRunId ?? null,
    });
    if (d.parentSessionId) {
      const p = this.session(d.parentSessionId);
      if (!p.children.includes(d.sessionId)) p.children.push(d.sessionId);
      s.depth = (p.depth ?? 0) + 1;
    }
    this.totals.sessionsSpawned += 1;
  },

  'session.state'(d) {
    const s = this.session(d.sessionId);
    s.state = d.state;
    if (d.detail !== undefined) s.stateDetail = d.detail;
  },

  'session.message'(d) {
    const s = this.session(d.sessionId);
    s.messageCount += 1;
    if (d.role === 'assistant' && d.text) s.lastSay = d.text.slice(0, 400);
  },

  'session.progress'(d) {
    const s = this.session(d.sessionId);
    s.progress = {
      done: Math.max(0, Number(d.done) || 0),
      total: Math.max(0, Number(d.total) || 0),
      steps: Array.isArray(d.steps) ? d.steps.slice(0, 100) : [],
    };
  },

  'session.tool_use'(d, evt) {
    const s = this.session(d.sessionId);
    s.toolCount += 1;
    s.lastTool = { name: d.name, summary: d.summary ?? '', ts: evt.ts };
    s.state = 'working';

    if (d.workspaceId && d.dir != null) {
      this.touchFolder(d.workspaceId, d.dir, evt.ts, d.sessionId);
      const w = this.workspaces.get(d.workspaceId);
      if (w) recordWorkspaceActivity(w, evt);
      // Reading or writing inside a workspace attaches the agent to that region.
      s.workspaceId = d.workspaceId;
      s.focusDir = d.dir;
    }
    if (d.path) {
      s.touched.push({ path: d.path, ts: evt.ts, tool: d.name });
      if (d.name === 'Edit' || d.name === 'Write' || d.name === 'NotebookEdit') s.editCount += 1;
    }
  },

  'session.tool_result'(d) {
    const s = this.session(d.sessionId);
    if (d.ok === false) s.lastError = (d.preview ?? 'tool failed').slice(0, 200);
  },

  'session.usage'(d) {
    const s = this.session(d.sessionId);
    const cumulative = (value, previous) => Number.isFinite(value) && value >= 0 ? Math.max(previous, value) : previous;
    const previous = s.tokens;
    s.tokens = {
      input: cumulative(d.inputTokens, previous.input),
      output: cumulative(d.outputTokens, previous.output),
      cacheRead: cumulative(d.cacheRead, previous.cacheRead),
    };
    const used = Number.isFinite(d.contextTokens) && d.contextTokens >= 0
      ? d.contextTokens : (s.tokens.input + s.tokens.cacheRead + s.tokens.output);
    s.contextPct = Math.max(0, Math.min(100, Math.round((used / MAX_CONTEXT_TOKENS) * 100)));
    const cost = cumulative(d.costUsd, s.costUsd);
    this.totals.costUsd += cost - s.costUsd;
    s.costUsd = cost;
    this.totals.inputTokens += s.tokens.input - previous.input;
    this.totals.outputTokens += s.tokens.output - previous.output;
  },

  'session.ended'(d, evt) {
    const s = this.session(d.sessionId);
    s.state = d.reason === 'error' ? 'error' : d.reason === 'cancelled' ? 'cancelled' : 'done';
    s.endedAt = evt.ts;
    s.error = d.error ?? null;
    s.result = d.result ?? null;
  },

  'session.delegated'(d, evt) {
    // A subagent created by the agent's own Task tool runs inside the harness process.
    // Field records that the delegation happened and what it was for, but does not
    // fabricate a session it cannot observe — that would leave a unit on the Field
    // stuck in `spawning` forever. Sessions Field spawns itself arrive as
    // `session.spawned` with a parentSessionId and are real units.
    const p = this.session(d.parentSessionId);
    p.delegations = p.delegations ?? [];
    if (!p.delegations.some((x) => x.id === d.childSessionId)) {
      p.delegations.push({
        id: d.childSessionId,
        description: d.description ?? null,
        type: d.subagentType ?? null,
        ts: evt.ts,
      });
    }
  },

  'assignment.created'(d, evt) {
    this.assignments.set(d.assignmentId, {
      id: d.assignmentId, sessionIds: d.sessionIds, targetType: d.targetType,
      targetId: d.targetId, targetLabel: d.targetLabel,
      workspaceId: d.workspaceId ?? null, orders: d.orders,
      status: 'active', createdAt: evt.ts,
    });
    for (const id of d.sessionIds) {
      const s = this.session(id);
      s.assignmentId = d.assignmentId;
      s.target = {
        type: d.targetType, id: d.targetId,
        label: d.targetLabel, workspaceId: d.workspaceId,
      };
      if (d.workspaceId) s.workspaceId = d.workspaceId;
    }
  },

  'assignment.completed'(d, evt) {
    const a = this.assignments.get(d.assignmentId);
    if (a) { a.status = 'completed'; a.reason = d.reason ?? null; a.endedAt = evt.ts; }
  },

  'assignment.cancelled'(d, evt) {
    const a = this.assignments.get(d.assignmentId);
    if (a) { a.status = 'cancelled'; a.reason = d.reason ?? null; a.endedAt = evt.ts; }
    for (const id of (a ? a.sessionIds : [])) {
      const s = this.sessions.get(id);
      if (s) s.target = null;
    }
  },

  'assignment.failed'(d, evt) {
    const a = this.assignments.get(d.assignmentId);
    if (a) { a.status = 'failed'; a.reason = d.reason ?? null; a.failedSessionId = d.sessionId ?? null; a.endedAt = evt.ts; }
  },

  'assignment.interrupted'(d, evt) {
    const a = this.assignments.get(d.assignmentId);
    if (a) { a.status = 'interrupted'; a.reason = d.reason ?? null; a.endedAt = evt.ts; }
  },

  'fs.changed'(d, evt) {
    const key = d.workspaceId + ':' + d.path;
    this.files.set(key, {
      key, workspaceId: d.workspaceId, path: d.path, dir: d.dir,
      change: d.change, lastTs: evt.ts, bySession: d.sessionId ?? null,
    });
    this.touchFolder(d.workspaceId, d.dir, evt.ts, d.sessionId);
    const w = this.workspaces.get(d.workspaceId);
    if (w) { w.changeCount += 1; recordWorkspaceActivity(w, evt); }
  },

  'git.status'(d) {
    const w = this.workspaces.get(d.workspaceId);
    if (w) w.git = { branch: d.branch, ahead: d.ahead, behind: d.behind, files: d.files };
  },

  'browser.navigated'(d, evt) {
    const s = this.session(d.sessionId);
    s.browser = { url: d.url, domain: d.domain, ts: evt.ts };
    if (!this.websites.has(d.domain)) {
      this.websites.set(d.domain, {
        domain: d.domain, label: d.domain, sessions: new Set(),
        lastUrl: null, lastTs: 0, hits: 0, discovered: true,
      });
    }
    const site = this.websites.get(d.domain);
    site.sessions.add(d.sessionId);
    site.lastUrl = d.url;
    site.lastTs = evt.ts;
    site.hits += 1;
  },

  'browser.closed'(d) {
    const s = this.sessions.get(d.sessionId);
    if (s) s.browser = null;
    for (const site of this.websites.values()) site.sessions.delete(d.sessionId);
  },

  'endpoint.health'(d, evt) {
    const e = this.endpoints.get(d.endpointId);
    if (!e) return;
    e.status = d.status;
    e.latencyMs = d.latencyMs ?? null;
    e.detail = d.detail ?? null;
    e.lastCheck = evt.ts;
    if (d.status === 'down') e.failures += 1; else e.failures = 0;
  },

  'endpoint.routed'(d) {
    const s = this.session(d.sessionId);
    s.endpointId = d.endpointId;
    s.model = d.model ?? s.model;
    s.rerouted = d.reason ?? null;
  },

  'permission.requested'(d, evt) {
    this.permissions.set(d.permissionId, {
      id: d.permissionId, sessionId: d.sessionId, toolName: d.toolName,
      input: d.input, status: 'pending', requestedAt: evt.ts, decidedAt: null,
    });
    const s = this.session(d.sessionId);
    s.state = 'waiting_permission';
    s.pendingPermission = d.permissionId;
  },

  'permission.decided'(d, evt) {
    const p = this.permissions.get(d.permissionId);
    if (p) { p.status = d.decision; p.decidedAt = evt.ts; }
    const s = this.sessions.get((p ? p.sessionId : null) ?? d.sessionId);
    if (s) {
      s.pendingPermission = null;
      if (s.state === 'waiting_permission') s.state = 'working';
    }
  },

  'work.verified'(d) {
    const s = this.sessions.get(d.sessionId);
    if (s) s.verified = d.result === 'verified' ? 'verified' : 'rejected';
    const a = this.assignments.get(d.assignmentId);
    if (a) a.verified = d.result;
  },

  'ui.position'(d) {
    this.positions.set(d.entityType + ':' + d.entityId, { x: d.x, y: d.y });
  },

  'world.capital_selected'(d, e) {
    this.world.capitalWorkspaceId = d.workspaceId;
    this.world.capitalSelectedAt = e.ts;
    this.world.revision += 1;
  },

  'world.territory_assigned'(d, e) {
    this.world.assignments[d.clusterKey] = {
      clusterKey: d.clusterKey,
      territoryId: d.territoryId,
      label: d.label || d.clusterKey,
      kind: d.kind || 'infrastructure',
      workspaceId: d.workspaceId || null,
      assignedAt: e.ts,
    };
    this.world.revision += 1;
  },

  'world.territory_released'(d) {
    delete this.world.assignments[d.clusterKey];
    this.world.revision += 1;
  },

  'ui.control_group'(d) {
    this.controlGroups[String(d.group)] = d.sessionIds;
  },

  'routine.schedule_claimed'(d) {
    const routine = this.routines.get(d.routineId);
    if (routine && typeof d.slot === 'string' && d.slot > (routine.lastScheduleSlot ?? '')) routine.lastScheduleSlot = d.slot;
  },

  'routine.triggered'(d, evt) {
    for (const id of d.sessionIds ?? []) this.session(id).routineId = d.routineId;
    const routine = this.routines.get(d.routineId);
    if (routine) {
      routine.lastRunAt = d.startedAt ?? evt.ts;
      routine.activeRunId = d.runId ?? null;
      if (d.runId && !(routine.activeRunIds ?? []).includes(d.runId)) {
        routine.activeRunIds = [...(routine.activeRunIds ?? []), d.runId];
      }
      routine.currentOwner = d.sessionIds?.[0] ?? null;
      routine.queuedRuns = routine.queuedRuns.filter((item) => item.runId !== d.runId);
      routine.history.push({ runId: d.runId ?? null, status: 'running', at: routine.lastRunAt, reason: d.reason ?? null });
    }
  },

  'routine.enabled'(d) {
    const routine = this.routines.get(d.routineId);
    if (routine) routine.enabled = !!d.enabled;
  },

  'routine.queued'(d, evt) {
    const routine = this.routines.get(d.routineId);
    if (routine && d.runId && !routine.queuedRuns.some((item) => item.runId === d.runId)) {
      routine.queuedRuns.push({ runId: d.runId, at: d.queuedAt ?? evt.ts, reason: d.reason ?? null, paths: d.paths ?? [], authorityHash: d.authorityHash ?? null, claimed: false });
      routine.queuedRuns = routine.queuedRuns.slice(-1);
    }
  },

  'routine.queue_claimed'(d) {
    const routine = this.routines.get(d.routineId);
    const item = routine?.queuedRuns.find((queued) => queued.runId === d.runId);
    if (item) item.claimed = true;
  },

  'routine.cooldown_started'(d) {
    const routine = this.routines.get(d.routineId);
    if (routine) routine.cooldownUntil = Number(d.until) || 0;
  },

  'routine.completed'(d, evt) {
    settleRoutine(this.routines.get(d.routineId), d, evt, 'completed');
  },

  'routine.failed'(d, evt) {
    settleRoutine(this.routines.get(d.routineId), d, evt, 'failed');
  },

  'routine.skipped'(d, evt) {
    const routine = this.routines.get(d.routineId);
    if (routine) {
      if (d.runId) routine.queuedRuns = routine.queuedRuns.filter((item) => item.runId !== d.runId);
      routine.history.push({ runId: d.runId ?? null, status: 'skipped', at: evt.ts, reason: d.reason ?? null });
    }
  },
};

function settleRoutine(routine, data, event, status) {
  if (!routine) return;
  routine.lastOutcome = status;
  routine.activeRunIds = (routine.activeRunIds ?? []).filter((id) => id !== data.runId);
  if (!routine.activeRunId || routine.activeRunId === data.runId) {
    routine.activeRunId = routine.activeRunIds[routine.activeRunIds.length - 1] ?? null;
    if (!routine.activeRunId) routine.currentOwner = null;
  }
  if (Number.isFinite(data.cooldownUntil)) routine.cooldownUntil = data.cooldownUntil;
  routine.queuedRuns = routine.queuedRuns.filter(item => item.runId !== data.runId);
  const row = routine.history.find((item) => data.runId && item.runId === data.runId);
  if (row) Object.assign(row, { status, endedAt: event.ts, reason: data.reason ?? null });
  else routine.history.push({ runId: data.runId ?? null, status, at: event.ts, endedAt: event.ts, reason: data.reason ?? null });
}

function recordWorkspaceActivity(workspace, event) {
  if (event.source != null && event.source !== 'observed') return;
  workspace.lastTs = event.ts;
  workspace.activityEvents.push({ seq: event.seq, ts: event.ts, kind: event.kind, subject: event.subject ?? null });
  if (workspace.activityEvents.length > 200) workspace.activityEvents.splice(0, workspace.activityEvents.length - 200);
}

function workspaceMaturity(workspace, campaigns, now) {
  const activityEvidence = workspace.activityEvents.filter((event) => now - event.ts <= ACTIVITY_WINDOW_MS);
  const relevant = campaigns.filter((campaign) => campaignTargetsWorkspace(campaign, workspace.id));
  const criteria = relevant.flatMap((campaign) => campaign.objectives
    .filter((objective) => objective.required !== false)
    .flatMap((objective) => objective.definitionOfDone.map((criterion) => ({ campaign, objective, criterion }))));
  const completed = criteria.filter(({ objective, criterion }) => objective.status === 'satisfied'
    && objective.criteriaEvidence?.some((item) => item.criterion === criterion && item.evidence));
  const verified = completed.filter(({ campaign }) => ['verified', 'promoted'].includes(campaign.phase)
    && campaign.verdicts.at(-1)?.verdict === 'verified');
  const persisted = verified.filter(({ campaign }) => campaign.checkpoints
    .some((checkpoint) => isContentRevision(checkpoint.revision)));
  const terminal = relevant.filter((campaign) => ['verified', 'promoted', 'failed', 'cancelled', 'rolled_back'].includes(campaign.phase));
  const reliable = terminal.filter((campaign) => ['verified', 'promoted'].includes(campaign.phase)).length;
  const ratio = (count) => criteria.length ? Math.round((count / criteria.length) * 100) : 0;
  const complete = ratio(completed.length);
  const verification = ratio(verified.length);
  const persistence = ratio(persisted.length);
  const score = Math.round(complete * 0.3 + verification * 0.4 + persistence * 0.3);
  return {
    score,
    tier: score >= 85 ? 4 : score >= 62 ? 3 : score >= 34 ? 2 : score > 0 ? 1 : 0,
    activity: Math.min(100, Math.round((activityEvidence.length / ACTIVITY_EVENTS_FOR_FULL_SIGNAL) * 100)),
    complete,
    verified: verification,
    persisted: persistence,
    reliability: terminal.length ? Math.round((reliable / terminal.length) * 100) : 0,
    evidence: [
      ...activityEvidence.slice(-5).map((event) => ({ type: 'activity', ...event })),
      ...completed.map(({ campaign, objective, criterion }) => ({
        type: 'completion', campaignId: campaign.id, objectiveId: objective.id, criterion,
      })),
      ...verified.map(({ campaign }) => ({
        type: 'verification', campaignId: campaign.id, verdictId: campaign.verdicts.at(-1).id,
      })),
      ...persisted.map(({ campaign }) => ({
        type: 'persistence', campaignId: campaign.id,
        checkpointId: campaign.checkpoints.find((checkpoint) => isContentRevision(checkpoint.revision))?.id,
      })),
    ].slice(-40),
  };
}

function campaignTargetsWorkspace(campaign, workspaceId) {
  const targets = [campaign.target, ...campaign.objectives.map((objective) => objective.target)].filter(Boolean);
  return targets.some((target) => target.workspaceId === workspaceId
    || (target.type === 'workspace' && target.id === workspaceId));
}

function isContentRevision(value) {
  return typeof value === 'string' && /^(?:sha256:)?[a-f0-9]{7,64}$/i.test(value.trim());
}
