// The registry owns every live harness session and turns operator commands into
// real harness actions. It is the only place allowed to spawn or kill a session.
import { randomUUID } from 'node:crypto';
import path from 'node:path';
import fs from 'node:fs';
import net from 'node:net';
import { fileURLToPath } from 'node:url';
import { HarnessSession } from './session.js';
import { KnossosSession } from './knossos-session.js';
import { composePrompt } from '../config.js';
import { resolveWorkspacePath } from '../workspace-path.js';
import { buildMatcher } from '../glob.js';
import { BudgetLedger } from '../budget-ledger.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PERMISSION_MCP = path.join(HERE, 'permission-mcp.mjs');

const EFFORT_ORDER = ['low', 'medium', 'high', 'xhigh', 'max'];

export class Registry {
  constructor({ cfg, emit, apiBase, permissionCapabilities, registerSecret, campaignPolicy, budgetLedger }) {
    this.cfg = cfg;
    this.emit = emit;
    this.apiBase = apiBase;
    this.permissionCapabilities = permissionCapabilities;
    this.registerSecret = registerSecret;
    this.campaignPolicy = campaignPolicy ?? (() => null);
    this.budgets = budgetLedger ?? new BudgetLedger();
    this.sessions = new Map();       // sessionId -> HarnessSession
    this.meta = new Map();           // sessionId -> { assignmentId, verifyFor, budgetUsd, costUsd }
    this.assignments = new Map();    // assignmentId -> { members, outcomes, settled }
    this.permissions = new Map();    // permissionId -> { resolve, sessionId, timer }
    this.endpointStatus = new Map(); // endpointId -> status
    this.campaignReportHandler = null;
    this.exhaustedCampaigns = new Set();
    this.sessionDeadlineTimers = new Map();
  }

  // ---------------------------------------------------------------- routing

  /** Pick a real endpoint. `auto` prefers the role default, then any healthy peer. */
  route(endpointId, roleId) {
    const eps = this.cfg.endpoints;
    const healthy = (id) => {
      const st = this.endpointStatus.get(id);
      return st === 'up' || st === undefined || st === 'unknown';
    };

    if (endpointId && endpointId !== 'auto') {
      if (healthy(endpointId)) return { endpointId, reason: 'operator choice' };
      const alt = eps.find((e) => e.kind === this.endpoint(endpointId)?.kind && healthy(e.id));
      if (alt) return { endpointId: alt.id, reason: `${endpointId} is down` };
      return { endpointId: null, requestedEndpointId: endpointId, reason: 'no healthy alternative' };
    }

    const role = this.cfg.roles.find((r) => r.id === roleId);
    const preferred = role?.default_endpoint;
    if (preferred && healthy(preferred)) return { endpointId: preferred, reason: 'role default' };

    const anyUp = eps.find((e) => healthy(e.id));
    if (anyUp) {
      return {
        endpointId: anyUp.id,
        reason: preferred ? `${preferred} unavailable` : 'first healthy endpoint',
      };
    }
    return { endpointId: null, requestedEndpointId: preferred ?? eps[0]?.id ?? null, reason: 'no endpoint reporting healthy' };
  }

  endpoint(id) { return this.cfg.endpoints.find((e) => e.id === id); }
  workspace(id) { return this.cfg.workspaces.find((w) => w.id === id); }

  /**
   * `adaptive` is resolved here from the real size of the target, not from a guess.
   * The chosen level is recorded on the spawn event.
   */
  resolveEffort(thinking, { roleId, workspaceId, targetPath }) {
    if (thinking && thinking !== 'adaptive') return { effort: thinking, reason: 'operator choice' };

    const role = this.cfg.roles.find((r) => r.id === roleId);
    const floor = role?.default_thinking ?? 'medium';

    let measured = 'medium';
    let reason = 'no measurable target';
    try {
      const ws = this.workspace(workspaceId);
      if (ws && targetPath != null) {
        const { abs } = resolveWorkspacePath(this.cfg, ws.id, targetPath, { type: 'any' });
        const st = fs.existsSync(abs) ? fs.statSync(abs) : null;
        if (st?.isFile()) {
          const lines = fs.readFileSync(abs, 'utf8').split('\n').length;
          measured = lines < 200 ? 'low' : lines < 800 ? 'medium' : 'high';
          reason = `target file is ${lines} lines`;
        } else if (st?.isDirectory()) {
          const n = countFiles(abs, 400);
          measured = n <= 12 ? 'low' : n <= 80 ? 'medium' : 'high';
          reason = `target holds ${n >= 400 ? '400+' : n} files`;
        }
      }
    } catch { /* fall back to the role floor */ }

    const effort = EFFORT_ORDER[Math.max(
      EFFORT_ORDER.indexOf(measured),
      EFFORT_ORDER.indexOf(floor),
    )] ?? 'medium';
    return { effort, reason: `adaptive: ${reason}` };
  }

  // ---------------------------------------------------------------- spawning

  spawn({
    agentId, workspaceId, missionId, orders, endpointId, thinking, target, assignmentId,
    verifyFor, campaignId, team, objectiveId, doctrine, environmentScope, budgetUsd,
  }) {
    // Follow-up work belongs to its original campaign even when the client omits
    // or supplies a different campaign ID. Resolve this before any admission.
    let inherited;
    if (verifyFor) {
      inherited = this.meta.get(verifyFor);
      if (!inherited) throw new Error('verification requires a known source session');
    } else if (assignmentId) {
      const assignment = this.assignments.get(assignmentId);
      if (!assignment || assignment.settled) throw new Error('reinforcement requires an active assignment');
      const members = [...assignment.members].map(id => this.meta.get(id));
      if (!members.length || members.some(meta => !meta)) throw new Error('assignment source policy is unavailable');
      if (new Set(members.map(meta => meta.campaignId ?? null)).size !== 1) {
        throw new Error('cannot reinforce an assignment spanning different campaigns');
      }
      inherited = members[0];
    }
    if (inherited) {
      if (campaignId != null && campaignId !== inherited.campaignId) throw new Error('follow-up campaign cannot differ from its source');
      campaignId = inherited.campaignId;
      missionId = inherited.missionId;
      team = inherited.team;
      objectiveId = inherited.objectiveId;
      environmentScope = inherited.environmentScope;
    }
    const agent = this.cfg.agents.find((a) => a.id === agentId);
    if (!agent) throw new Error(`unknown agent: ${agentId}`);

    const roleId = agent.role;
    const role = this.cfg.roles.find((r) => r.id === roleId);
    const ws = this.workspace(workspaceId ?? this.cfg.workspaces[0]?.id);
    if (!ws) throw new Error('no workspace mounted');
    if (!ws.mounted) throw new Error(`workspace ${ws.id} is not mounted at ${ws.path}`);

    const routed = this.route(endpointId ?? agent.endpoint, roleId);
    if (!routed.endpointId) {
      throw new Error(`no healthy inference endpoint for ${agentId}: ${routed.reason}`);
    }
    const ep = this.endpoint(routed.endpointId);
    this.assertCapacity({ endpointId: routed.endpointId, campaignId });
    const requestedBudget = budgetUsd ?? agent.budget_usd ?? this.cfg.defaults.budget_usd_per_session ?? 5;
    if (!Number.isFinite(requestedBudget) || requestedBudget <= 0) throw new Error('session dollar budget must be positive and finite');
    const campaign = campaignId ? this.campaignPolicy(campaignId) : null;
    if (campaignId && !campaign) throw new Error('campaign policy is unavailable');
    if (campaign && (!Number.isFinite(campaign.budgetUsd) || campaign.budgetUsd <= 0
      || !Number.isFinite(campaign.costUsd ?? 0) || (campaign.costUsd ?? 0) < 0)) {
      throw new Error('campaign budget policy is invalid');
    }
    const remaining = campaign && campaign.budgetUsd > 0
      ? this.budgets.campaign(campaignId, campaign.costUsd ?? 0, campaign.budgetUsd).remainingUsd : requestedBudget;
    const admittedBudget = Math.min(requestedBudget, remaining);
    if (admittedBudget <= 0) throw new Error('campaign budget is spent or reserved by other sessions');
    const eff = this.resolveEffort(thinking ?? agent.thinking, {
      roleId, workspaceId: ws.id, targetPath: target?.path ?? target?.id,
    });

    const id = randomUUID();
    const maxChildren = agent.max_children ?? this.cfg.defaults.max_children_per_session ?? 2;
    const effectiveTools = agent?.tools_allow ?? role?.tools_allow ?? [];

    let systemPrompt = composePrompt(this.cfg, { agentId, roleId, missionId, orders: null });
    systemPrompt += `\n\n---\n\n# Delegation limit\n\nYou may create at most ${maxChildren} subagents. ` +
      `Maximum delegation depth for this Field is ${this.cfg.defaults.max_delegation_depth ?? 2}.`;

    const SessionAdapter = ep?.kind === 'openai-compatible' ? KnossosSession : HarnessSession;
    const permissionToken = SessionAdapter === HarnessSession
      ? this.permissionCapabilities?.mint(id)
      : null;
    if (SessionAdapter === HarnessSession && !permissionToken) {
      throw new Error('session-scoped permission capability service is unavailable');
    }
    if (permissionToken) this.registerSecret?.(permissionToken);
    const mcpConfig = JSON.stringify({
      mcpServers: {
        field: {
          command: process.execPath,
          args: [PERMISSION_MCP],
          env: {
            FIELD_API: this.apiBase,
            FIELD_SESSION: id,
            FIELD_INTERNAL_TOKEN: permissionToken,
          },
        },
      },
    });

    const session = new SessionAdapter({
      id,
      agentId,
      name: agent.name ?? agentId,
      role: roleId,
      model: ep?.model ?? null,
      endpointId: routed.endpointId,
      effort: eff.effort,
      cwd: ws.path,
      workspaceId: ws.id,
      systemPrompt,
      addDirs: [],
      allowedTools: effectiveTools,
      disallowedTools: denyListFor(role, agent),
      mcpConfig,
      permissionTool: 'mcp__field__approve',
      workspaces: this.cfg.workspaces,
      engine: ep?.kind === 'openai-compatible' ? 'cameo' : 'anthropic',
      providerKind: ep?.kind,
      credentialEnvKeys: Array.isArray(ep?.credential_env)
        ? ep.credential_env
        : ep?.credential_env ? [ep.credential_env] : [],
      env: ep?.kind === 'openai-compatible' && (ep.base_url || ep.baseUrl)
        ? { CAMEO_BASE_URL: ep.base_url ?? ep.baseUrl, CAMEO_MODEL: ep.model ?? '' }
        : {},
    });

    let firstStart = true;
    session.beforeStart = () => {
      this.assertBudgetAllowsWork(id);
      this.admitExisting(id);
      this.armSessionDeadline(id);
      if (SessionAdapter === HarnessSession && !firstStart) {
        const token = this.permissionCapabilities?.mint(id);
        if (!token) throw new Error('session-scoped permission capability service is unavailable');
        this.registerSecret?.(token);
        const config = JSON.parse(session.mcpConfig);
        config.mcpServers.field.env.FIELD_INTERNAL_TOKEN = token;
        session.mcpConfig = JSON.stringify(config);
      }
      firstStart = false;
    };
    session.on('event', (kind, data) => this.onSessionEvent(session, kind, data));
    this.sessions.set(id, session);
    const maxOutputTokens = boundedPositiveInt(
      agent.completion_tokens ?? this.cfg.defaults.completion_tokens_per_session ?? 32_768,
      32_768,
    );
    const runtimeMinutes = boundedPositiveNumber(
      agent.wall_minutes ?? this.cfg.defaults.wall_minutes_per_session ?? 120,
      120,
    );
    const startedAt = Date.now();
    const runtimeMs = runtimeMinutes * 60_000;
    this.meta.set(id, {
      assignmentId: assignmentId ?? null,
      verifyFor: verifyFor ?? null,
      budgetUsd: admittedBudget,
      costUsd: 0,
      missionId: missionId ?? null,
      campaignId: campaignId ?? null,
      team: team ?? null,
      objectiveId: objectiveId ?? null,
      environmentScope: environmentScope ?? null,
      toolsAllow: [...effectiveTools],
      writeScope: [...(role?.write_scope ?? [])],
      outputTokens: 0,
      maxOutputTokens,
      startedAt,
      runtimeMs,
      deadlineAt: startedAt + runtimeMs,
      budgetStopped: false,
    });
    this.armSessionDeadline(id);

    const reservation = this.emit('budget.reserved', { sessionId: id, campaignId: campaignId ?? null, limitUsd: admittedBudget }, { subject: id, source: 'derived' });
    this.budgets.apply(reservation ?? { kind: 'budget.reserved', data: { sessionId: id, campaignId, limitUsd: admittedBudget } });

    this.emit('session.spawned', {
      sessionId: id,
      agentId,
      name: agent.name ?? agentId,
      role: roleId,
      model: ep?.model ?? null,
      endpointId: routed.endpointId,
      routeReason: routed.reason,
      thinking: eff.effort,
      thinkingReason: eff.reason,
      cwd: ws.path,
      workspaceId: ws.id,
      missionId: missionId ?? null,
      assignmentId: assignmentId ?? null,
      target: target ?? null,
      campaignId: campaignId ?? null,
      team: team ?? null,
      objectiveId: objectiveId ?? null,
      doctrine: doctrine ?? null,
      environmentScope: environmentScope ?? null,
      systemPrompt,
      initialOrders: orders ?? 'Await orders. Report ready and take no action yet.',
    }, { subject: id });

    if (assignmentId) this.assignments.get(assignmentId).members.add(id);
    session.start(orders ?? 'Await orders. Report ready and take no action yet.');
    return session;
  }

  onSessionEvent(session, kind, data) {
    const meta = this.meta.get(session.id) ?? {};
    if (kind === 'harness.permission_requested') {
      this.requestPermission({
        sessionId: session.id,
        toolName: data.toolName,
        input: data.input,
        toolUseId: data.requestId,
      }).then(({ decision }) => session.decidePermission?.(data.requestId, decision));
      return;
    }
    const contextual = {
      ...data,
      campaignId: data.campaignId ?? meta.campaignId ?? null,
      team: data.team ?? meta.team ?? null,
      objectiveId: data.objectiveId ?? meta.objectiveId ?? null,
    };

    if (kind === 'session.usage' && Number.isFinite(contextual.costUsd) && contextual.costUsd >= 0) {
      meta.costUsd = Math.max(meta.costUsd ?? 0, contextual.costUsd);
      if (meta.budgetUsd && meta.costUsd >= meta.budgetUsd) {
        this.stopSessionForBudget(session.id, 'dollar_cost', meta.costUsd, meta.budgetUsd);
      }
    }
    if (kind === 'session.usage' && Number.isFinite(contextual.outputTokens) && contextual.outputTokens >= 0) {
      meta.outputTokens = Math.max(meta.outputTokens ?? 0, contextual.outputTokens);
      if (meta.maxOutputTokens && meta.outputTokens >= meta.maxOutputTokens) {
        this.stopSessionForBudget(session.id, 'completion_tokens', meta.outputTokens, meta.maxOutputTokens);
      }
    }

    if (kind === 'session.turn_complete' && meta.verifyFor) {
      const text = String(contextual.result ?? '');
      const m = /VERDICT:\s*(verified|rejected)/i.exec(text);
      if (m) {
        this.emit('work.verified', {
          sessionId: meta.verifyFor,
          verifierSessionId: session.id,
          assignmentId: this.meta.get(meta.verifyFor)?.assignmentId ?? null,
          result: m[1].toLowerCase(),
          evidence: text.slice(0, 2000),
        }, { subject: meta.verifyFor });
      }
    }

    const stored = this.emit(kind, contextual, { subject: session.id, actor: session.agentId });
    this.budgets.apply(stored ?? { kind, data: contextual });
    if (kind === 'session.usage' && meta.campaignId) this.enforceCampaignBudget(meta.campaignId);
    if (kind === 'session.ended') {
      this.permissionCapabilities?.revoke(session.id);
      this.clearSessionDeadline(session.id);
      this.settleAssignmentMember(session.id, contextual.reason ?? 'exit');
    }
    if (kind === 'session.turn_complete') this.settleAssignmentMember(session.id, 'completed');
    if (kind === 'session.turn_complete' && meta.campaignId && this.campaignReportHandler) {
      const report = parseCampaignReport(contextual.result);
      if (report) {
        try {
          this.campaignReportHandler({
            sessionId: session.id,
            agentId: session.agentId,
            role: session.role,
            eventSeq: stored?.seq,
            campaignId: meta.campaignId,
            team: meta.team,
            objectiveId: meta.objectiveId,
            report,
          });
        } catch (error) {
          this.emit('campaign.report_rejected', {
            campaignId: meta.campaignId, sessionId: session.id,
            team: meta.team, objectiveId: meta.objectiveId,
            reason: error.message, reportKind: report.kind ?? null,
          }, { subject: meta.campaignId, actor: session.agentId });
        }
      }
    }
  }

  // ---------------------------------------------------------------- orders

  /** Compose real orders from a real target. */
  ordersFor(target, extra) {
    const ws = this.workspace(target.workspaceId);
    const lines = [];
    switch (target.type) {
      case 'workspace':
        lines.push(`Your workspace is ${ws?.name} at ${ws?.path}. Survey it before acting.`);
        break;
      case 'folder':
        lines.push(`Work inside \`${target.id || '.'}\` in the ${ws?.name} workspace. Stay in that subtree unless the orders say otherwise.`);
        break;
      case 'file': {
        const isMd = String(target.id).toLowerCase().endsWith('.md');
        lines.push(isMd
          ? `Read \`${target.id}\` in the ${ws?.name} workspace. It is an instruction document: execute the plan it contains, in order.`
          : `Your target is \`${target.id}\` in the ${ws?.name} workspace.`);
        break;
      }
      case 'mission': {
        const m = this.cfg.missions.find((x) => x.id === target.id);
        if (!m) { lines.push(`Mission ${target.id} is not defined.`); break; }
        lines.push(`Execute mission "${m.name ?? m.id}".`);
        if (m.target) lines.push(`Its target is \`${m.target}\` in the ${ws?.name} workspace.`);
        // A session that is already running was not spawned with this mission in its
        // system prompt, so the mission text travels with the orders.
        lines.push('', m.body.trim());
        if (m.definition_of_done?.length) {
          lines.push('', 'Definition of done:');
          for (const d of m.definition_of_done) lines.push(`- ${d}`);
        }
        break;
      }
      case 'website':
        lines.push(`Open ${target.url ?? `https://${target.id}`} and study it. Record the exact URL behind every claim you make.`);
        break;
      default:
        lines.push(`Target: ${target.label ?? target.id}`);
    }
    if (extra?.trim()) lines.push('', extra.trim());
    return lines.join('\n');
  }

  /** Assign one or more live sessions to a real target. */
  assign({ sessionIds, target, orders, endpointId, thinking }) {
    const assignmentId = randomUUID();
    const composed = this.ordersFor(target, orders);
    const skipped = [];
    const admitted = [];
    for (const id of sessionIds) {
      const s = this.sessions.get(id);
      const meta = this.meta.get(id);
      if (!s || !meta) {
        skipped.push({ sessionId: id, reason: 'session is not live' });
        continue;
      }
      try { this.assertBudgetAllowsWork(id); }
      catch (error) { skipped.push({ sessionId: id, reason: error.message }); continue; }

      // A running harness process is rooted at the cwd it was spawned with. It cannot
      // reach another workspace, so say so instead of issuing unfollowable orders.
      if (target.workspaceId && s.workspaceId && s.workspaceId !== target.workspaceId) {
        skipped.push({ sessionId: id, reason: `rooted in ${s.workspaceId}` });
        this.emit('session.state', {
          sessionId: id,
          state: 'blocked',
          detail: `cannot be assigned to ${target.workspaceId}: this session is rooted in ${s.workspaceId}. Spawn a new agent there instead.`,
        }, { subject: id });
        continue;
      }
      admitted.push(id);
    }

    this.emit('assignment.created', {
      assignmentId,
      sessionIds: admitted,
      targetType: target.type,
      targetId: target.id,
      targetLabel: target.label ?? target.id,
      workspaceId: target.workspaceId ?? null,
      orders: composed,
    }, { subject: assignmentId });
    this.assignments.set(assignmentId, {
      members: new Set(admitted), outcomes: new Map(), settled: false,
    });
    if (!admitted.length) {
      this.emit('assignment.cancelled', {
        assignmentId, reason: 'no eligible live sessions',
      }, { subject: assignmentId });
      this.assignments.get(assignmentId).settled = true;
      return { assignmentId, skipped };
    }

    for (const id of admitted) {
      const s = this.sessions.get(id);
      const meta = this.meta.get(id);

      if (meta.assignmentId && meta.assignmentId !== assignmentId) {
        this.settleAssignmentMember(id, 'reassigned');
      }
      meta.assignmentId = assignmentId;

      if (endpointId && endpointId !== 'auto') {
        const routed = this.route(endpointId, s.role);
        const ep = this.endpoint(routed.endpointId);
        if (ep && routed.endpointId !== s.endpointId) {
          s.endpointId = routed.endpointId;
          s.model = ep.model;
          this.emit('endpoint.routed', {
            sessionId: id, endpointId: routed.endpointId,
            model: ep.model, reason: routed.reason,
          }, { subject: id });
        }
      }
      if (thinking) {
        const eff = this.resolveEffort(thinking, {
          roleId: s.role, workspaceId: target.workspaceId, targetPath: target.id,
        });
        s.effort = eff.effort;
      }

      try {
        this.admitExisting(id);
        if (!s.proc) s.resume(composed);
        else s.send(composed);
      } catch (error) {
        skipped.push({ sessionId: id, reason: error.message });
        this.settleAssignmentMember(id, 'error');
      }
    }
    return { assignmentId, skipped };
  }

  settleAssignmentMember(sessionId, reason) {
    const assignmentId = this.meta.get(sessionId)?.assignmentId;
    const run = assignmentId ? this.assignments.get(assignmentId) : null;
    if (!run || run.settled || !run.members.has(sessionId)) return false;
    run.outcomes.set(sessionId, reason);
    const failed = [...run.outcomes.entries()].find(([, outcome]) => outcome === 'error');
    const cancelled = [...run.outcomes.entries()].find(([, outcome]) => ['cancelled', 'reassigned'].includes(outcome));
    if (failed || cancelled) {
      run.settled = true;
      const kind = failed ? 'assignment.failed' : 'assignment.cancelled';
      const [failedSessionId, outcome] = failed ?? cancelled;
      this.emit(kind, { assignmentId, sessionId: failedSessionId, reason: outcome }, { subject: assignmentId });
      return true;
    }
    if (run.outcomes.size === run.members.size) {
      run.settled = true;
      this.emit('assignment.completed', {
        assignmentId, sessionIds: [...run.members], reason: 'all assigned sessions exited successfully',
      }, { subject: assignmentId });
      return true;
    }
    return false;
  }

  enforceCampaignBudget(campaignId) {
    const campaign = this.campaignPolicy(campaignId);
    if (!campaign || !campaign.budgetExhausted || this.exhaustedCampaigns.has(campaignId)) return false;
    this.exhaustedCampaigns.add(campaignId);
    const sessionIds = [...this.meta.entries()]
      .filter(([, meta]) => meta.campaignId === campaignId)
      .map(([sessionId]) => sessionId);
    this.emit('campaign.budget_exhausted', {
      campaignId, costUsd: campaign.costUsd, budgetUsd: campaign.budgetUsd, sessionIds,
    }, { subject: campaignId, source: 'derived' });
    for (const sessionId of sessionIds) this.sessions.get(sessionId)?.pause();
    return true;
  }

  stopSessionForBudget(sessionId, budget, used, limit) {
    const meta = this.meta.get(sessionId);
    if (!meta || meta.budgetStopped) return false;
    meta.budgetStopped = true;
    this.emit('budget.exhausted', {
      sessionId, campaignId: meta.campaignId ?? null, budget, used, limit,
    }, { subject: sessionId, source: 'derived' });
    this.emit('session.state', {
      sessionId, state: 'blocked', campaignId: meta.campaignId ?? null,
      team: meta.team ?? null, objectiveId: meta.objectiveId ?? null,
      detail: `${budget.replace('_', ' ')} budget exhausted (${used} of ${limit})`,
    }, { subject: sessionId, source: 'derived' });
    this.sessions.get(sessionId)?.pause();
    return true;
  }

  armSessionDeadline(sessionId) {
    this.clearSessionDeadline(sessionId);
    const meta = this.meta.get(sessionId);
    if (!meta?.deadlineAt) return;
    const timer = setTimeout(() => {
      this.sessionDeadlineTimers.delete(sessionId);
      this.stopSessionForBudget(sessionId, 'wall_time_ms', Date.now() - meta.startedAt, meta.runtimeMs);
    }, Math.max(1, meta.deadlineAt - Date.now()));
    timer.unref?.();
    this.sessionDeadlineTimers.set(sessionId, timer);
  }

  clearSessionDeadline(sessionId) {
    const timer = this.sessionDeadlineTimers.get(sessionId);
    if (timer) clearTimeout(timer);
    this.sessionDeadlineTimers.delete(sessionId);
  }

  // ---------------------------------------------------------------- commands

  // Count owned processes, including idle harnesses and children still draining.
  // Admission and process creation are synchronous in this registry's event loop.
  assertCapacity({ sessionId, endpointId, campaignId }) {
    const limit = (value, fallback) => {
      if (value == null) return fallback;
      if (!Number.isSafeInteger(value) || value < 1) throw new Error('session concurrency limits must be positive integers');
      return value;
    };
    const globalLimit = limit(this.cfg.defaults.max_concurrent_sessions, 16);
    const endpointLimit = limit(this.endpoint(endpointId)?.max_concurrent_sessions, 4);
    const campaign = campaignId ? this.campaignPolicy(campaignId) : null;
    const campaignLimit = limit(campaign?.concurrency, globalLimit);
    const active = [...this.sessions.values()].filter(s => s.id !== sessionId && (s.proc || s.ownedProcesses?.size));
    const reason = active.length >= globalLimit ? 'global session concurrency limit'
      : active.filter(s => s.endpointId === endpointId).length >= endpointLimit ? 'endpoint session concurrency limit'
      : campaignId && active.filter(s => this.meta.get(s.id)?.campaignId === campaignId).length >= campaignLimit ? 'campaign session concurrency limit'
      : null;
    if (reason) {
      this.emit('capacity.denied', { sessionId, endpointId, campaignId, reason }, { subject: sessionId ?? campaignId ?? endpointId, source: 'derived' });
      throw new Error(reason);
    }
  }

  admitExisting(sessionId, endpointId) {
    const session = this.sessions.get(sessionId);
    if (!session) return;
    this.assertCapacity({ sessionId, endpointId: endpointId ?? session.endpointId, campaignId: this.meta.get(sessionId)?.campaignId });
  }

  assertBudgetAllowsWork(sessionId) {
    const meta = this.meta.get(sessionId);
    if (!meta) return;
    if (meta.budgetStopped || (meta.budgetUsd > 0 && meta.costUsd >= meta.budgetUsd)
      || (meta.deadlineAt && Date.now() >= meta.deadlineAt)
      || (meta.maxOutputTokens > 0 && meta.outputTokens >= meta.maxOutputTokens)) {
      throw new Error(`session ${sessionId} has exhausted its budget; create a new explicitly budgeted session`);
    }
    const campaign = meta.campaignId ? this.campaignPolicy(meta.campaignId) : null;
    if (campaign?.budgetExhausted) throw new Error('campaign budget is exhausted');
    const reservation = this.budgets.reservations.get(sessionId);
    if (reservation?.terminal && reservation.settled && reservation.spentUsd !== null) {
      const needed = Math.max(0, reservation.limitUsd - reservation.spentUsd);
      if (campaign?.budgetUsd > 0 && this.budgets.campaign(meta.campaignId, campaign.costUsd ?? 0, campaign.budgetUsd).remainingUsd < needed) {
        throw new Error('campaign budget is reserved by other sessions; resume denied');
      }
      const event = this.emit('budget.reactivated', { sessionId }, { subject: sessionId, source: 'derived' });
      this.budgets.apply(event ?? { kind: 'budget.reactivated', data: { sessionId } });
    }
  }

  command(kind, payload = {}) {
    const ids = payload.sessionIds ?? [];
    if (['resume', 'escalate', 'say', 'redirect'].includes(kind)) {
      for (const id of ids) this.assertBudgetAllowsWork(id);
    }
    this.emit('command.issued', { kind, ...payload }, { subject: ids[0] ?? null });

    switch (kind) {
      case 'pause':
        for (const id of ids) this.sessions.get(id)?.pause();
        return { paused: ids.length };

      case 'resume':
        for (const id of ids) { this.admitExisting(id); this.sessions.get(id)?.resume(payload.orders); }
        return { resumed: ids.length };

      case 'cancel':
        for (const id of ids) this.sessions.get(id)?.cancel();
        return { cancelled: ids.length };

      case 'redirect':
        return this.assign({ sessionIds: ids, target: payload.target, orders: payload.orders });

      case 'reinforce': {
        // Add real, newly spawned agents to an existing assignment.
        const spawned = [];
        for (const agentId of payload.agentIds ?? []) {
          const s = this.spawn({
            agentId,
            workspaceId: payload.target?.workspaceId,
            orders: this.ordersFor(payload.target, payload.orders),
            assignmentId: payload.assignmentId,
            target: payload.target,
            thinking: payload.thinking,
            endpointId: payload.endpointId,
          });
          spawned.push(s.id);
        }
        return { spawned };
      }

      case 'verify': {
        const verifierAgent = payload.verifierAgentId
          ?? this.cfg.agents.find((a) => a.role === 'verifier')?.id;
        if (!verifierAgent) throw new Error('no verifier agent is defined in field/agents');
        const out = [];
        for (const id of ids) {
          const s = this.sessions.get(id);
          if (!s) continue;
          const orders = [
            `Verify the work done by session ${s.name} (${id}) in workspace ${s.workspaceId}.`,
            'Re-derive the result yourself. Run the project checks and paste the real output.',
            'Do not fix anything you find.',
            'End your final message with exactly one line: `VERDICT: verified` or `VERDICT: rejected`.',
          ].join('\n');
          const v = this.spawn({
            agentId: verifierAgent,
            workspaceId: s.workspaceId,
            orders,
            verifyFor: id,
            thinking: 'high',
            target: { type: 'workspace', id: s.workspaceId, workspaceId: s.workspaceId },
          });
          out.push(v.id);
        }
        return { verifiers: out };
      }

      case 'escalate': {
        // Move to the most capable healthy endpoint and raise effort one step.
        const best = this.cfg.endpoints.find((e) => this.endpointStatus.get(e.id) !== 'down');
        for (const id of ids) {
          const s = this.sessions.get(id);
          if (!s) continue;
          this.admitExisting(id, best?.id);
          const next = EFFORT_ORDER[Math.min(EFFORT_ORDER.indexOf(s.effort) + 1, EFFORT_ORDER.length - 1)];
          s.effort = next;
          if (best) {
            s.endpointId = best.id;
            s.model = best.model;
            this.emit('endpoint.routed', {
              sessionId: id, endpointId: best.id, model: best.model, reason: 'escalated by operator',
            }, { subject: id });
          }
          this.emit('session.state', { sessionId: id, state: 'thinking', detail: `escalated to ${next}` }, { subject: id });
          if (s.proc) s.pause();
          s.resume(payload.orders ?? 'Escalated. Re-approach with more care and report what changed in your assessment.');
        }
        return { escalated: ids.length };
      }

      case 'say':
        for (const id of ids) { this.admitExisting(id); this.sessions.get(id)?.send(payload.text ?? ''); }
        return { sent: ids.length };

      default:
        throw new Error(`unknown command: ${kind}`);
    }
  }

  setControlGroup(group, sessionIds) {
    this.emit('ui.control_group', { group, sessionIds });
  }

  /** Safe live metadata for the campaign director. Never exposes the child process. */
  info(sessionId) {
    const session = this.sessions.get(sessionId);
    const meta = this.meta.get(sessionId);
    if (!session || !meta) return null;
    return {
      id: session.id, agentId: session.agentId, role: session.role,
      state: session.state, workspaceId: session.workspaceId,
      endpointId: session.endpointId, model: session.model,
      campaignId: meta.campaignId, team: meta.team, objectiveId: meta.objectiveId,
    };
  }

  setCampaignReportHandler(handler) {
    this.campaignReportHandler = typeof handler === 'function' ? handler : null;
  }

  // ---------------------------------------------------------------- permissions

  requestPermission({ sessionId, toolName, input, toolUseId, capabilitySessionId }) {
    if (capabilitySessionId && capabilitySessionId !== sessionId) {
      throw new Error('permission capability does not belong to this session');
    }
    const permissionId = randomUUID();
    this.emit('permission.requested', {
      permissionId, sessionId, toolName, input, toolUseId,
    }, { subject: sessionId });

    const session = this.sessions.get(sessionId);
    const meta = this.meta.get(sessionId) ?? {};
    const role = this.cfg.roles.find((item) => item.id === session?.role);
    const validateWritePath = session?.workspaceId
      ? (relativePath) => resolveWorkspacePath(this.cfg, session.workspaceId, relativePath, {
        operation: 'write', type: 'file', allowSensitive: false,
      })
      : null;
    const policy = evaluatePermissionPolicy({
      toolName,
      input,
      workspacePath: session?.cwd,
      readOnly: role?.read_only === true,
      environmentScope: meta.environmentScope,
      allowedTools: meta.toolsAllow ?? role?.tools_allow,
      writeScope: meta.writeScope ?? role?.write_scope,
      validateWritePath,
      allowedDomains: role?.network_domains ?? this.cfg.websites.map((site) => site.domain),
    });
    if (policy) {
      this.emit('permission.decided', {
        permissionId, sessionId, decision: 'deny', by: 'scope-policy', reason: policy.message,
      }, { subject: sessionId });
      return Promise.resolve(policy);
    }

    return new Promise((resolve) => {
      const timer = setTimeout(() => {
        if (!this.permissions.has(permissionId)) return;
        this.permissions.delete(permissionId);
        this.emit('permission.decided', {
          permissionId, sessionId, decision: 'deny', by: 'timeout',
        }, { subject: sessionId });
        resolve({ decision: 'deny', message: 'No operator decision within 10 minutes.' });
      }, 10 * 60 * 1000);

      this.permissions.set(permissionId, { resolve, sessionId, timer });
    });
  }

  decidePermission(permissionId, decision, message) {
    const entry = this.permissions.get(permissionId);
    if (!entry) return false;
    clearTimeout(entry.timer);
    this.permissions.delete(permissionId);
    this.emit('permission.decided', {
      permissionId, sessionId: entry.sessionId, decision, by: 'operator',
    }, { subject: entry.sessionId });
    entry.resolve({ decision, message });
    return true;
  }

  // ---------------------------------------------------------------- endpoints

  onEndpointHealth(endpointId, status) {
    const prev = this.endpointStatus.get(endpointId);
    this.endpointStatus.set(endpointId, status);
    if (status !== 'down' || prev === 'down') return;

    // A real endpoint failure pauses the sessions it powers and reroutes them.
    for (const [id, s] of this.sessions) {
      if (s.endpointId !== endpointId || !s.proc) continue;
      try { this.assertBudgetAllowsWork(id); }
      catch { s.pause(); continue; }
      const routed = this.route('auto', s.role);
      if (routed.endpointId && routed.endpointId !== endpointId) {
        try { this.admitExisting(id, routed.endpointId); }
        catch { s.pause(); continue; }
        s.pause();
        s.endpointId = routed.endpointId;
        s.model = this.endpoint(routed.endpointId)?.model ?? s.model;
        this.emit('endpoint.routed', {
          sessionId: id, endpointId: routed.endpointId, model: s.model,
          reason: `${endpointId} went down`,
        }, { subject: id });
        s.resume('Your endpoint failed and you were rerouted. Re-state where you were and continue.');
      } else {
        s.pause();
        this.emit('session.state', {
          sessionId: id, state: 'blocked', detail: `endpoint ${endpointId} is down, no alternative`,
        }, { subject: id });
      }
    }
  }

  shutdown() {
    for (const timer of this.sessionDeadlineTimers.values()) clearTimeout(timer);
    this.sessionDeadlineTimers.clear();
    for (const s of this.sessions.values()) {
      if (s.proc) s.cancel();
    }
  }
}

function boundedPositiveInt(value, fallback) {
  const number = Number(value);
  return Number.isSafeInteger(number) && number > 0 ? number : fallback;
}

function boundedPositiveNumber(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : fallback;
}

// Tools that can change the world. A read-only role is denied these outright, so the
// declaration in field/roles is a real constraint rather than a description.
const DIRECT_MUTATING_TOOLS = ['Edit', 'Write', 'NotebookEdit', 'PowerShell'];
const UNMANAGED_DELEGATION_TOOLS = ['Task', 'Agent'];

export function denyListFor(role, agent = null) {
  const denied = new Set(role?.tools_deny ?? []);
  if (role?.read_only) {
    // Shell is intentionally neither auto-approved nor disabled. It reaches the Field
    // permission policy below, which allows inspection/tests and rejects mutation.
    for (const tool of DIRECT_MUTATING_TOOLS) denied.add(tool);
  }
  // Per-agent loadouts can only narrow the role. The CLI's allowed-tools flag controls
  // auto-approval, so removed tools must also be passed through its hard deny list.
  if (Array.isArray(agent?.tools_allow)) {
    const equipped = new Set(agent.tools_allow);
    for (const tool of role?.tools_allow ?? []) if (!equipped.has(tool)) denied.add(tool);
  }
  // Claude's built-in children are not independently admitted by Field and cannot
  // be counted against campaign capacity. Keep delegation on Knossos's managed,
  // budgeted path instead of treating a prompt reminder as enforcement.
  for (const tool of UNMANAGED_DELEGATION_TOOLS) denied.add(tool);
  return denied.size ? [...denied] : null;
}

const TOOL_CATEGORY = new Map([
  ['read', 'read'], ['read_file', 'read'],
  ['grep', 'search'], ['search', 'search'], ['search_code', 'search'],
  ['glob', 'list'], ['list_dir', 'list'],
  ['edit', 'edit'], ['edit_file', 'edit'],
  ['write', 'write'], ['write_file', 'write'], ['notebookedit', 'write'],
  ['bash', 'shell'], ['powershell', 'shell'], ['shell', 'shell'], ['exec', 'shell'],
  ['run', 'shell'], ['run_command', 'shell'],
  ['webfetch', 'web_fetch'], ['websearch', 'web_search'],
  ['task', 'delegate'], ['agent', 'delegate'],
  ['verify', 'verify'], ['ask_user', 'ask'], ['askuserquestion', 'ask'],
]);
const ROLE_TOOL_CATEGORY = new Map([
  ['read', 'read'], ['grep', 'search'], ['glob', 'list'],
  ['edit', 'edit'], ['write', 'write'], ['notebookedit', 'write'],
  ['bash', 'shell'], ['powershell', 'shell'],
  ['webfetch', 'web_fetch'], ['websearch', 'web_search'],
  ['task', 'delegate'], ['agent', 'delegate'],
]);
const MUTATING_SHELL = /(^|[\s;&|])(rm|rmdir|del|erase|move|mv|copy|cp|mkdir|md|touch|tee|set-content|add-content|out-file|remove-item|move-item|copy-item|new-item|git\s+(add|commit|reset|checkout|restore|clean)|npm\s+(install|uninstall)|pnpm\s+(add|remove)|yarn\s+(add|remove)|pip\s+install|cargo\s+(install|fmt|fix))([\s;&|]|$)|(^|[^>])>{1,2}($|[^>])/i;

/**
 * Return a hard denial when a tool request violates an environment boundary. Returning
 * null means the request may proceed to the human approval queue; it is never auto-allowed.
 */
export function evaluatePermissionPolicy({
  toolName, input = {}, workspacePath, readOnly = false, environmentScope,
  allowedTools, writeScope, validateWritePath, allowedDomains,
} = {}) {
  const name = String(toolName ?? '');
  const category = TOOL_CATEGORY.get(name.toLowerCase()) ?? `unknown:${name.toLowerCase()}`;
  const mutating = ['edit', 'write', 'shell'].includes(category);
  const candidate = input.file_path ?? input.path ?? input.notebook_path ?? input.target_path ?? null;

  if (Array.isArray(allowedTools)) {
    const allowed = new Set(allowedTools.map((tool) => (
      ROLE_TOOL_CATEGORY.get(String(tool).toLowerCase()) ?? `exact:${String(tool).toLowerCase()}`
    )));
    // Internal Knossos verification is equivalent to the role's shell/test authority.
    if (category === 'verify' && allowed.has('shell')) allowed.add('verify');
    if (!allowed.has(category) && !allowed.has(`exact:${name.toLowerCase()}`)) {
      return { decision: 'deny', message: `Denied by Field: ${name || 'unknown tool'} is not in this role's tool capability set.` };
    }
  }

  if (category === 'web_fetch') {
    const egress = validateEgressUrl(input.url, allowedDomains);
    if (egress) return { decision: 'deny', message: `Denied by Field: ${egress}` };
  }

  if (category === 'shell') {
    const command = String(input.command ?? input.cmd ?? '');
    if (NETWORK_SHELL.test(command)) {
      return { decision: 'deny', message: 'Denied by Field: network-capable shell commands cannot enforce destination policy; use WebFetch with a declared domain.' };
    }
  }

  if (mutating && candidate && workspacePath) {
    const root = path.resolve(workspacePath);
    const resolved = path.resolve(root, String(candidate));
    const rootKey = process.platform === 'win32' ? root.toLowerCase() : root;
    const resolvedKey = process.platform === 'win32' ? resolved.toLowerCase() : resolved;
    if (resolvedKey !== rootKey && !resolvedKey.startsWith(rootKey + path.sep)) {
      return { decision: 'deny', message: 'Denied by Field: target escapes the assigned workspace.' };
    }
    if (['edit', 'write'].includes(category) && validateWritePath) {
      try {
        validateWritePath(path.relative(root, resolved));
      } catch {
        return { decision: 'deny', message: 'Denied by Field: target violates workspace path or sensitive-file policy.' };
      }
    }
  }

  if (mutating && ['snapshot', 'production-readonly'].includes(environmentScope)) {
    return { decision: 'deny', message: `Denied by Field: ${environmentScope} campaigns are read-only.` };
  }

  if (['edit', 'write'].includes(category) && Array.isArray(writeScope) && writeScope.length > 0) {
    if (!candidate || !workspacePath) {
      return { decision: 'deny', message: 'Denied by Field: the scoped write has no verifiable workspace path.' };
    }
    const root = path.resolve(workspacePath);
    const relative = path.relative(root, path.resolve(root, String(candidate))).replace(/\\/g, '/');
    if (!relative || relative === '..' || relative.startsWith('../') || !buildMatcher(writeScope)(relative)) {
      return { decision: 'deny', message: `Denied by Field: target is outside this role's write scope (${writeScope.join(', ')}).` };
    }
  }

  if (readOnly && ['edit', 'write'].includes(category)) {
    return { decision: 'deny', message: 'Denied by Field: this campaign role is read-only.' };
  }

  if (readOnly && category === 'shell') {
    const command = String(input.command ?? input.cmd ?? '');
    if (!command || MUTATING_SHELL.test(command) || /(^|[\s"'`=])\.\.([\\/]|$)/.test(command)) {
      return { decision: 'deny', message: 'Denied by Field: read-only shell requests may inspect or test, but may not mutate or escape scope.' };
    }
  }
  return null;
}

const NETWORK_SHELL = /(^|[\s;&|])(curl|wget|aria2c|fetch|invoke-webrequest|invoke-restmethod|iwr|irm|ssh|scp|sftp|ftp|telnet|nc|ncat|netcat|git\s+(clone|fetch|pull)|npm\s+(install|add)|pnpm\s+(install|add)|yarn\s+(install|add)|pip\s+install|cargo\s+install)([\s;&|]|$)/i;

export function validateEgressUrl(value, allowedDomains = []) {
  let url;
  try { url = new URL(String(value ?? '')); }
  catch { return 'network destination must be a complete HTTP(S) URL.'; }
  if (!['http:', 'https:'].includes(url.protocol) || url.username || url.password) {
    return 'network destination must be credential-free HTTP(S).';
  }
  const hostname = url.hostname.toLowerCase().replace(/\.$/, '');
  if (!hostname || isPrivateHost(hostname)) return 'loopback, private, link-local, and metadata destinations are blocked.';
  const declared = allowedDomains.map((domain) => String(domain).toLowerCase().replace(/^\*\./, '').replace(/\.$/, ''));
  if (!declared.some((domain) => hostname === domain || hostname.endsWith(`.${domain}`))) {
    return `destination ${hostname} is not declared in this role's network policy.`;
  }
  return null;
}

function isPrivateHost(hostname) {
  if (hostname === 'localhost' || hostname.endsWith('.localhost') || hostname === 'metadata.google.internal') return true;
  const bare = hostname.replace(/^\[|\]$/g, '');
  const ipVersion = net.isIP(bare);
  if (ipVersion === 4) {
    const [a, b] = bare.split('.').map(Number);
    return a === 0 || a === 10 || a === 127 || (a === 100 && b >= 64 && b <= 127)
      || (a === 169 && b === 254) || (a === 172 && b >= 16 && b <= 31)
      || (a === 192 && b === 168) || a >= 224;
  }
  if (ipVersion === 6) {
    const normalized = bare;
    return normalized === '::' || normalized === '::1' || normalized.startsWith('fc')
      || normalized.startsWith('fd') || /^fe[89ab]/.test(normalized)
      || normalized.startsWith('::ffff:');
  }
  return false;
}

function countFiles(dir, cap) {
  let n = 0;
  const stack = [dir];
  while (stack.length && n < cap) {
    let entries;
    try { entries = fs.readdirSync(stack.pop(), { withFileTypes: true }); } catch { continue; }
    for (const e of entries) {
      if (e.name === '.git' || e.name === 'node_modules' || e.name === 'target') continue;
      if (e.isDirectory()) stack.push(path.join(e.parentPath ?? dir, e.name));
      else if (++n >= cap) break;
    }
  }
  return n;
}

/**
 * Parse one structured campaign report from a final harness message. The surrounding
 * prose is ignored; domain validation happens in CampaignDirector before any report event
 * is accepted. Keeping the sentinel explicit avoids interpreting ordinary JSON examples
 * as operational commands.
 */
export function parseCampaignReport(value) {
  const text = String(value ?? '');
  const marker = text.lastIndexOf('FIELD_REPORT:');
  if (marker < 0) return null;
  let tail = text.slice(marker + 'FIELD_REPORT:'.length).trim();
  const fenced = /^```(?:json)?\s*([\s\S]*?)\s*```/i.exec(tail);
  if (fenced) tail = fenced[1];
  else tail = tail.split(/\r?\n/)[0];
  try {
    const parsed = JSON.parse(tail);
    return parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed : null;
  } catch {
    return null;
  }
}
