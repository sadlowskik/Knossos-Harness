// REST surface. Binds to loopback only: every route here can read and write real files
// and start real processes, so it is an operator-local interface, not a public API.
import fs from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { diffFile, log as gitLog, readCleanRevision, readStatus } from './watch/git.js';
import { DomainError } from './orchestration/model.js';
import { Projection } from './store/projection.js';
import { readEventRange, replayInto } from './store/replay.js';
import { updateFrontmatterFile } from './config.js';
import { readJsonBody } from './body.js';
import {
  readWorkspaceFile,
  resolveWorkspacePath,
  writeWorkspaceFile,
} from './workspace-path.js';
import {
  OutputBudget,
  redactCommand,
  stopProcessTree,
  TERMINAL_LIMITS,
  validateTerminalCommand,
} from './terminal-policy.js';
import { buildChildEnvironment } from './child-env.js';

const MAX_FILE_BYTES = 2 * 1024 * 1024;
const EVENT_PAGE_DEFAULT = 2000;
const EVENT_PAGE_MAX = 10_000;
const TERRITORY_IDS = [
  'italia', 'gallia', 'hispania', 'africa', 'aegyptus', 'britannia',
  'dacia', 'balkans', 'anatolia', 'levant', 'mesopotamia', 'cyrenaica',
];

function worldCluster(input) {
  const clusterKey = String(input?.clusterKey ?? '').trim().slice(0, 160);
  if (!clusterKey) throw new DomainError('invalid_id', 'clusterKey is required');
  return {
    clusterKey,
    label: String(input?.label ?? clusterKey).trim().slice(0, 96) || clusterKey,
    kind: String(input?.kind ?? 'infrastructure').trim().slice(0, 48) || 'infrastructure',
    workspaceId: input?.workspaceId ? String(input.workspaceId).slice(0, 96) : null,
  };
}

export function parseEventPage(url) {
  const rawFrom = url.searchParams.get('from');
  const rawLimit = url.searchParams.get('limit');
  const from = rawFrom == null || rawFrom === '' ? 0 : Number(rawFrom);
  const limit = rawLimit == null || rawLimit === '' ? EVENT_PAGE_DEFAULT : Number(rawLimit);
  if (!Number.isSafeInteger(from) || from < 0) throw new DomainError('invalid_pagination', 'from must be a non-negative integer');
  if (!Number.isSafeInteger(limit) || limit < 1 || limit > EVENT_PAGE_MAX) throw new DomainError('invalid_pagination', `limit must be an integer between 1 and ${EVENT_PAGE_MAX}`);
  return { from, limit };
}

function pageResult(events, { from, limit }, head = null, hasMore = events.length === limit) {
  const rows = events.slice(0, limit);
  const nextFrom = hasMore && rows.length ? rows.at(-1).seq : null;
  return { events: rows, head: head ?? (rows.at(-1)?.seq ?? from), nextFrom };
}

function json(res, code, body) {
  const text = JSON.stringify(body);
  res.writeHead(code, {
    'content-type': 'application/json; charset=utf-8',
    'content-length': Buffer.byteLength(text),
    'cache-control': 'no-store',
  });
  res.end(text);
}

const SKIP_DIRS = new Set(['.git', 'node_modules', 'target', 'dist', '.field-state']);

export function createApi({ cfg, projection, registry, director, simulator, routines, log, broadcast }) {
  const terminals = new Map();

  const routes = {
    'GET /api/state': async () => projection.snapshot(),

    'GET /api/world': async () => structuredClone(projection.world),

    'POST /api/world/capital': async (body) => {
      const workspaceId = String(body?.workspaceId ?? '').trim();
      const workspace = cfg.workspaces.find((item) => item.id === workspaceId && item.mounted);
      if (!workspace) throw new DomainError('not_found', `mounted workspace ${workspaceId || '(empty)'} does not exist`);
      log.append('world.capital_selected', { workspaceId }, { subject: workspaceId });
      return structuredClone(projection.world);
    },

    'POST /api/world/reconcile': async (body) => {
      const raw = Array.isArray(body?.clusters) ? body.clusters : [];
      if (raw.length > 96) throw new DomainError('invalid_request', 'at most 96 infrastructure clusters may be reconciled');
      const clusters = raw.map(worldCluster);
      const capitalWorkspaceId = projection.world.capitalWorkspaceId;
      if (!capitalWorkspaceId) throw new DomainError('invalid_state', 'choose a capital project before reconciling the world');

      const capitalKey = `workspace:${capitalWorkspaceId}`;
      const capitalCluster = clusters.find((item) => item.clusterKey === capitalKey) ?? {
        clusterKey: capitalKey,
        label: cfg.workspaces.find((item) => item.id === capitalWorkspaceId)?.name ?? capitalWorkspaceId,
        kind: 'project',
        workspaceId: capitalWorkspaceId,
      };
      const assignments = projection.world.assignments;
      const currentKeys = new Set(clusters.map((item) => item.clusterKey));
      currentKeys.add(capitalKey);
      for (const assignment of Object.values(assignments)) {
        if (currentKeys.has(assignment.clusterKey)) continue;
        log.append('world.territory_released', {
          clusterKey: assignment.clusterKey,
        }, { subject: assignment.clusterKey });
      }
      const currentItalia = Object.values(assignments).find((item) => item.territoryId === 'italia');
      const capitalPrevious = assignments[capitalKey]?.territoryId;

      if (currentItalia && currentItalia.clusterKey !== capitalKey) {
        const occupied = new Set(Object.values(assignments).map((item) => item.territoryId));
        const destination = capitalPrevious && capitalPrevious !== 'italia'
          ? capitalPrevious
          : TERRITORY_IDS.find((id) => id !== 'italia' && !occupied.has(id));
        if (destination) {
          log.append('world.territory_assigned', {
            ...currentItalia,
            territoryId: destination,
          }, { subject: currentItalia.clusterKey });
        } else {
          log.append('world.territory_released', {
            clusterKey: currentItalia.clusterKey,
          }, { subject: currentItalia.clusterKey });
        }
      }

      if (projection.world.assignments[capitalKey]?.territoryId !== 'italia') {
        log.append('world.territory_assigned', {
          ...capitalCluster,
          territoryId: 'italia',
        }, { subject: capitalKey });
      }

      const ordered = clusters
        .filter((item) => item.clusterKey !== capitalKey)
        .sort((a, b) => a.clusterKey.localeCompare(b.clusterKey));
      for (const cluster of ordered) {
        if (projection.world.assignments[cluster.clusterKey]) continue;
        const occupied = new Set(Object.values(projection.world.assignments).map((item) => item.territoryId));
        const territoryId = TERRITORY_IDS.find((id) => id !== 'italia' && !occupied.has(id));
        if (!territoryId) break;
        log.append('world.territory_assigned', { ...cluster, territoryId }, { subject: cluster.clusterKey });
      }
      return structuredClone(projection.world);
    },

    'GET /api/simulations': async () => ({
      enabled: !!simulator,
      scenarios: simulator?.scenarios() ?? [],
      active: simulator?.status() ?? null,
    }),

    'POST /api/simulations/run': async (body) => {
      if (!simulator) {
        throw new DomainError('simulation_disabled', 'Synthetic world rehearsals are disabled on this Field server');
      }
      return simulator.run(body?.scenario ?? 'operations-cycle', {
        speed: body?.speed,
        workspaceId: body?.workspaceId ?? projection.world.capitalWorkspaceId,
      });
    },

    'POST /api/simulations/stop': async () => simulator?.stop('operator') ?? { stopped: false },

    'GET /api/config': async () => ({
      field: cfg.field,
      defaults: cfg.defaults,
      workspaces: cfg.workspaces.map((w) => ({
        id: w.id, name: w.name, path: w.path, mounted: w.mounted, region: w.region,
      })),
      endpoints: cfg.endpoints,
      websites: cfg.websites,
      roles: cfg.roles.map(strip),
      agents: cfg.agents.map(strip),
      missions: cfg.missions.map((m) => ({ ...strip(m), body: m.body })),
      routines: cfg.routines,
      skills: cfg.skills.map(strip),
      memory: cfg.memory.map(strip),
      constitutions: cfg.constitutions.map(strip),
    }),

    'POST /api/sessions': async (body) => {
      const s = registry.spawn(body);
      return { sessionId: s.id };
    },

    'POST /api/agents/settings': async (body) => {
      const agentId = String(body?.agentId ?? '').trim();
      const agent = cfg.agents.find((item) => item.id === agentId);
      if (!agent) throw new DomainError('not_found', `agent ${agentId || '(empty)'} does not exist`);
      const role = cfg.roles.find((item) => item.id === agent.role);
      const roleTools = new Set(role?.tools_allow ?? []);
      const requested = Array.isArray(body?.toolsAllow) ? body.toolsAllow.map(String) : (agent.tools_allow ?? role?.tools_allow ?? []);
      const toolsAllow = [...new Set(requested)].filter((tool) => roleTools.has(tool));
      const name = String(body?.name ?? agent.name ?? agent.id).trim().slice(0, 64) || agent.id;
      const updates = { name, tools_allow: toolsAllow };
      updateFrontmatterFile(agent.file, updates);
      Object.assign(agent, updates);
      log.append('agent.settings_updated', { agentId, name, toolsAllow }, { subject: agentId });
      return { agent: strip(agent), appliesTo: 'future_sessions' };
    },

    'POST /api/assign': async (body) => registry.assign(body),

    'POST /api/command': async (body) => registry.command(body.kind, body),

    'POST /api/control-group': async (body) => {
      registry.setControlGroup(body.group, body.sessionIds ?? []);
      return { ok: true };
    },

    'POST /api/position': async (body) => {
      log.append('ui.position', body);
      return { ok: true };
    },

    'POST /api/permission/decide': async (body) => ({
      ok: registry.decidePermission(body.permissionId, body.decision, body.message),
    }),

    'GET /api/campaigns': async () => ({ campaigns: projection.campaigns.snapshot().campaigns }),

    'POST /api/campaigns/create': async (body) => director.create(body),

    'POST /api/campaigns/action': async (body) => {
      if (body?.kind === 'checkpoint') {
        if (body.confirmRisk !== true) {
          throw new DomainError('confirmation_required', 'recording a campaign checkpoint requires explicit operator confirmation');
        }
        const campaign = projection.campaigns.campaigns.get(String(body.campaignId ?? ''));
        if (!campaign) throw new DomainError('not_found', 'campaign does not exist');
        const workspaceId = campaign.target?.workspaceId
          ?? (campaign.target?.type === 'workspace' ? campaign.target.id : null);
        const workspace = cfg.workspaces.find((item) => item.id === workspaceId && item.mounted && item.git !== false);
        if (!workspace) throw new DomainError('checkpoint_unavailable', 'campaign target is not a mounted Git workspace');
        let state;
        try { state = await readCleanRevision(workspace.path); }
        catch (error) {
          throw new DomainError('dirty_workspace', `checkpoint refused: ${error.message}. Commit or intentionally discard the changes first.`);
        }
        return director.action({
          ...body,
          revision: state.revision,
          workspaceId,
          branch: state.branch,
          checkpointMode: 'record_only',
        });
      }
      if (body?.kind === 'rollback' && body.confirmRisk !== true) {
        throw new DomainError('confirmation_required', 'recording a rollback requires explicit operator confirmation');
      }
      return director.action(body);
    },

    'GET /api/campaigns/trace': async (_body, url) => {
      const campaignId = url.searchParams.get('campaignId');
      if (!campaignId) throw new DomainError('invalid_id', 'campaignId is required');
      const campaign = projection.campaigns.campaigns.get(campaignId);
      if (!campaign) throw new DomainError('not_found', `campaign ${campaignId} does not exist`);
      const ids = new Set([
        campaignId, ...campaign.objectiveIds, ...campaign.findingIds,
        ...campaign.mitigationIds, ...campaign.verdictIds, ...campaign.checkpointIds,
      ]);
      const page = parseEventPage(url);
      const events = readEventRange(log)
        .filter((evt) => evt.seq > page.from && (ids.has(evt.subject) || evt.data?.campaignId === campaignId))
        .slice(0, page.limit + 1);
      return { campaignId, ...pageResult(events, page, log.size, events.length > page.limit) };
    },

    'GET /api/campaigns/replay': async (_body, url) => {
      const campaignId = url.searchParams.get('campaignId');
      if (!campaignId) throw new DomainError('invalid_id', 'campaignId is required');
      if (!projection.campaigns.campaigns.has(campaignId)) {
        throw new DomainError('not_found', `campaign ${campaignId} does not exist`);
      }
      const requested = Number(url.searchParams.get('seq') ?? log.size);
      if (!Number.isSafeInteger(requested) || requested < 0 || requested > log.size) {
        throw new DomainError('invalid_seq', `seq must be an integer between 0 and ${log.size}`);
      }
      const historical = new Projection(cfg);
      const { count, lastEvent } = replayInto(log, historical, { to: requested });
      const campaign = historical.campaigns.campaigns.get(campaignId);
      const now = lastEvent?.ts ?? Date.now();
      return {
        campaignId,
        requestedSeq: requested,
        actualSeq: lastEvent?.seq ?? 0,
        replayedEvents: count,
        campaign: campaign ? historical.campaigns.campaignView(campaign) : null,
        graph: historical.graph.snapshot(now, { maxNodes: 4000, maxEdges: 8000 }),
      };
    },

    // Called by the permission MCP server running inside a harness session.
    // Holds the request open until an operator decides.
    'POST /api/internal/permission': async (body, _url, req) =>
      registry.requestPermission({
        ...body,
        sessionId: req.fieldAuth?.sessionId ?? body.sessionId,
        capabilitySessionId: req.fieldAuth?.sessionId,
      }),

    'GET /api/events': async (_b, url) => {
      const page = parseEventPage(url);
      const events = log.read(page.from, page.limit + 1);
      return pageResult(events, page, log.size, events.length > page.limit);
    },

    'GET /api/trace': async (_b, url) => {
      const subject = url.searchParams.get('subject');
      if (!subject) throw new DomainError('invalid_id', 'subject is required');
      const page = parseEventPage(url);
      const events = log.bySubject(subject, { fromSeq: page.from, limit: page.limit + 1 });
      return { subject, ...pageResult(events, page, log.size, events.length > page.limit) };
    },

    'GET /api/fs/tree': async (_b, url) => {
      const wsId = url.searchParams.get('ws');
      const rel = url.searchParams.get('dir') ?? '';
      const { abs, ws } = resolveWorkspacePath(cfg, wsId, rel, { type: 'directory' });
      const entries = fs.readdirSync(abs, { withFileTypes: true })
        .filter((e) => !SKIP_DIRS.has(e.name))
        .filter((e) => ws.isVisible(path.posix.join(rel.replace(/\\/g, '/'), e.name)))
        .map((e) => {
          const childRel = path.posix.join(rel.replace(/\\/g, '/'), e.name);
          let size = null;
          if (e.isFile()) {
            try { size = fs.statSync(path.join(abs, e.name)).size; } catch { size = null; }
          }
          return { name: e.name, path: childRel, dir: e.isDirectory(), size };
        })
        .sort((a, b) => (a.dir === b.dir ? a.name.localeCompare(b.name) : a.dir ? -1 : 1));
      return { ws: wsId, dir: rel, entries };
    },

    'GET /api/fs/file': async (_b, url) => {
      const wsId = url.searchParams.get('ws');
      const rel = url.searchParams.get('path');
      const resolved = resolveWorkspacePath(cfg, wsId, rel, { type: 'file' });
      const st = resolved.stat;
      if (st.size > MAX_FILE_BYTES) {
        return { ws: wsId, path: rel, tooLarge: true, size: st.size, content: null };
      }
      return {
        ws: wsId, path: rel, size: st.size, tooLarge: false,
        content: readWorkspaceFile(resolved),
      };
    },

    'PUT /api/fs/file': async (body) => {
      const resolved = resolveWorkspacePath(cfg, body.ws, body.path, { operation: 'write', type: 'file' });
      writeWorkspaceFile(resolved, body.content ?? '');
      // The fs watcher will observe this write and emit the change event itself.
      return { ok: true, bytes: Buffer.byteLength(body.content ?? '') };
    },

    'GET /api/git/diff': async (_b, url) => {
      const wsId = url.searchParams.get('ws');
      const file = url.searchParams.get('path');
      const { ws } = resolveWorkspacePath(cfg, wsId, file, { type: 'file' });
      return { ws: wsId, path: file, diff: await diffFile(ws.path, file) };
    },

    'GET /api/git/log': async (_b, url) => {
      const wsId = url.searchParams.get('ws');
      const { ws } = resolveWorkspacePath(cfg, wsId, '', { type: 'directory' });
      return { ws: wsId, commits: await gitLog(ws.path, 40) };
    },

    'GET /api/git/status': async (_b, url) => {
      const wsId = url.searchParams.get('ws');
      const { ws } = resolveWorkspacePath(cfg, wsId, '', { type: 'directory' });
      return { ws: wsId, ...(await readStatus(ws.path)) };
    },

    'POST /api/terminal/run': async (body) => {
      const { ws, abs: cwd } = resolveWorkspacePath(cfg, body.ws, body.cwd ?? '', { type: 'directory' });
      if (terminals.size >= TERMINAL_LIMITS.concurrent) {
        throw new DomainError('terminal_capacity', `Field allows at most ${TERMINAL_LIMITS.concurrent} concurrent terminals`);
      }
      const command = validateTerminalCommand(body.command);
      const id = body.terminalId ?? randomUUID();
      if (terminals.has(id)) throw new DomainError('terminal_exists', 'terminal id is already active');
      const shell = process.platform === 'win32' ? 'powershell.exe' : 'bash';
      const args = process.platform === 'win32'
        ? ['-NoProfile', '-NonInteractive', '-Command', command]
        : ['-lc', command];

      const proc = spawn(shell, args, {
        cwd,
        env: buildChildEnvironment(),
        detached: process.platform !== 'win32',
        windowsHide: true,
      });
      const output = new OutputBudget();
      let stoppedForLimit = false;
      const timer = setTimeout(() => {
        stoppedForLimit = true;
        broadcast({ type: 'terminal', id, stream: 'meta', data: '\n[terminated: time limit]\n' });
        stopProcessTree(proc);
      }, TERMINAL_LIMITS.timeoutMs);
      terminals.set(id, { proc, timer });

      broadcast({ type: 'terminal', id, stream: 'meta', data: `[${body.ws}:${body.cwd || '.'}] $ ${redactCommand(command)}\n` });
      const forward = (stream) => (chunk) => {
        const accepted = output.accept(chunk);
        if (accepted.text) broadcast({ type: 'terminal', id, stream, data: accepted.text });
        if (accepted.exceeded && !stoppedForLimit) {
          stoppedForLimit = true;
          broadcast({ type: 'terminal', id, stream: 'meta', data: '\n[terminated: output limit]\n' });
          stopProcessTree(proc);
        }
      };
      proc.stdout.on('data', forward('out'));
      proc.stderr.on('data', forward('err'));
      proc.on('close', (code) => {
        clearTimeout(timer);
        terminals.delete(id);
        broadcast({ type: 'terminal', id, stream: 'meta', data: `\n[exit ${code}]\n`, exit: code });
      });
      proc.on('error', (e) => {
        clearTimeout(timer);
        terminals.delete(id);
        broadcast({ type: 'terminal', id, stream: 'err', data: `failed to start: ${e.message}\n`, exit: -1 });
      });

      log.append('terminal.run', { workspaceId: body.ws, command: redactCommand(command), terminalId: id });
      return { terminalId: id };
    },

    'POST /api/terminal/kill': async (body) => {
      const terminal = terminals.get(body.terminalId);
      if (terminal) {
        clearTimeout(terminal.timer);
        stopProcessTree(terminal.proc);
      }
      return { ok: !!terminal };
    },

    'POST /api/routines/run': async (body) => {
      const result = routines.run(String(body?.routineId ?? ''));
      if (!result.admitted) throw new DomainError('routine_not_admitted', result.reason);
      return { sessionId: result.sessionId, runId: result.runId };
    },
  };

  return async function handle(req, res) {
    const url = new URL(req.url, 'http://127.0.0.1');
    const key = `${req.method} ${url.pathname}`;
    const route = routes[key];
    if (!route) return false;

    try {
      const body = req.method === 'POST' || req.method === 'PUT' ? await readJsonBody(req) : null;
      const result = await route(body, url, req);
      json(res, 200, result);
    } catch (e) {
      const status = e.status ?? (e instanceof DomainError
        ? (e.code === 'not_found' ? 404 : ['gate_blocked', 'invalid_transition', 'role_conflict'].includes(e.code) ? 409 : 400)
        : 400);
      json(res, status, {
        error: e.message,
        code: e.code ?? 'bad_request',
        ...(e.detail ?? {}),
      });
    }
    return true;
  };
}

function strip(o) {
  const { body, file, ...rest } = o;
  return rest;
}
