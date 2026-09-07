// Field server. Loads the git-backed configuration, replays the event log into live
// state, then opens the operator interface.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import { loadConfig } from './config.js';
import { EventLog } from './store/db.js';
import { Projection } from './store/projection.js';
import { replayInto } from './store/replay.js';
import { Registry } from './harness/registry.js';
import { CampaignDirector } from './orchestration/director.js';
import { createApi } from './api.js';
import { createHub } from './ws.js';
import { startFsWatchers } from './watch/fs.js';
import { startGitWatchers } from './watch/git.js';
import { startEndpointProbes } from './endpoints.js';
import { startRoutines } from './routines.js';
import { FieldSimulator } from './simulation/field-simulator.js';
import { createControlSecurity } from './security.js';
import { readJsonBody } from './body.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '../..');
const FIELD_DIR = process.env.FIELD_DIR || path.join(ROOT, 'field');
const STATE_DIR = process.env.FIELD_STATE || path.join(ROOT, '.field-state');
const DIST_DIR = path.join(ROOT, 'web', 'dist');

const cfg = loadConfig(FIELD_DIR);
const PORT = Number(process.env.FIELD_PORT || cfg.field.api_port || 7749);
const API_BASE = `http://127.0.0.1:${PORT}`;
const devUiOrigin = process.env.FIELD_UI_ORIGIN
  || (process.env.npm_lifecycle_event === 'dev' ? 'http://127.0.0.1:7748' : null);
const security = createControlSecurity({
  port: PORT,
  trustedOrigins: devUiOrigin ? [devUiOrigin] : [],
  bootstrapRedirect: devUiOrigin ?? '/',
  frameSources: cfg.websites.flatMap((site) => [
    `https://${site.domain}`,
    `http://${site.domain}`,
  ]),
});

// ---------------------------------------------------------------- state

const configuredSecrets = Object.entries(process.env)
  .filter(([key, value]) => value && /(api[_-]?key|token|secret|password|authorization|cookie)/i.test(key))
  .map(([, value]) => value);
configuredSecrets.push(security.bootstrapToken, security.browserToken);
const log = new EventLog(STATE_DIR, { secrets: configuredSecrets });
const projection = new Projection(cfg);

// Replay everything that ever happened. Live state and a historical replay are
// produced by the same fold, so they can never drift apart.
const replayStart = Date.now();
const { count: replayed } = replayInto(log, projection);

// The hub is created once the http server exists; until then broadcasts are dropped,
// because there is nothing connected yet to receive them.
let hub = { broadcast() {}, pushEvent() {}, close() {} };

// Every appended event updates live state and reaches every connected operator.
log.subscribe((evt) => {
  projection.apply(evt);
  hub.pushEvent(evt);
});

const emit = (kind, data, meta) => log.append(kind, data, meta);

// Sessions do not survive a restart: the harness processes are gone. Mark any session
// the log still believes is live as interrupted, rather than showing a ghost.
for (const s of [...projection.sessions.values()]) {
  if (['spawning', 'ready', 'idle', 'thinking', 'working', 'waiting_permission'].includes(s.state)) {
    emit('session.state', {
      sessionId: s.id, state: 'interrupted', detail: 'field server restarted',
    }, { subject: s.id });
  }
}

for (const assignment of projection.assignments.values()) {
  if (assignment.status !== 'active') continue;
  const allUnavailable = assignment.sessionIds.length > 0 && assignment.sessionIds.every((id) => {
    const session = projection.sessions.get(id);
    return !session || ['done', 'cancelled', 'error', 'interrupted'].includes(session.state);
  });
  if (allUnavailable) {
    emit('assignment.interrupted', {
      assignmentId: assignment.id,
      reason: 'Field restarted before assignment completion was durably recorded',
    }, { subject: assignment.id, source: 'derived' });
  }
}

// Explicitly external test actors have campaign membership but no harness session event.
// Keep them usable in deterministic campaigns without ever presenting them as live units.
for (const campaign of projection.campaigns.campaigns.values()) {
  for (const [team, formation] of Object.entries(campaign.teams)) {
    for (const member of formation.members) {
      if (member.status === 'active' && !projection.sessions.has(member.sessionId)) {
        emit('team.member_state', {
          campaignId: campaign.id, team, sessionId: member.sessionId,
          status: 'external', sessionState: 'external',
        }, { subject: campaign.id });
      }
    }
  }
}

const registry = new Registry({
  cfg,
  emit,
  apiBase: API_BASE,
  permissionCapabilities: {
    mint: security.mintHarnessToken,
    revoke: security.revokeHarnessToken,
  },
  registerSecret: (secret) => log.addSecret(secret),
  campaignPolicy: (campaignId) => projection.campaigns.campaigns.get(campaignId) ?? null,
  budgetLedger: projection.budgets,
});
const director = new CampaignDirector({
  projection: projection.campaigns,
  registry,
  emit,
  eventHead: () => log.size,
});
registry.setCampaignReportHandler((report) => director.ingestSessionReport(report));
const routines = startRoutines({ cfg, registry, log, emit, projection });
// Rehearsals are synthetic projection events only: no provider calls, executable tools, or
// cost. They are available by default on this loopback operator UI and can be disabled.
const simulator = process.env.FIELD_SIMULATION === '0'
  ? null
  : new FieldSimulator({ emit, workspaces: cfg.workspaces });

// ---------------------------------------------------------------- http

const api = createApi({
  cfg, projection, registry, director, simulator, routines, log,
  broadcast: (msg) => hub.broadcast(msg),
});

const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8', '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml', '.woff2': 'font/woff2', '.png': 'image/png',
  '.webp': 'image/webp', '.ico': 'image/x-icon',
};

const server = http.createServer(async (req, res) => {
  security.applyHeaders(res);
  const url = new URL(req.url, 'http://127.0.0.1');

  const bootstrap = security.consumeBootstrap(req, url);
  if (bootstrap.handled) {
    if (!bootstrap.ok) return sendError(res, bootstrap);
    if (bootstrap.cookie) res.setHeader('set-cookie', bootstrap.cookie);
    res.writeHead(303, { location: bootstrap.redirect });
    return res.end();
  }

  if (req.method === 'GET' && url.pathname === '/healthz') {
    const network = security.authorizeNetwork(req);
    if (!network.ok) return sendError(res, network);
    res.writeHead(200, { 'content-type': 'application/json; charset=utf-8' });
    return res.end(JSON.stringify({ ok: true }));
  }

  const auth = security.authorizeRequest(req, url);
  if (!auth.ok) return sendError(res, auth);
  req.fieldAuth = auth;

  if (req.method === 'POST' && url.pathname === '/api/logout') {
    res.setHeader('set-cookie', security.revokeBrowserSession());
    hub.revokeClients();
    res.writeHead(200, { 'content-type': 'application/json' });
    return res.end(JSON.stringify({ ok: true }));
  }

  // Routine toggling needs the live routine controller, so it is handled here.
  if (req.method === 'POST' && url.pathname === '/api/routines/toggle') {
    try {
      const body = await readJsonBody(req);
      const ok = routines.setEnabled(body.routineId, !!body.enabled, {
        confirmRisk: body.confirmRisk === true,
      });
      res.writeHead(ok ? 200 : 400, { 'content-type': 'application/json' });
      res.end(JSON.stringify(ok ? {
        ok, states: routines.states(), details: routines.details(),
      } : { error: 'unknown routine' }));
    } catch (error) {
      sendError(res, {
        status: error.status ?? 400,
        code: error.code ?? 'bad_request',
        error: error.message,
      });
    }
    return;
  }
  if (req.method === 'GET' && url.pathname === '/api/routines/states') {
    res.writeHead(200, { 'content-type': 'application/json' });
    return res.end(JSON.stringify({ states: routines.states(), details: routines.details() }));
  }

  if (await api(req, res)) return;

  // Static: the built operator interface, when it has been built.
  if (fs.existsSync(DIST_DIR)) {
    const rel = url.pathname === '/' ? 'index.html' : url.pathname.slice(1);
    const file = path.resolve(DIST_DIR, rel);
    if (file.startsWith(DIST_DIR) && fs.existsSync(file) && fs.statSync(file).isFile()) {
      res.writeHead(200, { 'content-type': MIME[path.extname(file)] ?? 'application/octet-stream' });
      return fs.createReadStream(file).pipe(res);
    }
    const index = path.join(DIST_DIR, 'index.html');
    if (fs.existsSync(index)) {
      res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
      return fs.createReadStream(index).pipe(res);
    }
  }

  res.writeHead(404, { 'content-type': 'application/json' });
  res.end(JSON.stringify({ error: 'not found', hint: 'run `npm run web` for the dev interface' }));
});

// Attach the websocket hub now that the http server exists.
hub = createHub(server, { projection, authorizeUpgrade: security.authorizeUpgrade });
server.maxConnections = 64;
server.headersTimeout = 10_000;
server.requestTimeout = 30_000;
server.keepAliveTimeout = 5_000;
server.on('clientError', (_error, socket) => {
  if (!socket.writable) return;
  socket.end('HTTP/1.1 400 Bad Request\r\nConnection: close\r\nContent-Length: 0\r\n\r\n');
});

const stopFs = startFsWatchers(cfg, emit);
const stopGit = startGitWatchers(cfg, emit);
const stopProbes = startEndpointProbes(cfg, emit, registry);

server.listen(PORT, '127.0.0.1', () => {
  routines.start();
  const mounted = cfg.workspaces.filter((w) => w.mounted).length;
  console.log(`\n  Field server  ${API_BASE}`);
  console.log(`  open          ${security.bootstrapUrl}`);
  console.log(`  store         ${log.backend} (${log.size} events, replayed ${replayed} in ${Date.now() - replayStart}ms)`);
  console.log(`  workspaces    ${mounted}/${cfg.workspaces.length} mounted`);
  console.log(`  agents        ${cfg.agents.length} defined · ${cfg.roles.length} roles`);
  console.log(`  endpoints     ${cfg.endpoints.length} configured`);
  console.log(`  routines      ${cfg.routines.filter((r) => r.enabled).length} enabled of ${cfg.routines.length}`);
  for (const w of cfg.workspaces.filter((x) => !x.mounted)) {
    console.log(`  ! workspace "${w.id}" is not mounted: ${w.path} does not exist`);
  }
  console.log('');
});

function shutdown() {
  console.log('\n  stopping sessions…');
  simulator?.stop('shutdown');
  registry.shutdown();
  routines.stop();
  stopFs();
  stopGit();
  stopProbes();
  server.close(() => { log.close(); process.exit(0); });
  setTimeout(() => process.exit(0), 1500).unref();
}
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);

function sendError(res, failure) {
  const body = JSON.stringify({ error: failure.error, code: failure.code });
  res.writeHead(failure.status, {
    'content-type': 'application/json; charset=utf-8',
    'content-length': Buffer.byteLength(body),
  });
  res.end(body);
}
