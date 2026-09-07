// Transport to the Field server. One websocket for state and events, plain fetch for commands.

const listeners = {
  snapshot: new Set(),
  event: new Set(),
  terminal: new Set(),
  status: new Set(),
};

let socket = null;
let retry = 0;
let retryTimer = null;
let stopped = false;

export function on(kind, fn) {
  listeners[kind]?.add(fn);
  return () => listeners[kind]?.delete(fn);
}

function fire(kind, payload) {
  for (const fn of listeners[kind] ?? []) {
    try { fn(payload); } catch (e) { console.error(`[client:${kind}]`, e); }
  }
}

export function connect() {
  if (stopped) return;
  if (socket && (socket.readyState === 0 || socket.readyState === 1)) return;
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  socket = new WebSocket(`${proto}://${location.host}/ws`);

  socket.onopen = () => { retry = 0; fire('status', { connected: true }); };

  socket.onmessage = (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.type === 'snapshot') fire('snapshot', msg.state);
    else if (msg.type === 'event') fire('event', msg.event);
    else if (msg.type === 'terminal') fire('terminal', msg);
  };

  socket.onclose = () => {
    fire('status', { connected: false });
    if (stopped) return;
    // The server may simply be restarting; back off but keep trying.
    const delay = Math.min(500 * 2 ** retry++, 8000);
    clearTimeout(retryTimer);
    retryTimer = setTimeout(connect, delay);
  };

  socket.onerror = () => socket?.close();
}

async function request(method, url, body) {
  const res = await fetch(url, {
    method,
    headers: body ? { 'content-type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({ error: `${res.status} ${res.statusText}` }));
  if (!res.ok || data.error) throw new Error(data.error ?? `request failed (${res.status})`);
  return data;
}

export const api = {
  logout: async () => {
    await request('POST', '/api/logout');
    stopped = true;
    clearTimeout(retryTimer);
    socket?.close();
  },
  config:        () => request('GET', '/api/config'),
  state:         () => request('GET', '/api/state'),
  world:         () => request('GET', '/api/world'),
  selectCapital: (workspaceId) => request('POST', '/api/world/capital', { workspaceId }),
  reconcileWorld:(clusters) => request('POST', '/api/world/reconcile', { clusters }),
  events:        (from = 0, limit = 2000) => request('GET', `/api/events?from=${from}&limit=${limit}`),
  trace:         (subject, from = 0, limit = 2000) => request('GET', `/api/trace?subject=${encodeURIComponent(subject)}&from=${from}&limit=${limit}`),
  campaigns:     () => request('GET', '/api/campaigns'),
  simulations:   () => request('GET', '/api/simulations'),
  runSimulation: (scenario = 'operations-cycle', speed = 1, workspaceId = null) =>
                   request('POST', '/api/simulations/run', { scenario, speed, workspaceId }),
  stopSimulation:() => request('POST', '/api/simulations/stop'),
  campaignTrace: (campaignId, from = 0, limit = 2000) => request('GET', `/api/campaigns/trace?campaignId=${encodeURIComponent(campaignId)}&from=${from}&limit=${limit}`),
  campaignReplay:(campaignId, seq) => request('GET', `/api/campaigns/replay?campaignId=${encodeURIComponent(campaignId)}&seq=${encodeURIComponent(seq)}`),
  createCampaign:(body) => request('POST', '/api/campaigns/create', body),
  campaignAction:(body) => request('POST', '/api/campaigns/action', body),

  spawn:         (body) => request('POST', '/api/sessions', body),
  updateAgent:   (body) => request('POST', '/api/agents/settings', body),
  assign:        (body) => request('POST', '/api/assign', body),
  command:       (kind, body) => request('POST', '/api/command', { kind, ...body }),
  controlGroup:  (group, sessionIds) => request('POST', '/api/control-group', { group, sessionIds }),
  position:      (body) => request('POST', '/api/position', body),
  decide:        (permissionId, decision, message) =>
                   request('POST', '/api/permission/decide', { permissionId, decision, message }),

  tree:          (ws, dir = '') => request('GET', `/api/fs/tree?ws=${ws}&dir=${encodeURIComponent(dir)}`),
  readFile:      (ws, path) => request('GET', `/api/fs/file?ws=${ws}&path=${encodeURIComponent(path)}`),
  writeFile:     (ws, path, content) => request('PUT', '/api/fs/file', { ws, path, content }),

  diff:          (ws, path) => request('GET', `/api/git/diff?ws=${ws}&path=${encodeURIComponent(path)}`),
  gitLog:        (ws) => request('GET', `/api/git/log?ws=${ws}`),

  runCommand:    (ws, command, terminalId) => request('POST', '/api/terminal/run', { ws, command, terminalId }),
  killCommand:   (terminalId) => request('POST', '/api/terminal/kill', { terminalId }),

  runRoutine:    (routineId) => request('POST', '/api/routines/run', { routineId }),
  toggleRoutine: (routineId, enabled, confirmRisk = false) =>
                   request('POST', '/api/routines/toggle', { routineId, enabled, confirmRisk }),
  routineStates: () => request('GET', '/api/routines/states'),
};
