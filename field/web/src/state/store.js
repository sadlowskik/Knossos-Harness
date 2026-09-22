// A small external store. The server owns the truth; this holds the last snapshot
// plus purely local operator state (selection, camera, mode).
import { useSyncExternalStore } from 'react';
import { api, connect, on } from '../net/client.js';

const EMPTY = {
  seq: 0, now: Date.now(), sessions: [], workspaces: [], folders: [], files: [],
  websites: [], endpoints: [], assignments: [], permissions: [],
  campaigns: [], capabilities: [], checkpoints: [], graph: { nodes: [], edges: [], visibleNodeKeys: [] },
  controlGroups: {}, positions: {}, world: { capitalWorkspaceId: null, assignments: {}, revision: 0 },
  totals: { costUsd: 0 },
};

/* Two destinations. Rome is the territory — where the work is happening — and Atlas is
   every conversation at once. The Map, Project, Plans, Routines and History screens were
   each a slice of one of those two, so they are gone and their capability moved inside.
   Anything a bookmark, a stored value or an old link still names lands on whichever of
   the two absorbed it. */
const MODE_IDS = new Set(['rome', 'atlas']);
const MODE_ALIASES = {
  board: 'atlas', theater: 'atlas', field: 'atlas',
  rts: 'rome', map: 'rome', workspace: 'rome', project: 'rome',
  campaigns: 'rome', plans: 'rome', routines: 'rome', traces: 'rome', history: 'rome',
};

/** A mode id that exists, for any value at all. Unknown and absent both mean Rome. */
export function normalizeMode(value) {
  const id = typeof value === 'string' ? value.trim().toLowerCase() : '';
  if (MODE_IDS.has(id)) return id;
  return MODE_ALIASES[id] ?? 'rome';
}

const requestedMode = typeof location === 'undefined'
  ? null
  : new URLSearchParams(location.search).get('mode');

let state = {
  connected: false,
  snap: EMPTY,
  config: null,
  events: [],            // rolling tail, newest last
  mode: normalizeMode(requestedMode),
  selection: [],         // session ids
  activeWorkspaceId: null,
  activeSessionId: null,
  activeCampaignId: null,
  hover: null,
  focus: null,           // { type, id, ... } — what the last "open this" asked for
  /* What Rome should show when it next renders, set by anything anywhere that says
     "open this there". `seq` rises on every request so repeating one still lands. */
  rome: null,            // { seq, workspaceId, dir, expanded, path, view, pane, url, plans, campaignId }
  camera: { x: 0, y: 0, z: 0.85 },
  lastEventBySession: {},
  error: null,
};

const listeners = new Set();
let scheduled = false;

function emit() {
  if (scheduled) return;
  scheduled = true;
  queueMicrotask(() => { scheduled = false; for (const fn of listeners) fn(); });
}

export function setState(patch) {
  state = typeof patch === 'function' ? { ...state, ...patch(state) } : { ...state, ...patch };
  emit();
}

export function getState() { return state; }

function subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); }

export function useField() {
  return useSyncExternalStore(subscribe, getState, getState);
}

const MAX_EVENT_TAIL = 600;

export async function bootstrap() {
  on('status', ({ connected }) => setState({ connected }));

  on('snapshot', (snap) => setState({ snap }));

  on('event', (evt) => {
    const tail = state.events.length >= MAX_EVENT_TAIL
      ? state.events.slice(state.events.length - MAX_EVENT_TAIL + 1)
      : state.events.slice();
    tail.push(evt);

    // Route pulses are driven by real events, never by a timer.
    const sid = evt.data?.sessionId ?? (evt.subject || null);
    const stamps = sid ? { ...state.lastEventBySession, [sid]: evt.ts } : state.lastEventBySession;
    setState({ events: tail, lastEventBySession: stamps });
    // Agent definitions live in config, not the snapshot: re-read them when one changes.
    if (typeof evt.kind === 'string' && evt.kind.startsWith('agent.')) refreshConfig();
  });

  connect();

  try {
    const [config, snap] = await Promise.all([api.config(), api.state()]);
    setState({ config, snap });
  } catch (e) {
    setState({ error: e.message });
  }
}

/** Re-read `/api/config` (roles, agents, endpoints); resolves to the new config. */
export async function refreshConfig() {
  const config = await api.config();
  setState({ config });
  return config;
}

// ---------------------------------------------------------------- selection

export function selectOnly(ids) { setState({ selection: [...new Set(ids)] }); }

export function toggleSelection(id) {
  const has = state.selection.includes(id);
  setState({ selection: has ? state.selection.filter((x) => x !== id) : [...state.selection, id] });
}

export function addSelection(ids) {
  setState({ selection: [...new Set([...state.selection, ...ids])] });
}

export function clearSelection() { setState({ selection: [] }); }

export function setMode(mode) { setState({ mode: normalizeMode(mode) }); }

// ---------------------------------------------------------------- going to Rome

const dirOf = (path) => {
  const clean = typeof path === 'string' ? path.replaceAll('\\', '/').replace(/^\/+|\/+$/g, '') : '';
  return clean.includes('/') ? clean.slice(0, clean.lastIndexOf('/')) : '';
};

let romeSeq = 0;

/** Ask Rome to open something. It is the only destination that shows a place. */
export function goToRome(request = {}) {
  romeSeq += 1;
  setState({ mode: 'rome', rome: { seq: romeSeq, ...request } });
}

/** Rome has handled the request; stop replaying it on every render. */
export function romeRequestHandled(seq) {
  if (state.rome?.seq === seq) setState({ rome: null });
}

export function selectProject(workspaceId, { open = false, path = null, view = null } = {}) {
  if (!workspaceId) return;
  setState({
    activeWorkspaceId: workspaceId,
    focus: { type: 'workspace', workspaceId, ...(path ? { path } : {}), ...(view ? { view } : {}) },
  });
  if (open) {
    goToRome({
      workspaceId, dir: path ? dirOf(path) : '', expanded: true,
      ...(path ? { path } : {}), ...(view ? { view } : {}),
    });
  }
}

export function selectAgent(sessionId, { open = false } = {}) {
  if (!sessionId) return;
  const session = state.snap.sessions.find((item) => item.id === sessionId);
  setState({
    activeSessionId: sessionId,
    activeWorkspaceId: session?.workspaceId ?? state.activeWorkspaceId,
    activeCampaignId: session?.campaignId ?? state.activeCampaignId,
    selection: [sessionId],
    focus: { type: 'session', id: sessionId, workspaceId: session?.workspaceId ?? null },
  });
  if (open && session?.workspaceId) {
    goToRome({ workspaceId: session.workspaceId, dir: dirOf(session.focusPath) || session.focusDir || '', sessionId });
  }
}

export function clearActiveAgent() { setState({ activeSessionId: null }); }

export function selectCampaign(campaignId, { open = false } = {}) {
  if (!campaignId) return;
  const campaign = state.snap.campaigns.find((item) => item.id === campaignId);
  const workspaceId = campaign?.target?.workspaceId ?? campaign?.target?.id ?? state.activeWorkspaceId;
  setState({ activeCampaignId: campaignId, activeWorkspaceId: workspaceId });
  if (open) goToRome({ plans: true, campaignId, workspaceId });
}

/** Open the project's files — Rome, at its root, expanded. */
export function openCity(workspaceId, path = null) { selectProject(workspaceId, { open: true, path }); }

/** Open the project's change review (the git Changes list) in Rome's expanded folder. */
export function openChanges(workspaceId) { selectProject(workspaceId, { open: true, view: 'changes' }); }

/** Open the plans overlay on Rome, on one plan or one agent when given. */
export function openPlans(sessionId = null, campaignId = null) {
  if (sessionId) {
    selectAgent(sessionId);
    const session = state.snap.sessions.find((item) => item.id === sessionId);
    goToRome({ plans: true, campaignId: session?.campaignId ?? campaignId ?? null, sessionId });
    return;
  }
  goToRome({ plans: true, campaignId });
}

/**
 * "Open this over there", from a context menu, a card or a conversation. There is one
 * "over there" now: Rome, at the folder that holds it, with the files expanded.
 */
export function openInWorkspace(focus) {
  if (focus?.type === 'session') {
    const session = state.snap.sessions.find((item) => item.id === focus.id);
    setState({
      focus,
      activeSessionId: focus.id,
      activeWorkspaceId: session?.workspaceId ?? state.activeWorkspaceId,
      selection: [focus.id],
    });
    goToRome({
      workspaceId: session?.workspaceId ?? state.activeWorkspaceId,
      dir: dirOf(session?.focusPath) || session?.focusDir || '',
      sessionId: focus.id,
    });
    return;
  }
  if (focus?.type === 'browser') {
    setState({ focus });
    goToRome({ workspaceId: state.activeWorkspaceId, dir: '', expanded: true, pane: 'browser', url: focus.url });
    return;
  }
  const dir = focus?.type === 'folder' ? (focus.path ?? '') : '';
  setState({ focus, activeWorkspaceId: focus?.workspaceId ?? state.activeWorkspaceId });
  goToRome({ workspaceId: focus?.workspaceId ?? state.activeWorkspaceId, dir, expanded: true });
}

export function selectedSessions() {
  const byId = new Map(state.snap.sessions.map((s) => [s.id, s]));
  return state.selection.map((id) => byId.get(id)).filter(Boolean);
}
