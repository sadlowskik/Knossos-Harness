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

const MODE_IDS = new Set(['theater', 'field', 'campaigns', 'workspace', 'routines', 'traces']);
const requestedMode = typeof location === 'undefined'
  ? null
  : new URLSearchParams(location.search).get('mode');

let state = {
  connected: false,
  snap: EMPTY,
  config: null,
  events: [],            // rolling tail, newest last
  mode: MODE_IDS.has(requestedMode) ? requestedMode : 'field',
  selection: [],         // session ids
  activeWorkspaceId: null,
  activeSessionId: null,
  activeCampaignId: null,
  hover: null,
  focus: null,           // { type, id, ... } opened in Workspace mode
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
  });

  connect();

  try {
    const [config, snap] = await Promise.all([api.config(), api.state()]);
    setState({ config, snap });
  } catch (e) {
    setState({ error: e.message });
  }
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

export function setMode(mode) { setState({ mode }); }

export function selectProject(workspaceId, { open = false, path = null } = {}) {
  if (!workspaceId) return;
  setState({
    activeWorkspaceId: workspaceId,
    focus: { type: 'workspace', workspaceId, ...(path ? { path } : {}) },
    ...(open ? { mode: 'workspace' } : {}),
  });
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
    ...(open ? { mode: 'campaigns' } : {}),
  });
}

export function clearActiveAgent() { setState({ activeSessionId: null }); }

export function selectCampaign(campaignId, { open = false } = {}) {
  if (!campaignId) return;
  const campaign = state.snap.campaigns.find((item) => item.id === campaignId);
  const workspaceId = campaign?.target?.workspaceId ?? campaign?.target?.id ?? state.activeWorkspaceId;
  setState({
    activeCampaignId: campaignId,
    activeWorkspaceId: workspaceId,
    ...(open ? { mode: 'campaigns' } : {}),
  });
}

export function openCity(workspaceId, path = null) { selectProject(workspaceId, { open: true, path }); }

export function openSenate(sessionId = null, campaignId = null) {
  if (sessionId) selectAgent(sessionId, { open: true });
  else if (campaignId) selectCampaign(campaignId, { open: true });
  else setState({ mode: 'campaigns' });
}

export function openInWorkspace(focus) {
  if (focus?.type === 'session') {
    const session = state.snap.sessions.find((item) => item.id === focus.id);
    setState({
      focus,
      mode: 'workspace',
      activeSessionId: focus.id,
      activeWorkspaceId: session?.workspaceId ?? state.activeWorkspaceId,
      selection: [focus.id],
    });
    return;
  }
  setState({
    focus,
    mode: 'workspace',
    activeWorkspaceId: focus?.workspaceId ?? state.activeWorkspaceId,
  });
}

export function selectedSessions() {
  const byId = new Map(state.snap.sessions.map((s) => [s.id, s]));
  return state.selection.map((id) => byId.get(id)).filter(Boolean);
}
