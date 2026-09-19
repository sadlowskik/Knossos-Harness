// Cities — a city IS a workspace. Everything here is derived from the live projection and
// the event log; nothing is written back to field/ (operational state stays in the log).
//
// Leveling reuses the existing per-workspace `maturity` score (campaign completion,
// verification, persistence) rather than inventing a parallel XP system, so a city's rank
// reflects the same real evidence the rest of Field already trusts.

import { workspaceMaturity } from './store/projection.js';

const ACTIVE_STATES = new Set(['spawning', 'thinking', 'working', 'waiting_permission']);
export function isActiveState(state) { return ACTIVE_STATES.has(state); }

// maturity.tier is 0..4; name the ranks a civilization would recognise.
const TIER_NAMES = ['outpost', 'outpost', 'town', 'city', 'capital'];

function citySessions(projection, workspaceId) {
  return [...projection.sessions.values()].filter((s) => s.workspaceId === workspaceId);
}

function rankOf(maturity) {
  const t = maturity?.tier ?? 0;
  return { tier: TIER_NAMES[t] ?? 'outpost', level: t, score: maturity?.score ?? 0 };
}

function summariseSession(s) {
  return {
    sessionId: s.id, name: s.name, role: s.role, state: s.state,
    endpointId: s.endpointId, contextPct: s.contextPct, costUsd: s.costUsd,
    verified: s.verified, lastSay: s.lastSay ?? null, active: isActiveState(s.state),
  };
}

/** Compact card for every city — cheap enough for a snapshot summary. */
export function listCities(projection, now = Date.now()) {
  const campaigns = projection.campaigns.snapshot().campaigns;
  return [...projection.workspaces.values()].map((w) => {
    const maturity = workspaceMaturity(w, campaigns, now);
    const sessions = citySessions(projection, w.id);
    return {
      id: w.id, name: w.name, mounted: w.mounted,
      ...rankOf(maturity),
      agentCount: sessions.filter((s) => isActiveState(s.state)).length,
      sessionCount: sessions.length,
    };
  });
}

/** Full city view including roster and the chat/order feed. */
export function cityDetail(projection, log, id, now = Date.now()) {
  const w = projection.workspaces.get(id);
  if (!w) return null;
  const campaigns = projection.campaigns.snapshot().campaigns;
  const maturity = workspaceMaturity(w, campaigns, now);
  const sessions = citySessions(projection, id);
  return {
    id: w.id, name: w.name, path: w.path, mounted: w.mounted,
    ...rankOf(maturity), maturity,
    agents: sessions.map(summariseSession),
    agentCount: sessions.filter((s) => isActiveState(s.state)).length,
    feed: cityFeed(projection, log, id, 60),
  };
}

export function activeCitySessionIds(projection, workspaceId) {
  return citySessions(projection, workspaceId)
    .filter((s) => isActiveState(s.state))
    .map((s) => s.id);
}

/** Resolve a default agent for a role, so the operator can deploy without picking one. */
export function resolveCityAgent(cfg, role) {
  const want = role || cfg.defaults?.role || 'builder';
  return (cfg.agents.find((a) => a.role === want) ?? cfg.agents[0])?.id ?? null;
}

// ---- feed: a bounded tail of the event log, filtered to this city's sessions ----

const FEED_KINDS = new Set([
  'session.message', 'session.spawned', 'session.ended', 'work.verified', 'command.issued',
]);
const FEED_TAIL_EVENTS = 1500;

function feedText(kind, d) {
  switch (kind) {
    case 'session.message': return String(d.text ?? '').slice(0, 600);
    case 'session.spawned': return `deployed ${d.name ?? d.agentId ?? 'agent'}`;
    case 'session.ended': return `ended (${d.reason ?? 'done'})`;
    case 'work.verified': return `verification: ${d.result ?? '?'}`;
    case 'command.issued': return `order: ${d.kind ?? ''}${d.text ? ' — ' + String(d.text).slice(0, 200) : ''}`;
    default: return '';
  }
}

function cityFeed(projection, log, workspaceId, limit = 60) {
  const ids = new Set(citySessions(projection, workspaceId).map((s) => s.id));
  if (!ids.size) return [];
  const from = Math.max(0, (log.size ?? 0) - FEED_TAIL_EVENTS);
  const events = log.read(from, FEED_TAIL_EVENTS);
  const feed = [];
  for (const e of events) {
    if (!FEED_KINDS.has(e.kind)) continue;
    const d = e.data ?? {};
    const sid = d.sessionId ?? (Array.isArray(d.sessionIds) ? d.sessionIds[0] : null);
    if (!sid || !ids.has(sid)) continue;
    feed.push({ at: e.ts, seq: e.seq, kind: e.kind, sessionId: sid, role: d.role ?? null, text: feedText(e.kind, d) });
  }
  return feed.slice(-limit);
}
