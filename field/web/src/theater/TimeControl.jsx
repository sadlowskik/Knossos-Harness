/* Rome's time control.

   History used to be a destination: a list of subjects on the left, an event log on the
   right and a scrubber over it. But a scrubber over a list is a list; the question it was
   really answering — where was everyone an hour ago, and which corners had already gone
   quiet — is a question about the map. So the scrubber is on the map, along its bottom
   edge. Live is the right edge. Drag left and Rome redraws from the same event log the
   server folded live: agents stand in the folders they were in, districts carry the
   staleness they had, and one click on "Live" leaves it.

   Nothing here simulates. It folds `GET /api/events` with the same rules the server's
   projection uses, up to the event you stopped on. */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api, on } from '../net/client.js';
import ReplayRail from '../ui/ReplayRail.jsx';

const MAX_EVENTS = 4000;

/**
 * The world as of one event: sessions where they stood, folders with the timestamps they
 * had then, files changed by then. Shaped exactly like the snapshot's own fields so the
 * map draws replay and live with the same code.
 */
export function foldWorld(events) {
  const sessions = new Map();
  const folders = new Map();
  const files = new Map();
  let at = 0;

  const session = (id) => {
    if (!sessions.has(id)) {
      sessions.set(id, {
        id, name: id.slice(0, 6), role: 'builder', model: null, endpointId: null,
        workspaceId: null, state: 'spawning', focusDir: '', focusPath: null,
        lastTool: null, toolCount: 0, editCount: 0, costUsd: 0, contextPct: 0,
        startedAt: 0, progress: null, campaignId: null, verified: 'unverified',
      });
    }
    return sessions.get(id);
  };
  const touchFolder = (workspaceId, dir, ts, sessionId) => {
    if (!workspaceId || dir == null) return;
    const key = `${workspaceId}:${dir}`;
    if (!folders.has(key)) folders.set(key, { key, workspaceId, dir, hits: 0, lastTs: 0, agents: new Set() });
    const folder = folders.get(key);
    folder.hits += 1;
    folder.lastTs = Math.max(folder.lastTs, ts);
    if (sessionId) folder.agents.add(sessionId);
  };

  for (const evt of events) {
    const d = evt.data ?? {};
    at = Math.max(at, evt.ts ?? 0);
    switch (evt.kind) {
      case 'session.spawned': {
        const s = session(d.sessionId);
        Object.assign(s, {
          agentId: d.agentId ?? null, name: d.name ?? s.name, role: d.role ?? s.role,
          model: d.model ?? null, endpointId: d.endpointId ?? null,
          workspaceId: d.workspaceId ?? null, state: 'spawning', startedAt: evt.ts,
          target: d.target ?? null, campaignId: d.campaignId ?? null,
        });
        break;
      }
      case 'session.state': {
        const s = session(d.sessionId);
        s.state = d.state;
        if (d.detail !== undefined) s.stateDetail = d.detail;
        break;
      }
      case 'session.progress': {
        const s = session(d.sessionId);
        s.progress = { done: Math.max(0, Number(d.done) || 0), total: Math.max(0, Number(d.total) || 0) };
        break;
      }
      case 'session.tool_use': {
        const s = session(d.sessionId);
        s.toolCount += 1;
        s.lastTool = { name: d.name, summary: d.summary ?? '', ts: evt.ts };
        s.state = 'working';
        if (d.workspaceId && d.dir != null) {
          touchFolder(d.workspaceId, d.dir, evt.ts, d.sessionId);
          s.workspaceId = d.workspaceId;
          s.focusDir = d.dir;
        }
        if (d.path) {
          s.focusPath = d.path;
          if (['Edit', 'Write', 'NotebookEdit'].includes(d.name)) s.editCount += 1;
        }
        break;
      }
      case 'session.usage': {
        const s = session(d.sessionId);
        if (Number.isFinite(d.costUsd)) s.costUsd = Math.max(s.costUsd, d.costUsd);
        break;
      }
      case 'session.ended': {
        const s = session(d.sessionId);
        s.state = d.reason === 'error' ? 'error' : d.reason === 'cancelled' ? 'cancelled' : 'done';
        s.endedAt = evt.ts;
        break;
      }
      case 'fs.changed': {
        const key = `${d.workspaceId}:${d.path}`;
        files.set(key, { key, workspaceId: d.workspaceId, path: d.path, dir: d.dir, lastTs: evt.ts });
        touchFolder(d.workspaceId, d.dir, evt.ts, d.sessionId);
        break;
      }
      default: break;
    }
  }

  return {
    at,
    sessions: [...sessions.values()],
    folders: [...folders.values()].map((f) => ({ ...f, agents: [...f.agents] })),
    files: [...files.values()],
  };
}

const stamp = (ts) => (ts ? new Date(ts).toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '—');

export default function TimeControl({ onReplay }) {
  const [events, setEvents] = useState([]);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState(null);
  // null means live. Otherwise the number of events folded, 1..events.length.
  const [pos, setPos] = useState(null);
  const report = useRef(onReplay);
  report.current = onReplay;

  useEffect(() => {
    let alive = true;
    api.events(0, MAX_EVENTS)
      .then((r) => { if (alive) { setEvents(r.events ?? []); setLoaded(true); } })
      .catch((e) => { if (alive) { setError(e.message); setLoaded(true); } });
    return () => { alive = false; };
  }, []);

  // The log keeps growing while you are looking at it. Live stays live; a replay position
  // is an index into the prefix, so new events only extend the rail to the right.
  useEffect(() => on('event', (evt) => {
    setEvents((prev) => (prev.some((item) => item.seq === evt.seq)
      ? prev
      : [...prev.slice(Math.max(0, prev.length - MAX_EVENTS + 1)), evt]));
  }), []);

  const folded = useMemo(
    () => (pos == null ? null : foldWorld(events.slice(0, pos))),
    [pos, events],
  );

  useEffect(() => { report.current(folded); }, [folded]);

  const toLive = useCallback(() => setPos(null), []);

  if (!loaded) return null;
  if (error) {
    return <div className="time-rail" role="status"><span className="label">time</span><span className="mono">history unavailable — {error}</span></div>;
  }
  if (!events.length) {
    return (
      <div className="time-rail" role="status">
        <span className="label">time</span>
        <span className="time-hint">Nothing has happened yet. As agents work, this rail becomes the history of the map.</span>
      </div>
    );
  }

  const last = events.length;
  const value = pos ?? last;
  const replaying = pos != null && pos < last;

  return (
    <div className={`time-rail${replaying ? ' replaying' : ''}`}>
      {replaying && (
        <span className="time-badge" role="status">
          Replay · {stamp(folded?.at)}
        </span>
      )}
      <ReplayRail
        className="time-scrubber"
        label={replaying ? 'replay' : 'live'}
        ariaLabel="Time on the map"
        min={1}
        max={last}
        value={value}
        onChange={(next) => setPos(next >= last ? null : next)}
        onEnd={toLive}
        detail={replaying ? stamp(folded?.at) : 'now'}
      />
      {replaying && (
        <button type="button" className="btn sm time-live" onClick={toLive}>Back to live</button>
      )}
    </div>
  );
}
