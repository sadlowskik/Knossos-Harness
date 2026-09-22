/* The one transcript. Workspace mode, the Board's conversations and the Map's unit
   inspector used to each fetch `/api/trace`, subscribe to the websocket and write their
   own row markup. This is that renderer, once: a hook for the events, a hook for the
   "only autoscroll when the reader is already at the bottom" rule, and two row styles —
   `full` (the Workspace transcript) and `compact` (the Map's fold). */

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { api, on } from '../net/client.js';
import VerdictLadder from './VerdictLadder.jsx';

/** The event kinds a transcript shows. Everything else is Traces' business. */
export const TRANSCRIPT_KINDS = new Set([
  'session.spawned', 'session.message', 'session.thinking', 'session.tool_use',
  'session.tool_result', 'session.ended', 'permission.requested', 'permission.decided',
  'work.verified', 'session.verification',
]);

/** One session's events: the stored trace, then every live event for it. */
export function useSessionTrace(sessionId, limit = 2000) {
  const [events, setEvents] = useState([]);
  const [error, setError] = useState(null);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    if (!sessionId) { setEvents([]); setError(null); setLoaded(true); return undefined; }
    let alive = true;
    setLoaded(false);
    api.trace(sessionId, 0, limit)
      .then((result) => { if (alive) { setEvents(result.events ?? []); setError(null); setLoaded(true); } })
      .catch((e) => { if (alive) { setEvents([]); setError(e.message); setLoaded(true); } });
    return () => { alive = false; };
  }, [sessionId, limit]);

  useEffect(() => {
    if (!sessionId) return undefined;
    return on('event', (evt) => {
      if (evt.subject !== sessionId && evt.data?.sessionId !== sessionId) return;
      setEvents((prev) => (prev.some((item) => item.seq === evt.seq) ? prev : [...prev, evt]));
    });
  }, [sessionId]);

  return { events, error, loaded };
}

/* Autoscroll that respects the reader. A feed that yanks itself to the bottom while you
   are reading three screens up is a feed you cannot read; this follows the tail only
   while the tail is what you are looking at. */
export function useStickyScroll(key, resetKey = null) {
  const ref = useRef(null);
  const pinnedRef = useRef(true);
  const [pinned, setPinned] = useState(true);

  const onScroll = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
    if (atBottom !== pinnedRef.current) { pinnedRef.current = atBottom; setPinned(atBottom); }
  }, []);

  const scrollToEnd = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
    pinnedRef.current = true;
    setPinned(true);
  }, []);

  // A different session is a different conversation: start at its tail.
  useLayoutEffect(() => { pinnedRef.current = true; setPinned(true); }, [resetKey]);
  useLayoutEffect(() => {
    if (!pinnedRef.current) return;
    const el = ref.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [key, resetKey]);

  return { ref, onScroll, pinned, scrollToEnd };
}

/** The objective / state line above a transcript. */
export function TranscriptBrief({ session, campaigns = [] }) {
  const objective = campaigns
    .flatMap((campaign) => campaign.objectives ?? [])
    .find((item) => item.id === session?.objectiveId);
  return (
    <div className="agent-brief">
      <span className="label">current objective</span>
      <b>{objective?.statement ?? session?.target?.label ?? session?.target?.id ?? 'Awaiting assignment'}</b>
      <span className="mono">
        {session?.state?.replaceAll('_', ' ') ?? 'unknown'}
        {objective?.status ? ` · ${objective.status}` : ''}
        {session?.costUsd != null ? ` · $${Number(session.costUsd).toFixed(4)}` : ''}
      </span>
      {objective?.progress?.total ? (
        <progress max={objective.progress.total} value={objective.progress.done ?? 0} />
      ) : null}
    </div>
  );
}

/* ---------------------------------------------------------------- rows: full */

export function TranscriptRow({ evt }) {
  const d = evt.data ?? {};
  switch (evt.kind) {
    case 'session.spawned':
      return (
        <div className="tr prompt">
          <div className="who">initial prompt</div>
          <details>
            <summary>{d.initialOrders ?? 'Open prompt'}</summary>
            <pre className="body">{d.systemPrompt}</pre>
          </details>
        </div>
      );
    case 'session.message':
      return (
        <div className={`tr ${d.role}`}>
          <div className="who">{d.role}</div>
          <div className="body">{d.text}</div>
        </div>
      );
    case 'session.thinking':
      return (
        <div className="tr tool">
          <div className="who">thinking</div>
          <div className="body thinking">{d.text?.slice(0, 600)}</div>
        </div>
      );
    case 'session.tool_use':
      return (
        <div className="tr tool">
          <div className="who">{d.name}</div>
          <div className="body">{d.summary}</div>
        </div>
      );
    case 'session.tool_result':
      return (
        <div className={`tr ${d.ok === false ? 'error' : 'tool'}`}>
          <div className="who">{d.ok === false ? 'tool failed' : 'result'}</div>
          <div className="body">{(d.preview ?? '').slice(0, 400)}</div>
        </div>
      );
    case 'permission.requested':
      return (
        <div className="tr tool">
          <div className="who">approval requested</div>
          <div className="body">{d.toolName}</div>
        </div>
      );
    case 'permission.decided':
      return (
        <div className="tr tool">
          <div className="who">approval {d.decision}</div>
          <div className="body">by {d.by}</div>
        </div>
      );
    case 'work.verified':
      return (
        <div className="tr tool">
          <div className="who">verified</div>
          <div className="body">{d.result}{d.tier ? ` · ${d.tier}` : ''}{d.summary ? ` — ${d.summary}` : ''}</div>
        </div>
      );
    case 'session.verification':
      return (
        <div className="tr tool">
          <div className="who">verdict</div>
          <div className="body"><VerdictLadder verdict={{ ...d, ts: evt.ts }} compact /></div>
        </div>
      );
    case 'session.ended':
      return (
        <div className={`tr ${d.reason === 'error' ? 'error' : 'user'}`}>
          <div className="who">session {d.reason}</div>
          {d.error && <div className="body">{d.error}</div>}
        </div>
      );
    default:
      return null;
  }
}

/* -------------------------------------------------------------- rows: compact */

export function TraceLine({ evt }) {
  const data = evt.data ?? {};
  if (evt.kind === 'session.message') {
    const role = data.role ?? 'note';
    return (
      <p className={`atlas-line ${role}`}>
        <span className="atlas-role">{role === 'assistant' ? 'agent' : role}</span>
        <span className="atlas-text">{String(data.text ?? '').slice(0, 500)}</span>
      </p>
    );
  }
  if (evt.kind === 'session.tool_use') {
    return (
      <p className="atlas-line tool">
        <span className="atlas-role">tool</span>
        <span className="atlas-text"><code>{data.name}</code>{data.summary ? <> {data.summary}</> : null}</span>
      </p>
    );
  }
  if (evt.kind === 'session.tool_result') {
    const failed = data.ok === false;
    return (
      <p className={`atlas-line ${failed ? 'error' : 'result'}`}>
        <span className="atlas-role">{failed ? 'failed' : 'result'}</span>
        <span className="atlas-text mono">{String(data.preview ?? '').slice(0, 240)}</span>
      </p>
    );
  }
  if (evt.kind === 'permission.requested') {
    return (
      <p className="atlas-line approval">
        <span className="atlas-role">approval</span>
        <span className="atlas-text">Asked to run <code>{data.toolName}</code></span>
      </p>
    );
  }
  if (evt.kind === 'session.ended') {
    return (
      <p className={`atlas-line ${data.reason === 'error' ? 'error' : 'ended'}`}>
        <span className="atlas-role">ended</span>
        <span className="atlas-text">{data.reason}{data.error ? ` · ${data.error}` : ''}</span>
      </p>
    );
  }
  return null;
}

/* ------------------------------------------------------------------ the feed */

/**
 * A session's transcript, scrolling itself. `variant` picks the row style; the container
 * autoscrolls only while the reader is at the bottom, and offers to jump back when not.
 */
export default function SessionTranscript({
  session,
  variant = 'full',
  campaigns = null,
  limit = 2000,
  max = 0,
  brief = false,
  connected = true,
  className = '',
  emptyText = 'Nothing reported yet. Messages and tool calls appear here as the agent works.',
}) {
  const sessionId = session?.id ?? null;
  const { events, error, loaded } = useSessionTrace(sessionId, limit);
  const rows = events.filter((evt) => TRANSCRIPT_KINDS.has(evt.kind));
  const shown = max > 0 ? rows.slice(-max) : rows;
  const { ref, onScroll, pinned, scrollToEnd } = useStickyScroll(shown.length, sessionId);
  const Row = variant === 'compact' ? TraceLine : TranscriptRow;

  return (
    <div className={`convo-feed-wrap${className ? ` ${className}` : ''}`}>
      <div className={`convo-feed ${variant}`} ref={ref} onScroll={onScroll}>
        {brief && session && <TranscriptBrief session={session} campaigns={campaigns ?? []} />}
        {error && (
          <p className="convo-feed-note bad" role="alert">
            The transcript could not be read: {error}
          </p>
        )}
        {!error && !connected && (
          <p className="convo-feed-note bad" role="status">
            Not connected to the Field server. This is the last state that reached the browser.
          </p>
        )}
        {shown.length > 0
          ? shown.map((evt) => <Row key={evt.seq} evt={evt} />)
          : (!error && loaded && <p className="convo-feed-empty">{emptyText}</p>)}
      </div>
      {!pinned && shown.length > 0 && (
        <button type="button" className="convo-jump" onClick={scrollToEnd}>Jump to the latest</button>
      )}
    </div>
  );
}

/** The Map's fold: the compact feed under an objective line. */
export function ColumnTranscript({ session, campaigns = [] }) {
  const objective = (campaigns ?? [])
    .flatMap((campaign) => campaign.objectives ?? [])
    .find((item) => item.id === session?.objectiveId);
  return (
    <div className="atlas-trace">
      <p className="atlas-objective">
        {objective?.statement ?? session?.target?.label ?? session?.stateDetail ?? 'Waiting for an assignment'}
      </p>
      <SessionTranscript session={session} variant="compact" limit={400} max={80} className="atlas-feed-wrap" />
    </div>
  );
}
