import { useEffect, useMemo, useRef, useState } from 'react';
import { api } from '../net/client.js';
import { setMode, useField } from '../state/store.js';
import VerdictLadder from '../ui/VerdictLadder.jsx';
import ReplayRail from '../ui/ReplayRail.jsx';
import { plainState } from '../ui/WorkCard.jsx';
import EmptyState from '../ui/EmptyState.jsx';

// Traces replay the real event log. The slider does not simulate anything: it folds the
// same events the Field folded live, up to the chosen point.
export default function TracesMode() {
  const st = useField();
  const [subject, setSubject] = useState(null);
  const [events, setEvents] = useState([]);
  const [pos, setPos] = useState(1);
  const [loading, setLoading] = useState(false);
  const [nextFrom, setNextFrom] = useState(null);
  const requestGeneration = useRef(0);

  const subjects = useMemo(() => {
    const rows = st.snap.sessions.map((s) => ({
      id: s.id, kind: 'session', title: s.name ?? s.id.slice(0, 8),
      sub: `${s.role} · ${s.state} · ${s.toolCount} tools · $${(s.costUsd ?? 0).toFixed(4)}`,
      state: s.state, ts: s.startedAt,
    }));
    const assigns = st.snap.assignments.map((a) => ({
      id: a.id, kind: 'assignment', title: a.targetLabel ?? a.targetId,
      sub: `${a.targetType} · ${a.sessionIds.length} agent(s) · ${a.status}`,
      state: a.status, ts: a.createdAt,
    }));
    return [...rows, ...assigns].sort((x, y) => (y.ts ?? 0) - (x.ts ?? 0));
  }, [st.snap.sessions, st.snap.assignments]);

  useEffect(() => {
    const generation = ++requestGeneration.current;
    if (!subject) return;
    setLoading(true);
    api.trace(subject, 0, 2000)
      .then((r) => { if (generation !== requestGeneration.current) return; setEvents(r.events); setNextFrom(r.nextFrom); setPos(r.events.length || 1); })
      .catch(() => { if (generation === requestGeneration.current) setEvents([]); })
      .finally(() => setLoading(false));
  }, [subject]);

  async function loadMore() {
    if (nextFrom == null || loading) return;
    const generation = requestGeneration.current;
    const activeSubject = subject;
    setLoading(true);
    try {
      const r = await api.trace(activeSubject, nextFrom, 2000);
      if (generation !== requestGeneration.current) return;
      setEvents((current) => [...current, ...r.events]);
      setNextFrom(r.nextFrom);
    } finally { setLoading(false); }
  }

  const visible = events.slice(0, pos);
  const folded = useMemo(() => fold(visible), [visible]);

  return (
    <div className="trace-layout">
      <div className="trace-list">
        {subjects.length === 0 && (
          <EmptyState
            title="Nothing has run yet"
            action={<button type="button" className="btn" onClick={() => setMode('theater')}>Go to the Board</button>}
          >Start an agent and its full event history lands here, replayable in order.</EmptyState>
        )}
        {subjects.map((s) => (
          <button
            key={s.id}
            className={`trace-item${subject === s.id ? ' on' : ''}`}
            onClick={() => setSubject(s.id)}
            type="button"
          >
            <div className="t1">
              <span className={`dot tone-${plainState({ state: s.state }).tone}`} />
              {s.title}
              <span className="label" style={{ marginLeft: 'auto' }}>{s.kind}</span>
            </div>
            <div className="t2">{s.sub}</div>
          </button>
        ))}
      </div>

      <div className="trace-main">
        {!subject ? (
          <EmptyState title="Pick a session on the left">
            Every event it produced is replayable in order: drag the scrubber to move through it.
          </EmptyState>
        ) : (
          <>
            <ReplayRail
              min={1}
              max={Math.max(1, events.length)}
              value={pos}
              onChange={setPos}
              detail={visible.length ? new Date(visible[visible.length - 1].ts).toLocaleTimeString() : '—'}
            />

            <div className="replay-stats">
              <Stat k="state" v={folded.state} />
              <Stat k="tools" v={folded.tools} />
              <Stat k="edits" v={folded.edits} />
              <Stat k="files" v={folded.files.size} />
              <Stat k="cost" v={`$${folded.cost.toFixed(4)}`} />
              <Stat k="approvals" v={`${folded.approved}/${folded.requested}`} />
              <Stat k="verified" v={folded.verified ?? '—'} />
              {folded.budget != null && <Stat k="budget" v={`${folded.budget.toFixed(2)}${folded.exhausted ? ' · exhausted' : ''}`} />}
            </div>
            {folded.verdict && (
              <div className="trace-verdict">
                <VerdictLadder verdict={folded.verdict} />
              </div>
            )}

            <div className="trace-events">
              {loading && <EmptyState status title="Loading the trace…">Folding the event log up to the point you chose.</EmptyState>}
              {visible.map((e) => (
                <div className={`evt ${e.kind.replace('.', '-')}`} key={e.seq}>
                  <span className="t">{new Date(e.ts).toLocaleTimeString()}</span>
                  <span className="k">{e.kind}</span>
                  <span className="d" title={JSON.stringify(e.data)}>{describe(e)}</span>
                </div>
              ))}
              {nextFrom != null && <button className="btn" type="button" onClick={loadMore} disabled={loading}>Load more events</button>}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function Stat({ k, v }) {
  return (
    <div className="replay-stat">
      <span className="label">{k}</span>
      <span className="mono">{v}</span>
    </div>
  );
}

function fold(events) {
  const out = {
    state: '—', tools: 0, edits: 0, files: new Set(),
    cost: 0, requested: 0, approved: 0, verified: null,
    budget: null, exhausted: false, verdict: null,
  };
  for (const e of events) {
    const d = e.data ?? {};
    switch (e.kind) {
      case 'session.state': out.state = d.state; break;
      case 'session.spawned': out.state = 'spawning'; break;
      case 'session.tool_use':
        out.tools += 1;
        if (['Edit', 'Write', 'NotebookEdit'].includes(d.name)) out.edits += 1;
        if (d.path) out.files.add(d.path);
        break;
      case 'session.usage': if (typeof d.costUsd === 'number') out.cost = d.costUsd; break;
      case 'permission.requested': out.requested += 1; break;
      case 'permission.decided': if (d.decision === 'allow') out.approved += 1; break;
      case 'work.verified': out.verified = d.result; break;
      case 'budget.reserved': if (typeof d.limitUsd === 'number') out.budget = d.limitUsd; break;
      case 'budget.exhausted': out.exhausted = true; break;
      case 'budget.reactivated': out.exhausted = false; break;
      case 'session.verification':
        out.verdict = { ...d, ts: e.ts, residualRisk: [], recovery: [], turnSettled: false };
        break;
      case 'session.turn_complete':
        if (out.verdict && !out.verdict.turnSettled) {
          out.verdict = {
            ...out.verdict,
            residualRisk: Array.isArray(d.residualRisk) ? d.residualRisk : [],
            recovery: Array.isArray(d.recovery) ? d.recovery : [],
            turnSettled: true,
          };
        }
        break;
      case 'session.ended': out.state = d.reason === 'error' ? 'error' : 'ended'; break;
      default: break;
    }
  }
  return out;
}

function describe(e) {
  const d = e.data ?? {};
  switch (e.kind) {
    case 'session.spawned': return `${d.name} · ${d.role} · ${d.model} · effort ${d.thinking} (${d.thinkingReason ?? ''})`;
    case 'session.state': return `${d.state}${d.detail ? ` — ${d.detail}` : ''}`;
    case 'session.message': return `${d.role}: ${(d.text ?? '').slice(0, 120)}`;
    case 'session.thinking': return (d.text ?? '').slice(0, 120);
    case 'session.tool_use': return d.summary ?? d.name;
    case 'session.tool_result': return `${d.ok === false ? 'FAILED — ' : ''}${(d.preview ?? '').slice(0, 110)}`;
    case 'session.usage': return `in ${d.inputTokens ?? 0} out ${d.outputTokens ?? 0} cache ${d.cacheRead ?? 0}${d.costUsd != null ? ` · $${d.costUsd.toFixed(4)}` : ''}`;
    case 'session.ended': return `${d.reason}${d.error ? ` — ${d.error}` : ''}`;
    case 'session.delegated': return `→ ${d.subagentType}: ${d.description}`;
    case 'assignment.created': return `${d.sessionIds.length} agent(s) → ${d.targetType} ${d.targetLabel}`;
    case 'fs.changed': return `${d.change} ${d.path}`;
    case 'git.status': return `${d.branch} · ${d.files?.length ?? 0} changed`;
    case 'endpoint.health': return `${d.endpointId} ${d.status}${d.latencyMs ? ` ${d.latencyMs}ms` : ''} — ${d.detail ?? ''}`;
    case 'endpoint.routed': return `→ ${d.endpointId} (${d.reason ?? ''})`;
    case 'permission.requested': return `${d.toolName}`;
    case 'permission.decided': return `${d.decision} by ${d.by}`;
    case 'browser.navigated': return d.url;
    case 'work.verified': return `${d.result}`;
    case 'session.verification': {
      const tiers = Array.isArray(d.tiers) ? d.tiers : [];
      const ladder = tiers.map((t) => `${t.label ?? `tier ${t.tier}`}:${t.skipped ? 'skip' : t.forgiven ? 'forgiven' : t.passed ? 'pass' : 'FAIL'}`).join(' ');
      return `${d.passed ? 'verified' : 'not verified'}${d.reachedTier != null ? ` · reached tier ${d.reachedTier}` : ''}${ladder ? ` · ${ladder}` : ''}`;
    }
    case 'budget.reserved': return `${Number(d.limitUsd ?? 0).toFixed(2)} reserved`;
    case 'budget.exhausted': return `${String(d.budget ?? 'budget').replaceAll('_', ' ')} exhausted (${d.used} of ${d.limit})`;
    case 'git.committed': return `${String(d.revision ?? '').slice(0, 7)} ${d.message ?? ''} · ${(d.paths ?? []).length} file(s)`;
    case 'git.reverted': return `reverted ${(d.paths ?? []).join(', ').slice(0, 100)}`;
    case 'command.issued': return `${d.kind} × ${(d.sessionIds ?? []).length}`;
    case 'routine.triggered': return `${d.routineId} — ${d.reason ?? 'manual'}`;
    case 'terminal.run': return d.command;
    default: return JSON.stringify(d).slice(0, 120);
  }
}
