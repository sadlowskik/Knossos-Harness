import { useEffect } from 'react';
import { openCity, openInWorkspace, useField } from '../state/store.js';
import { ColumnTranscript } from '../ui/Transcript.jsx';
import { StatusPill, plainState } from '../ui/WorkCard.jsx';
import AgentControls from '../hud/AgentControls.jsx';
import VerdictLadder from '../ui/VerdictLadder.jsx';

// "$spent / $budget · $left" when a reservation exists, plain spend otherwise.
export function budgetLine(session) {
  const spent = `${(session.costUsd ?? 0).toFixed(4)}`;
  if (!session.budgetUsd) return spent;
  const budget = `${Number(session.budgetUsd).toFixed(2)}`;
  if (session.budgetExhausted) return `${spent} / ${budget} · exhausted`;
  const left = session.budgetRemainingUsd != null ? ` · ${Number(session.budgetRemainingUsd).toFixed(2)} left` : '';
  return `${spent} / ${budget}${left}`;
}

// A selected unit's panel on the Map: live transcript (reusing Atlas's ColumnTranscript
// fold), where it is working, and the shared talk / pause / stop controls. It overlays the
// Field and never navigates away, so the operator keeps watching while they talk.
export default function UnitInspector({ session, onClose }) {
  const st = useField();

  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  if (!session) return null;

  const workspace = st.snap.workspaces.find((w) => w.id === session.workspaceId);
  const where = session.focusPath ?? session.focusDir ?? null;
  const endpoint = st.snap.endpoints.find((e) => e.id === session.endpointId);
  const elapsed = session.startedAt ? Math.max(0, Math.round(((st.snap.now ?? Date.now()) - session.startedAt) / 60000)) : null;

  return (
    <aside
      className="unit-inspector"
      role="dialog"
      aria-label={`${session.name ?? session.id} details`}
      onPointerDown={(e) => e.stopPropagation()}
      onClick={(e) => e.stopPropagation()}
    >
      <header className="unit-inspector-head">
        <b className="unit-inspector-name">{session.name ?? session.id}</b>
        <StatusPill compact state={plainState(session)} />
        <button type="button" className="unit-inspector-close" onClick={onClose} aria-label="Close agent details">×</button>
      </header>

      <dl className="unit-inspector-facts mono">
        <div><dt>role</dt><dd>{session.role ?? 'agent'}</dd></div>
        <div><dt>model</dt><dd>{endpoint?.name ?? session.endpointId ?? '—'}{session.model ? ` · ${session.model}` : ''}</dd></div>
        <div><dt>project</dt><dd>{workspace?.name ?? session.workspaceId ?? 'staging'}</dd></div>
        {where && <div><dt>file</dt><dd title={where}>{where}</dd></div>}
        {session.lastTool?.name && <div><dt>tool</dt><dd title={session.lastTool.summary}>{session.lastTool.name}</dd></div>}
        <div className={session.budgetExhausted ? 'budget-exhausted' : ''}>
          <dt>{session.budgetExhausted ? 'budget' : 'cost'}</dt>
          <dd title={session.budgetExhausted ? `${String(session.budgetExhaustedReason ?? 'budget').replaceAll('_', ' ')} exhausted; the agent is paused` : 'spent / budget'}>
            {budgetLine(session)}
          </dd>
        </div>
        <div><dt>context</dt><dd>{session.contextPct ?? 0}%</dd></div>
        {elapsed != null && <div><dt>elapsed</dt><dd>{elapsed < 1 ? '<1m' : `${elapsed}m`}</dd></div>}
        {session.verified && session.verified !== 'unverified' && <div><dt>verified</dt><dd>{session.verified}</dd></div>}
      </dl>
      {session.budgetExhausted && (
        <p className="unit-inspector-error" role="alert">
          {String(session.budgetExhaustedReason ?? 'budget').replaceAll('_', ' ')} budget exhausted — the agent is paused until you raise it or stop it.
        </p>
      )}
      {session.error && <p className="unit-inspector-error" role="alert">{session.error}</p>}
      {session.lastVerdict && <div className="unit-inspector-verdict"><VerdictLadder verdict={session.lastVerdict} /></div>}

      <div className="unit-inspector-feed">
        <ColumnTranscript session={session} campaigns={st.snap.campaigns ?? []} />
      </div>

      <div className="unit-inspector-foot">
        <AgentControls
          session={session}
          onOpen={() => { openInWorkspace({ type: 'session', id: session.id }); onClose(); }}
        />
        {workspace?.mounted && (
          <button type="button" className="btn sm ghost" onClick={() => { openCity(session.workspaceId); onClose(); }}>Open project files</button>
        )}
      </div>
    </aside>
  );
}
