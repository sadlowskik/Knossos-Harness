import { api } from '../net/client.js';
import { openInWorkspace, selectOnly, useField } from '../state/store.js';

const STATE_LABEL = {
  spawning: 'spawning', ready: 'ready', idle: 'idle', thinking: 'thinking',
  working: 'working', waiting_permission: 'awaiting approval', blocked: 'blocked',
  error: 'error', interrupted: 'interrupted', paused: 'paused', done: 'done',
  cancelled: 'cancelled',
};

export default function SelectionHUD() {
  const st = useField();
  const selected = st.snap.sessions.filter((s) => st.selection.includes(s.id));
  if (!selected.length) return null;

  const cmd = (kind, extra) =>
    api.command(kind, { sessionIds: st.selection, ...extra }).catch((e) => console.error(e));

  const single = selected.length === 1 ? selected[0] : null;
  const endpointOf = (id) => st.snap.endpoints.find((e) => e.id === id);

  return (
    <div className="hud">
      <div className="hud-head">
        <span className="label">
          {selected.length === 1 ? 'agent' : `${selected.length} agents selected`}
        </span>
        {single && <span className="hud-name">{single.name}</span>}
      </div>

      {single ? (
        <div className="hud-body">
          <Row k="role" v={single.role} />
          <Row k="state" v={STATE_LABEL[single.state] ?? single.state} accent={single.state === 'working'} />
          <Row k="endpoint" v={`${endpointOf(single.endpointId)?.name ?? single.endpointId ?? '—'}`} />
          <Row k="model" v={single.model ?? '—'} mono />
          <Row k="thinking" v={single.thinking ?? '—'} />
          <Row k="target" v={single.target?.label ?? single.focusDir ?? '—'} mono />
          <Row k="context" v={`${single.contextPct}%`} bar={single.contextPct / 100} />
          <Row k="cost" v={`$${(single.costUsd ?? 0).toFixed(4)}`} mono />
          <Row k="tools" v={`${single.toolCount} · ${single.editCount} edits`} mono />
          <Row k="verified" v={single.verified} accent={single.verified === 'verified'} />
          {single.children?.length > 0 && <Row k="children" v={String(single.children.length)} />}
          {single.delegations?.length > 0 && (
            <Row
              k="delegated"
              v={single.delegations.map((d) => d.type ?? 'subagent').join(', ')}
            />
          )}
          {single.lastTool && (
            <div className="hud-last">
              <span className="label">last action</span>
              <div className="mono hud-last-text">{single.lastTool.summary}</div>
            </div>
          )}
          {single.lastSay && (
            <div className="hud-last">
              <span className="label">last message</span>
              <div className="hud-say">{single.lastSay.slice(0, 220)}</div>
            </div>
          )}
          {single.error && <div className="hud-err">{single.error}</div>}
        </div>
      ) : (
        <div className="hud-list">
          {selected.map((s) => (
            <button key={s.id} className="hud-row" onClick={() => selectOnly([s.id])} type="button">
              <span className={`dot ${s.state}`} />
              <span className="hud-row-name">{s.name}</span>
              <span className="hud-row-role label">{s.role}</span>
              <span className="mono hud-row-ctx">{s.contextPct}%</span>
            </button>
          ))}
          <div className="hud-total mono">
            total ${selected.reduce((m, s) => m + (s.costUsd ?? 0), 0).toFixed(4)}
          </div>
        </div>
      )}

      <div className="hud-actions">
        <button className="btn" onClick={() => cmd('pause')} type="button">Pause</button>
        <button className="btn" onClick={() => cmd('resume')} type="button">Resume</button>
        <button className="btn" onClick={() => cmd('verify')} type="button">Verify</button>
        <button className="btn" onClick={() => cmd('escalate')} type="button">Escalate</button>
        <button className="btn danger" onClick={() => cmd('cancel')} type="button">Cancel</button>
        {single && (
          <button
            className="btn primary"
            type="button"
            onClick={() => openInWorkspace({ type: 'session', id: single.id })}
          >Open</button>
        )}
      </div>
    </div>
  );
}

function Row({ k, v, mono, accent, bar }) {
  return (
    <div className="hud-kv">
      <span className="label">{k}</span>
      <span className={`hud-v${mono ? ' mono' : ''}${accent ? ' accent' : ''}`}>{v}</span>
      {bar != null && (
        <div className="hud-bar"><i style={{ width: `${Math.min(100, bar * 100)}%` }} /></div>
      )}
    </div>
  );
}
