import { useEffect, useRef, useState } from 'react';
import { api } from '../net/client.js';
import { getState, openInWorkspace, selectOnly, useField } from '../state/store.js';

const THINKING = ['low', 'medium', 'high', 'adaptive'];

export default function ContextMenu({ screen, target, onClose }) {
  const st = useField();
  const ref = useRef(null);
  const [pane, setPane] = useState(null);      // null | 'assign' | 'spawn'
  const [endpoint, setEndpoint] = useState('auto');
  const [thinking, setThinking] = useState('adaptive');
  const [orders, setOrders] = useState('');
  const [agentId, setAgentId] = useState('');
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);

  const selection = st.selection;
  const sessions = st.snap.sessions;
  const selected = sessions.filter((s) => selection.includes(s.id));
  const config = st.config;

  useEffect(() => {
    const onDown = (e) => { if (!ref.current?.contains(e.target)) onClose(); };
    const onKey = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('mousedown', onDown);
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('mousedown', onDown);
      window.removeEventListener('keydown', onKey);
    };
  }, [onClose]);

  useEffect(() => {
    if (config?.agents?.length && !agentId) setAgentId(config.agents[0].id);
  }, [config, agentId]);

  const isAssignable = ['folder', 'workspace', 'website', 'file', 'mission'].includes(target.type);
  const targetLabel = target.label ?? target.id ?? '—';

  async function run(fn) {
    setBusy(true); setErr(null);
    try { await fn(); onClose(); }
    catch (e) { setErr(e.message); }
    finally { setBusy(false); }
  }

  const doAssign = () => run(async () => {
    const res = await api.assign({
      sessionIds: selection,
      target: {
        type: target.type,
        id: target.type === 'workspace' ? target.workspaceId : target.id,
        label: targetLabel,
        workspaceId: target.workspaceId ?? null,
        url: target.url ?? null,
      },
      orders,
      endpointId: endpoint,
      thinking,
    });
    // Agents rooted in another workspace cannot take this target. Say which, and
    // keep the menu open so the operator can act on it.
    if (res.skipped?.length) {
      const names = res.skipped.map((s) => {
        const sess = sessions.find((x) => x.id === s.sessionId);
        return `${sess?.name ?? s.sessionId.slice(0, 6)} (${s.reason})`;
      });
      throw new Error(`Assigned ${selection.length - res.skipped.length}. Could not assign: ${names.join(', ')}.`);
    }
  });

  const doSpawn = () => run(async () => {
    const wsId = target.workspaceId ?? config?.workspaces?.[0]?.id;
    const r = await api.spawn({
      agentId,
      workspaceId: wsId,
      endpointId: endpoint === 'auto' ? undefined : endpoint,
      thinking,
      orders: orders || undefined,
      target: isAssignable
        ? { type: target.type, id: target.id, label: targetLabel, workspaceId: wsId }
        : undefined,
    });
    selectOnly([r.sessionId]);
  });

  const cmd = (kind, extra) => run(() => api.command(kind, { sessionIds: selection, ...extra }));

  const style = {
    left: Math.min(screen.x, (window.innerWidth || 1200) - 300),
    top: Math.min(screen.y, (window.innerHeight || 800) - 260),
  };

  return (
    <div className="ctx" style={style} ref={ref}>
      <div className="ctx-head">
        <span className="ctx-kind label">{target.type}</span>
        <span className="ctx-title">{targetLabel}</span>
      </div>

      {pane === null && (
        <div className="ctx-items">
          {target.type === 'agent' && (
            <>
              <Item onClick={() => { selectOnly([target.id]); onClose(); }}>Select only this agent</Item>
              <Item onClick={() => { openInWorkspace({ type: 'session', id: target.id }); onClose(); }}>
                Open transcript & workspace
              </Item>
              <Sep />
              <Item disabled={!selection.length} onClick={() => cmd('pause')}>Pause</Item>
              <Item disabled={!selection.length} onClick={() => cmd('resume')}>Resume</Item>
              <Item disabled={!selection.length} onClick={() => cmd('verify')}>Verify work</Item>
              <Item disabled={!selection.length} onClick={() => cmd('escalate')}>Escalate</Item>
              <Sep />
              <Item danger disabled={!selection.length} onClick={() => cmd('cancel')}>Cancel session</Item>
            </>
          )}

          {isAssignable && (
            <>
              {selection.length > 0 ? (
                <Item accent onClick={() => setPane('assign')}>
                  Assign {selection.length} agent{selection.length > 1 ? 's' : ''} here
                </Item>
              ) : (
                <div className="ctx-note">Select agents first, or spawn one here.</div>
              )}
              <Item onClick={() => setPane('spawn')}>Spawn a new agent here…</Item>
              <Sep />
              {target.workspaceId && (
                <Item onClick={() => {
                  openInWorkspace({
                    type: target.type === 'workspace' ? 'workspace' : 'folder',
                    workspaceId: target.workspaceId,
                    path: target.type === 'workspace' ? '' : target.id,
                  });
                  onClose();
                }}>Open in Workspace</Item>
              )}
              {target.type === 'website' && target.url && (
                <Item onClick={() => { openInWorkspace({ type: 'browser', url: target.url }); onClose(); }}>
                  Open live browser surface
                </Item>
              )}
            </>
          )}

          {target.type === 'empty' && (
            <>
              <Item onClick={() => setPane('spawn')}>Spawn a new agent…</Item>
              {selection.length > 0 && <Item onClick={() => cmd('pause')}>Pause selection</Item>}
            </>
          )}
        </div>
      )}

      {(pane === 'assign' || pane === 'spawn') && (
        <div className="ctx-form">
          {pane === 'spawn' && (
            <label className="fld">
              <span className="label">agent</span>
              <select value={agentId} onChange={(e) => setAgentId(e.target.value)}>
                {(config?.agents ?? []).map((a) => (
                  <option key={a.id} value={a.id}>{a.name} · {a.role}</option>
                ))}
              </select>
            </label>
          )}

          <label className="fld">
            <span className="label">endpoint</span>
            <select value={endpoint} onChange={(e) => setEndpoint(e.target.value)}>
              <option value="auto">auto — route by health</option>
              {(config?.endpoints ?? []).map((e) => {
                const live = st.snap.endpoints.find((x) => x.id === e.id);
                return (
                  <option key={e.id} value={e.id}>
                    {e.name} {live ? `· ${live.status}` : ''}
                  </option>
                );
              })}
            </select>
          </label>

          <label className="fld">
            <span className="label">thinking</span>
            <div className="seg">
              {THINKING.map((t) => (
                <button
                  key={t}
                  className={t === thinking ? 'on' : ''}
                  onClick={() => setThinking(t)}
                  type="button"
                >{t}</button>
              ))}
            </div>
          </label>

          <label className="fld">
            <span className="label">orders</span>
            <textarea
              rows={4}
              value={orders}
              placeholder={
                pane === 'assign'
                  ? 'What should they do here? Target context is added automatically.'
                  : 'Optional opening orders.'
              }
              onChange={(e) => setOrders(e.target.value)}
            />
          </label>

          {err && <div className="ctx-err">{err}</div>}

          <div className="ctx-actions">
            <button className="btn ghost" onClick={() => setPane(null)} type="button">Back</button>
            <button
              className="btn primary"
              disabled={busy}
              onClick={pane === 'assign' ? doAssign : doSpawn}
              type="button"
            >
              {busy ? 'Working…' : pane === 'assign' ? `Assign ${selection.length}` : 'Spawn'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function Item({ children, onClick, disabled, danger, accent }) {
  return (
    <button
      type="button"
      className={`ctx-item${danger ? ' danger' : ''}${accent ? ' accent' : ''}`}
      disabled={disabled}
      onClick={onClick}
    >{children}</button>
  );
}

function Sep() { return <div className="ctx-sep" />; }
