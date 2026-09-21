import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { api } from '../net/client.js';
import { getState, openInWorkspace, selectOnly, useField } from '../state/store.js';

const THINKING = ['low', 'medium', 'high', 'adaptive'];

export default function ContextMenu({ screen, target, onClose, initialPane = null, fixed = false }) {
  const st = useField();
  const ref = useRef(null);
  const [pane, setPane] = useState(initialPane);      // null | 'assign' | 'spawn'
  const [endpoint, setEndpoint] = useState('auto');
  const [thinking, setThinking] = useState('adaptive');
  const [orders, setOrders] = useState('');
  const [agentId, setAgentId] = useState('');
  const [wsId, setWsId] = useState(target.workspaceId ?? '');
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const [shift, setShift] = useState({ x: 0, y: 0 });

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
  useEffect(() => {
    if (config?.workspaces?.length && !wsId) setWsId(config.workspaces[0].id);
  }, [config, wsId]);

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
    const r = await api.spawn({
      agentId,
      workspaceId: wsId,
      endpointId: endpoint === 'auto' ? undefined : endpoint,
      thinking,
      orders: orders || undefined,
      target: target.type === 'workspace' || target.type === 'empty'
        ? { type: 'workspace', id: wsId, label: config?.workspaces?.find((w) => w.id === wsId)?.name ?? targetLabel, workspaceId: wsId }
        : isAssignable
          ? { type: target.type, id: target.id, label: targetLabel, workspaceId: wsId }
          : undefined,
    });
    selectOnly([r.sessionId]);
  });

  const cmd = (kind, extra) => run(() => api.command(kind, { sessionIds: selection, ...extra }));

  // Keep the whole menu on screen whatever pane is open: measure after layout and pull
  // it back inside the viewport by however much it overflows.
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const dx = Math.min(0, (window.innerWidth || 1200) - 8 - (r.right - shift.x));
    const dy = Math.min(0, (window.innerHeight || 800) - 8 - (r.bottom - shift.y));
    if (dx !== shift.x || dy !== shift.y) setShift({ x: dx, y: dy });
  }, [pane, err, screen.x, screen.y]);

  const style = {
    left: Math.max(8, screen.x + shift.x),
    top: Math.max(8, screen.y + shift.y),
  };
  const noEndpoints = !(config?.endpoints?.length || st.snap.endpoints?.length);

  return (
    <div className={`ctx${fixed ? ' ctx-fixed' : ''}`} style={style} ref={ref} role="dialog" aria-label={pane === 'spawn' ? 'Start an agent' : 'Actions'}>
      <div className="ctx-head">
        <span className="ctx-kind label">{pane === 'spawn' ? 'start an agent on' : target.type}</span>
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
                <div className="ctx-note">Select agents first, or start one here.</div>
              )}
              <Item onClick={() => setPane('spawn')}>Start a new agent here…</Item>
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
              <Item onClick={() => setPane('spawn')}>Start a new agent…</Item>
              {selection.length > 0 && <Item onClick={() => cmd('pause')}>Pause selection</Item>}
            </>
          )}
        </div>
      )}

      {(pane === 'assign' || pane === 'spawn') && (
        <div className="ctx-form">
          {pane === 'spawn' && ['workspace', 'empty'].includes(target.type) && (config?.workspaces?.length ?? 0) > 1 && (
            <label className="fld">
              <span className="label">project</span>
              <select value={wsId} onChange={(e) => setWsId(e.target.value)}>
                {config.workspaces.map((w) => <option key={w.id} value={w.id}>{w.name}</option>)}
              </select>
            </label>
          )}
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

          {pane === 'spawn' && noEndpoints && (
            <div className="ctx-note warn">No model is set up yet. Add one under Models first, or the agent will fail to start.</div>
          )}
          {err && <div className="ctx-err" role="alert">{err}</div>}

          <div className="ctx-actions">
            <button className="btn ghost" onClick={() => (initialPane ? onClose() : setPane(null))} type="button">{initialPane ? 'Cancel' : 'Back'}</button>
            <button
              className="btn primary"
              disabled={busy}
              onClick={pane === 'assign' ? doAssign : doSpawn}
              type="button"
            >
              {busy ? 'Starting…' : pane === 'assign' ? `Assign ${selection.length}` : 'Start agent'}
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
