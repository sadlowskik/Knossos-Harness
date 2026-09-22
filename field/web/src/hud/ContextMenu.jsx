import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { Pencil, Plus } from 'lucide-react';
import { api } from '../net/client.js';
import { getState, openInWorkspace, refreshConfig, selectOnly, useField } from '../state/store.js';

const THINKING = ['low', 'medium', 'high', 'adaptive'];
// An agent definition carries a fixed level; "adaptive" is a per-session choice.
const AGENT_THINKING = ['low', 'medium', 'high'];

function emptyDraft(config) {
  return {
    id: null,
    name: '',
    role: config?.roles?.[0]?.id ?? '',
    endpoint: config?.endpoints?.[0]?.id ?? '',
    model: '',
    thinking: 'medium',
    orders: '',
  };
}

/* `initialAgentId` preselects the agent definition: the Board's "start another one like
   this" on a finished conversation opens the same agent on the same project. */
export default function ContextMenu({ screen, target, onClose, initialPane = null, initialAgentId = null, fixed = false, onOpenModels = null }) {
  const st = useField();
  const ref = useRef(null);
  const [pane, setPane] = useState(initialPane);      // null | 'assign' | 'spawn' | 'new-agent'
  const [endpoint, setEndpoint] = useState('auto');
  const [thinking, setThinking] = useState('adaptive');
  const [orders, setOrders] = useState('');
  const [agentId, setAgentId] = useState(initialAgentId ?? '');
  const [wsId, setWsId] = useState(target.workspaceId ?? '');
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState(null);
  const [shift, setShift] = useState({ x: 0, y: 0 });
  // The agent being defined or edited in the "new agent" pane, plus the
  // one-line task for "Save and start".
  const [draft, setDraft] = useState(() => (initialPane === 'new-agent' ? emptyDraft(getState().config) : null));
  const [task, setTask] = useState('');
  const [confirmDelete, setConfirmDelete] = useState(false);

  const selection = st.selection;
  const sessions = st.snap.sessions;
  const selected = sessions.filter((s) => selection.includes(s.id));
  const config = st.config;
  const roles = config?.roles ?? [];
  const endpoints = config?.endpoints ?? [];

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
    // A preselected agent that no longer exists in config falls back to the first one,
    // rather than leaving the select on a definition the server cannot spawn.
    if (config?.agents?.length && !config.agents.some((agent) => agent.id === agentId)) {
      setAgentId(config.agents[0].id);
    }
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

  // Where a spawned session is pointed: the project itself, or the folder,
  // file or site the menu was opened on.
  const spawnTarget = () => (
    target.type === 'workspace' || target.type === 'empty'
      ? { type: 'workspace', id: wsId, label: config?.workspaces?.find((w) => w.id === wsId)?.name ?? targetLabel, workspaceId: wsId }
      : isAssignable
        ? { type: target.type, id: target.id, label: targetLabel, workspaceId: wsId }
        : undefined
  );

  const doSpawn = () => run(async () => {
    const r = await api.spawn({
      agentId,
      workspaceId: wsId,
      endpointId: endpoint === 'auto' ? undefined : endpoint,
      thinking,
      orders: orders || undefined,
      target: spawnTarget(),
    });
    selectOnly([r.sessionId]);
  });

  const cmd = (kind, extra) => run(() => api.command(kind, { sessionIds: selection, ...extra }));

  // ---- the "new agent" pane ------------------------------------------------

  function openNewAgent() {
    setErr(null); setConfirmDelete(false);
    setDraft(emptyDraft(config));
    setPane('new-agent');
  }

  // Edit prefills from the full record (`/api/config` strips the orders).
  async function openEditAgent(id) {
    setErr(null); setConfirmDelete(false);
    const fromConfig = config?.agents?.find((a) => a.id === id);
    if (!fromConfig) return;
    setDraft({ ...emptyDraft(config), ...pick(fromConfig), id });
    setPane('new-agent');
    try {
      const { agents } = await api.agents();
      const full = agents.find((a) => a.id === id);
      if (full) setDraft((current) => (current?.id === id ? { ...current, ...pick(full), id } : current));
    } catch (e) { setErr(e.message); }
  }

  const leaveAgentPane = () => (initialPane === 'new-agent' ? onClose() : setPane('spawn'));

  async function saveAgent({ start = false } = {}) {
    if (!draft) return;
    setBusy(true); setErr(null);
    try {
      const body = {
        name: draft.name,
        role: draft.role,
        endpoint: draft.endpoint,
        model: draft.model.trim() || undefined,
        thinking: draft.thinking || undefined,
        orders: draft.orders,
      };
      const res = draft.id ? await api.saveAgent({ id: draft.id, ...body }) : await api.createAgent(body);
      await refreshConfig();
      const id = res.agent.id;
      setAgentId(id);
      if (start) {
        if (!wsId) throw new Error('Saved. No project is mounted to start it on.');
        const r = await api.spawn({ agentId: id, workspaceId: wsId, orders: task.trim() || undefined, target: spawnTarget() });
        selectOnly([r.sessionId]);
        onClose();
      } else {
        leaveAgentPane();
      }
    } catch (e) { setErr(e.message); }
    finally { setBusy(false); }
  }

  async function deleteAgent() {
    if (!draft?.id) return;
    setBusy(true); setErr(null);
    try {
      await api.deleteAgent(draft.id);
      await refreshConfig();
      if (agentId === draft.id) setAgentId('');
      setConfirmDelete(false);
      leaveAgentPane();
    } catch (e) { setErr(e.message); setConfirmDelete(false); }
    finally { setBusy(false); }
  }

  // Keep the whole menu on screen whatever pane is open: measure after layout and pull
  // it back inside the viewport by however much it overflows.
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const dx = Math.min(0, (window.innerWidth || 1200) - 8 - (r.right - shift.x));
    const dy = Math.min(0, (window.innerHeight || 800) - 8 - (r.bottom - shift.y));
    if (dx !== shift.x || dy !== shift.y) setShift({ x: dx, y: dy });
  }, [pane, err, confirmDelete, screen.x, screen.y]);

  const style = {
    left: Math.max(8, screen.x + shift.x),
    top: Math.max(8, screen.y + shift.y),
  };
  const noEndpoints = !(config?.endpoints?.length || st.snap.endpoints?.length);
  const editing = pane === 'new-agent';
  const headKind = editing ? (draft?.id ? 'edit agent' : 'new agent') : pane === 'spawn' ? 'start an agent on' : target.type;
  const headTitle = editing ? (draft?.name?.trim() || (draft?.id ? draft.id : 'unnamed')) : targetLabel;
  const dialogLabel = editing ? (draft?.id ? 'Edit agent' : 'New agent') : pane === 'spawn' ? 'Start an agent' : 'Actions';

  return (
    <div className={`ctx${fixed ? ' ctx-fixed' : ''}${editing ? ' ctx-wide' : ''}`} style={style} ref={ref} role="dialog" aria-label={dialogLabel}>
      <div className="ctx-head">
        <span className="ctx-kind label">{headKind}</span>
        <span className="ctx-title">{headTitle}</span>
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
              <Item onClick={openNewAgent}>Define a new agent…</Item>
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
              <Item onClick={openNewAgent}>Define a new agent…</Item>
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
            <div className="fld">
              <span className="label">agent</span>
              <div className="fld-row">
                <select value={agentId} onChange={(e) => setAgentId(e.target.value)} aria-label="Agent">
                  {(config?.agents ?? []).map((a) => (
                    <option key={a.id} value={a.id}>{a.name} · {a.role}</option>
                  ))}
                </select>
                <button
                  type="button"
                  className="btn sm ghost icon"
                  title="Edit this agent"
                  aria-label="Edit this agent"
                  disabled={!agentId}
                  onClick={() => openEditAgent(agentId)}
                ><Pencil aria-hidden="true" /></button>
                <button type="button" className="btn sm ghost" onClick={openNewAgent}><Plus aria-hidden="true" />New agent</button>
              </div>
              {!(config?.agents?.length) && <span className="fld-hint">No agents defined yet. Define one to start.</span>}
            </div>
          )}

          <label className="fld">
            <span className="label">endpoint</span>
            <select value={endpoint} onChange={(e) => setEndpoint(e.target.value)}>
              <option value="auto">auto — route by health</option>
              {endpoints.map((e) => {
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
              disabled={busy || (pane === 'spawn' && !agentId)}
              onClick={pane === 'assign' ? doAssign : doSpawn}
              type="button"
            >
              {busy ? 'Starting…' : pane === 'assign' ? `Assign ${selection.length}` : 'Start agent'}
            </button>
          </div>
        </div>
      )}

      {editing && draft && (
        <AgentForm
          draft={draft}
          setDraft={setDraft}
          roles={roles}
          endpoints={endpoints}
          liveEndpoints={st.snap.endpoints ?? []}
          canStart={Boolean(wsId)}
          projectName={config?.workspaces?.find((w) => w.id === wsId)?.name ?? targetLabel}
          task={task}
          setTask={setTask}
          busy={busy}
          err={err}
          confirmDelete={confirmDelete}
          setConfirmDelete={setConfirmDelete}
          onCancel={leaveAgentPane}
          onSave={() => saveAgent()}
          onSaveAndStart={() => saveAgent({ start: true })}
          onDelete={deleteAgent}
          onOpenModels={onOpenModels}
        />
      )}
    </div>
  );
}

function pick(agent) {
  return {
    name: agent.name ?? '',
    role: agent.role ?? '',
    endpoint: agent.endpoint ?? '',
    model: agent.model ?? '',
    thinking: AGENT_THINKING.includes(agent.thinking) ? agent.thinking : 'medium',
    orders: agent.orders ?? '',
  };
}

function modelHint(endpoint, endpoints) {
  const ep = endpoints.find((e) => e.id === endpoint);
  const fallback = ep?.model ? ` Empty uses the endpoint's default, ${ep.model}.` : ' Empty uses the endpoint’s default.';
  if (ep?.kind === 'anthropic') return `An Anthropic model id, e.g. claude-sonnet-4-5.${fallback}`;
  return `For Ollama or a Cameo box, the model name, e.g. a Hugging Face id like Qwen/Qwen2.5-Coder-7B-Instruct.${fallback}`;
}

function AgentForm({
  draft, setDraft, roles, endpoints, liveEndpoints, canStart, projectName, task, setTask,
  busy, err, confirmDelete, setConfirmDelete, onCancel, onSave, onSaveAndStart, onDelete, onOpenModels,
}) {
  const set = (key) => (e) => setDraft((d) => ({ ...d, [key]: e.target.value }));
  const missing = [];
  if (!roles.length) missing.push('role');
  if (!endpoints.length) missing.push('endpoint');
  const ready = missing.length === 0;
  const live = liveEndpoints.find((e) => e.id === draft.endpoint);
  const valid = ready && draft.name.trim() && draft.role && draft.endpoint;
  const nameRef = useRef(null);
  useEffect(() => { nameRef.current?.focus(); }, []);

  return (
    <div className="ctx-form">
      {!ready && (
        <div className="ctx-note warn">
          {!roles.length && <div>No roles are defined. Add a role file under <code>field/roles/</code> first; a role is what an agent is allowed to do.</div>}
          {!endpoints.length && (
            <div>
              No model endpoint is set up, so there is nothing for an agent to run on.
              {onOpenModels && <> <button type="button" className="btn sm" onClick={onOpenModels}>Set up a model</button></>}
            </div>
          )}
        </div>
      )}

      <label className="fld">
        <span className="label">name</span>
        <input ref={nameRef} type="text" value={draft.name} maxLength={64} placeholder="e.g. Rhea" onChange={set('name')} />
        {!draft.id && <span className="fld-hint">The id, colour and initials follow from the name.</span>}
      </label>

      <label className="fld">
        <span className="label">role</span>
        <select value={draft.role} onChange={set('role')} disabled={!roles.length}>
          {!roles.length && <option value="">no roles defined</option>}
          {roles.map((r) => <option key={r.id} value={r.id}>{r.name ?? r.id}</option>)}
        </select>
      </label>

      <div className="fld">
        <span className="label">endpoint</span>
        <div className="fld-row">
          <span className={`dot ${live?.status ?? 'unknown'}`} title={live ? live.status : 'not probed yet'} aria-hidden="true" />
          <select value={draft.endpoint} onChange={set('endpoint')} disabled={!endpoints.length} aria-label="Endpoint">
            {!endpoints.length && <option value="">no endpoint set up</option>}
            {endpoints.map((e) => {
              const status = liveEndpoints.find((x) => x.id === e.id)?.status;
              return <option key={e.id} value={e.id}>{e.name ?? e.id}{status ? ` · ${status}` : ''}</option>;
            })}
          </select>
        </div>
      </div>

      <label className="fld">
        <span className="label">model <small>optional</small></span>
        <input type="text" value={draft.model} maxLength={128} placeholder="endpoint default" onChange={set('model')} spellCheck={false} />
        <span className="fld-hint">{modelHint(draft.endpoint, endpoints)}</span>
      </label>

      <label className="fld">
        <span className="label">thinking</span>
        <div className="seg">
          {AGENT_THINKING.map((t) => (
            <button key={t} type="button" className={t === draft.thinking ? 'on' : ''} onClick={() => setDraft((d) => ({ ...d, thinking: t }))}>{t}</button>
          ))}
        </div>
      </label>

      <label className="fld">
        <span className="label">standing orders</span>
        <textarea rows={5} value={draft.orders} placeholder="What this agent always keeps in mind. Markdown; becomes part of its system prompt." onChange={set('orders')} />
      </label>

      {canStart && (
        <label className="fld">
          <span className="label">first task <small>for “Save and start” on {projectName}</small></span>
          <input type="text" value={task} placeholder="Optional opening orders for the first session." onChange={(e) => setTask(e.target.value)} />
        </label>
      )}

      {err && <div className="ctx-err" role="alert">{err}</div>}

      {confirmDelete ? (
        <div className="ctx-actions ctx-confirm">
          <span className="ctx-note">Delete {draft.name || draft.id}? Its file is removed; running sessions keep it until they end.</span>
          <button type="button" className="btn ghost" disabled={busy} onClick={() => setConfirmDelete(false)}>Keep</button>
          <button type="button" className="btn danger" disabled={busy} onClick={onDelete}>{busy ? 'Deleting…' : 'Delete'}</button>
        </div>
      ) : (
        <div className="ctx-actions">
          {draft.id && <button type="button" className="btn ghost danger" disabled={busy} onClick={() => setConfirmDelete(true)}>Delete</button>}
          <span className="ctx-spacer" />
          <button type="button" className="btn ghost" disabled={busy} onClick={onCancel}>Cancel</button>
          <button type="button" className="btn ghost" disabled={busy || !valid} onClick={onSave}>{busy ? 'Saving…' : 'Save'}</button>
          {canStart && (
            <button type="button" className="btn primary" disabled={busy || !valid} onClick={onSaveAndStart}>{busy ? 'Saving…' : 'Save and start'}</button>
          )}
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
