import { useEffect, useMemo, useRef, useState } from 'react';
import { api, on } from '../net/client.js';
import { openChanges, openInWorkspace, selectAgent, useField } from '../state/store.js';
import {
  identityFor,
  initials,
  identityHue,
  saveFieldSettings,
} from '../theater/fieldPreferences.js';
import PermissionRequests, { isPrivilegedTool } from '../hud/PermissionRequests.jsx';
import PowerSources from '../setup/PowerSources.jsx';
import ContextMenu from '../hud/ContextMenu.jsx';
import AgentControls from '../hud/AgentControls.jsx';

const TERMINAL = new Set(['done', 'cancelled', 'interrupted', 'error']);
const ATTENTION = new Set(['blocked', 'error', 'waiting_permission']);
const TRACE_KINDS = new Set([
  'session.spawned', 'session.message', 'session.thinking', 'session.tool_use',
  'session.tool_result', 'session.ended', 'permission.requested', 'permission.decided',
  'work.verified',
]);

// Raw session states → plain language + a semantic tone the stylesheet knows about.
// tone: working | attention | done | failed | idle
export function plainState(session) {
  const state = session?.state ?? 'idle';
  if (!session) return { label: 'idle', tone: 'idle' };
  if (state === 'waiting_permission' || session.pendingPermission) return { label: 'needs approval', tone: 'attention' };
  if (state === 'blocked') return { label: 'blocked', tone: 'attention' };
  if (state === 'error') return { label: 'failed', tone: 'failed' };
  if (state === 'done') return { label: 'done', tone: 'done' };
  if (state === 'cancelled' || state === 'interrupted') return { label: 'stopped', tone: 'idle' };
  if (state === 'thinking') return { label: 'thinking', tone: 'working' };
  if (['running', 'working', 'active', 'spawning', 'starting'].includes(state)) return { label: 'working', tone: 'working' };
  if (state === 'paused') return { label: 'paused', tone: 'idle' };
  return { label: 'idle', tone: 'idle' };
}

export default function AtlasMode({ settings, setSettings }) {
  const st = useField();
  const selectedId = st.activeSessionId;
  const [modelsOpen, setModelsOpen] = useState(false);
  const [starter, setStarter] = useState(null); // { workspace, screen }
  const workspaces = st.snap.workspaces.filter((item) => item.mounted);
  const startAgent = (workspace, e, pane = 'spawn') => setStarter({
    workspace,
    pane,
    screen: { x: e?.clientX ?? window.innerWidth / 2 - 140, y: e?.clientY ?? 120 },
  });
  // Defining an agent needs no project; it is anchored to the first one only for "Save and start".
  const newAgent = (e) => startAgent(workspaces[0] ?? null, e, 'new-agent');
  const sessions = useMemo(
    () => [...st.snap.sessions].sort((a, b) => (b.startedAt ?? 0) - (a.startedAt ?? 0)),
    [st.snap.sessions],
  );
  const live = sessions.filter((session) => !TERMINAL.has(session.state));
  const attention = live.filter((session) => ATTENTION.has(session.state) || session.pendingPermission);
  const endpoints = st.snap.endpoints?.length ? st.snap.endpoints : st.config?.endpoints ?? [];
  const identities = useMemo(
    () => new Map(sessions.map((session) => [session.id, identityFor(session, endpoints, settings)])),
    [sessions, endpoints, settings],
  );
  const columns = useMemo(() => buildColumns(workspaces, live, sessions), [workspaces, live, sessions]);
  const pendingCount = st.snap.permissions?.length ?? 0;

  useEffect(() => { saveFieldSettings(settings); }, [settings]);

  return (
    <div className="atlas-board">
      <header className="atlas-head">
        <div className="atlas-title">
          <span className="atlas-kicker">Board</span>
          <h1>Agents and projects</h1>
        </div>
        <div className="atlas-stats" aria-live="polite">
          <span className="atlas-chip tone-working"><i aria-hidden="true" /><b>{live.length}</b> working</span>
          <span className={`atlas-chip${attention.length ? ' tone-attention' : ''}`}><i aria-hidden="true" /><b>{attention.length}</b> need you</span>
          <span className="atlas-chip"><i aria-hidden="true" /><b>{workspaces.length}</b> {workspaces.length === 1 ? 'project' : 'projects'}</span>
        </div>
        <div className="atlas-actions">
          {workspaces.length > 0 && (
            <button type="button" className="btn primary" onClick={(e) => startAgent(workspaces[0], e)}>Start an agent</button>
          )}
          <button type="button" className="btn ghost" onClick={newAgent}>New agent</button>
          <button type="button" className="btn ghost" onClick={() => setModelsOpen(true)}>Models</button>
          <button type="button" className="btn ghost" onClick={() => setSettings((current) => ({ ...current, theme: 'rome' }))}>
            Rome map
          </button>
        </div>
      </header>
      <div className="atlas-scroll">
        {pendingCount > 0 && (
          <section className="atlas-approvals" aria-label="Approvals waiting">
            <h2 className="atlas-section-title">
              <i aria-hidden="true" />
              {pendingCount === 1 ? 'One request needs your decision' : `${pendingCount} requests need your decision`}
            </h2>
            <PermissionRequests />
          </section>
        )}
        {columns.length > 0 && endpoints.length === 0 && (
          <section className="atlas-approvals atlas-nudge" aria-label="No model yet">
            <h2 className="atlas-section-title"><i aria-hidden="true" />No model yet</h2>
            <p>Agents cannot start until a model is set up: a Cameo box, Ollama on this machine, or a provider key.</p>
            <div className="atlas-empty-actions">
              <button type="button" className="btn" onClick={() => setModelsOpen(true)}>Set up a model</button>
            </div>
          </section>
        )}
        {columns.length ? (
          <div className="atlas-grid" role="list">
            {columns.map((column) => (
              <AgentColumn
                key={column.key}
                column={column}
                identity={column.session ? identities.get(column.session.id) : null}
                selected={column.session?.id === selectedId}
                permissions={st.snap.permissions ?? []}
                campaigns={st.snap.campaigns ?? []}
                now={st.snap.now}
                onStart={startAgent}
              />
            ))}
          </div>
        ) : (
          <div className="atlas-empty">
            <span className="atlas-empty-mark" aria-hidden="true" />
            <b>No project open</b>
            <p>Add a project folder under <code>workspaces</code> in <code>field/field.yaml</code>, then restart Field. Each project gets a card here, and each agent working on it gets its own.</p>
            <div className="atlas-empty-actions">
              <button type="button" className="btn" onClick={() => setModelsOpen(true)}>Set up a model</button>
            </div>
          </div>
        )}
      </div>
      {modelsOpen && <PowerSources onClose={() => setModelsOpen(false)} />}
      {starter && (
        <ContextMenu
          fixed
          initialPane={starter.pane ?? 'spawn'}
          screen={starter.screen}
          target={starter.workspace
            ? { type: 'workspace', id: starter.workspace.id, workspaceId: starter.workspace.id, label: starter.workspace.name }
            : { type: 'empty', id: null, label: 'no project mounted' }}
          onClose={() => setStarter(null)}
          onOpenModels={() => { setStarter(null); setModelsOpen(true); }}
        />
      )}
    </div>
  );
}

function buildColumns(workspaces, live, allSessions) {
  const byWorkspace = new Map(workspaces.map((workspace) => [workspace.id, { workspace, agents: [] }]));
  for (const session of live) {
    const bucket = byWorkspace.get(session.workspaceId) ?? byWorkspace.get(workspaces[0]?.id);
    if (bucket) bucket.agents.push(session);
  }
  const columns = [];
  for (const { workspace, agents } of byWorkspace.values()) {
    if (agents.length) {
      for (const session of agents) {
        columns.push({ key: session.id, workspace, session });
      }
    } else {
      columns.push({ key: `ws:${workspace.id}`, workspace, session: null });
    }
  }
  if (!columns.length && allSessions.length) {
    for (const session of allSessions.slice(0, 8)) {
      columns.push({
        key: session.id,
        workspace: workspaces.find((item) => item.id === session.workspaceId) ?? null,
        session,
      });
    }
  }
  return columns;
}

function AgentColumn({ column, identity, selected, permissions, campaigns, now, onStart }) {
  const { workspace, session } = column;
  const pending = permissions.filter((item) => item.sessionId === session?.id);
  const state = plainState(session);
  const needsYou = Boolean(session && (ATTENTION.has(session.state) || pending.length));
  const lastTool = typeof session?.lastTool === 'string' ? session.lastTool : session?.lastTool?.name;
  const cost = session?.costUsd ? `$${Number(session.costUsd).toFixed(3)}` : '';
  const elapsed = session?.startedAt ? Math.max(0, Math.round(((now ?? Date.now()) - session.startedAt) / 60000)) : null;
  const pct = session?.progress?.total ? Math.round(((session.progress.done ?? 0) / session.progress.total) * 100) : null;
  return (
    <article
      className={`atlas-card tone-${state.tone}${selected ? ' selected' : ''}${needsYou ? ' needs-you' : ''}${session ? '' : ' no-agent'}`}
      role="listitem"
      aria-label={session ? `${identity?.displayName ?? session.name ?? session.id}, ${state.label}` : `${workspace?.name ?? 'Project'}, no agent yet`}
    >
      <header className="atlas-card-head">
        {session ? (
          <button type="button" className="atlas-who" onClick={() => selectAgent(session.id)} title="Select this agent">
            <Mark identity={identity} />
            <span className="atlas-who-text">
              <b>{identity?.displayName ?? session.name ?? session.id}</b>
              <small className="atlas-model">{identity?.endpointAlias ?? session.model ?? 'no model'}</small>
            </span>
          </button>
        ) : (
          <div className="atlas-who idle">
            <span className="atlas-mark atlas-mark-project" aria-hidden="true">{initials(workspace?.name ?? '?')}</span>
            <span className="atlas-who-text">
              <b>{workspace?.name ?? 'Project'}</b>
              <small className="atlas-model">no agent yet</small>
            </span>
          </div>
        )}
        <span className={`atlas-pill tone-${state.tone}`}><i aria-hidden="true" />{state.label}</span>
      </header>
      {(workspace || session?.cwd) && (
        <div className="atlas-context">
          <span className="atlas-ws" title={workspace?.path}>{workspace?.name ?? session?.cwd}</span>
          {workspace?.git?.branch && (
            <span className="atlas-git">
              {workspace.git.branch}
              {(workspace.git.files?.length ?? workspace.changeCount) ? (
                <>
                  {' · '}
                  <button type="button" className="atlas-changes" onClick={() => openChanges(workspace.id)} title="Review, accept or revert the changes">
                    {workspace.git.files?.length ?? workspace.changeCount} changed
                  </button>
                </>
              ) : ''}
            </span>
          )}
        </div>
      )}
      {pending.map((permission) => (
        <div className="atlas-perm" key={permission.id} role="note">
          <span className="atlas-perm-label">
            <i aria-hidden="true" />
            {isPrivilegedTool(permission.toolName, permission.input) ? 'Privileged approval' : 'Approval'} · <code>{permission.toolName}</code>
          </span>
          <code className="atlas-perm-input">{summarize(permission.toolName, permission.input)}</code>
          <button
            type="button"
            className="atlas-perm-jump"
            onClick={() => document.querySelector('.atlas-approvals')?.scrollIntoView({ block: 'start', behavior: 'smooth' })}
          >Decide at the top of the board</button>
        </div>
      ))}
      <div className="atlas-body">
        {session
          ? <ColumnTranscript session={session} campaigns={campaigns} />
          : (
            <div className="atlas-idle">
              <b>No agent working on this project yet.</b>
              <p>Start one with orders, a model and a thinking level. It shows up here the moment it begins; routines and plans can also assign agents.</p>
              <div className="atlas-idle-actions">
                <button type="button" className="btn primary" onClick={(e) => onStart(workspace, e)}>Start an agent</button>
                <button type="button" className="btn ghost" onClick={() => openInWorkspace({ type: 'workspace', workspaceId: workspace?.id, path: '' })}>Open files</button>
              </div>
            </div>
          )}
      </div>
      {session && (
        <footer className="atlas-foot">
          <div className="atlas-foot-row">
            {lastTool && <span className="atlas-foot-tool"><span className="atlas-foot-label">tool</span><code>{lastTool}</code></span>}
            {pct != null && <span className="atlas-foot-tool"><span className="atlas-foot-label">progress</span><code>{pct}%</code></span>}
            {elapsed != null && <span className="atlas-foot-tool"><span className="atlas-foot-label">elapsed</span><code>{elapsed < 1 ? '<1m' : `${elapsed}m`}</code></span>}
            {session.contextPct != null && <span className="atlas-foot-tool"><span className="atlas-foot-label">context</span><code>{session.contextPct}%</code></span>}
            <span className={`atlas-foot-cost${session.budgetExhausted ? ' budget-exhausted' : ''}`} title={session.budgetExhausted ? 'budget exhausted; the agent is paused' : 'spent / budget'}>
              <span className="atlas-foot-label">{session.budgetExhausted ? 'budget' : 'cost'}</span>
              <code>{cost || '$0.000'}{session.budgetUsd ? ` / ${Number(session.budgetUsd).toFixed(2)}` : ''}{session.budgetExhausted ? ' · exhausted' : ''}</code>
            </span>
          </div>
          {session.error && <p className="atlas-foot-error" role="alert">{session.error}</p>}
          <AgentControls session={session} compact onOpen={() => openInWorkspace({ type: 'session', id: session.id })} />
        </footer>
      )}
    </article>
  );
}

function Mark({ identity }) {
  const hue = identityHue(identity?.displayName ?? '?');
  if (identity?.iconUrl) return <img className="atlas-mark" src={identity.iconUrl} alt="" />;
  return <span className="atlas-mark" style={{ background: `hsl(${hue} 30% 26%)`, color: `hsl(${hue} 60% 86%)` }}>{initials(identity?.displayName ?? '?')}</span>;
}

export function ColumnTranscript({ session, campaigns }) {
  const sessionId = session.id;
  const [events, setEvents] = useState([]);
  const bottomRef = useRef(null);

  useEffect(() => {
    let alive = true;
    api.trace(sessionId, 0, 400).then((result) => { if (alive) setEvents(result.events ?? []); }).catch(() => { if (alive) setEvents([]); });
    return () => { alive = false; };
  }, [sessionId, session.messageCount, session.toolCount, session.state]);

  useEffect(() => on('event', (evt) => {
    if (evt.subject !== sessionId && evt.data?.sessionId !== sessionId) return;
    setEvents((prev) => (prev.some((item) => item.seq === evt.seq) ? prev : [...prev, evt]));
  }), [sessionId]);

  useEffect(() => { bottomRef.current?.scrollIntoView({ block: 'end' }); }, [events.length]);

  const objective = campaigns.flatMap((campaign) => campaign.objectives ?? []).find((item) => item.id === session.objectiveId);
  const rows = events.filter((evt) => TRACE_KINDS.has(evt.kind)).slice(-80);

  return (
    <div className="atlas-trace">
      <p className="atlas-objective">{objective?.statement ?? session.target?.label ?? session.stateDetail ?? 'Waiting for an assignment'}</p>
      {rows.length ? (
        <div className="atlas-feed">
          {rows.map((evt) => <TraceLine key={evt.seq} evt={evt} />)}
          <div ref={bottomRef} />
        </div>
      ) : (
        <p className="atlas-feed-empty">Nothing reported yet. Messages and tool calls will appear here as the agent works.</p>
      )}
    </div>
  );
}

function TraceLine({ evt }) {
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

function summarize(toolName, input) {
  if (!input || typeof input !== 'object') return String(input ?? toolName);
  if (typeof input.command === 'string') return input.command;
  if (typeof input.file_path === 'string') return input.file_path;
  if (typeof input.path === 'string') return input.path;
  if (typeof input.url === 'string') return input.url;
  return toolName;
}
