import { useEffect, useMemo, useRef, useState } from 'react';
import { api, on } from '../net/client.js';
import { selectAgent, useField } from '../state/store.js';
import {
  identityFor,
  initials,
  identityHue,
  saveFieldSettings,
} from '../theater/fieldPreferences.js';
import PermissionRequests, { isPrivilegedTool } from '../hud/PermissionRequests.jsx';

const TERMINAL = new Set(['done', 'cancelled', 'interrupted', 'error']);
const ATTENTION = new Set(['blocked', 'error', 'waiting_permission']);
const TRACE_KINDS = new Set([
  'session.spawned', 'session.message', 'session.thinking', 'session.tool_use',
  'session.tool_result', 'session.ended', 'permission.requested', 'permission.decided',
  'work.verified',
]);

export default function AtlasMode({ settings, setSettings }) {
  const st = useField();
  const selectedId = st.activeSessionId;
  const workspaces = st.snap.workspaces.filter((item) => item.mounted);
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

  useEffect(() => { saveFieldSettings(settings); }, [settings]);

  return (
    <div className="atlas-board">
      <header className="atlas-head">
        <div>
          <span>ATLAS</span>
          <h1>Agents and workspaces</h1>
        </div>
        <div className="atlas-stats" aria-live="polite">
          <span><b>{live.length}</b> live</span>
          <span className={attention.length ? 'attention' : ''}><b>{attention.length}</b> need you</span>
          <span><b>{workspaces.length}</b> workspaces</span>
        </div>
        <div className="atlas-actions">
          <button type="button" onClick={() => setSettings((current) => ({ ...current, theme: 'rome' }))}>
            Rome map
          </button>
        </div>
      </header>
      <PermissionRequests />
      <div className="atlas-cols" role="list">
        {columns.length ? columns.map((column) => (
          <AgentColumn
            key={column.key}
            column={column}
            identity={column.session ? identities.get(column.session.id) : null}
            selected={column.session?.id === selectedId}
            permissions={st.snap.permissions ?? []}
            campaigns={st.snap.campaigns ?? []}
          />
        )) : (
          <div className="atlas-empty">
            <b>No mounted workspace.</b>
            <p>Add paths in field/field.yaml. Atlas shows one column per live agent, grouped by workspace.</p>
          </div>
        )}
      </div>
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

function AgentColumn({ column, identity, selected, permissions, campaigns }) {
  const { workspace, session } = column;
  const pending = permissions.filter((item) => item.sessionId === session?.id);
  return (
    <article className={`atlas-col${selected ? ' selected' : ''}${session && ATTENTION.has(session.state) ? ' needs-you' : ''}`} role="listitem">
      <header className="atlas-col-head">
        {session ? (
          <button type="button" className="atlas-who" onClick={() => selectAgent(session.id)}>
            <Mark identity={identity} />
            <span>
              <b>{identity?.displayName ?? session.name ?? session.id}</b>
              <small>{identity?.endpointAlias ?? session.model ?? 'unassigned'}</small>
            </span>
          </button>
        ) : (
          <div className="atlas-who idle">
            <span>
              <b>{workspace?.name ?? 'Workspace'}</b>
              <small>idle</small>
            </span>
          </div>
        )}
        <div className="atlas-col-meta">
          <span className={`atlas-state state-${session?.state ?? 'idle'}`}>{session?.state ?? 'idle'}</span>
          <span className="atlas-ws" title={workspace?.path}>{workspace?.name ?? session?.cwd ?? '—'}</span>
        </div>
      </header>
      {workspace?.git?.branch && <p className="atlas-git">{workspace.git.branch}{workspace.changeCount ? ` · ${workspace.changeCount} changed` : ''}</p>}
      {pending.map((permission) => (
        <div className="atlas-perm" key={permission.id}>
          <span>{isPrivilegedTool(permission.toolName, permission.input) ? 'Privileged approval' : 'Approval'} · {permission.toolName}</span>
          <code>{summarize(permission.toolName, permission.input)}</code>
        </div>
      ))}
      <div className="atlas-body">
        {session
          ? <ColumnTranscript session={session} campaigns={campaigns} />
          : <p className="atlas-idle">No agent in this workspace yet.</p>}
      </div>
      {session?.lastTool && <footer className="atlas-foot">{session.lastTool}{session.costUsd ? ` · $${Number(session.costUsd).toFixed(3)}` : ''}</footer>}
    </article>
  );
}

function Mark({ identity }) {
  const hue = identityHue(identity?.displayName ?? '?');
  if (identity?.iconUrl) return <img className="atlas-mark" src={identity.iconUrl} alt="" />;
  return <span className="atlas-mark" style={{ background: `hsl(${hue} 28% 24%)` }}>{initials(identity?.displayName ?? '?')}</span>;
}

function ColumnTranscript({ session, campaigns }) {
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
      <p className="atlas-objective">{objective?.statement ?? session.target?.label ?? session.stateDetail ?? 'Awaiting assignment'}</p>
      {rows.map((evt) => <TraceLine key={evt.seq} evt={evt} />)}
      <div ref={bottomRef} />
    </div>
  );
}

function TraceLine({ evt }) {
  const data = evt.data ?? {};
  if (evt.kind === 'session.message') {
    return <p className={`atlas-line ${data.role ?? ''}`}><span>{data.role}</span>{String(data.text ?? '').slice(0, 500)}</p>;
  }
  if (evt.kind === 'session.tool_use') {
    return <p className="atlas-line tool"><span>{data.name}</span>{data.summary}</p>;
  }
  if (evt.kind === 'session.tool_result') {
    return <p className={`atlas-line ${data.ok === false ? 'error' : 'tool'}`}><span>{data.ok === false ? 'failed' : 'result'}</span>{String(data.preview ?? '').slice(0, 240)}</p>;
  }
  if (evt.kind === 'permission.requested') {
    return <p className="atlas-line tool"><span>approval</span>{data.toolName}</p>;
  }
  if (evt.kind === 'session.ended') {
    return <p className={`atlas-line ${data.reason === 'error' ? 'error' : ''}`}><span>ended</span>{data.reason}{data.error ? ` · ${data.error}` : ''}</p>;
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
