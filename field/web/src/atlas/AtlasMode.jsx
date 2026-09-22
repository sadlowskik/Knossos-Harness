import { useEffect, useMemo, useRef, useState } from 'react';
import { Plus, Settings } from 'lucide-react';
import { api, on } from '../net/client.js';
import { openChanges, openInWorkspace, selectAgent, useField } from '../state/store.js';
import { identityFor } from '../theater/fieldPreferences.js';
import useFieldSettings from '../theater/useFieldSettings.js';
import FieldSettings from '../theater/FieldSettings.jsx';
import PermissionRequests, { isPrivilegedTool } from '../hud/PermissionRequests.jsx';
import PowerSources from '../setup/PowerSources.jsx';
import ContextMenu from '../hud/ContextMenu.jsx';
import AgentControls from '../hud/AgentControls.jsx';
import EmptyState from '../ui/EmptyState.jsx';
import WorkCard, { AgentMark, MetaRow, StatusPill, plainState } from '../ui/WorkCard.jsx';

const TERMINAL = new Set(['done', 'cancelled', 'interrupted', 'error']);
const ATTENTION = new Set(['blocked', 'error', 'waiting_permission']);
const TRACE_KINDS = new Set([
  'session.spawned', 'session.message', 'session.thinking', 'session.tool_use',
  'session.tool_result', 'session.ended', 'permission.requested', 'permission.decided',
  'work.verified',
]);

// plainState lives with the card it labels; re-exported because the Map imports it here.
export { plainState };

export default function AtlasMode() {
  const st = useField();
  const [settings, setSettings] = useFieldSettings();
  const selectedId = st.activeSessionId;
  const [modelsOpen, setModelsOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [starter, setStarter] = useState(null); // { workspace, screen }
  const workspaces = st.snap.workspaces.filter((item) => item.mounted);
  const startAgent = (workspace, e, pane = 'spawn') => setStarter({
    workspace,
    pane,
    screen: { x: e?.clientX ?? window.innerWidth / 2 - 140, y: e?.clientY ?? 120 },
  });
  // Ctrl+Alt+N from App.jsx: open the starter on the first mounted project. Defining a new
  // agent now lives inside that dialog rather than as a second header button.
  useEffect(() => {
    const open = () => {
      if (!workspaces.length) return;
      setStarter({ workspace: workspaces[0], pane: 'spawn', screen: { x: Math.max(8, window.innerWidth / 2 - 180), y: 120 } });
    };
    window.addEventListener('field:start-agent', open);
    return () => window.removeEventListener('field:start-agent', open);
  }, [workspaces]);
  const sessions = useMemo(
    () => [...st.snap.sessions].sort((a, b) => (b.startedAt ?? 0) - (a.startedAt ?? 0)),
    [st.snap.sessions],
  );
  const live = sessions.filter((session) => !TERMINAL.has(session.state));
  const endpoints = st.snap.endpoints?.length ? st.snap.endpoints : st.config?.endpoints ?? [];
  const identities = useMemo(
    () => new Map(sessions.map((session) => [session.id, identityFor(session, endpoints, settings)])),
    [sessions, endpoints, settings],
  );
  const columns = useMemo(() => buildColumns(workspaces, live, sessions), [workspaces, live, sessions]);
  const pendingCount = st.snap.permissions?.length ?? 0;

  return (
    <div className="atlas-board">
      {/* One row: the primary action and the gear. The live counters are in the topbar,
          and the heading only repeated what the nav already says. */}
      <header className="atlas-head">
        <div className="atlas-actions">
          {workspaces.length > 0 && (
            <button
              type="button"
              className="btn primary"
              onClick={(e) => startAgent(workspaces[0], e)}
              aria-keyshortcuts="Control+Alt+N"
            >Start an agent<kbd className="btn-hint">Ctrl+Alt+N</kbd></button>
          )}
          <button type="button" className="btn ghost icon" aria-label="Field settings" onClick={() => setSettingsOpen(true)}>
            <Settings aria-hidden="true" />
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
          <EmptyState
            title="No project open"
            action={<button type="button" className="btn primary" onClick={() => setModelsOpen(true)}>Set up a model</button>}
          >
            Add a project folder under <code>workspaces</code> in <code>field/field.yaml</code>, then restart Field. Each
            project gets a card here, and each agent working on it gets its own.
          </EmptyState>
        )}
      </div>
      {modelsOpen && <PowerSources onClose={() => setModelsOpen(false)} settings={settings} setSettings={setSettings} />}
      {settingsOpen && (
        <FieldSettings
          standalone
          settings={settings}
          setSettings={setSettings}
          selected={sessions.find((session) => session.id === selectedId) ?? null}
          config={st.config}
          onClose={() => setSettingsOpen(false)}
          onOpenModels={() => { setSettingsOpen(false); setModelsOpen(true); }}
        />
      )}
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
  // Cards that need a decision sort to the front of the grid. That, and a hairline in the
  // attention colour, replaces the border that used to pulse forever.
  const needsYou = (column) => Number(Boolean(
    column.session && (ATTENTION.has(column.session.state) || column.session.pendingPermission),
  ));
  return columns.sort((a, b) => needsYou(b) - needsYou(a));
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
  const changed = workspace?.git?.files?.length ?? workspace?.changeCount ?? 0;

  const meta = (workspace || session?.cwd) ? (
    <MetaRow
      className="work-card-context"
      items={[
        { key: 'project', value: workspace?.name ?? session?.cwd, title: workspace?.path },
        workspace?.git?.branch && { key: 'branch', value: workspace.git.branch },
        workspace?.git?.branch && changed
          ? { key: 'changed', value: `${changed} changed`, onClick: () => openChanges(workspace.id), title: 'Review, accept or revert the changes' }
          : null,
      ]}
    />
  ) : null;

  const notices = pending.length ? (
    <div className="work-card-notices">
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
    </div>
  ) : null;

  const footer = session ? (
    <>
      <MetaRow
        items={[
          lastTool && { key: 'tool', label: 'tool', value: lastTool },
          pct != null && { key: 'progress', label: 'progress', value: `${pct}%` },
          elapsed != null && { key: 'elapsed', label: 'elapsed', value: elapsed < 1 ? '<1m' : `${elapsed}m` },
          session.contextPct != null && { key: 'context', label: 'context', value: `${session.contextPct}%` },
          {
            key: 'cost',
            label: session.budgetExhausted ? 'budget' : 'cost',
            tone: session.budgetExhausted ? 'failed' : null,
            title: session.budgetExhausted ? 'budget exhausted; the agent is paused' : 'spent / budget',
            value: `${cost || '$0.000'}${session.budgetUsd ? ` / ${Number(session.budgetUsd).toFixed(2)}` : ''}${session.budgetExhausted ? ' · exhausted' : ''}`,
          },
        ]}
        className="work-meta-foot"
      />
      {session.error && <p className="atlas-foot-error" role="alert">{session.error}</p>}
      <AgentControls session={session} compact onOpen={() => openInWorkspace({ type: 'session', id: session.id })} />
    </>
  ) : null;

  return (
    <WorkCard
      tone={state.tone}
      selected={selected}
      needsYou={needsYou}
      hollow={!session}
      ariaLabel={session ? `${identity?.displayName ?? session.name ?? session.id}, ${state.label}` : `${workspace?.name ?? 'Project'}, no agent yet`}
      mark={session
        ? <AgentMark identity={identity} />
        : <AgentMark project name={workspace?.name ?? '?'} />}
      title={session ? (identity?.displayName ?? session.name ?? session.id) : (workspace?.name ?? 'Project')}
      subtitle={session ? (identity?.endpointAlias ?? session.model ?? 'no model') : 'no agent yet'}
      onTitleClick={session ? () => selectAgent(session.id) : null}
      titleTitle={session ? 'Select this agent' : undefined}
      pill={<StatusPill state={state} />}
      action={!session && (
        <button
          type="button"
          className="btn quiet-add"
          aria-label={`Start an agent on ${workspace?.name ?? 'this project'}`}
          title={`Start an agent on ${workspace?.name ?? 'this project'}`}
          onClick={(e) => onStart(workspace, e)}
        ><Plus aria-hidden="true" /></button>
      )}
      meta={meta}
      notices={notices}
      footer={footer}
    >
      {session
        ? <ColumnTranscript session={session} campaigns={campaigns} />
        : (
          <div className="atlas-idle">
            <b>No agent working on this project yet.</b>
            <p>Use the + above to start one with orders, a model and a thinking level. It shows up here the moment it begins; routines and plans can also assign agents.</p>
            <div className="atlas-idle-actions">
              <button type="button" className="btn ghost" onClick={() => openInWorkspace({ type: 'workspace', workspaceId: workspace?.id, path: '' })}>Open files</button>
            </div>
          </div>
        )}
    </WorkCard>
  );
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
