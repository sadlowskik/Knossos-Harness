/* Atlas is every conversation at once.

   It used to be a grid of project cards with the agents working on them folded inside,
   which is backwards: you do not talk to a repository. One panel per agent — running,
   waiting, or finished in the last half hour — each one a chat window with the files
   that agent has in scope. Projects did not disappear; they became the filter at the
   top, because "show me only what is happening in Cameo" is a question about a set of
   conversations, not a reason to draw a card for a folder.

   It is one of the two destinations. Rome answers "where is the work happening"; this
   answers "what is every agent doing", which is why an all-conversations view belongs
   here and not on a screen of its own. The panel itself lives in ui/Conversation.jsx so
   Rome can mount the same thing in its folder detail; this file composes them into
   columns. */

import { useEffect, useMemo, useState } from 'react';
import { Settings } from 'lucide-react';
import { useField } from '../state/store.js';
import useFieldSettings from '../theater/useFieldSettings.js';
import FieldSettings from '../theater/FieldSettings.jsx';
import PermissionRequests from '../hud/PermissionRequests.jsx';
import PowerSources from '../setup/PowerSources.jsx';
import ContextMenu from '../hud/ContextMenu.jsx';
import EmptyState from '../ui/EmptyState.jsx';
import Conversation, { sessionsInScope, sortConversations } from '../ui/Conversation.jsx';

// A conversation that ended stays on Atlas for this long: long enough to read what
// happened and start another agent on the same work, not so long that Atlas becomes a
// graveyard. Everything older is on Rome's time rail.
const RECENTLY_FINISHED_MS = 30 * 60 * 1000;

export default function AtlasMode() {
  const st = useField();
  const [settings, setSettings] = useFieldSettings();
  const [modelsOpen, setModelsOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [starter, setStarter] = useState(null);   // { workspace, pane, screen, agentId }
  const [project, setProject] = useState('all');  // the slim project filter
  const [focusedId, setFocusedId] = useState(null);
  const [collapsedIds, setCollapsedIds] = useState(() => new Set());

  const workspaces = st.snap.workspaces.filter((item) => item.mounted);
  const endpoints = st.snap.endpoints?.length ? st.snap.endpoints : st.config?.endpoints ?? [];
  const pendingCount = st.snap.permissions?.length ?? 0;

  const startAgent = (workspace, e, { pane = 'spawn', agentId = null } = {}) => setStarter({
    workspace,
    pane,
    agentId,
    screen: { x: e?.clientX ?? window.innerWidth / 2 - 140, y: e?.clientY ?? 120 },
  });

  // Ctrl+Alt+N from App.jsx: start a conversation on the filtered project, or the first.
  useEffect(() => {
    const open = () => {
      if (!workspaces.length) return;
      const workspace = workspaces.find((item) => item.id === project) ?? workspaces[0];
      setStarter({ workspace, pane: 'spawn', agentId: null, screen: { x: Math.max(8, window.innerWidth / 2 - 180), y: 120 } });
    };
    window.addEventListener('field:start-agent', open);
    return () => window.removeEventListener('field:start-agent', open);
  }, [workspaces, project]);

  // Running, waiting, and recently finished — in that order, decisions first.
  const all = useMemo(() => sortConversations(
    sessionsInScope(st.snap.sessions, {
      finishedWithinMs: RECENTLY_FINISHED_MS,
      now: st.snap.now ?? Date.now(),
    }),
    st.snap.permissions ?? [],
  ), [st.snap.sessions, st.snap.permissions, st.snap.now]);

  const shown = useMemo(
    () => (project === 'all' ? all : all.filter((session) => session.workspaceId === project)),
    [all, project],
  );

  // A focused conversation takes the whole width; otherwise up to three columns, and one
  // wide column when there is only one conversation to read.
  const focused = focusedId ? shown.find((session) => session.id === focusedId) ?? null : null;
  const panels = focused ? [focused] : shown;
  const columns = focused ? 1 : Math.min(3, panels.length || 1);
  const projectOf = (session) => workspaces.find((item) => item.id === session.workspaceId) ?? null;

  const counts = useMemo(() => {
    const map = new Map();
    for (const session of all) map.set(session.workspaceId, (map.get(session.workspaceId) ?? 0) + 1);
    return map;
  }, [all]);

  return (
    <div className="atlas-board">
      {/* One row: which project you are looking at, the primary action, the gear. */}
      <header className="atlas-head">
        {workspaces.length > 1 && all.length > 0 && (
          <label className="convo-filter">
            <span className="label">project</span>
            <select value={project} onChange={(e) => { setProject(e.target.value); setFocusedId(null); }}>
              <option value="all">All projects · {all.length}</option>
              {workspaces.map((workspace) => (
                <option key={workspace.id} value={workspace.id}>
                  {workspace.name}{counts.get(workspace.id) ? ` · ${counts.get(workspace.id)}` : ' · none'}
                </option>
              ))}
            </select>
          </label>
        )}
        <span className="grow" />
        <div className="atlas-actions">
          {workspaces.length > 0 && all.length > 0 && (
            <button
              type="button"
              className="btn primary"
              onClick={(e) => startAgent(workspaces.find((item) => item.id === project) ?? workspaces[0], e)}
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

        {workspaces.length > 0 && endpoints.length === 0 && (
          <section className="atlas-approvals atlas-nudge" aria-label="No model yet">
            <h2 className="atlas-section-title"><i aria-hidden="true" />No model yet</h2>
            <p>Agents cannot start until a model is set up: a Cameo box, Ollama on this machine, or a provider key.</p>
            <div className="atlas-empty-actions">
              <button type="button" className="btn" onClick={() => setModelsOpen(true)}>Set up a model</button>
            </div>
          </section>
        )}

        {panels.length > 0 && (
          <div className="atlas-convos" data-columns={columns} role="list">
            {panels.map((session) => (
              <Conversation
                key={session.id}
                session={session}
                workspace={projectOf(session)}
                focused={focusedId === session.id}
                collapsed={collapsedIds.has(session.id) && focusedId !== session.id}
                showProject={project === 'all'}
                onToggleFocus={() => setFocusedId((id) => (id === session.id ? null : session.id))}
                onToggleCollapse={() => setCollapsedIds((prev) => {
                  const next = new Set(prev);
                  if (next.has(session.id)) next.delete(session.id); else next.add(session.id);
                  return next;
                })}
                onStartSimilar={(ended) => startAgent(
                  projectOf(ended) ?? workspaces[0],
                  null,
                  { agentId: ended.agentId ?? null },
                )}
              />
            ))}
          </div>
        )}

        {panels.length === 0 && workspaces.length === 0 && (
          <EmptyState
            title="No project open"
            action={<button type="button" className="btn primary" onClick={() => setModelsOpen(true)}>Set up a model</button>}
          >
            Add a project folder under <code>workspaces</code> in <code>field/field.yaml</code>, then restart Field.
            Every agent you start on it gets a conversation here.
          </EmptyState>
        )}

        {panels.length === 0 && workspaces.length > 0 && all.length === 0 && (
          <EmptyState
            title="No agents running"
            action={(
              <button
                type="button"
                className="btn primary"
                onClick={(e) => startAgent(workspaces.find((item) => item.id === project) ?? workspaces[0], e)}
                aria-keyshortcuts="Control+Alt+N"
              >Start an agent<kbd className="btn-hint">Ctrl+Alt+N</kbd></button>
            )}
          >
            Atlas is a conversation per agent: talk to it, choose the files it should work on, and pause or stop
            it from the same panel. Start one to open the first conversation.
          </EmptyState>
        )}

        {panels.length === 0 && all.length > 0 && (
          <EmptyState
            title={`No agents on ${workspaces.find((item) => item.id === project)?.name ?? 'this project'}`}
            action={<button type="button" className="btn" onClick={() => setProject('all')}>Show every project</button>}
          >
            {all.length === 1 ? 'One conversation is running' : `${all.length} conversations are running`} on other projects.
          </EmptyState>
        )}

        {!st.connected && panels.length === 0 && (
          <p className="convo-offline" role="status">Not connected to the Field server, so this is the last state the browser received.</p>
        )}
      </div>

      {modelsOpen && <PowerSources onClose={() => setModelsOpen(false)} settings={settings} setSettings={setSettings} />}
      {settingsOpen && (
        <FieldSettings
          standalone
          settings={settings}
          setSettings={setSettings}
          selected={st.snap.sessions.find((session) => session.id === st.activeSessionId) ?? null}
          config={st.config}
          onClose={() => setSettingsOpen(false)}
          onOpenModels={() => { setSettingsOpen(false); setModelsOpen(true); }}
        />
      )}
      {starter && (
        <ContextMenu
          fixed
          initialPane={starter.pane ?? 'spawn'}
          initialAgentId={starter.agentId ?? null}
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
