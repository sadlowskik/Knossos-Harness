/* One agent, one conversation.

   The Board is a set of these; the Map's folder detail mounts the same component for
   whoever is working under the folder you clicked. It therefore assumes nothing about
   the Board's layout or its settings: give it a session and it renders the header, the
   files in scope, the transcript, the composer and the shared pause / resume / escalate
   / stop verbs, reading the snapshot and the presentation settings itself.

   The composer sends the same `POST /api/command { kind: "say" }` as AgentControls. The
   server has no notion of an attached file, so "files in scope" is a line prepended to
   the next message — explicit, visible in the transcript, and nothing invented. */

import { useEffect, useMemo, useRef, useState } from 'react';
import { ChevronDown, ChevronUp, Maximize2, Minimize2, Paperclip, RotateCcw, X } from 'lucide-react';
import { api } from '../net/client.js';
import { openChanges, openInWorkspace, selectAgent, setMode, useField } from '../state/store.js';
import { MAX_SESSION_FILES, identityFor, sessionFilesFor, withSessionFiles } from '../theater/fieldPreferences.js';
import useFieldSettings from '../theater/useFieldSettings.js';
import AgentControls from '../hud/AgentControls.jsx';
import SessionTranscript from './Transcript.jsx';
import FilePicker from './FilePicker.jsx';
import VerdictLadder from './VerdictLadder.jsx';
import { AgentMark, MetaRow, StatusPill, plainActivity, plainState } from './WorkCard.jsx';

export const TERMINAL_STATES = new Set(['done', 'cancelled', 'interrupted', 'error']);
export const ATTENTION_STATES = new Set(['blocked', 'error', 'waiting_permission']);

const normalizePath = (value) => (typeof value === 'string' ? value.replaceAll('\\', '/').replace(/^\/+|\/+$/g, '') : '');
const under = (path, dir) => {
  const p = normalizePath(path);
  return Boolean(p) && (dir === '' || p === dir || p.startsWith(`${dir}/`));
};

/** Whether this conversation is waiting on the operator. */
export function conversationNeedsYou(session, permissions = []) {
  if (!session) return false;
  return ATTENTION_STATES.has(session.state)
    || Boolean(session.pendingPermission)
    || permissions.some((item) => item.sessionId === session.id);
}

/**
 * Which sessions belong to a project, or to a folder inside it.
 *
 * `dir` matches by prefix against the file the agent is on, the folder it is anchored
 * to, every file it has touched and every file the operator attached — so "who is
 * working under src/field" has one answer, shared by the Board and the Map.
 */
export function sessionsInScope(sessions = [], {
  workspaceId = null,
  dir = null,
  includeFinished = true,
  finishedWithinMs = 0,
  now = Date.now(),
  attachedFiles = null,
} = {}) {
  const prefix = dir == null ? null : normalizePath(dir);
  return sessions.filter((session) => {
    if (workspaceId && session.workspaceId !== workspaceId) return false;
    if (TERMINAL_STATES.has(session.state)) {
      if (!includeFinished) return false;
      if (finishedWithinMs > 0) {
        const ended = session.endedAt ?? session.lastEventTs ?? session.startedAt ?? 0;
        if (ended && now - ended > finishedWithinMs) return false;
      }
    }
    if (!prefix) return true;
    const attached = attachedFiles?.[session.id] ?? [];
    return under(session.focusPath, prefix)
      || under(session.focusDir, prefix)
      || (session.touched ?? []).some((item) => under(item?.path, prefix))
      || attached.some((path) => under(path, prefix));
  });
}

/** Conversations that need a decision first, then the most recently started. */
export function sortConversations(sessions = [], permissions = []) {
  return [...sessions].sort((a, b) => {
    const attention = Number(conversationNeedsYou(b, permissions)) - Number(conversationNeedsYou(a, permissions));
    if (attention) return attention;
    const live = Number(!TERMINAL_STATES.has(b.state)) - Number(!TERMINAL_STATES.has(a.state));
    if (live) return live;
    return (b.startedAt ?? 0) - (a.startedAt ?? 0);
  });
}

const baseName = (path) => String(path).split('/').filter(Boolean).at(-1) ?? path;

export default function Conversation({
  session,
  workspace = null,
  startDir = '',
  collapsed = false,
  focused = false,
  onToggleCollapse = null,
  onToggleFocus = null,
  onStartSimilar = null,
  showProject = true,
  className = '',
}) {
  const st = useField();
  const [settings, setSettings] = useFieldSettings();
  const [text, setText] = useState('');
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState(null);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [filesOpen, setFilesOpen] = useState(false);
  const [detailOpen, setDetailOpen] = useState(false);
  const [reviewing, setReviewing] = useState(false);
  const [sentFiles, setSentFiles] = useState(null);
  const inputRef = useRef(null);

  const sessionId = session?.id ?? null;
  useEffect(() => {
    setText(''); setNote(null); setSentFiles(null);
    setPickerOpen(false); setFilesOpen(false); setDetailOpen(false); setReviewing(false);
  }, [sessionId]);

  const endpoints = st.snap.endpoints?.length ? st.snap.endpoints : st.config?.endpoints ?? [];
  const identity = useMemo(
    () => (session ? identityFor(session, endpoints, settings) : null),
    [session, endpoints, settings],
  );
  const attached = sessionFilesFor(settings, sessionId);

  if (!session) return null;

  const project = workspace ?? st.snap.workspaces.find((item) => item.id === session.workspaceId) ?? null;
  const state = plainState(session);
  const permissions = (st.snap.permissions ?? []).filter((item) => item.sessionId === session.id);
  const terminal = TERMINAL_STATES.has(session.state);
  const connected = st.connected !== false;
  const needsYou = conversationNeedsYou(session, st.snap.permissions ?? []);
  const name = identity?.displayName ?? session.name ?? session.id;

  const lastTool = typeof session.lastTool === 'string' ? session.lastTool : session.lastTool?.name;
  const elapsed = session.startedAt
    ? Math.max(0, Math.round(((st.snap.now ?? Date.now()) - session.startedAt) / 60000))
    : null;
  const pct = session.progress?.total
    ? Math.round(((session.progress.done ?? 0) / session.progress.total) * 100)
    : null;
  const cost = session.costUsd ? `$${Number(session.costUsd).toFixed(3)}` : '$0.000';
  const changed = project?.git?.files?.length ?? project?.changeCount ?? 0;
  const current = normalizePath(session.focusPath);
  const errorText = session.error ?? session.lastError ?? null;
  const activity = plainActivity(session, { connected });
  const verdict = session.lastVerdict ?? null;

  /* One number, and only when it is worth a number: a few tenths of a cent is noise, a
     budget you are spending against is not. The exact figure stays under Details. */
  const spend = Number(session.costUsd ?? 0);
  const showSpend = spend >= 0.01 || Boolean(session.budgetUsd) || session.budgetExhausted;

  /* The second decision moment. Changes on the project's branch are the thing the
     operator came to accept, so they get a card of their own rather than a cell in a
     fact row. */
  const decide = Boolean(project?.git?.branch && changed > 0);

  /* Who this agent is actually working with. The selection HUD on the canvas Map and the
     senate chamber in Plans each showed a version of this; both screens are gone, so the
     facts live on the conversation, which is the one place an agent is described. */
  const working = (() => {
    const ids = new Set();
    for (const edge of st.snap.graph?.edges ?? []) {
      if (edge.type !== 'communicates_with') continue;
      const from = String(edge.from ?? '').replace(/^agent:/, '');
      const to = String(edge.to ?? '').replace(/^agent:/, '');
      if (from === session.id) ids.add(to);
      if (to === session.id) ids.add(from);
    }
    for (const child of session.children ?? []) ids.add(child);
    return [...ids]
      .map((id) => st.snap.sessions.find((item) => item.id === id)?.name)
      .filter(Boolean);
  })();
  const delegated = (session.delegations ?? []).map((item) => item.type ?? 'subagent');

  const contextPending = attached.length > 0 && JSON.stringify(attached) !== JSON.stringify(sentFiles);

  const toggleFile = (path) => setSettings((prev) => {
    const list = sessionFilesFor(prev, session.id);
    const next = list.includes(path)
      ? list.filter((item) => item !== path)
      : [...list, path].slice(0, MAX_SESSION_FILES);
    return withSessionFiles(prev, session.id, next);
  });

  async function send() {
    const body = text.trim();
    if (!body || busy) return;
    setBusy(true); setNote(null);
    const prefix = contextPending
      ? `Files in scope:\n${attached.map((path) => `- ${path}`).join('\n')}\n\n`
      : '';
    try {
      const res = await api.command('say', { sessionIds: [session.id], text: `${prefix}${body}` });
      setText('');
      if (prefix) setSentFiles(attached);
      setNote(res?.sent === false
        ? { tone: 'bad', text: 'No live process received the message.' }
        : { tone: 'ok', text: prefix ? `Sent to ${name} with ${attached.length} file${attached.length === 1 ? '' : 's'} in scope.` : `Sent to ${name}.` });
      inputRef.current?.focus();
    } catch (e) {
      setNote({ tone: 'bad', text: e.message });
    } finally {
      setBusy(false);
    }
  }

  const head = (
    <header className="convo-head">
      {/* The model used to be printed under every name. It is one fact about an agent,
          not the headline, so it moved into Details and stayed here as the tooltip. */}
      <button
        type="button"
        className="convo-who"
        onClick={() => selectAgent(session.id)}
        title={`${name} — ${identity?.endpointAlias ?? session.model ?? 'no model'}`}
      >
        <AgentMark identity={identity} size="sm" />
        <span className="convo-who-text"><b>{name}</b></span>
      </button>
      <StatusPill state={state} compact />
      {showProject && project && (
        <span className="convo-project mono" title={project.path}>{project.name}</span>
      )}
      <span className="convo-head-actions">
        {onToggleFocus && (
          <button
            type="button"
            className="btn sm ghost icon"
            aria-pressed={focused}
            aria-label={focused ? `Restore ${name} to the column layout` : `Give ${name} the full width`}
            title={focused ? 'Back to columns' : 'Full width'}
            onClick={onToggleFocus}
          >{focused ? <Minimize2 aria-hidden="true" /> : <Maximize2 aria-hidden="true" />}</button>
        )}
        {onToggleCollapse && (
          <button
            type="button"
            className="btn sm ghost icon"
            aria-expanded={!collapsed}
            aria-label={collapsed ? `Expand the conversation with ${name}` : `Collapse the conversation with ${name}`}
            title={collapsed ? 'Expand' : 'Collapse'}
            onClick={onToggleCollapse}
          >{collapsed ? <ChevronDown aria-hidden="true" /> : <ChevronUp aria-hidden="true" />}</button>
        )}
      </span>
    </header>
  );

  const shell = `convo tone-${state.tone}${needsYou ? ' needs-you' : ''}${focused ? ' focused' : ''}${collapsed ? ' collapsed' : ''}${className ? ` ${className}` : ''}`;

  if (collapsed) {
    return (
      <article className={shell} role="listitem" aria-label={`${name}, ${state.label}`}>
        {head}
        {/* Collapsed used to be four mono facts on one line. It is now the same sentence
            the open panel leads with; the facts are one expand away. */}
        <p className="convo-oneline">{activity}</p>
      </article>
    );
  }

  return (
    <article className={shell} role="listitem" aria-label={`${name}, ${state.label}`}>
      {head}

      {/* What this conversation is working on: the operator's attachments, plus the file
          the agent is actually touching right now, which is never silently the same.

          The strip used to be permanently open, usually saying "none attached yet". It
          now opens from the paperclip beside the composer, which carries the count. */}
      {filesOpen && (
      <div className="convo-files">
        <span className="convo-files-label">
          <Paperclip aria-hidden="true" />files
        </span>
        {current && (
          <button
            type="button"
            className={`convo-file current${attached.includes(current) ? ' on' : ''}`}
            title={attached.includes(current) ? `${current} — open now and in scope` : `${current} — open now; click to put it in scope`}
            onClick={() => (attached.includes(current) ? null : toggleFile(current))}
          >
            <span className="convo-file-tag">open now</span>
            <span className="convo-file-name">{baseName(current)}</span>
          </button>
        )}
        {attached.filter((path) => path !== current).map((path) => (
          <span key={path} className="convo-file" title={path}>
            <span className="convo-file-name">{baseName(path)}</span>
            <button
              type="button"
              className="convo-file-x"
              aria-label={`Remove ${path} from the files in scope`}
              onClick={() => toggleFile(path)}
            ><X aria-hidden="true" /></button>
          </span>
        ))}
        {!current && !attached.length && <span className="convo-files-none">none attached yet</span>}
        <span className="convo-files-pick">
          <button
            type="button"
            className="btn sm ghost"
            aria-expanded={pickerOpen}
            disabled={!project}
            title={project ? 'Choose the files this agent should work on' : 'This agent has no project folder'}
            onClick={() => setPickerOpen((open) => !open)}
          >Add files</button>
          {pickerOpen && project && (
            <FilePicker
              wsId={project.id}
              startDir={startDir || normalizePath(session.focusDir)}
              attached={attached}
              onToggle={toggleFile}
              onClose={() => setPickerOpen(false)}
            />
          )}
        </span>
      </div>
      )}

      {/* The full request — tool, input, diff — is on the approval card at the top of
          the board. Restating it here was the same decision printed twice. */}
      {permissions.length > 0 && (
        <button
          type="button"
          className="convo-waiting"
          onClick={() => {
            // Atlas has the card on screen; from Rome's folder sheet, go to it.
            const card = document.querySelector('.perm');
            if (card) card.scrollIntoView({ block: 'center', behavior: 'smooth' });
            else setMode('atlas');
          }}
        >
          <i aria-hidden="true" />
          {permissions.length === 1 ? 'Waiting on your approval' : `Waiting on ${permissions.length} approvals`}
        </button>
      )}

      <p className="atlas-objective convo-objective">
        {session.target?.label ?? session.stateDetail ?? (terminal ? 'No further orders.' : 'Waiting for an assignment')}
      </p>

      <SessionTranscript
        session={session}
        variant="full"
        limit={600}
        max={200}
        connected={connected}
        className="convo-transcript"
        emptyText={terminal
          ? 'This conversation ended without reporting anything.'
          : 'Nothing reported yet. Messages, thinking and tool calls appear here as the agent works.'}
      />

      <footer className="convo-foot">
        {/* One line saying what is happening, one bar for how far along, one number for
            what it has cost — in place of the nine-cell fact row that used to open the
            footer. Every cell of that row is still below, under Details. */}
        <div className="convo-now">
          <p className="convo-now-text">{activity}</p>
          {showSpend && (
            <span
              className={`convo-spend mono${session.budgetExhausted ? ' spent-out' : ''}`}
              title={session.budgetExhausted ? 'Budget exhausted; the agent is paused.' : 'Spent so far'}
            >
              {spend >= 0.01 ? `$${spend.toFixed(2)}` : cost}
              {session.budgetUsd ? <small>{` of $${Number(session.budgetUsd).toFixed(2)}`}</small> : null}
            </span>
          )}
        </div>
        {pct != null && (
          <div
            className="convo-bar"
            style={{ '--pct': `${pct}%` }}
            role="progressbar"
            aria-valuenow={pct}
            aria-valuemin={0}
            aria-valuemax={100}
            aria-label={`${pct}% of the way through`}
          ><i aria-hidden="true" /></div>
        )}

        {/* Decision moment: changes the operator has to accept or throw away. */}
        {decide && (
          <div className={`decide${reviewing ? ' flashed' : ''}`} role="group" aria-label="Changes waiting for you">
            <span className="decide-count mono">{changed}</span>
            <span className="decide-text">
              <b>{changed === 1 ? 'file changed' : 'files changed'}</b>
              <small>{project.name} · {project.git.branch}</small>
            </span>
            {verdict && (
              <span className={`decide-verdict ${verdict.passed ? 'passed' : 'failed'}`}>
                {verdict.passed ? 'verified' : 'not verified'}
              </span>
            )}
            <button
              type="button"
              className="btn decide-go"
              onClick={() => { setReviewing(true); openChanges(project.id); setTimeout(() => setReviewing(false), 900); }}
            >Review the changes</button>
            {reviewing && <span className="decide-said" role="status">Opened in the workspace.</span>}
          </div>
        )}

        {errorText && <p className="convo-error" role="alert">{errorText}</p>}
        {!connected && (
          <p className="convo-error" role="status">
            Not connected to the Field server, so nothing here is live and messages will not send.
          </p>
        )}

        {terminal ? (
          /* What it finished with is already the line at the top of this footer, so
             what is left here is the two things you can do about it. */
          <div className="convo-ended">
            {onStartSimilar && (
              <button type="button" className="btn sm" onClick={() => onStartSimilar(session)}>
                <RotateCcw aria-hidden="true" />
                Start another {name} on {project?.name ?? 'this project'}
              </button>
            )}
            <button type="button" className="btn sm ghost" onClick={() => openInWorkspace({ type: 'session', id: session.id })}>
              Open the full transcript
            </button>
          </div>
        ) : (
          <>
            <AgentControls
              session={session}
              compact
              showSay={false}
              onOpen={() => openInWorkspace({ type: 'session', id: session.id })}
            />
            <div className="convo-composer">
              <textarea
                ref={inputRef}
                rows={2}
                value={text}
                placeholder={`Tell ${name} what to do next… (Ctrl+Enter)`}
                aria-label={`Message ${name}`}
                onChange={(e) => setText(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); send(); } }}
              />
              <div className="convo-composer-row">
                <button
                  type="button"
                  className={`convo-clip${filesOpen ? ' on' : ''}`}
                  aria-expanded={filesOpen}
                  aria-label={attached.length
                    ? `Files in scope: ${attached.length}. Show the list.`
                    : 'Choose the files this agent should work on'}
                  title="Files in scope"
                  onClick={() => setFilesOpen((open) => !open)}
                >
                  <Paperclip aria-hidden="true" />
                  {attached.length > 0 && <b className="mono">{attached.length}</b>}
                </button>
                <span className="convo-composer-hint">
                  {contextPending
                    ? `The next message carries ${attached.length} file${attached.length === 1 ? '' : 's'} in scope.`
                    : ''}
                </span>
                <button
                  type="button"
                  className="btn sm primary"
                  disabled={busy || !text.trim() || !connected}
                  onClick={send}
                >{busy ? 'Sending…' : 'Send'}</button>
              </div>
            </div>
          </>
        )}
        {note && <p className={`agent-controls-note ${note.tone}`} role="status">{note.text}</p>}

        {/* Nothing was deleted. Everything the footer used to shout — the verification
            ladder, the budget, the tool and edit counts, elapsed, context, role,
            delegations, collaborators, the project path — is here, one tap down. */}
        <button
          type="button"
          className="convo-more"
          aria-expanded={detailOpen}
          onClick={() => setDetailOpen((open) => !open)}
        >{detailOpen ? <ChevronUp aria-hidden="true" /> : <ChevronDown aria-hidden="true" />}Details</button>
        {detailOpen && (
          <div className="convo-detail">
            <MetaRow
              className="work-meta-foot"
              items={[
                showProject && project && { key: 'project', label: 'project', value: project.name, title: project.path },
                { key: 'model', label: 'model', value: identity?.endpointAlias ?? session.model ?? 'no model' },
                project?.git?.branch && changed
                  ? { key: 'changed', label: 'changed', value: `${changed} files`, onClick: () => openChanges(project.id), title: 'Review, accept or revert the changes' }
                  : null,
                session.role && { key: 'role', label: 'role', value: session.role },
                lastTool && { key: 'tool', label: 'tool', value: lastTool },
                session.toolCount != null && {
                  key: 'tools', label: 'tools',
                  value: `${session.toolCount}${session.editCount ? ` · ${session.editCount} edits` : ''}`,
                },
                pct != null && { key: 'progress', label: 'progress', value: `${pct}%` },
                elapsed != null && { key: 'elapsed', label: 'elapsed', value: elapsed < 1 ? '<1m' : `${elapsed}m` },
                session.contextPct != null && { key: 'context', label: 'context', value: `${session.contextPct}%` },
                delegated.length > 0 && { key: 'delegated', label: 'delegated', value: delegated.join(', ') },
                working.length > 0 && { key: 'working-with', label: 'working with', value: working.join(', '), title: 'Agents this one is talking to or has delegated to' },
                {
                  key: 'cost',
                  label: session.budgetExhausted ? 'budget' : 'cost',
                  tone: session.budgetExhausted ? 'failed' : null,
                  title: session.budgetExhausted ? 'budget exhausted; the agent is paused' : 'spent / budget',
                  value: `${cost}${session.budgetUsd ? ` / ${Number(session.budgetUsd).toFixed(2)}` : ''}${session.budgetExhausted ? ' · exhausted' : ''}`,
                },
                attached.length > 0 && { key: 'scope', label: 'in scope', value: `${attached.length} file${attached.length === 1 ? '' : 's'}` },
              ]}
            />
            {verdict && <VerdictLadder verdict={verdict} compact />}
          </div>
        )}
      </footer>
    </article>
  );
}
