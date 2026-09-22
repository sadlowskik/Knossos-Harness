import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api, on } from '../net/client.js';
import { selectProject, setState, useField } from '../state/store.js';
import { renderMarkdown } from './md.js';
import { validateBrowserUrl } from './browser-url.js';
import VerdictLadder from '../ui/VerdictLadder.jsx';

export default function WorkspaceMode() {
  const st = useField();
  const workspaces = st.snap.workspaces.filter((w) => w.mounted);
  const focus = st.focus;

  const focusSession = focus?.type === 'session'
    ? st.snap.sessions.find((s) => s.id === focus.id)
    : null;

  const [wsId, setWsId] = useState(st.activeWorkspaceId);
  const [openPath, setOpenPath] = useState(null);
  // Two destinations. Markdown and diff are not peers of the file: they are ways of
  // looking at the one that is open, so they live in a segment inside the file pane.
  const [centerTab, setCenterTab] = useState('file');
  const [fileView, setFileView] = useState('source');
  const [rightTab, setRightTab] = useState('terminal');
  const [leftTab, setLeftTab] = useState('files');

  // Follow whatever the operator opened from the Field.
  useEffect(() => {
    if (!focus) return;
    if (focus.type === 'session' && focusSession) {
      setWsId(focusSession.workspaceId);
      setCenterTab('transcript');
    } else if (focus.workspaceId) {
      setWsId(focus.workspaceId);
      if (focus.path) setOpenPath(focus.path);
      if (focus.view === 'changes') setLeftTab('changes');
    } else if (focus.type === 'browser') {
      setRightTab('browser');
    }
  }, [focus, focusSession]);

  useEffect(() => {
    if (st.activeWorkspaceId && workspaces.some((workspace) => workspace.id === st.activeWorkspaceId)) {
      setWsId(st.activeWorkspaceId);
    }
  }, [st.activeWorkspaceId, workspaces]);

  useEffect(() => {
    if (!wsId && workspaces.length) setWsId(workspaces[0].id);
  }, [workspaces, wsId]);

  const ws = st.snap.workspaces.find((w) => w.id === wsId);
  // Rendered is offered for Markdown; Diff only for a path git actually reports as changed.
  const canRender = Boolean(openPath && openPath.toLowerCase().endsWith('.md'));
  const canDiff = Boolean(openPath && (ws?.git?.files ?? []).some((f) => f.path === openPath));
  const view = (fileView === 'rendered' && !canRender) || (fileView === 'diff' && !canDiff) ? 'source' : fileView;
  const VIEWS = [
    { id: 'source', label: 'Source', on: true },
    { id: 'rendered', label: 'Rendered', on: canRender },
    { id: 'diff', label: 'Diff', on: canDiff },
  ].filter((item) => item.on);

  if (!workspaces.length) {
    return <div className="empty"><b>No project is mounted.</b>Add a folder under <code>workspaces</code> in field/field.yaml and restart Field.</div>;
  }

  return (
    <div className="panes">
      <div className="pane" style={{ flex: '0 0 244px' }}>
        <div className="pane-head">
          <select
            className="ws-select"
            value={wsId ?? ''}
            onChange={(e) => {
              setWsId(e.target.value);
              setOpenPath(null);
              selectProject(e.target.value);
            }}
            style={{
              background: 'transparent', border: 0, color: 'var(--ink)',
              fontSize: 12, outline: 'none', flex: '1 1 auto',
            }}
          >
            {workspaces.map((w) => <option key={w.id} value={w.id}>{w.name}</option>)}
          </select>
          {ws?.git && <span className="label">{ws.git.branch}</span>}
        </div>
        <div className="pane-head pane-head-sub">
          <div className="tabs">
            {[['files', 'files'], ['changes', `changes${ws?.git?.files?.length ? ` · ${ws.git.files.length}` : ''}`]].map(([t, label]) => (
              <button
                key={t}
                className={`tab${leftTab === t ? ' on' : ''}`}
                onClick={() => setLeftTab(t)}
                type="button"
              >{label}</button>
            ))}
          </div>
        </div>
        <div className="pane-body">
          {leftTab === 'changes'
            ? (
              <Changes
                wsId={wsId}
                openPath={openPath}
                onOpen={(p) => { setOpenPath(p); setCenterTab('file'); setFileView('diff'); }}
              />
            )
            : (
              <FileTree
                wsId={wsId}
                git={ws?.git}
                openPath={openPath}
                onOpen={(p) => {
                  setOpenPath(p);
                  setCenterTab('file');
                  setFileView(p.toLowerCase().endsWith('.md') ? 'rendered' : 'source');
                }}
              />
            )}
        </div>
      </div>

      <div className="pane" style={{ flex: '1 1 auto' }}>
        <div className="pane-head">
          <div className="tabs">
            {[['file', 'File'], ['transcript', 'Transcript']].map(([t, label]) => (
              <button
                key={t}
                className={`tab${centerTab === t ? ' on' : ''}`}
                onClick={() => setCenterTab(t)}
                type="button"
              >{label}</button>
            ))}
          </div>
          <span className="grow pane-head-path label">
            {centerTab === 'transcript' ? (focusSession?.name ?? 'no agent selected') : (openPath ?? '—')}
          </span>
          {centerTab === 'file' && VIEWS.length > 1 && (
            <div className="viewctl" role="group" aria-label="File view">
              {VIEWS.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  className={`viewctl-item${view === item.id ? ' on' : ''}`}
                  aria-pressed={view === item.id}
                  onClick={() => setFileView(item.id)}
                >{item.label}</button>
              ))}
            </div>
          )}
        </div>
        <div className="pane-body">
          {centerTab === 'transcript'
            ? <Transcript session={focusSession} />
            : view === 'diff'
              ? <Diff wsId={wsId} path={openPath} />
              : <FileView wsId={wsId} path={openPath} markdown={view === 'rendered'} />}
        </div>
      </div>

      <div className="pane" style={{ flex: '0 0 400px' }}>
        <div className="pane-head">
          <div className="tabs">
            {['terminal', 'browser'].map((t) => (
              <button
                key={t}
                className={`tab${rightTab === t ? ' on' : ''}`}
                onClick={() => setRightTab(t)}
                type="button"
              >{t}</button>
            ))}
          </div>
        </div>
        <div className="pane-body" style={{ overflow: 'hidden' }}>
          {rightTab === 'terminal'
            ? <Terminal wsId={wsId} />
            : <BrowserPane initialUrl={focus?.type === 'browser' ? focus.url : null} />}
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ tree */

function FileTree({ wsId, git, openPath, onOpen }) {
  const [expanded, setExpanded] = useState(new Set(['']));
  const [dirs, setDirs] = useState({});
  const changed = useMemo(
    () => new Set((git?.files ?? []).map((f) => f.path)),
    [git],
  );

  const load = useCallback(async (dir) => {
    if (!wsId) return;
    try {
      const r = await api.tree(wsId, dir);
      setDirs((d) => ({ ...d, [dir]: r.entries }));
    } catch (e) {
      setDirs((d) => ({ ...d, [dir]: [] }));
    }
  }, [wsId]);

  useEffect(() => { setDirs({}); setExpanded(new Set([''])); if (wsId) load(''); }, [wsId, load]);

  const toggle = (dir) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(dir)) next.delete(dir);
      else { next.add(dir); if (!dirs[dir]) load(dir); }
      return next;
    });
  };

  const rows = [];
  const walk = (dir, depth) => {
    for (const e of dirs[dir] ?? []) {
      const isOpen = expanded.has(e.path);
      rows.push(
        <button
          key={e.path}
          className={`tree-row${openPath === e.path ? ' on' : ''}`}
          style={{ paddingLeft: 10 + depth * 11 }}
          onClick={() => (e.dir ? toggle(e.path) : onOpen(e.path))}
          type="button"
        >
          <span className="caret">{e.dir ? (isOpen ? '▾' : '▸') : ''}</span>
          <span className="nm">{e.name}</span>
          {changed.has(e.path) && <span className="chg" title="changed in git" />}
          {!e.dir && e.size != null && <span className="sz">{fmtSize(e.size)}</span>}
        </button>,
      );
      if (e.dir && isOpen) walk(e.path, depth + 1);
    }
  };
  walk('', 0);

  return <div className="tree">{rows.length ? rows : <div className="empty">empty</div>}</div>;
}

function fmtSize(n) {
  if (n < 1024) return `${n}b`;
  if (n < 1024 * 1024) return `${Math.round(n / 1024)}k`;
  return `${(n / 1048576).toFixed(1)}m`;
}

/* ------------------------------------------------------------------ changes */

const DEFAULT_COMMIT_MESSAGE = 'Field: accept agent changes';

// Review what the agents changed in the working tree: every file with its line
// counts, the per-file diff one click away, and the two operator decisions,
// accept (commit) or revert. Both go through the server and land in History.
function Changes({ wsId, openPath, onOpen }) {
  const [changes, setChanges] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(null);
  const [dialog, setDialog] = useState(null); // { kind: 'commit' | 'revert' }
  const [message, setMessage] = useState(DEFAULT_COMMIT_MESSAGE);
  const generation = useRef(0);

  const refresh = useCallback(async () => {
    if (!wsId) { setChanges(null); return; }
    const mine = ++generation.current;
    try {
      const r = await api.gitChanges(wsId);
      if (mine === generation.current) { setChanges(r); setError(null); }
    } catch (e) {
      if (mine === generation.current) { setChanges(null); setError(e.message); }
    }
  }, [wsId]);

  // Re-read on open and whenever the server reports the tree changed; git.status
  // arrives from the watcher, the other two from our own accept / revert.
  useEffect(() => { refresh(); }, [refresh]);
  useEffect(() => on('event', (evt) => {
    if (['git.status', 'git.reverted', 'git.committed'].includes(evt.kind) && (evt.data?.workspaceId ?? evt.data?.ws) === wsId) refresh();
  }), [wsId, refresh]);

  const files = changes?.files ?? [];

  const commit = async () => {
    const text = message.trim() || DEFAULT_COMMIT_MESSAGE;
    setBusy('commit');
    try {
      await api.gitCommit(wsId, text);
      setDialog(null);
      setMessage(DEFAULT_COMMIT_MESSAGE);
      await refresh();
    } catch (e) {
      setError(e.message);
    } finally { setBusy(null); }
  };

  const revert = async () => {
    setBusy('revert');
    try {
      await api.gitRevert(wsId);
      setDialog(null);
      await refresh();
    } catch (e) {
      setError(e.message);
    } finally { setBusy(null); }
  };

  if (!wsId) return <div className="empty">No project selected.</div>;

  return (
    <div className="changes">
      <div className="changes-summary mono">
        {changes
          ? <><b>{files.length}</b> changed · <span className="add">+{changes.additions ?? 0}</span> <span className="del">−{changes.deletions ?? 0}</span></>
          : error ? 'git unavailable' : 'reading…'}
      </div>
      {error && <div className="changes-error" role="alert">{error}</div>}
      <div className="changes-list">
        {files.map((f) => (
          <button
            key={f.path}
            type="button"
            className={`tree-row change-row${openPath === f.path ? ' on' : ''}`}
            onClick={() => onOpen(f.path)}
            title={`${f.status}${f.from ? ` from ${f.from}` : ''}: click to open the diff`}
          >
            <span className={`change-status st-${f.status}`}>{STATUS_MARK[f.status] ?? '•'}</span>
            <span className="nm">{f.path}</span>
            <span className="change-counts">
              {f.additions > 0 && <span className="add">+{f.additions}</span>}
              {f.deletions > 0 && <span className="del">−{f.deletions}</span>}
            </span>
          </button>
        ))}
        {changes && !files.length && <div className="empty"><b>Working tree is clean.</b>Nothing to accept or revert.</div>}
      </div>
      <div className="changes-actions">
        <button
          type="button"
          className="btn primary sm"
          disabled={!files.length || busy != null}
          onClick={() => setDialog({ kind: 'commit' })}
        >Accept: commit</button>
        <button
          type="button"
          className="btn danger sm"
          disabled={!files.length || busy != null}
          onClick={() => setDialog({ kind: 'revert' })}
        >Revert</button>
      </div>
      {dialog?.kind === 'commit' && (
        <div className="changes-dialog" role="dialog" aria-label="Commit changes">
          <span className="label">commit {files.length} file{files.length === 1 ? '' : 's'}</span>
          <input
            className="changes-message mono"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') commit(); if (e.key === 'Escape') setDialog(null); }}
            placeholder={DEFAULT_COMMIT_MESSAGE}
            autoFocus
          />
          <div className="changes-actions">
            <button type="button" className="btn ghost sm" disabled={busy != null} onClick={() => setDialog(null)}>Cancel</button>
            <button type="button" className="btn primary sm" disabled={busy != null} onClick={commit}>{busy === 'commit' ? 'Committing…' : 'Commit'}</button>
          </div>
        </div>
      )}
      {dialog?.kind === 'revert' && (
        <div className="changes-dialog danger" role="dialog" aria-label="Revert changes">
          <span className="label">discard changes in {files.length} file{files.length === 1 ? '' : 's'}</span>
          <ul className="changes-dialog-files mono">
            {files.slice(0, 12).map((f) => <li key={f.path}>{f.path}</li>)}
            {files.length > 12 && <li>… and {files.length - 12} more</li>}
          </ul>
          <p className="perm-warn">Tracked files go back to HEAD; new files are deleted. This cannot be undone.</p>
          <div className="changes-actions">
            <button type="button" className="btn ghost sm" disabled={busy != null} onClick={() => setDialog(null)}>Keep changes</button>
            <button type="button" className="btn danger sm" disabled={busy != null} onClick={revert}>{busy === 'revert' ? 'Reverting…' : 'Revert all'}</button>
          </div>
        </div>
      )}
    </div>
  );
}

const STATUS_MARK = { modified: 'M', added: 'A', deleted: 'D', renamed: 'R', untracked: '?', copied: 'C', conflicted: 'U' };

/* ------------------------------------------------------------------ file */

function FileView({ wsId, path, markdown }) {
  const [content, setContent] = useState('');
  const [loaded, setLoaded] = useState(null);
  const [dirty, setDirty] = useState(false);
  const [note, setNote] = useState(null);

  useEffect(() => {
    if (!wsId || !path) { setContent(''); setLoaded(null); return; }
    let live = true;
    api.readFile(wsId, path)
      .then((r) => {
        if (!live) return;
        if (r.tooLarge) { setContent(`(file is ${fmtSize(r.size)} — too large to open here)`); setLoaded(null); }
        else { setContent(r.content); setLoaded(r.content); }
        setDirty(false);
      })
      .catch((e) => { if (live) { setContent(`(${e.message})`); setLoaded(null); } });
    return () => { live = false; };
  }, [wsId, path]);

  const save = async () => {
    try {
      await api.writeFile(wsId, path, content);
      setLoaded(content); setDirty(false);
      setNote('saved'); setTimeout(() => setNote(null), 1600);
    } catch (e) { setNote(e.message); }
  };

  useEffect(() => {
    const onKey = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 's') { e.preventDefault(); if (dirty) save(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  if (!path) return <div className="empty"><b>No file open.</b>Pick one from the tree.</div>;

  if (markdown) {
    return (
      <div className="md-view" dangerouslySetInnerHTML={{ __html: renderMarkdown(content) }} />
    );
  }

  return (
    <div className="editor">
      <textarea
        value={content}
        spellCheck={false}
        onChange={(e) => { setContent(e.target.value); setDirty(e.target.value !== loaded); }}
      />
      {(dirty || note) && (
        <div style={{
          position: 'absolute', right: 12, bottom: 10, display: 'flex', gap: 6, alignItems: 'center',
        }}>
          {note && <span className="label">{note}</span>}
          {dirty && <button className="btn primary" onClick={save} type="button">Save ⌘S</button>}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ diff */

function Diff({ wsId, path }) {
  const [text, setText] = useState('');
  const load = useCallback(() => {
    if (!wsId || !path) { setText(''); return; }
    api.diff(wsId, path).then((r) => setText(r.diff)).catch((e) => setText(`(${e.message})`));
  }, [wsId, path]);
  useEffect(() => { load(); }, [load]);
  // An accept or revert changes what the diff shows without changing the path.
  useEffect(() => on('event', (evt) => {
    if (['git.reverted', 'git.committed'].includes(evt.kind) && (evt.data?.workspaceId ?? evt.data?.ws) === wsId) load();
  }), [wsId, load]);

  if (!path) return <div className="empty"><b>No file open.</b>Diffs show real git output.</div>;

  return (
    <div className="diff">
      {text.split('\n').map((line, i) => {
        const cls = line.startsWith('+') && !line.startsWith('+++') ? 'add'
          : line.startsWith('-') && !line.startsWith('---') ? 'del'
            : line.startsWith('@@') ? 'hunk'
              : line.startsWith('diff ') || line.startsWith('index ') ? 'meta' : '';
        return <div key={i} className={cls}>{line || ' '}</div>;
      })}
    </div>
  );
}

/* ------------------------------------------------------------------ transcript */

function Transcript({ session }) {
  const sessionId = session?.id;
  const [events, setEvents] = useState([]);
  const st = useField();
  const bottomRef = useRef(null);

  useEffect(() => {
    if (!sessionId) { setEvents([]); return; }
    let alive = true;
    api.trace(sessionId, 0, 2000).then((r) => { if (alive) setEvents(r.events); }).catch(() => { if (alive) setEvents([]); });
    return () => { alive = false; };
  }, [sessionId]);

  // Append live events for this session as they arrive.
  useEffect(() => {
    if (!sessionId) return;
    return on('event', (evt) => {
      if (evt.subject !== sessionId && evt.data?.sessionId !== sessionId) return;
      setEvents((prev) => (prev.some((e) => e.seq === evt.seq) ? prev : [...prev, evt]));
    });
  }, [sessionId]);

  useEffect(() => { bottomRef.current?.scrollIntoView({ block: 'end' }); }, [events.length]);

  if (!sessionId) {
    return <div className="empty"><b>No agent open.</b>Pick an agent on the Board or Map and choose Open transcript.</div>;
  }

  const rows = events.filter((e) => [
    'session.spawned', 'session.message', 'session.thinking', 'session.tool_use', 'session.tool_result',
    'session.ended', 'permission.requested', 'permission.decided', 'work.verified', 'session.verification',
  ].includes(e.kind));
  const objective = st.snap.campaigns
    .flatMap((campaign) => campaign.objectives ?? [])
    .find((item) => item.id === session?.objectiveId);

  return (
    <div className="transcript">
      {events.length >= 2000 && <div className="label">Showing the first 2,000 events. Open Traces to load the rest.</div>}
      <div className="agent-brief">
        <span className="label">current objective</span>
        <b>{objective?.statement ?? session?.target?.label ?? session?.target?.id ?? 'Awaiting assignment'}</b>
        <span className="mono">{session?.state?.replaceAll('_', ' ') ?? 'unknown'}{objective?.status ? ` · ${objective.status}` : ''}{session?.costUsd != null ? ` · $${session.costUsd.toFixed(4)}` : ''}</span>
        {objective?.progress?.total && (
          <progress max={objective.progress.total} value={objective.progress.done ?? 0} />
        )}
      </div>
      {rows.map((e) => <TranscriptRow key={e.seq} evt={e} />)}
      <div ref={bottomRef} />
    </div>
  );
}

function TranscriptRow({ evt }) {
  const d = evt.data ?? {};
  switch (evt.kind) {
    case 'session.spawned':
      return (
        <div className="tr prompt">
          <div className="who">initial prompt</div>
          <details>
            <summary>{d.initialOrders ?? 'Open prompt'}</summary>
            <pre className="body">{d.systemPrompt}</pre>
          </details>
        </div>
      );
    case 'session.message':
      return (
        <div className={`tr ${d.role}`}>
          <div className="who">{d.role}</div>
          <div className="body">{d.text}</div>
        </div>
      );
    case 'session.thinking':
      return (
        <div className="tr tool">
          <div className="who">thinking</div>
          <div className="body" style={{ opacity: .72, fontStyle: 'italic' }}>{d.text?.slice(0, 600)}</div>
        </div>
      );
    case 'session.tool_use':
      return (
        <div className="tr tool">
          <div className="who">{d.name}</div>
          <div className="body">{d.summary}</div>
        </div>
      );
    case 'session.tool_result':
      return (
        <div className={`tr ${d.ok === false ? 'error' : 'tool'}`}>
          <div className="who">{d.ok === false ? 'tool failed' : 'result'}</div>
          <div className="body">{(d.preview ?? '').slice(0, 400)}</div>
        </div>
      );
    case 'permission.requested':
      return (
        <div className="tr tool">
          <div className="who">approval requested</div>
          <div className="body">{d.toolName}</div>
        </div>
      );
    case 'permission.decided':
      return (
        <div className="tr tool">
          <div className="who">approval {d.decision}</div>
          <div className="body">by {d.by}</div>
        </div>
      );
    case 'work.verified':
      return (
        <div className="tr tool">
          <div className="who">verified</div>
          <div className="body">{d.result}{d.tier ? ` · ${d.tier}` : ''}{d.summary ? ` — ${d.summary}` : ''}</div>
        </div>
      );
    case 'session.verification':
      return (
        <div className="tr tool">
          <div className="who">verdict</div>
          <div className="body"><VerdictLadder verdict={{ ...d, ts: evt.ts }} compact /></div>
        </div>
      );
    case 'session.ended':
      return (
        <div className={`tr ${d.reason === 'error' ? 'error' : 'user'}`}>
          <div className="who">session {d.reason}</div>
          {d.error && <div className="body">{d.error}</div>}
        </div>
      );
    default:
      return null;
  }
}

/* ------------------------------------------------------------------ terminal */

function Terminal({ wsId }) {
  const [lines, setLines] = useState([]);
  const [cmd, setCmd] = useState('');
  const [history, setHistory] = useState([]);
  const [hIndex, setHIndex] = useState(-1);
  const idRef = useRef(`term-${Math.random().toString(36).slice(2, 9)}`);
  const outRef = useRef(null);

  useEffect(() => on('terminal', (msg) => {
    if (msg.id !== idRef.current) return;
    setLines((prev) => [...prev.slice(-500), { stream: msg.stream, data: msg.data }]);
  }), []);

  useEffect(() => { if (outRef.current) outRef.current.scrollTop = outRef.current.scrollHeight; }, [lines]);

  const run = async () => {
    if (!cmd.trim() || !wsId) return;
    setHistory((h) => [...h, cmd]);
    setHIndex(-1);
    try { await api.runCommand(wsId, cmd, idRef.current); }
    catch (e) { setLines((prev) => [...prev, { stream: 'err', data: `${e.message}\n` }]); }
    setCmd('');
  };

  return (
    <div className="term">
      <div className="term-out" ref={outRef}>
        {lines.length === 0 && (
          <span style={{ color: 'var(--ghost)' }}>
            Real shell in the selected workspace. Output streams from the Field server.
          </span>
        )}
        {lines.map((l, i) => (
          <span key={i} className={l.stream === 'err' ? 'err' : l.stream === 'meta' ? 'meta' : ''}>
            {l.data}
          </span>
        ))}
      </div>
      <div className="term-in">
        <span>$</span>
        <input
          value={cmd}
          placeholder={wsId ? 'run a command…' : 'no workspace'}
          onChange={(e) => setCmd(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter') run();
            else if (e.key === 'ArrowUp') {
              e.preventDefault();
              const i = hIndex < 0 ? history.length - 1 : Math.max(0, hIndex - 1);
              if (history[i] != null) { setHIndex(i); setCmd(history[i]); }
            } else if (e.key === 'ArrowDown') {
              e.preventDefault();
              const i = hIndex < 0 ? -1 : hIndex + 1;
              if (i >= history.length || i < 0) { setHIndex(-1); setCmd(''); }
              else { setHIndex(i); setCmd(history[i]); }
            }
          }}
        />
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ browser */

function BrowserPane({ initialUrl }) {
  const st = useField();
  const [url, setUrl] = useState(initialUrl ?? '');
  const [loadedUrl, setLoadedUrl] = useState('');
  const [browserError, setBrowserError] = useState(null);
  const allowedDomains = useMemo(
    () => st.snap.websites.filter((site) => !site.discovered).map((site) => site.domain),
    [st.snap.websites],
  );

  const navigate = useCallback((candidate) => {
    const result = validateBrowserUrl(candidate, allowedDomains);
    if (!result.ok) { setBrowserError(result.error); return; }
    setBrowserError(null);
    setUrl(result.url);
    setLoadedUrl(result.url);
  }, [allowedDomains]);

  useEffect(() => {
    if (initialUrl) { setUrl(initialUrl); navigate(initialUrl); }
  }, [initialUrl, navigate]);

  const routes = st.snap.sessions
    .filter((s) => s.browser?.url)
    .map((s) => ({ name: s.name, url: s.browser.url, domain: s.browser.domain }));

  return (
    <div className="term" style={{ background: 'var(--bg)' }}>
      <div className="term-in" style={{ borderTop: 0, borderBottom: '1px solid var(--line)' }}>
        <span style={{ fontSize: 10 }}>URL</span>
        <input
          value={url}
          placeholder="https://…"
          onChange={(e) => setUrl(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') navigate(url); }}
        />
      </div>

      {routes.length > 0 && (
        <div style={{ padding: '6px 10px', borderBottom: '1px solid var(--line)' }}>
          <span className="label">live agent browser routes</span>
          {routes.map((r, i) => (
            <button
              key={i}
              className="tree-row"
              style={{ padding: '2px 0' }}
              onClick={() => navigate(r.url)}
              type="button"
            >
              <span className="nm">{r.name} → {r.domain}</span>
            </button>
          ))}
        </div>
      )}

      {browserError && (
        <div role="alert" className="label" style={{ padding: '7px 10px', color: 'var(--danger)' }}>
          {browserError}
        </div>
      )}

      {loadedUrl ? (
        <>
          <iframe
            title="browser surface"
            src={loadedUrl}
            style={{ flex: '1 1 auto', border: 0, background: '#fff', minHeight: 0 }}
            sandbox="allow-scripts"
            referrerPolicy="no-referrer"
          />
          <div className="label" style={{ padding: '5px 10px', borderTop: '1px solid var(--line)' }}>
            sites that refuse framing will stay blank — open externally instead
          </div>
        </>
      ) : (
        <div className="empty">
          <b>No browser surface open.</b>
          Enter a URL, or pick a route an agent is actually using.
        </div>
      )}
    </div>
  );
}
