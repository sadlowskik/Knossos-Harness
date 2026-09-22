/* The working parts of what used to be the Project destination.

   Project was a screen you navigated to in order to see one folder's files, its diffs and
   a shell in it — which is a folder's detail, not a place of its own. The screen is gone;
   these are its panes, unchanged in behaviour, mounted by Rome inside the folder you
   opened. Each one takes a workspace id and a root folder, so "the files here" means the
   folder you clicked rather than always the project root. */

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api, on } from '../net/client.js';
import { renderMarkdown } from './md.js';
import { validateBrowserUrl } from './browser-url.js';
import { useField } from '../state/store.js';

export function fmtSize(n) {
  if (n < 1024) return `${n}b`;
  if (n < 1024 * 1024) return `${Math.round(n / 1024)}k`;
  return `${(n / 1048576).toFixed(1)}m`;
}

/* ------------------------------------------------------------------ tree */

export function FileTree({ wsId, root = '', git, openPath, onOpen }) {
  const [expanded, setExpanded] = useState(() => new Set([root]));
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
    } catch {
      setDirs((d) => ({ ...d, [dir]: [] }));
    }
  }, [wsId]);

  useEffect(() => { setDirs({}); setExpanded(new Set([root])); if (wsId) load(root); }, [wsId, root, load]);

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
  walk(root, 0);

  return <div className="tree">{rows.length ? rows : <div className="empty">empty</div>}</div>;
}

/* ------------------------------------------------------------------ changes */

const DEFAULT_COMMIT_MESSAGE = 'Field: accept agent changes';
const STATUS_MARK = { modified: 'M', added: 'A', deleted: 'D', renamed: 'R', untracked: '?', copied: 'C', conflicted: 'U' };

/* Review what the agents changed in the working tree: every file with its line counts,
   the per-file diff one click away, and the two operator decisions, accept (commit) or
   revert. Both go through the server and land in the event log, so Rome's time control
   can scrub back through them. Accept and Revert are whole-project verbs even when you
   opened them from inside a folder, and the summary line says so. */
export function Changes({ wsId, scopeDir = '', openPath, onOpen }) {
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
  const here = scopeDir
    ? files.filter((f) => f.path === scopeDir || f.path.startsWith(`${scopeDir}/`)).length
    : files.length;

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
      {changes && scopeDir && (
        <p className="changes-scope">
          {here === 0 ? 'None of them are in this folder.' : `${here} of them ${here === 1 ? 'is' : 'are'} in this folder.`}
          {' '}Accept and Revert cover the whole project.
        </p>
      )}
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
          className="btn sm"
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

/* ------------------------------------------------------------------ file */

export function FileView({ wsId, path, markdown, primary = false }) {
  const [content, setContent] = useState('');
  const [loaded, setLoaded] = useState(null);
  const [dirty, setDirty] = useState(false);
  const [note, setNote] = useState(null);
  const saveRef = useRef(null);

  useEffect(() => {
    if (!wsId || !path) { setContent(''); setLoaded(null); return undefined; }
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
  saveRef.current = save;

  useEffect(() => {
    const onKey = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 's') { e.preventDefault(); if (dirty) saveRef.current?.(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [dirty]);

  if (!path) return <div className="empty"><b>No file open.</b>Pick one from the tree.</div>;

  if (markdown) {
    return <div className="md-view" dangerouslySetInnerHTML={{ __html: renderMarkdown(content) }} />;
  }

  return (
    <div className="editor">
      <textarea
        value={content}
        spellCheck={false}
        aria-label={`Contents of ${path}`}
        onChange={(e) => { setContent(e.target.value); setDirty(e.target.value !== loaded); }}
      />
      {(dirty || note) && (
        <div className="editor-save">
          {note && <span className="label">{note}</span>}
          {dirty && <button className={`btn sm${primary ? ' primary' : ''}`} onClick={save} type="button">Save ⌘S</button>}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ diff */

export function Diff({ wsId, path }) {
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

/* ------------------------------------------------------------------ terminal */

export function Terminal({ wsId }) {
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
          <span className="term-hint">Real shell in this project. Output streams from the Field server.</span>
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
          aria-label="Run a command"
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

export function BrowserPane({ initialUrl }) {
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
    <div className="term browser-pane">
      <div className="term-in browser-url">
        <span>URL</span>
        <input
          value={url}
          aria-label="Browser address"
          placeholder="https://…"
          onChange={(e) => setUrl(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter') navigate(url); }}
        />
      </div>

      {routes.length > 0 && (
        <div className="browser-routes">
          <span className="label">live agent browser routes</span>
          {routes.map((r, i) => (
            <button key={i} className="tree-row" onClick={() => navigate(r.url)} type="button">
              <span className="nm">{r.name} → {r.domain}</span>
            </button>
          ))}
        </div>
      )}

      {browserError && <div role="alert" className="label browser-error">{browserError}</div>}

      {loadedUrl ? (
        <>
          <iframe
            title="browser surface"
            className="browser-frame"
            src={loadedUrl}
            sandbox="allow-scripts"
            referrerPolicy="no-referrer"
          />
          <div className="label browser-note">
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
