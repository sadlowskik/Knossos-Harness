import { useState } from 'react';
import { api } from '../net/client.js';
import { selectOnly, useField } from '../state/store.js';

const PRIVILEGED_TOOLS = new Set([
  'bash', 'shell', 'powershell', 'write', 'edit', 'notebookedit',
  'write_file', 'str_replace', 'strreplace',
]);

export function isPrivilegedTool(toolName, input) {
  const name = String(toolName ?? '').toLowerCase();
  if (PRIVILEGED_TOOLS.has(name)) return true;
  if (typeof input?.command === 'string' && input.command.trim()) return true;
  return false;
}

const leaf = (path) => String(path ?? '').replaceAll('\\', '/').split('/').filter(Boolean).at(-1) ?? '';

/* What the agent is asking for, as the end of a sentence that starts with its name.

   The card used to open with a flag, a name · role pair, a "wants to run" label and a
   tool identifier, then the project name and its absolute path, before you reached the
   thing you were actually deciding on. This is that stack, said once. */
function askPhrase(request) {
  const context = request.context;
  if (context?.kind === 'diff') return context.path ? `edit ${leaf(context.path)}` : 'edit a file';
  if (context?.kind === 'command') return 'run a command';
  if (typeof request.input?.command === 'string') return 'run a command';
  const path = context?.path ?? request.input?.file_path ?? request.input?.path;
  if (typeof path === 'string' && path) return `work on ${leaf(path)}`;
  if (typeof request.input?.url === 'string') {
    try { return `read ${new URL(request.input.url).hostname}`; } catch { return 'read a web page'; }
  }
  return `use ${request.toolName}`;
}

export default function PermissionRequests() {
  const st = useField();
  const [busy, setBusy] = useState({});
  const [armed, setArmed] = useState({});
  const [why, setWhy] = useState({});
  // The beat. A decided request leaves the snapshot immediately, so the card that
  // replaces it is held here for a moment: you see the decision land.
  const [settled, setSettled] = useState([]);
  const pending = st.snap.permissions ?? [];
  if (!pending.length && !settled.length) return null;

  const decide = async (id, decision) => {
    setBusy((b) => ({ ...b, [id]: true }));
    try {
      await api.decide(id, decision, decision === 'deny' ? 'Denied by the Field operator.' : undefined);
      setSettled((rows) => [...rows, { id, word: decision === 'deny' ? 'Denied' : 'Allowed' }]);
      setTimeout(() => setSettled((rows) => rows.filter((row) => row.id !== id)), 1600);
    } catch (e) {
      console.error(e);
    } finally {
      setBusy((b) => ({ ...b, [id]: false }));
      setArmed((a) => ({ ...a, [id]: false }));
    }
  };

  return (
    <section className="atlas-approvals" aria-label="Approvals waiting">
      {pending.length > 0 && (
        <h2 className="atlas-section-title">
          <i aria-hidden="true" />
          {pending.length === 1 ? 'One request needs you' : `${pending.length} requests need you`}
        </h2>
      )}
      <div className="perms">
        {settled.map((row) => (
          <p className={`perm-settled ${row.word.toLowerCase()}`} key={`settled-${row.id}`} role="status">
            <i aria-hidden="true" />{row.word}
          </p>
        ))}
        {pending.map((p) => {
          const session = st.snap.sessions.find((s) => s.id === p.sessionId);
          const workspace = st.snap.workspaces.find((w) => w.id === session?.workspaceId);
          const detail = summarize(p.toolName, p.input);
          const privileged = isPrivilegedTool(p.toolName, p.input);
          const name = session?.name ?? p.sessionId.slice(0, 6);
          const said = session?.lastSay ?? session?.stateDetail ?? null;
          return (
            <div className={`perm${privileged ? ' privileged' : ''}`} key={p.id} role="group" aria-label={`Approval request: ${p.toolName}`}>
              <p className="perm-ask">
                <button
                  className="perm-who"
                  type="button"
                  onClick={() => selectOnly([p.sessionId])}
                  title="Select this agent"
                >{name}</button>
                {' wants to '}
                <b>{askPhrase(p)}</b>
                <code className="mono" title={`Tool: ${p.toolName}`}>{p.toolName}</code>
              </p>

              <PermissionContext context={p.context} fallback={detail} />

              {privileged && !armed[p.id] && (
                <p className="perm-warn">This can write files, run a shell, or change the project. Review it before allowing.</p>
              )}

              <div className="perm-actions">
                {/* The project, the folder it runs in and whatever the agent last said
                    are the second question, so they are behind the second question. */}
                <button
                  type="button"
                  className="perm-why-toggle"
                  aria-expanded={Boolean(why[p.id])}
                  onClick={() => setWhy((w) => ({ ...w, [p.id]: !w[p.id] }))}
                >Why?</button>
                <span className="grow" />
                <button
                  className="btn danger"
                  disabled={busy[p.id]}
                  onClick={() => decide(p.id, 'deny')}
                  type="button"
                >Deny</button>
                {privileged && !armed[p.id] ? (
                  <button
                    className="btn"
                    disabled={busy[p.id]}
                    onClick={() => setArmed((a) => ({ ...a, [p.id]: true }))}
                    type="button"
                  >Review, then allow</button>
                ) : (
                  <button
                    className="btn primary"
                    disabled={busy[p.id]}
                    onClick={() => decide(p.id, 'allow')}
                    type="button"
                  >{privileged ? 'Allow privileged' : 'Allow'}</button>
                )}
              </div>

              {why[p.id] && (
                <div className="perm-more">
                  {workspace && <p className="perm-ws mono">{workspace.name} · {workspace.path}</p>}
                  {session?.role && <p className="perm-why"><span className="perm-tool-label">role</span>{session.role}</p>}
                  {said && <p className="perm-why"><span className="perm-tool-label">agent said</span>{String(said).slice(0, 240)}</p>}
                  {!workspace && !session?.role && !said && <p className="perm-why">Nothing else is known about this request.</p>}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </section>
  );
}

// What the operator is deciding on: the unified diff for an edit, the command with
// its directory for a shell, the tool input otherwise. The server built this from the
// request; secrets registered with the event log were already scrubbed from it.
function PermissionContext({ context, fallback }) {
  if (!context || typeof context !== 'object') return <pre className="perm-input mono">{fallback}</pre>;
  if (context.kind === 'diff') {
    const lines = String(context.diff ?? '').split('\n');
    return (
      <div className="perm-context">
        <div className="perm-context-head mono">
          <span className="perm-context-path" title={context.path}>{context.path}</span>
          <span className="perm-context-counts">
            <span className="add">+{context.additions ?? 0}</span> <span className="del">−{context.deletions ?? 0}</span>
          </span>
        </div>
        <div className="diff perm-diff">
          {lines.map((line, i) => {
            const cls = line.startsWith('+') && !line.startsWith('+++') ? 'add'
              : line.startsWith('-') && !line.startsWith('---') ? 'del'
                : line.startsWith('@@') ? 'hunk'
                  : line.startsWith('…') ? 'meta' : '';
            return <div key={i} className={cls}>{line || ' '}</div>;
          })}
        </div>
        {context.truncated && <div className="perm-context-note">Diff truncated; the full change lands in the workspace diff once allowed.</div>}
      </div>
    );
  }
  if (context.kind === 'command') {
    return (
      <div className="perm-context">
        <div className="perm-context-head mono">
          <span className="perm-tool-label">in</span>
          <span className="perm-context-path" title={context.cwd}>{context.cwd || '.'}</span>
        </div>
        <pre className="perm-input perm-command mono">{context.command}</pre>
        {context.truncated && <div className="perm-context-note">Command truncated.</div>}
      </div>
    );
  }
  return (
    <div className="perm-context">
      {context.path && <div className="perm-context-head mono"><span className="perm-context-path">{context.path}</span></div>}
      <pre className="perm-input mono">{context.text ?? fallback}</pre>
      {context.truncated && <div className="perm-context-note">Input truncated.</div>}
    </div>
  );
}

function summarize(toolName, input) {
  if (!input || typeof input !== 'object') return String(input ?? '');
  if (typeof input.command === 'string') return input.command;
  if (typeof input.file_path === 'string') {
    const extra = input.old_string != null ? '\n\n— replacing —\n' + String(input.old_string).slice(0, 300) : '';
    return input.file_path + extra;
  }
  if (typeof input.path === 'string') return input.path;
  if (typeof input.url === 'string') return input.url;
  const text = JSON.stringify(input, null, 2);
  return text.length > 900 ? text.slice(0, 900) + '\n…' : text;
}
