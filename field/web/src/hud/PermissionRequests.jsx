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

export default function PermissionRequests() {
  const st = useField();
  const [busy, setBusy] = useState({});
  const [armed, setArmed] = useState({});
  const pending = st.snap.permissions ?? [];
  if (!pending.length) return null;

  const decide = async (id, decision) => {
    setBusy((b) => ({ ...b, [id]: true }));
    try {
      await api.decide(id, decision, decision === 'deny' ? 'Denied by the Field operator.' : undefined);
    } catch (e) {
      console.error(e);
    } finally {
      setBusy((b) => ({ ...b, [id]: false }));
      setArmed((a) => ({ ...a, [id]: false }));
    }
  };

  return (
    <div className="perms">
      {pending.map((p) => {
        const session = st.snap.sessions.find((s) => s.id === p.sessionId);
        const workspace = st.snap.workspaces.find((w) => w.id === session?.workspaceId);
        const detail = summarize(p.toolName, p.input);
        const privileged = isPrivilegedTool(p.toolName, p.input);
        return (
          <div className={`perm${privileged ? ' privileged' : ''}`} key={p.id} role="group" aria-label={`Approval request: ${p.toolName}`}>
            <div className="perm-top">
              <span className="perm-flag label">
                <i aria-hidden="true" />
                {privileged ? 'Privileged · needs your approval' : 'Needs your approval'}
              </span>
              <button
                className="perm-who"
                type="button"
                onClick={() => selectOnly([p.sessionId])}
                title="Select this agent"
              >
                {session?.name ?? p.sessionId.slice(0, 6)} · {session?.role ?? '—'}
              </button>
            </div>
            <div className="perm-tool"><span className="perm-tool-label">wants to run</span><code className="mono">{p.toolName}</code></div>
            {workspace && <div className="perm-ws mono">{workspace.name} · {workspace.path}</div>}
            <PermissionContext context={p.context} fallback={detail} />
            {(session?.lastSay || session?.stateDetail) && (
              <p className="perm-why"><span className="perm-tool-label">agent said</span>{String(session.lastSay ?? session.stateDetail).slice(0, 240)}</p>
            )}
            {privileged && !armed[p.id] && (
              <p className="perm-warn">This can write files, run a shell, or change the project. Review it before allowing.</p>
            )}
            <div className="perm-actions">
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
          </div>
        );
      })}
    </div>
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
