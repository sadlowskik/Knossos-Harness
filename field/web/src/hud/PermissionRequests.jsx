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
          <div className={`perm${privileged ? ' privileged' : ''}`} key={p.id}>
            <div className="perm-top">
              <span className="perm-flag label">
                {privileged ? 'privileged · operator approval' : 'operator approval required'}
              </span>
              <button
                className="perm-who"
                type="button"
                onClick={() => selectOnly([p.sessionId])}
                title="select this agent"
              >
                {session?.name ?? p.sessionId.slice(0, 6)} · {session?.role ?? '—'}
              </button>
            </div>
            {workspace && <div className="perm-ws mono">{workspace.name} · {workspace.path}</div>}
            <div className="perm-tool mono">{p.toolName}</div>
            <pre className="perm-input mono">{detail}</pre>
            {privileged && !armed[p.id] && (
              <p className="perm-warn">This can write files, run a shell, or change the workspace. Confirm before approve.</p>
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
                >Review privileged</button>
              ) : (
                <button
                  className="btn primary"
                  disabled={busy[p.id]}
                  onClick={() => decide(p.id, 'allow')}
                  type="button"
                >{privileged ? 'Approve privileged' : 'Approve'}</button>
              )}
            </div>
          </div>
        );
      })}
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
