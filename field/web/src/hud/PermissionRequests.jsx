import { useState } from 'react';
import { api } from '../net/client.js';
import { selectOnly, useField } from '../state/store.js';

// A pending permission is a real harness process blocked on a human. It gets the most
// prominent treatment in the interface, and it does not go away on its own.
export default function PermissionRequests() {
  const st = useField();
  const [busy, setBusy] = useState({});
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
    }
  };

  return (
    <div className="perms">
      {pending.map((p) => {
        const session = st.snap.sessions.find((s) => s.id === p.sessionId);
        const detail = summarize(p.toolName, p.input);
        return (
          <div className="perm" key={p.id}>
            <div className="perm-top">
              <span className="perm-flag label">operator approval required</span>
              <button
                className="perm-who"
                type="button"
                onClick={() => selectOnly([p.sessionId])}
                title="select this agent on the Field"
              >
                {session?.name ?? p.sessionId.slice(0, 6)} · {session?.role ?? '—'}
              </button>
            </div>
            <div className="perm-tool mono">{p.toolName}</div>
            <pre className="perm-input mono">{detail}</pre>
            <div className="perm-actions">
              <button
                className="btn danger"
                disabled={busy[p.id]}
                onClick={() => decide(p.id, 'deny')}
                type="button"
              >Deny</button>
              <button
                className="btn primary"
                disabled={busy[p.id]}
                onClick={() => decide(p.id, 'allow')}
                type="button"
              >Approve</button>
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
  if (typeof input.url === 'string') return input.url;
  const text = JSON.stringify(input, null, 2);
  return text.length > 900 ? text.slice(0, 900) + '\n…' : text;
}
