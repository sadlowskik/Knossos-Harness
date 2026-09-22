import { useEffect, useRef, useState } from 'react';
import { api } from '../net/client.js';

const TERMINAL = new Set(['done', 'cancelled', 'interrupted', 'error']);

// One place to talk to an agent: a message box wired to the mid-session `say` command
// plus pause / resume / escalate / stop. Used by the Board's conversations, the Rome
// inspector and the Map unit inspector so every surface offers the same verbs.
// `showSay={false}` keeps only the verbs: the Board's conversation has its own composer,
// which prepends the files in scope before sending the same `say` command.
export default function AgentControls({ session, onOpen, compact = false, showSay = true }) {
  const [text, setText] = useState('');
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState(null);
  const inputRef = useRef(null);

  useEffect(() => { setText(''); setNote(null); }, [session?.id]);

  if (!session) return null;
  const name = session.name ?? session.id;
  const terminal = TERMINAL.has(session.state);
  const paused = session.state === 'paused';

  async function run(kind, extra = {}, okText = null) {
    setBusy(true); setNote(null);
    try {
      const res = await api.command(kind, { sessionIds: [session.id], ...extra });
      if (okText) setNote({ tone: 'ok', text: typeof okText === 'function' ? okText(res) : okText });
    } catch (e) {
      setNote({ tone: 'bad', text: e.message });
    } finally {
      setBusy(false);
    }
  }

  async function say() {
    const body = text.trim();
    if (!body || busy) return;
    await run('say', { text: body }, (res) => (res?.sent === false ? 'No live process received the message.' : `Sent to ${name}.`));
    setText('');
    inputRef.current?.focus();
  }

  return (
    <div className={`agent-controls${compact ? ' compact' : ''}`} onClick={(e) => e.stopPropagation()}>
      {!terminal && showSay && (
        <textarea
          ref={inputRef}
          rows={compact ? 1 : 2}
          value={text}
          placeholder={`Tell ${name} what to do next… (Ctrl+Enter)`}
          aria-label={`Message ${name}`}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); say(); } }}
        />
      )}
      <div className="agent-controls-row">
        {!terminal && (paused
          ? <button type="button" className="btn sm" disabled={busy} onClick={() => run('resume', {}, 'Resumed.')}>Resume</button>
          : <button type="button" className="btn sm" disabled={busy} onClick={() => run('pause', {}, 'Paused.')}>Pause</button>)}
        {!terminal && <button type="button" className="btn sm ghost" disabled={busy} onClick={() => run('escalate', {}, 'Escalated to a stronger model.')}>Escalate</button>}
        {!terminal && <button type="button" className="btn sm danger" disabled={busy} onClick={() => run('cancel', {}, 'Stopped.')}>Stop</button>}
        {onOpen && <button type="button" className="btn sm ghost" onClick={onOpen}>Open transcript</button>}
        <span className="grow" />
        {!terminal && showSay && <button type="button" className="btn sm primary" disabled={busy || !text.trim()} onClick={say}>{busy ? 'Working…' : 'Send'}</button>}
      </div>
      {note && <p className={`agent-controls-note ${note.tone}`} role="status">{note.text}</p>}
    </div>
  );
}
