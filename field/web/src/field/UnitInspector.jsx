import { useEffect, useRef, useState } from 'react';
import { api } from '../net/client.js';
import { openCity, useField } from '../state/store.js';
import { ColumnTranscript } from '../atlas/AtlasMode.jsx';

// The unit's "city": a panel on the RTS lens that shows a selected unit's live
// transcript (reusing Atlas's ColumnTranscript fold), where it is working, and an
// order box wired to the real mid-session `say` command. It overlays the Field and
// never navigates away, so the operator keeps watching the parade while they talk.
export default function UnitInspector({ session, onClose }) {
  const st = useField();
  const [text, setText] = useState('');
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState('');
  const inputRef = useRef(null);

  useEffect(() => { setText(''); setNote(''); }, [session?.id]);
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  if (!session) return null;

  const workspace = st.snap.workspaces.find((w) => w.id === session.workspaceId);
  const where = session.focusPath ?? session.focusDir ?? null;
  const simulated = !!session.simulated;

  async function send() {
    const body = text.trim();
    if (!body || busy) return;
    setBusy(true); setNote('');
    try {
      const res = await api.command('say', { sessionIds: [session.id], text: body });
      setText('');
      // A synthetic unit runs off a scripted timeline and is not a live process, so the
      // order is accepted but there is no agent to receive it. Say so rather than imply
      // a round-trip that will not happen.
      setNote(simulated
        ? 'Order recorded. Rehearsal units replay a fixed script and will not respond.'
        : `Sent to ${session.name}.`);
      if (!res?.sent && !simulated) setNote('No live process received the order.');
    } catch (e) {
      setNote(e.message);
    } finally {
      setBusy(false);
      inputRef.current?.focus();
    }
  }

  return (
    <aside
      className="field-unit-city"
      onPointerDown={(e) => e.stopPropagation()}
      onClick={(e) => e.stopPropagation()}
      style={{
        position: 'absolute', top: 12, right: 12, width: 360, maxHeight: 'calc(100% - 24px)',
        display: 'flex', flexDirection: 'column', zIndex: 30,
        background: 'rgba(12,11,9,0.94)', border: '1px solid #302D28', borderRadius: 8,
        color: '#D8D2C8', boxShadow: '0 10px 30px rgba(0,0,0,0.45)', overflow: 'hidden',
      }}
    >
      <header style={{
        display: 'flex', alignItems: 'center', gap: 8, padding: '10px 12px',
        borderBottom: '1px solid #24211C',
      }}>
        <span style={{ color: '#FFD08A', fontSize: 13, fontWeight: 600, flex: 1 }}>
          {session.name ?? session.id}
          {simulated && (
            <span className="label" style={{ marginLeft: 8, color: '#7E8B94', fontWeight: 400 }}>
              rehearsal
            </span>
          )}
        </span>
        <button
          type="button"
          onClick={onClose}
          aria-label="Close unit city"
          style={{
            background: 'none', border: 'none', color: '#7C776D', cursor: 'pointer',
            fontSize: 18, lineHeight: 1, padding: 2,
          }}
        >×</button>
      </header>

      <div className="mono" style={{ padding: '8px 12px', fontSize: 11, color: '#9B948A', display: 'flex', flexWrap: 'wrap', gap: '4px 10px' }}>
        <span>{session.role ?? 'agent'}</span>
        <span style={{ color: '#5F5A51' }}>·</span>
        <span>{session.state ?? '—'}</span>
        <span style={{ color: '#5F5A51' }}>·</span>
        <span>city: <b style={{ color: '#C9C2B7' }}>{workspace?.name ?? session.workspaceId ?? 'staging'}</b></span>
        {where && (
          <>
            <span style={{ color: '#5F5A51' }}>·</span>
            <span>file: <b style={{ color: '#C9C2B7' }}>{where}</b></span>
          </>
        )}
      </div>

      <div style={{ flex: 1, minHeight: 120, overflow: 'auto', padding: '0 12px 8px', borderTop: '1px solid #24211C' }}>
        <ColumnTranscript session={session} campaigns={st.snap.campaigns ?? []} />
      </div>

      <div style={{ padding: 10, borderTop: '1px solid #24211C', display: 'flex', flexDirection: 'column', gap: 6 }}>
        {note && <div className="label" style={{ color: simulated ? '#7E8B94' : '#8FB07A', fontSize: 11 }}>{note}</div>}
        <textarea
          ref={inputRef}
          rows={2}
          value={text}
          placeholder={`Tell ${session.name ?? 'this unit'} what needs doing…`}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); send(); } }}
          style={{
            resize: 'none', width: '100%', boxSizing: 'border-box', padding: '6px 8px',
            background: '#181611', border: '1px solid #302D28', borderRadius: 5,
            color: '#E4DED4', fontSize: 12, fontFamily: 'inherit',
          }}
        />
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          {workspace?.mounted && (
            <button
              type="button"
              className="btn ghost"
              onClick={() => { openCity(session.workspaceId); onClose(); }}
            >Open City ↗</button>
          )}
          <span style={{ flex: 1 }} />
          <button type="button" className="btn primary" disabled={busy || !text.trim()} onClick={send}>
            {busy ? 'Sending…' : 'Send order'}
          </button>
        </div>
      </div>
    </aside>
  );
}
