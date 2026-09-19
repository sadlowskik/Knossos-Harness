import { useEffect, useState } from 'react';
import { Cpu, Plus, Trash2, Wifi, X } from 'lucide-react';
import { api } from '../net/client.js';

const KINDS = [
  { value: 'anthropic', label: 'Anthropic (Claude)' },
  { value: 'openai-compatible', label: 'OpenAI-compatible / Ollama / Cameo' },
];
const EMPTY = { id: '', name: '', kind: 'anthropic', model: '', base_url: '', key: '' };

// Power sources = the models behind your units. Keys are write-only: they are sent once,
// stored locally, and never returned to the UI.
export default function PowerSources({ onClose }) {
  const [list, setList] = useState([]);
  const [form, setForm] = useState(EMPTY);
  const [busy, setBusy] = useState(false);
  const [testing, setTesting] = useState('');
  const [error, setError] = useState('');
  const [note, setNote] = useState('');

  async function load() {
    try { const r = await api.endpoints(); setList(r.endpoints ?? []); } catch (e) { setError(e.message); }
  }
  useEffect(() => { load(); const t = setInterval(load, 6000); return () => clearInterval(t); }, []);

  const patch = (p) => setForm((f) => ({ ...f, ...p }));
  const needsUrl = form.kind === 'openai-compatible';

  async function add() {
    setBusy(true); setError(''); setNote('');
    try {
      const r = await api.addEndpoint({
        id: form.id || form.name, name: form.name || form.id, kind: form.kind,
        model: form.model || undefined, base_url: form.base_url || undefined, key: form.key || undefined,
      });
      setNote(`Added ${r.id}${r.hasKey ? ' (key stored locally)' : ''}.`);
      setForm(EMPTY);
      load();
    } catch (e) { setError(e.message); } finally { setBusy(false); }
  }
  async function test(id) {
    setTesting(id); setError(''); setNote('');
    try {
      const r = await api.testEndpoint(id);
      setNote(r.ok ? `${id}: reachable${r.status ? ' (' + r.status + ')' : ''}${r.note ? ' — ' + r.note : ''}` : `${id}: ${r.error ?? 'unreachable'}`);
    } catch (e) { setError(e.message); } finally { setTesting(''); }
  }
  async function remove(id) {
    setBusy(true); setError(''); setNote('');
    try { await api.deleteEndpoint(id); setNote(`Removed ${id}.`); load(); }
    catch (e) { setError(e.message); } finally { setBusy(false); }
  }

  return (
    <div className="cityhub-veil" onClick={onClose}>
      <div className="cityhub" role="dialog" aria-modal="true" onClick={(e) => e.stopPropagation()}>
        <header className="cityhub-head">
          <span><Cpu /> Power sources</span>
          <button type="button" onClick={onClose} aria-label="Close power sources"><X /></button>
        </header>
        <div className="psrc-body">
          <section className="psrc-list">
            {list.map((e) => (
              <div key={e.id} className={`psrc-row status-${e.status}`}>
                <div>
                  <b>{e.name}</b>
                  <small>{e.kind}{e.model ? ' · ' + e.model : ''}{e.base_url ? ' · ' + e.base_url : ''}</small>
                  <small>{e.source === 'user' ? 'added by you' : 'built-in'} · {e.hasKey ? 'key set' : 'no key'} · {e.status}</small>
                </div>
                <div className="psrc-row-actions">
                  <button type="button" onClick={() => test(e.id)} disabled={testing === e.id}><Wifi /> {testing === e.id ? '…' : 'Test'}</button>
                  {e.source === 'user' && <button type="button" className="danger" onClick={() => remove(e.id)} aria-label={`Remove ${e.id}`}><Trash2 /></button>}
                </div>
              </div>
            ))}
            {!list.length && <p className="cityhub-empty">No power sources yet.</p>}
          </section>

          <section className="psrc-form">
            <h3>Add a model</h3>
            <label>Kind
              <select value={form.kind} onChange={(e) => patch({ kind: e.target.value })}>
                {KINDS.map((k) => <option key={k.value} value={k.value}>{k.label}</option>)}
              </select>
            </label>
            <label>Name<input value={form.name} onChange={(e) => patch({ name: e.target.value })} placeholder="My model" /></label>
            <label>Id<input value={form.id} onChange={(e) => patch({ id: e.target.value })} placeholder="auto from name" /></label>
            <label>Model<input value={form.model} onChange={(e) => patch({ model: e.target.value })} placeholder={needsUrl ? 'llama3 / local' : 'claude-sonnet-5'} /></label>
            {needsUrl && <label>Base URL<input value={form.base_url} onChange={(e) => patch({ base_url: e.target.value })} placeholder="http://127.0.0.1:11434/v1" /></label>}
            <label>API key <small>(write-only; stored locally, never shown again)</small>
              <input type="password" value={form.key} onChange={(e) => patch({ key: e.target.value })} placeholder={needsUrl ? 'optional for local' : 'sk-ant-…'} autoComplete="off" />
            </label>
            {error && <p className="cityhub-error">{error}</p>}
            {note && <p className="cityhub-note">{note}</p>}
            <button type="button" className="primary" disabled={busy || (!form.name && !form.id)} onClick={add}><Plus /> Add power source</button>
          </section>
        </div>
      </div>
    </div>
  );
}
