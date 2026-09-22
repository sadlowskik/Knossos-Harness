import { useEffect, useRef, useState } from 'react';
import { ChevronDown, Cpu, Plus, Trash2, Wifi, X } from 'lucide-react';
import { api } from '../net/client.js';
import { normalizeFieldSettings, registryKeyFor } from '../theater/fieldPreferences.js';

/* How an endpoint is named wherever an agent shows it: the alias, the model it really
   serves, its Hugging Face repo, and — for a teacher endpoint — the student it feeds. */
function EndpointNaming({ endpoint, settings, setSettings }) {
  const key = registryKeyFor(endpoint);
  const entry = settings?.modelRegistry?.[key] ?? {};
  const patch = (change) => setSettings((current) => normalizeFieldSettings({
    ...current,
    modelRegistry: { ...current.modelRegistry, [key]: { ...current.modelRegistry?.[key], ...change } },
  }));
  const field = (label, name, placeholder) => (
    <label className="psrc-field">
      <span className="psrc-field-label">{label}</span>
      <input className="mono" value={entry[name] ?? ''} placeholder={placeholder} onChange={(event) => patch({ [name]: event.target.value })} />
    </label>
  );
  return (
    <div className="psrc-naming">
      {field('Alias', 'endpointAlias', endpoint.name || endpoint.id)}
      {field('Model served', 'servedModel', endpoint.model || 'as reported')}
      {field('HF repository', 'hfRepo', 'owner/model')}
      {field('Student target', 'studentTarget', 'none')}
      <label className="psrc-check">
        <input
          type="checkbox"
          checked={!!entry.collectTeacherTraces}
          onChange={(event) => patch({ collectTeacherTraces: event.target.checked, distillationRole: event.target.checked ? 'teacher' : '' })}
        />
        <span>Collect traces from this endpoint as distillation source data</span>
      </label>
      <p className="psrc-help">Naming is stored on this device. It changes what agents are labelled, not what they run on.</p>
    </div>
  );
}

const KINDS = [
  { value: 'anthropic', label: 'Anthropic (Claude)' },
  { value: 'openai-compatible', label: 'OpenAI-compatible / Ollama / Cameo' },
];
const EMPTY = { id: '', name: '', kind: 'anthropic', model: '', base_url: '', key: '' };
const STATUS_LABEL = { up: 'reachable', down: 'unreachable', unknown: 'not tested' };
const statusLabel = (status) => STATUS_LABEL[status] ?? status ?? 'not tested';

/* Power sources = the models behind your units. Keys are write-only: they are sent once,
   stored locally, and never returned to the UI.

   This is also the only place an endpoint is described. The endpoint rail beneath the Map
   keeps status and spend and opens this modal on the row you clicked; the naming fields
   (alias, model served, HF repo, distillation) used to be a separate Models tab in the
   settings panel and now sit beside the endpoint they describe. */
export default function PowerSources({ onClose, focus = null, settings = null, setSettings = null }) {
  const [list, setList] = useState([]);
  const [form, setForm] = useState(EMPTY);
  const [busy, setBusy] = useState(false);
  const [testing, setTesting] = useState('');
  const [error, setError] = useState('');
  const [note, setNote] = useState('');
  const [backend, setBackend] = useState('');
  const [openRow, setOpenRow] = useState(focus);
  const focusRef = useRef(null);

  useEffect(() => { focusRef.current?.scrollIntoView({ block: 'nearest' }); }, [list.length]);

  async function load() {
    try {
      const r = await api.endpoints();
      setList(r.endpoints ?? []);
      setBackend(r.keyBackend ?? '');
    } catch (e) { setError(e.message); }
  }
  const keyHome = backend === 'keychain' ? 'your system keychain' : 'a private file on this machine';
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
    <div className="cityhub-veil psrc-veil" onClick={onClose}>
      <div className="cityhub psrc-modal" role="dialog" aria-modal="true" aria-labelledby="psrc-title" onClick={(e) => e.stopPropagation()}>
        <header className="cityhub-head psrc-head">
          <span className="psrc-title-wrap">
            <Cpu aria-hidden="true" />
            <span>
              <b id="psrc-title">Models</b>
              <small>The models your agents run on. Keys go to {keyHome} and never leave it.</small>
            </span>
          </span>
          <button type="button" className="psrc-close" onClick={onClose} aria-label="Close models"><X /></button>
        </header>
        <div className="psrc-body">
          <section className="psrc-list" aria-label="Configured models">
            {list.map((e) => (
              <div
                key={e.id}
                className={`psrc-row status-${e.status}${focus === e.id ? ' focused' : ''}`}
                ref={focus === e.id ? focusRef : null}
              >
                <div className="psrc-row-main">
                  <i className={`psrc-dot status-${e.status ?? 'unknown'}`} aria-hidden="true" />
                  <div className="psrc-row-text">
                    <b>{e.name}</b>
                    <small className="mono">{e.kind}{e.model ? ' · ' + e.model : ''}{e.base_url ? ' · ' + e.base_url : ''}</small>
                    <small>{e.source === 'user' ? 'Added by you' : 'Built in'} · {e.hasKey ? 'key set' : 'no key'} · {statusLabel(e.status)}</small>
                  </div>
                  <div className="psrc-row-actions">
                    {setSettings && (
                      <button
                        type="button"
                        className={`btn ghost psrc-expand${openRow === e.id ? ' on' : ''}`}
                        aria-expanded={openRow === e.id}
                        onClick={() => setOpenRow(openRow === e.id ? null : e.id)}
                      ><ChevronDown aria-hidden="true" /> Naming</button>
                    )}
                    <button type="button" className="btn ghost" onClick={() => test(e.id)} disabled={testing === e.id}><Wifi aria-hidden="true" /> {testing === e.id ? 'Testing…' : 'Test'}</button>
                    {e.source === 'user' && <button type="button" className="btn ghost danger" onClick={() => remove(e.id)} aria-label={`Remove ${e.id}`}><Trash2 aria-hidden="true" /></button>}
                  </div>
                </div>
                {setSettings && openRow === e.id && (
                  <EndpointNaming endpoint={e} settings={settings} setSettings={setSettings} />
                )}
              </div>
            ))}
            {!list.length && (
              <div className="psrc-empty">
                <b>No models yet</b>
                <p>Use the form to add one. Anthropic needs an API key; a local server such as Ollama or Cameo needs its base URL.</p>
              </div>
            )}
          </section>

          <section className="psrc-form" aria-label="Add a model">
            <h3>Add a model</h3>
            <label className="psrc-field">
              <span className="psrc-field-label">Provider</span>
              <select value={form.kind} onChange={(e) => patch({ kind: e.target.value })}>
                {KINDS.map((k) => <option key={k.value} value={k.value}>{k.label}</option>)}
              </select>
            </label>
            <label className="psrc-field">
              <span className="psrc-field-label">Name</span>
              <input value={form.name} onChange={(e) => patch({ name: e.target.value })} placeholder="My model" />
            </label>
            <label className="psrc-field">
              <span className="psrc-field-label">Id <small>optional</small></span>
              <input className="mono" value={form.id} onChange={(e) => patch({ id: e.target.value })} placeholder="Made from the name if empty" />
            </label>
            <label className="psrc-field">
              <span className="psrc-field-label">Model</span>
              <input className="mono" value={form.model} onChange={(e) => patch({ model: e.target.value })} placeholder={needsUrl ? 'llama3 / local' : 'claude-sonnet-5'} />
            </label>
            {needsUrl && (
              <label className="psrc-field">
                <span className="psrc-field-label">Base URL</span>
                <input className="mono" value={form.base_url} onChange={(e) => patch({ base_url: e.target.value })} placeholder="http://127.0.0.1:11434/v1" />
              </label>
            )}
            <label className="psrc-field">
              <span className="psrc-field-label">API key {needsUrl && <small>optional for local servers</small>}</span>
              <input type="password" value={form.key} onChange={(e) => patch({ key: e.target.value })} placeholder={needsUrl ? 'Leave empty for local' : 'sk-ant-…'} autoComplete="off" />
              <span className="psrc-help">Kept in {keyHome}. Sent once, never shown again.</span>
            </label>
            {error && <p className="cityhub-error psrc-msg" role="alert">{error}</p>}
            {note && <p className="cityhub-note psrc-msg" role="status">{note}</p>}
            <div className="psrc-form-actions">
              <button type="button" className="btn ghost" onClick={onClose}>Close</button>
              <button type="button" className="btn primary" disabled={busy || (!form.name && !form.id)} onClick={add}><Plus aria-hidden="true" /> Add model</button>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
