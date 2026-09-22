import { useState } from 'react';
import { Check, RadioTower, Upload, X } from 'lucide-react';
import { api } from '../net/client.js';
import ToolIcon from '../ui/ToolIcon.jsx';
import {
  CUSTOM_FONT_MAX_BYTES,
  DEFAULT_FIELD_SETTINGS,
  agentPreferenceKey,
  loadCustomFont,
  normalizeFieldSettings,
  readFileAsDataUrl,
  saveFieldSettings,
} from './fieldPreferences.js';

export function ChoiceGroup({ value, options, onChange }) {
  return <div className="settings-choice">{options.map((option) => (
    <button
      type="button"
      key={option.value}
      className={value === option.value ? 'on' : ''}
      aria-pressed={value === option.value}
      onClick={() => onChange(option.value)}
    >{option.label}</button>
  ))}</div>;
}

/* The one settings panel. It used to live inside the Rome branch of TheaterMode, which
   left the Board with no settings at all; both screens now open this from their gear. */
export default function FieldSettings({
  settings, setSettings, selected, config, onClose, onOpenModels = null, standalone = false,
}) {
  const [tab, setTab] = useState('world');
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState('');
  const key = selected ? agentPreferenceKey(selected) : null;
  const override = key ? settings.agentOverrides?.[key] ?? {} : {};
  const configuredAgent = selected ? config?.agents?.find((agent) => agent.id === selected.agentId) : null;
  const role = selected ? config?.roles?.find((item) => item.id === selected.role) : null;
  const roleTools = role?.tools_allow ?? [];
  const selectedTools = override.toolsAllow ?? configuredAgent?.tools_allow ?? roleTools;
  const oxAlpha = settings.modelRegistry['ox-alpha'];

  function patchSettings(patch) { setSettings((current) => normalizeFieldSettings({ ...current, ...patch })); }
  function patchAgent(patch) { if (key) patchSettings({ agentOverrides: { ...settings.agentOverrides, [key]: { ...override, ...patch } } }); }
  function patchOx(patch) { patchSettings({ modelRegistry: { ...settings.modelRegistry, 'ox-alpha': { ...oxAlpha, ...patch } } }); }
  function toggleTool(tool) { patchAgent({ toolsAllow: selectedTools.includes(tool) ? selectedTools.filter((item) => item !== tool) : [...selectedTools, tool] }); }

  function upload(event) {
    const file = event.target.files?.[0];
    if (!file) return;
    if (file.size > 256 * 1024) { setMessage('Icon must be smaller than 256 KB.'); return; }
    const reader = new FileReader();
    reader.onload = () => patchAgent({ iconDataUrl: String(reader.result), iconUrl: '' });
    reader.readAsDataURL(file);
  }

  function chooseTypeface(typeface) {
    if (typeface === 'custom' && !settings.customFont?.dataUrl) {
      setMessage('Choose a font file first.');
      return;
    }
    setMessage('');
    patchSettings({ typeface });
  }

  // The file is read, parsed as a real FontFace and only then stored. A file that is
  // not a font never reaches the stylesheet or localStorage.
  async function pickFont(event) {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file) return;
    if (file.size > CUSTOM_FONT_MAX_BYTES) {
      setMessage('That font is over 1.5 MB. Settings live in this browser’s storage, so pick a smaller file.');
      return;
    }
    setMessage('Loading the font…');
    try {
      const dataUrl = await readFileAsDataUrl(file);
      await loadCustomFont(dataUrl);
      patchSettings({ typeface: 'custom', customFont: { name: file.name, dataUrl } });
      setMessage(`Using ${file.name}. Monospaced data stays on IBM Plex Mono.`);
    } catch {
      patchSettings({ typeface: DEFAULT_FIELD_SETTINGS.typeface, customFont: { name: '', dataUrl: '' } });
      setMessage('That file didn’t load as a font. Keeping the default.');
    }
  }

  async function save() {
    setSaving(true); setMessage('');
    try {
      const persisted = saveFieldSettings(settings); setSettings(persisted);
      if (configuredAgent) await api.updateAgent({ agentId: configuredAgent.id, name: override.displayName || configuredAgent.name, toolsAllow: selectedTools });
      setMessage(configuredAgent ? 'Saved. Tool authority applies to future deployments.' : 'Presentation saved on this device.');
    } catch (error) { setMessage(error.message); } finally { setSaving(false); }
  }

  return <aside
    className={`field-side-panel settings-panel${standalone ? ' settings-standalone' : ''}`}
    role="dialog"
    aria-label="Field settings"
    onClick={(event) => event.stopPropagation()}
  >
    <header><div><span>FIELD</span><h2>Settings</h2></div><button type="button" onClick={onClose} aria-label="Close settings"><X /></button></header>
    <nav>{['world', 'identity', 'models'].map((item) => (
      <button type="button" key={item} className={tab === item ? 'on' : ''} onClick={() => setTab(item)}>{item}</button>
    ))}</nav>
    <div className="settings-scroll">
      {tab === 'world' && <>
        <section>
          <label>Theme</label>
          <p className="settings-help">Atlas is the parallel agent board. Rome is the operations map.</p>
          <ChoiceGroup value={settings.theme} options={[{ value: 'atlas', label: 'Atlas' }, { value: 'rome', label: 'Rome' }]} onChange={(theme) => patchSettings({ theme })} />
        </section>
        <section>
          <label>Typeface</label>
          <p className="settings-help">
            Plex ships with Field, so it renders the same with no network. Monospaced data — costs,
            paths, tool names — always stays on IBM Plex Mono.
          </p>
          <ChoiceGroup
            value={settings.typeface}
            options={[{ value: 'plex', label: 'Plex' }, { value: 'system', label: 'System' }, { value: 'custom', label: 'Custom' }]}
            onChange={chooseTypeface}
          />
          <label className="upload-control">
            <Upload />{settings.customFont?.name ? `Replace ${settings.customFont.name}` : 'Choose a font file'}
            <input type="file" accept=".woff2,.ttf,.otf,font/woff2,font/ttf,font/otf" onChange={pickFont} />
          </label>
          <small>woff2, ttf or otf, under 1.5 MB. It is stored on this device with the rest of your settings.</small>
        </section>
        {onOpenModels && (
          <section>
            <label>Models</label>
            <p className="settings-help">Cameo boxes, Ollama and provider keys: what agents actually run on.</p>
            <button type="button" className="btn ghost" onClick={onOpenModels}><RadioTower aria-hidden="true" />Set up models</button>
          </section>
        )}
        <section>
          <label>World density</label>
          <ChoiceGroup value={settings.density} options={[{ value: 'quiet', label: 'Quiet' }, { value: 'balanced', label: 'Balanced' }, { value: 'dense', label: 'Dense' }]} onChange={(density) => patchSettings({ density })} />
        </section>
        <section className="settings-toggle">
          <div><label>World motion</label><p>Animate active routes and status pulses.</p></div>
          <button type="button" className={settings.motion ? 'on' : ''} aria-pressed={settings.motion} aria-label="World motion" onClick={() => patchSettings({ motion: !settings.motion })}><i /></button>
        </section>
      </>}
      {tab === 'identity' && <>
        <section><label>Agent identity</label><ChoiceGroup value={settings.identityMode} options={[{ value: 'portrait', label: 'Portrait' }, { value: 'model', label: 'Model' }, { value: 'both', label: 'Both' }]} onChange={(identityMode) => patchSettings({ identityMode })} /></section>
        <section><label>Map markers</label><ChoiceGroup value={settings.markerMode} options={[{ value: 'person', label: 'Person' }, { value: 'model', label: 'Model' }, { value: 'both', label: 'Both' }]} onChange={(markerMode) => patchSettings({ markerMode })} /></section>
        <section><label>Emblem source</label><select value={settings.emblemSource} aria-label="Emblem source" onChange={(event) => patchSettings({ emblemSource: event.target.value })}><option value="auto">Auto</option><option value="huggingface">Hugging Face</option><option value="endpoint">Endpoint</option><option value="upload">Upload</option><option value="initials">Initials</option></select><small>HF avatar › endpoint › upload › initials</small></section>
        {selected ? <section className="agent-settings"><label>Selected agent</label><h3>{override.displayName || selected.name}</h3><div className="settings-field"><span>Display name</span><input value={override.displayName ?? selected.name ?? ''} onChange={(event) => patchAgent({ displayName: event.target.value })} /></div><div className="settings-field"><span>Endpoint alias</span><input value={override.endpointAlias ?? ''} placeholder="Use endpoint name" onChange={(event) => patchAgent({ endpointAlias: event.target.value })} /></div><div className="settings-field"><span>HF repository</span><input value={override.hfRepo ?? ''} placeholder="owner/model" onChange={(event) => patchAgent({ hfRepo: event.target.value })} /></div><div className="settings-field"><span>Icon URL</span><input value={override.iconUrl ?? ''} placeholder="https://…" onChange={(event) => patchAgent({ iconUrl: event.target.value, iconDataUrl: '' })} /></div><label className="upload-control"><Upload />Upload icon<input type="file" accept="image/*" onChange={upload} /></label></section> : <section><p>Select an agent to edit its persistent name and emblem.</p></section>}
        {selected && <section><label>Tool authority</label><p className="settings-help">A configured agent may receive a subset of its role&apos;s real allowlist. Changes apply on its next deployment.</p><div className="authority-grid">{roleTools.map((tool) => <button type="button" key={tool} className={selectedTools.includes(tool) ? 'on' : ''} aria-pressed={selectedTools.includes(tool)} onClick={() => toggleTool(tool)}><ToolIcon name={tool} />{tool}<Check /></button>)}</div>{!configuredAgent && <small>Ad-hoc sessions cannot persist tool changes.</small>}</section>}
      </>}
      {tab === 'models' && <section className="model-registry-card"><label>Distillation teacher</label><h3>OX Alpha is GLM 5.3 Flash</h3><p>OX Alpha is the endpoint alias. It serves GLM 5.3 Flash and supplies teacher traces for the Ornith student.</p><div className="settings-field"><span>Teacher endpoint</span><input value={oxAlpha.endpointAlias} onChange={(event) => patchOx({ endpointAlias: event.target.value })} /></div><div className="settings-field"><span>Model served</span><input value={oxAlpha.servedModel} onChange={(event) => patchOx({ servedModel: event.target.value })} /></div><div className="settings-field"><span>HF repository</span><input value={oxAlpha.hfRepo} onChange={(event) => patchOx({ hfRepo: event.target.value })} /></div><div className="settings-field"><span>Student target</span><input value={oxAlpha.studentTarget} onChange={(event) => patchOx({ studentTarget: event.target.value })} /></div><div className="settings-toggle"><div><label>Collect teacher traces</label><p>Mark OX Alpha traces as distillation source data.</p></div><button type="button" className={oxAlpha.collectTeacherTraces ? 'on' : ''} aria-pressed={oxAlpha.collectTeacherTraces} aria-label="Collect teacher traces" onClick={() => patchOx({ collectTeacherTraces: !oxAlpha.collectTeacherTraces })}><i /></button></div></section>}
    </div>
    <footer>
      <button type="button" className="restore" onClick={() => setSettings(normalizeFieldSettings(DEFAULT_FIELD_SETTINGS))}>Restore defaults</button>
      <span role="status">{message}</span>
      <button type="button" className="save" disabled={saving} onClick={save}>{saving ? 'Saving…' : 'Save settings'}</button>
    </footer>
  </aside>;
}
