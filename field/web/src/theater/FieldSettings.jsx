import { useState } from 'react';
import { Check, MapPin, RadioTower, Upload, X } from 'lucide-react';
import { api } from '../net/client.js';
import ToolIcon from '../ui/ToolIcon.jsx';
import RoutinesPanel from '../routines/RoutinesPanel.jsx';
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

/* The one settings panel, opened from the gear on either screen.

   Its third tab is Routines, which used to be a destination of its own. Standing work —
   what is armed, when it fires, what came of it last — is a setting, not a place, so the
   whole panel moved in here rather than keeping a nav button alive for it. */
export default function FieldSettings({
  settings, setSettings, selected, config, onClose,
  onOpenModels = null, onChooseCapital = null,
  standalone = false,
}) {
  const [tab, setTab] = useState('field');
  const [saving, setSaving] = useState(false);
  const [message, setMessage] = useState('');
  const key = selected ? agentPreferenceKey(selected) : null;
  const override = key ? settings.agentOverrides?.[key] ?? {} : {};
  const configuredAgent = selected ? config?.agents?.find((agent) => agent.id === selected.agentId) : null;
  const role = selected ? config?.roles?.find((item) => item.id === selected.role) : null;
  const roleTools = role?.tools_allow ?? [];
  const selectedTools = override.toolsAllow ?? configuredAgent?.tools_allow ?? roleTools;
  // Rome hands in its world handlers; the Board does not have a world to configure.
  const worldRows = [onOpenModels, onChooseCapital].some(Boolean);

  function patchSettings(patch) { setSettings((current) => normalizeFieldSettings({ ...current, ...patch })); }
  function patchAgent(patch) { if (key) patchSettings({ agentOverrides: { ...settings.agentOverrides, [key]: { ...override, ...patch } } }); }
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
    <nav>{['field', 'agents', 'routines'].map((item) => (
      <button type="button" key={item} className={tab === item ? 'on' : ''} onClick={() => setTab(item)}>{item}</button>
    ))}</nav>
    <div className="settings-scroll">
      {tab === 'field' && <>
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
        {/* What used to be three buttons in Rome's header. They open dialogs, so they are
            rows here rather than settings, and only appear on the screen that has a world. */}
        {worldRows && (
          <section className="settings-rows">
            <label>{onChooseCapital ? 'World' : 'Models'}</label>
            {onOpenModels && <button type="button" className="btn ghost" onClick={onOpenModels}><RadioTower aria-hidden="true" />Models<small>Cameo boxes, Ollama, provider keys</small></button>}
            {onChooseCapital && <button type="button" className="btn ghost" onClick={onChooseCapital}><MapPin aria-hidden="true" />Capital<small>The project the world is anchored on</small></button>}
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
      {tab === 'agents' && <>
        <section>
          <label>Show an agent as</label>
          <p className="settings-help">One answer for the whole app: the cards, the panels and the map markers all use it.</p>
          <ChoiceGroup value={settings.identity} options={[{ value: 'person', label: 'Person' }, { value: 'model', label: 'Model' }, { value: 'both', label: 'Both' }]} onChange={(identity) => patchSettings({ identity })} />
        </section>
        <section><label>Emblem source</label><select value={settings.emblemSource} aria-label="Emblem source" onChange={(event) => patchSettings({ emblemSource: event.target.value })}><option value="auto">Auto</option><option value="huggingface">Hugging Face</option><option value="endpoint">Endpoint</option><option value="upload">Upload</option><option value="initials">Initials</option></select><small>HF avatar › endpoint › upload › initials</small></section>
        {selected ? <section className="agent-settings"><label>Selected agent</label><h3>{override.displayName || selected.name}</h3><div className="settings-field"><span>Display name</span><input value={override.displayName ?? selected.name ?? ''} onChange={(event) => patchAgent({ displayName: event.target.value })} /></div><div className="settings-field"><span>Endpoint alias</span><input value={override.endpointAlias ?? ''} placeholder="Use endpoint name" onChange={(event) => patchAgent({ endpointAlias: event.target.value })} /></div><div className="settings-field"><span>HF repository</span><input value={override.hfRepo ?? ''} placeholder="owner/model" onChange={(event) => patchAgent({ hfRepo: event.target.value })} /></div><div className="settings-field"><span>Icon URL</span><input value={override.iconUrl ?? ''} placeholder="https://…" onChange={(event) => patchAgent({ iconUrl: event.target.value, iconDataUrl: '' })} /></div><label className="upload-control"><Upload />Upload icon<input type="file" accept="image/*" onChange={upload} /></label></section> : <section><p>Select an agent to edit its persistent name and emblem.</p></section>}
        {selected && <section><label>Tool authority</label><p className="settings-help">A configured agent may receive a subset of its role&apos;s real allowlist. Changes apply on its next deployment.</p><div className="authority-grid">{roleTools.map((tool) => <button type="button" key={tool} className={selectedTools.includes(tool) ? 'on' : ''} aria-pressed={selectedTools.includes(tool)} onClick={() => toggleTool(tool)}><ToolIcon name={tool} />{tool}<Check /></button>)}</div>{!configuredAgent && <small>Ad-hoc sessions cannot persist tool changes.</small>}</section>}
      </>}
      {tab === 'routines' && <RoutinesPanel />}
    </div>
    <footer>
      <button type="button" className="restore" onClick={() => setSettings(normalizeFieldSettings(DEFAULT_FIELD_SETTINGS))}>Restore defaults</button>
      <span role="status">{message}</span>
      <button type="button" className="save" disabled={saving} onClick={save}>{saving ? 'Saving…' : 'Save settings'}</button>
    </footer>
  </aside>;
}
