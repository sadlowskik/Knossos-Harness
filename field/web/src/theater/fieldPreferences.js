/* v4 dropped `theme` (the Board / Rome choice is navigation, not a preference) and
   folded identityMode + markerMode into one `identity`. The key is bumped so a stale
   v3 blob carrying `theme: 'rome'` cannot resurrect a branch that no longer exists. */
const STORAGE_KEY = 'knossos.field.presentation.v4';

// The sans stacks the typeface setting switches between. --font-mono is never touched:
// numbers, paths and tool names stay on IBM Plex Mono whatever the operator picks.
export const CUSTOM_FONT_FAMILY = 'Field Custom';
export const CUSTOM_FONT_MAX_BYTES = 1_572_864; // ~1.5 MB; the blob shares localStorage
const PLEX_SANS_STACK = '"IBM Plex Sans", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';
const SYSTEM_SANS_STACK = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif';

export const DEFAULT_FIELD_SETTINGS = Object.freeze({
  typeface: 'plex',
  customFont: { name: '', dataUrl: '' },
  // person | model | both — one answer, used by the Board cards and the Rome markers.
  identity: 'both',
  emblemSource: 'auto',
  density: 'balanced',
  motion: true,
  agentOverrides: {},
  // Which files each conversation has in scope: { [sessionId]: ['src/a.rs', …] }.
  // Client-side only — the server has no notion of an attached file; the Board prepends
  // the list to the next message it sends.
  sessionFiles: {},
  modelRegistry: {
    'ox-alpha': {
      endpointAlias: 'OX Alpha',
      servedModel: 'GLM 5.3 Flash',
      hfRepo: 'zai-org/GLM-5.3-Flash',
      distillationRole: 'teacher',
      studentTarget: 'Ornith 35B A3B',
      collectTeacherTraces: true,
    },
  },
});

function safeObject(value) {
  return value && typeof value === 'object' && !Array.isArray(value) ? value : {};
}

// A stored custom font is a name plus a data: URL. Anything else is dropped, so a
// corrupted or hand-edited entry can never be handed to FontFace.
function safeCustomFont(value) {
  const input = safeObject(value);
  const name = typeof input.name === 'string' ? input.name.slice(0, 120) : '';
  const dataUrl = typeof input.dataUrl === 'string' && /^data:[^;,]*;base64,/.test(input.dataUrl)
    ? input.dataUrl
    : '';
  return dataUrl ? { name, dataUrl } : { name: '', dataUrl: '' };
}

// identityMode ('portrait'|'model'|'both') and markerMode ('person'|'model'|'both') asked
// the same three-way question twice. Read both once, here, so a v3 blob still lands
// somewhere sensible; afterwards only `identity` exists.
function migratedIdentity(input) {
  const raw = input.identity ?? input.markerMode ?? input.identityMode;
  const value = raw === 'portrait' ? 'person' : raw;
  return ['person', 'model', 'both'].includes(value) ? value : DEFAULT_FIELD_SETTINGS.identity;
}

export const MAX_SESSION_FILES = 24;
const MAX_SESSION_FILE_PATH = 400;
const MAX_SESSIONS_WITH_FILES = 200;
const NO_FILES = Object.freeze([]);

/* The attached-file map is operator input that survives a reload, so it is normalized
   the same way the other stored keys are: anything that is not a list of workspace-
   relative paths is dropped rather than handed back to the picker or to an agent. */
function safeSessionFiles(value) {
  const input = safeObject(value);
  const out = {};
  for (const [id, list] of Object.entries(input).slice(0, MAX_SESSIONS_WITH_FILES)) {
    if (!Array.isArray(list) || !id) continue;
    const paths = [...new Set(list
      .filter((path) => typeof path === 'string' && path && path.length <= MAX_SESSION_FILE_PATH)
      .map((path) => path.replaceAll('\\', '/').replace(/^\/+/, '').trim())
      .filter(Boolean))].slice(0, MAX_SESSION_FILES);
    if (paths.length) out[String(id)] = paths;
  }
  return out;
}

/** The files attached to one conversation. Stable empty array, so memos stay stable. */
export function sessionFilesFor(settings, sessionId) {
  return settings?.sessionFiles?.[sessionId] ?? NO_FILES;
}

/** Settings with one conversation's attached files replaced (empty removes the key). */
export function withSessionFiles(settings, sessionId, paths) {
  const next = { ...(settings?.sessionFiles ?? {}) };
  if (paths?.length) next[sessionId] = paths;
  else delete next[sessionId];
  return { ...settings, sessionFiles: next };
}

export function normalizeFieldSettings(value = {}) {
  const input = safeObject(value);
  // The removed keys are destructured away so a stored blob cannot smuggle them back in.
  const { theme: _theme, identityMode: _identityMode, markerMode: _markerMode, ...rest } = input;
  const customFont = safeCustomFont(input.customFont);
  const typeface = ['plex', 'system', 'custom'].includes(input.typeface) ? input.typeface : DEFAULT_FIELD_SETTINGS.typeface;
  return {
    ...DEFAULT_FIELD_SETTINGS,
    ...rest,
    // "custom" without a usable blob is just the default.
    typeface: typeface === 'custom' && !customFont.dataUrl ? DEFAULT_FIELD_SETTINGS.typeface : typeface,
    customFont,
    identity: migratedIdentity(input),
    emblemSource: ['auto', 'huggingface', 'endpoint', 'upload', 'initials'].includes(input.emblemSource) ? input.emblemSource : DEFAULT_FIELD_SETTINGS.emblemSource,
    density: ['quiet', 'balanced', 'dense'].includes(input.density) ? input.density : DEFAULT_FIELD_SETTINGS.density,
    motion: input.motion !== false,
    agentOverrides: safeObject(input.agentOverrides),
    sessionFiles: safeSessionFiles(input.sessionFiles),
    modelRegistry: {
      ...DEFAULT_FIELD_SETTINGS.modelRegistry,
      ...safeObject(input.modelRegistry),
    },
  };
}

export function loadFieldSettings() {
  if (typeof window === 'undefined') return normalizeFieldSettings();
  try { return normalizeFieldSettings(JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? '{}')); }
  catch { return normalizeFieldSettings(); }
}

export function saveFieldSettings(settings) {
  const normalized = normalizeFieldSettings(settings);
  if (typeof window !== 'undefined') window.localStorage.setItem(STORAGE_KEY, JSON.stringify(normalized));
  return normalized;
}

// ---- typeface ---------------------------------------------------------------

export function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error('unreadable'));
    reader.onload = () => resolve(String(reader.result));
    reader.readAsDataURL(file);
  });
}

// Rejects when the bytes are not a font the engine can parse, which is how the
// settings panel knows to keep the default.
export async function loadCustomFont(dataUrl) {
  if (typeof window === 'undefined' || !window.FontFace || !document.fonts) throw new Error('no font loader');
  for (const face of [...document.fonts]) {
    if (face.family === CUSTOM_FONT_FAMILY) document.fonts.delete(face);
  }
  const face = new FontFace(CUSTOM_FONT_FAMILY, `url("${dataUrl}")`);
  await face.load();
  document.fonts.add(face);
  return face;
}

// Applies the stored choice to --font-sans and answers with the typeface that is
// actually in force, so a blob that no longer loads can be reconciled back to Plex.
export async function applyTypeface(settings = DEFAULT_FIELD_SETTINGS) {
  if (typeof document === 'undefined') return 'plex';
  const root = document.documentElement;
  if (settings.typeface === 'system') {
    root.style.setProperty('--font-sans', SYSTEM_SANS_STACK);
    return 'system';
  }
  if (settings.typeface === 'custom' && settings.customFont?.dataUrl) {
    try {
      await loadCustomFont(settings.customFont.dataUrl);
      root.style.setProperty('--font-sans', `"${CUSTOM_FONT_FAMILY}", ${PLEX_SANS_STACK}`);
      return 'custom';
    } catch { /* fall through to the default below */ }
  }
  root.style.removeProperty('--font-sans');
  return 'plex';
}

export function agentPreferenceKey(session) {
  return String(session?.agentId || session?.id || 'unknown');
}

const REPO_HINTS = [
  [/glm|ox[-_ ]?alpha/i, 'zai-org/GLM-5.3-Flash'],
  [/qwen/i, 'Qwen/Qwen3'],
  [/deepseek/i, 'deepseek-ai/DeepSeek-V3'],
  [/nemotron/i, 'nvidia/Llama-3.1-Nemotron'],
];

export function inferredRepo(model = '', endpoint = {}) {
  const explicit = endpoint.hf_repo ?? endpoint.hfRepo ?? endpoint.repository;
  if (explicit) return String(explicit);
  const haystack = `${model} ${endpoint.id ?? ''} ${endpoint.name ?? ''}`;
  return REPO_HINTS.find(([pattern]) => pattern.test(haystack))?.[1] ?? '';
}

function endpointIcon(endpoint = {}) {
  return endpoint.icon_url ?? endpoint.iconUrl ?? endpoint.avatar_url ?? endpoint.avatarUrl ?? endpoint.logo_url ?? endpoint.logoUrl ?? '';
}

function hfAvatar(repo = '') {
  const owner = String(repo).split('/')[0];
  return owner ? `https://huggingface.co/avatars/${encodeURIComponent(owner)}.svg` : '';
}

const OX_ALPHA = /glm[-_ ]?5[._ -]?3.*flash|ox[-_ ]?alpha/i;

/* Which modelRegistry entry describes this endpoint. The naming fields used to live in
   a Models tab of the settings panel; they now sit beside the endpoint in PowerSources,
   keyed by endpoint id — except the built-in OX Alpha teacher, which keeps its name. */
export function registryKeyFor(endpoint = {}) {
  const haystack = `${endpoint.model ?? ''} ${endpoint.id ?? ''} ${endpoint.name ?? ''}`;
  return OX_ALPHA.test(haystack) ? 'ox-alpha' : String(endpoint.id ?? '');
}

export function identityFor(session, endpoints = [], settings = DEFAULT_FIELD_SETTINGS) {
  const endpoint = endpoints.find((item) => item.id === session?.endpointId) ?? {};
  const key = agentPreferenceKey(session);
  const override = settings.agentOverrides?.[key] ?? {};
  const rawModel = override.servedModel || session?.model || endpoint.model || 'unassigned';
  const oxAlpha = OX_ALPHA.test(`${rawModel} ${endpoint.id ?? ''} ${endpoint.name ?? ''}`);
  const registryKey = oxAlpha ? 'ox-alpha' : String(endpoint.id ?? '');
  const registered = settings.modelRegistry?.[registryKey]
    ?? (oxAlpha ? DEFAULT_FIELD_SETTINGS.modelRegistry['ox-alpha'] : {});
  const servedModel = override.servedModel || registered.servedModel || rawModel;
  const endpointAlias = override.endpointAlias || registered.endpointAlias || endpoint.name || endpoint.id || servedModel;
  const hfRepo = override.hfRepo || registered.hfRepo || inferredRepo(servedModel, endpoint);
  const source = settings.emblemSource;
  const uploaded = override.iconDataUrl || override.iconUrl || '';
  const fromEndpoint = endpointIcon(endpoint);
  let iconUrl = '';
  if (source === 'upload') iconUrl = uploaded;
  else if (source === 'endpoint') iconUrl = fromEndpoint;
  else if (source === 'huggingface') iconUrl = hfAvatar(hfRepo);
  else if (source !== 'initials') iconUrl = uploaded || fromEndpoint || hfAvatar(hfRepo);
  const displayName = override.displayName || session?.name || session?.agentId || 'Unnamed agent';
  return {
    key,
    displayName,
    endpointAlias,
    servedModel,
    hfRepo,
    iconUrl,
    source: uploaded ? 'custom' : fromEndpoint ? 'endpoint' : hfRepo ? 'huggingface' : 'initials',
    distillationRole: registered.distillationRole || override.distillationRole || '',
    studentTarget: registered.studentTarget || override.studentTarget || '',
    collectTeacherTraces: registered.collectTeacherTraces ?? !!override.collectTeacherTraces,
  };
}

export function initials(value = '') {
  const parts = String(value).trim().split(/[\s/_-]+/).filter(Boolean);
  return (parts.length > 1 ? `${parts[0][0]}${parts.at(-1)[0]}` : parts[0]?.slice(0, 2) || '?').toUpperCase();
}

export function identityHue(value = '') {
  let hash = 0;
  for (const char of String(value)) hash = ((hash << 5) - hash + char.charCodeAt(0)) | 0;
  return Math.abs(hash) % 360;
}

export function verifiedContribution(cluster, agents = []) {
  if (cluster?.metrics) return cluster.metrics;
  return {
    score: 0, tier: 0, activity: 0, complete: 0, verified: 0,
    persisted: 0, reliability: 0, evidence: [],
  };
}
