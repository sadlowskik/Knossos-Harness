const STORAGE_KEY = 'knossos.field.presentation.v3';

export const DEFAULT_FIELD_SETTINGS = Object.freeze({
  theme: 'atlas',
  identityMode: 'both',
  markerMode: 'both',
  emblemSource: 'auto',
  density: 'balanced',
  motion: true,
  agentOverrides: {},
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

export function normalizeFieldSettings(value = {}) {
  const input = safeObject(value);
  return {
    ...DEFAULT_FIELD_SETTINGS,
    ...input,
    theme: ['rome', 'atlas'].includes(input.theme) ? input.theme : DEFAULT_FIELD_SETTINGS.theme,
    identityMode: ['portrait', 'model', 'both'].includes(input.identityMode) ? input.identityMode : DEFAULT_FIELD_SETTINGS.identityMode,
    markerMode: ['person', 'model', 'both'].includes(input.markerMode) ? input.markerMode : DEFAULT_FIELD_SETTINGS.markerMode,
    emblemSource: ['auto', 'huggingface', 'endpoint', 'upload', 'initials'].includes(input.emblemSource) ? input.emblemSource : DEFAULT_FIELD_SETTINGS.emblemSource,
    density: ['quiet', 'balanced', 'dense'].includes(input.density) ? input.density : DEFAULT_FIELD_SETTINGS.density,
    motion: input.motion !== false,
    agentOverrides: safeObject(input.agentOverrides),
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

export function identityFor(session, endpoints = [], settings = DEFAULT_FIELD_SETTINGS) {
  const endpoint = endpoints.find((item) => item.id === session?.endpointId) ?? {};
  const key = agentPreferenceKey(session);
  const override = settings.agentOverrides?.[key] ?? {};
  const rawModel = override.servedModel || session?.model || endpoint.model || 'unassigned';
  const oxAlpha = /glm[-_ ]?5[._ -]?3.*flash|ox[-_ ]?alpha/i.test(`${rawModel} ${endpoint.id ?? ''} ${endpoint.name ?? ''}`);
  const teacher = settings.modelRegistry?.['ox-alpha'] ?? DEFAULT_FIELD_SETTINGS.modelRegistry['ox-alpha'];
  const servedModel = oxAlpha ? teacher.servedModel : rawModel;
  const endpointAlias = override.endpointAlias || (oxAlpha ? teacher.endpointAlias : endpoint.name || endpoint.id || servedModel);
  const hfRepo = override.hfRepo || (oxAlpha ? teacher.hfRepo : inferredRepo(servedModel, endpoint));
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
    distillationRole: oxAlpha ? teacher.distillationRole : override.distillationRole || '',
    studentTarget: oxAlpha ? teacher.studentTarget : override.studentTarget || '',
    collectTeacherTraces: oxAlpha ? teacher.collectTeacherTraces : !!override.collectTeacherTraces,
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

/**
 * Build the temporary world shown during a rehearsal. Production collections are
 * never merged into synthetic collections, so demo activity cannot inflate real
 * scores and synthetic history disappears when the active run ends.
 */
export function rehearsalSnapshot(snapshot, activeRun) {
  const rehearsal = snapshot?.rehearsal;
  if (!activeRun || !rehearsal) return snapshot;
  return {
    ...snapshot,
    sessions: rehearsal.sessions ?? [],
    workspaces: rehearsal.workspaces ?? snapshot.workspaces,
    folders: rehearsal.folders ?? [],
    files: rehearsal.files ?? [],
    websites: rehearsal.websites ?? [],
    assignments: rehearsal.assignments ?? [],
    permissions: [],
    campaigns: [],
    checkpoints: [],
    graph: rehearsal.graph ?? { nodes: [], edges: [], visibleNodeKeys: [] },
    totals: rehearsal.totals ?? { costUsd: 0 },
    rehearsalMode: true,
  };
}

export function verifiedContribution(cluster, agents = []) {
  if (cluster?.metrics) return cluster.metrics;
  return {
    score: 0, tier: 0, activity: 0, complete: 0, verified: 0,
    persisted: 0, reliability: 0, evidence: [],
  };
}
