import { useEffect, useMemo, useRef, useState } from 'react';
import {
  BookOpen, Box, Check, Container, Database, GitBranch, Globe2,
  Landmark, Network, Play, RadioTower, Settings, ShieldCheck, Sparkles, Square,
  TerminalSquare, Upload, UserRound, Wrench, X,
} from 'lucide-react';
import { api } from '../net/client.js';
import { clearActiveAgent, openCity, openSenate, selectAgent, useField } from '../state/store.js';
import { useModalFocus } from '../ui/useModalFocus.js';
import AtlasMode from '../atlas/AtlasMode.jsx';
import {
  DEFAULT_FIELD_SETTINGS,
  agentPreferenceKey,
  identityFor,
  identityHue,
  initials,
  loadFieldSettings,
  normalizeFieldSettings,
  rehearsalSnapshot,
  saveFieldSettings,
  verifiedContribution,
} from './fieldPreferences.js';

const TERMINAL_STATES = new Set(['done', 'cancelled', 'interrupted', 'error']);
const ATTENTION_STATES = new Set(['blocked', 'error', 'waiting_permission']);
const PROJECT_SLOTS = [
  { x: 35, y: 18, w: 17, h: 22 },
  { x: 13, y: 25, w: 17, h: 21 }, { x: 66, y: 32, w: 17, h: 21 },
  { x: 20, y: 63, w: 17, h: 20 }, { x: 62, y: 62, w: 17, h: 20 },
];
const FRONT_OFFSETS = [
  { dx: -27, dy: -18, w: 22, h: 24 }, { dx: 24, dy: -17, w: 22, h: 24 },
  { dx: -27, dy: 24, w: 22, h: 24 }, { dx: 24, dy: 24, w: 22, h: 24 },
  { dx: 0, dy: -29, w: 22, h: 22 }, { dx: 0, dy: 34, w: 22, h: 22 },
];
const OUTPOST_SLOTS = [
  { x: 2, y: 38, w: 12, h: 16 }, { x: 86, y: 38, w: 12, h: 16 },
  { x: 27, y: 3, w: 13, h: 15 }, { x: 61, y: 3, w: 13, h: 15 },
  { x: 28, y: 79, w: 13, h: 15 }, { x: 60, y: 79, w: 13, h: 15 },
];
const CITY_SLOTS = [[28, 58], [68, 57], [48, 72], [61, 31]];
const AGENT_SLOTS = [
  [17, 30], [34, 20], [52, 18], [70, 24], [82, 38], [84, 58],
  [71, 73], [53, 79], [34, 76], [18, 64], [14, 47], [50, 49],
];
const GATEWAY_SLOTS = [[5, 22], [93, 23], [94, 75], [50, 92], [4, 76]];
const LANDMARK_KIND = {
  model: 'model', runtime: 'runtime', knowledge: 'archive', verification: 'verification',
  interface: 'interface', storage: 'storage', core: 'core', general: 'project',
};
const KIND_ICON = { project: Landmark, workfront: GitBranch, gateway: RadioTower, service: Container, model: BookOpen, runtime: Wrench, archive: Database, verification: ShieldCheck, interface: Globe2, storage: Box, core: Network };
const SETTLEMENT_MODULES = [
  { id: 'forum', src: '/assets/living-rome/modules/forum.webp', label: 'civic forum' },
  { id: 'works', src: '/assets/living-rome/modules/works.webp', label: 'engineering works' },
  { id: 'archive', src: '/assets/living-rome/modules/archive.webp', label: 'archive and observatory' },
  { id: 'gate', src: '/assets/living-rome/modules/gate.webp', label: 'fortified gate' },
];
const SETTLEMENT_LAYOUTS = [
  [{ x: 50, y: 50, size: 82, rotate: -1 }, { x: 24, y: 67, size: 48, rotate: 2 }, { x: 78, y: 69, size: 43, rotate: -2 }],
  [{ x: 48, y: 54, size: 80, rotate: 1 }, { x: 76, y: 35, size: 46, rotate: -2 }, { x: 25, y: 72, size: 44, rotate: 1 }],
  [{ x: 51, y: 48, size: 79, rotate: 0 }, { x: 23, y: 38, size: 45, rotate: -2 }, { x: 76, y: 73, size: 47, rotate: 2 }],
  [{ x: 49, y: 53, size: 81, rotate: -1 }, { x: 75, y: 42, size: 44, rotate: 1 }, { x: 23, y: 70, size: 46, rotate: -2 }],
];

function settlementSeed(value = '') {
  const token = String(value).normalize('NFKD').replace(/[^a-z0-9]/gi, '').slice(0, 4).toUpperCase().padEnd(4, 'X');
  let hash = 2166136261;
  for (const character of token) {
    hash ^= character.charCodeAt(0);
    hash = Math.imul(hash, 16777619);
  }
  return { token, value: hash >>> 0 };
}

function settlementPlan(name, isCapital, tier) {
  const seed = settlementSeed(name);
  const layout = SETTLEMENT_LAYOUTS[seed.value % SETTLEMENT_LAYOUTS.length];
  const moduleOffset = (seed.value >>> 3) % SETTLEMENT_MODULES.length;
  const modules = SETTLEMENT_MODULES.map((_, index) => SETTLEMENT_MODULES[(index + moduleOffset) % SETTLEMENT_MODULES.length]);
  const count = isCapital ? 3 : tier >= 3 ? 2 : 1;
  return {
    seed,
    terrain: (seed.value >>> 7) % 4,
    rotation: ((seed.value >>> 11) % 7) - 3,
    roadA: 18 + ((seed.value >>> 14) % 22),
    roadB: 138 + ((seed.value >>> 19) % 28),
    modules: modules.slice(0, count).map((module, index) => ({ ...module, ...layout[index] })),
  };
}

function classify(value = '') {
  const text = String(value).toLowerCase();
  if (/(model|inference|llm|weights|adapter|lora)/.test(text)) return 'model';
  if (/(docker|podman|container|k8s|kube|deploy|runtime|service)/.test(text)) return 'runtime';
  if (/(docs|memory|context|knowledge|research)/.test(text)) return 'knowledge';
  if (/(test|verify|eval|audit|security|red.?team|blue.?team)/.test(text)) return 'verification';
  if (/(site|web|frontend|ui|client|landing)/.test(text)) return 'interface';
  if (/(db|database|store|storage|data|vector|volume|cache)/.test(text)) return 'storage';
  if (/(api|server|core|src|engine|orchestrat|harness)/.test(text)) return 'core';
  return 'general';
}
function basename(value = '') { return String(value).replaceAll('\\', '/').split('/').filter(Boolean).at(-1) || 'root'; }
function stateLabel(value = 'unknown') { return String(value).replaceAll('_', ' '); }
function clipLabel(value = '', length = 42) { const text = String(value).trim().replace(/\s+/g, ' '); return text.length > length ? `${text.slice(0, length - 1).trim()}…` : text; }

function frontKeyFor(session) {
  if (session.objectiveId) return `objective:${session.objectiveId}`;
  if (session.campaignId) return `campaign:${session.campaignId}`;
  if (session.simulationRunId) return `rehearsal:${session.simulationRunId}`;
  if (session.assignmentId) return `assignment:${session.assignmentId}`;
  return null;
}

function objectiveLabel(assignment, sessions, campaigns, workspaceById) {
  const first = sessions[0];
  if (first?.objectiveId) {
    for (const campaign of campaigns) {
      const objective = campaign.objectives?.find((item) => item.id === first.objectiveId);
      if (objective?.statement) return clipLabel(objective.statement);
    }
  }
  if (first?.campaignId) {
    const campaign = campaigns.find((item) => item.id === first.campaignId);
    if (campaign?.name) return clipLabel(campaign.name);
  }
  if (first?.simulationRunId) {
    const workspace = workspaceById.get(first.workspaceId);
    return `${workspace?.name ?? 'Field'} rehearsal`;
  }
  if (assignment?.targetType === 'mission' && assignment.targetLabel) return clipLabel(assignment.targetLabel);
  const paragraphs = String(assignment?.orders ?? '').split(/\n\s*\n/).map((item) => item.trim()).filter(Boolean);
  if (paragraphs.length > 1) return clipLabel(paragraphs.at(-1).split(/(?<=[.!?])\s/)[0]);
  return 'Active objective';
}

function activityMetrics(activity = 0) {
  return {
    score: 0, tier: 0, activity: Math.max(0, Math.min(100, activity)),
    complete: 0, verified: 0, persisted: 0, reliability: 0, evidence: [],
  };
}

function infrastructureClusters(snap, capitalWorkspaceId) {
  const workspaceById = new Map(snap.workspaces.map((item) => [item.id, item]));
  const sessionById = new Map(snap.sessions.map((item) => [item.id, item]));
  const liveSessions = snap.sessions.filter((session) => !TERMINAL_STATES.has(session.state));
  const liveSessionIds = new Set(liveSessions.map((session) => session.id));
  const foldersByWorkspace = new Map();
  for (const folder of snap.folders) {
    if (!workspaceById.has(folder.workspaceId)) continue;
    const rows = foldersByWorkspace.get(folder.workspaceId) ?? [];
    rows.push(folder);
    foldersByWorkspace.set(folder.workspaceId, rows);
  }

  const projects = snap.workspaces.filter((item) => item.mounted).map((workspace) => {
    const history = snap.sessions.filter((item) => item.workspaceId === workspace.id);
    const folders = [...(foldersByWorkspace.get(workspace.id) ?? [])].sort((a, b) => (b.hits ?? 0) - (a.hits ?? 0));
    const resources = folders.slice(0, 4).map((folder) => {
      const occupants = (folder.agents ?? []).map((id) => sessionById.get(id)).filter(Boolean);
      const state = folder.agents?.some((id) => liveSessionIds.has(id))
        ? 'running'
        : occupants.some((item) => item.verified === 'verified')
          ? 'verified'
          : occupants.some((item) => item.state === 'done')
            ? 'completed'
          : folder.hits ? 'active' : 'discovered';
      return { id: folder.key, label: basename(folder.dir), kind: LANDMARK_KIND[classify(folder.dir)], state };
    });
    const failures = history.filter((item) => ['error', 'blocked'].includes(item.state)).length;
    return {
      clusterKey: `workspace:${workspace.id}`, label: workspace.name, displayLabel: workspace.name,
      kind: 'project', workspaceId: workspace.id, parentKey: null,
      activity: (workspace.changeCount ?? 0) + 20 + folders.reduce((sum, item) => sum + (item.hits ?? 0), 0),
      resources, metrics: workspace.maturity ?? activityMetrics(),
      damaged: failures > 0, failureCount: failures,
      stateLabel: workspace.id === capitalWorkspaceId ? 'capital project' : 'project settlement',
    };
  }).sort((a, b) => Number(b.workspaceId === capitalWorkspaceId) - Number(a.workspaceId === capitalWorkspaceId));

  const assignmentById = new Map((snap.assignments ?? []).map((item) => [item.id, item]));
  const frontsByKey = new Map();
  for (const session of liveSessions) {
    const key = frontKeyFor(session);
    if (!key) continue;
    const front = frontsByKey.get(key) ?? { key, sessions: [], workspaceId: session.workspaceId };
    front.sessions.push(session);
    front.workspaceId ||= session.workspaceId;
    frontsByKey.set(key, front);
  }
  const fronts = [...frontsByKey.values()].map((front) => {
    const assignment = assignmentById.get(front.sessions[0]?.assignmentId);
    const resources = new Map();
    for (const session of front.sessions) {
      const path = session.focusDir ?? session.target?.id;
      if (!path) continue;
      const id = `${front.workspaceId}:${path}`;
      resources.set(id, { id, label: basename(path), kind: LANDMARK_KIND[classify(path)], state: 'running' });
    }
    const teams = new Set(front.sessions.map((item) => item.team || (item.role === 'challenger' ? 'red' : ['verifier', 'referee'].includes(item.role) ? 'blue' : 'operator')));
    const damaged = front.sessions.some((item) => ATTENTION_STATES.has(item.state));
    const contested = damaged || (teams.has('red') && teams.has('blue'));
    const defended = teams.has('blue') || front.sessions.some((item) => item.role === 'verifier');
    const label = objectiveLabel(assignment, front.sessions, snap.campaigns ?? [], workspaceById);
    const resourceList = [...resources.values()].slice(0, 2);
    return {
      clusterKey: front.key, label, displayLabel: label, kind: `workfront${contested ? ' contested' : ''}${defended ? ' defended' : ''}${damaged ? ' damaged' : ''}`, baseKind: 'workfront', workspaceId: front.workspaceId,
      parentKey: `workspace:${front.workspaceId || capitalWorkspaceId}`,
      activity: front.sessions.length * 18, resources: resourceList,
      metrics: activityMetrics(Math.min(100, front.sessions.length * 20)), teams: [...teams], contested, defended, damaged,
      stateLabel: damaged ? 'requires intervention' : contested ? 'red / blue contest' : defended ? 'verification front' : 'active operation',
    };
  }).sort((a, b) => b.activity - a.activity);

  const frontForSession = new Map();
  for (const front of frontsByKey.values()) for (const session of front.sessions) frontForSession.set(session.id, front.key);
  const gateways = (snap.websites ?? []).flatMap((site) => {
    const activeVisitors = (site.sessions ?? []).filter((id) => liveSessionIds.has(id));
    if (!activeVisitors.length) return [];
    const first = sessionById.get(activeVisitors[0]);
    return [{
      clusterKey: `gateway:${site.domain}`, label: site.label || site.domain, displayLabel: site.label || site.domain,
      kind: 'gateway', workspaceId: first?.workspaceId ?? null,
      parentKey: frontForSession.get(activeVisitors[0]) ?? `workspace:${first?.workspaceId || capitalWorkspaceId}`,
      activity: activeVisitors.length * 9, resources: [{ id: site.domain, label: site.domain, kind: 'gateway', state: 'running' }],
      metrics: activityMetrics(Math.min(100, activeVisitors.length * 25)),
      stateLabel: 'observed external site',
    }];
  });
  const services = (snap.endpoints ?? []).flatMap((endpoint) => {
    const operators = liveSessions.filter((item) => item.endpointId === endpoint.id);
    if (!operators.length || endpoint.status === 'down') return [];
    const first = operators[0];
    return [{
      clusterKey: `service:${endpoint.id}`, label: endpoint.name || endpoint.model || endpoint.id,
      displayLabel: endpoint.name || endpoint.model || endpoint.id, kind: 'service', workspaceId: first.workspaceId,
      parentKey: frontForSession.get(first.id) ?? `workspace:${first.workspaceId || capitalWorkspaceId}`,
      activity: operators.length * 7, resources: [{ id: endpoint.id, label: endpoint.model || 'model endpoint', kind: 'service', state: 'running' }],
      metrics: activityMetrics(Math.min(100, operators.length * 25)), stateLabel: 'serving active agents',
    }];
  });

  return [...projects, ...fronts, ...gateways, ...services].slice(0, 16);
}

function livingWorldPositions(clusters, capitalWorkspaceId) {
  const projects = clusters.filter((item) => item.kind === 'project');
  const capitalKey = `workspace:${capitalWorkspaceId}`;
  projects.sort((a, b) => Number(b.clusterKey === capitalKey) - Number(a.clusterKey === capitalKey));
  const positions = [];
  const projectPosition = new Map();
  const occupiedProjectSlots = new Set();
  projects.forEach((cluster, index) => {
    const preferred = settlementSeed(cluster.displayLabel || cluster.label).value % PROJECT_SLOTS.length;
    let slotIndex = preferred;
    while (occupiedProjectSlots.has(slotIndex) && occupiedProjectSlots.size < PROJECT_SLOTS.length) slotIndex = (slotIndex + 1) % PROJECT_SLOTS.length;
    occupiedProjectSlots.add(slotIndex);
    const slot = PROJECT_SLOTS[slotIndex] ?? PROJECT_SLOTS[Math.min(index, PROJECT_SLOTS.length - 1)];
    const position = { ...slot, id: cluster.clusterKey, clusterKey: cluster.clusterKey };
    positions.push(position); projectPosition.set(cluster.clusterKey, position);
  });
  const frontCounts = new Map();
  for (const cluster of clusters.filter((item) => (item.baseKind ?? item.kind) === 'workfront')) {
    const parent = projectPosition.get(cluster.parentKey) ?? projectPosition.get(capitalKey) ?? positions[0];
    if (!parent) continue;
    const count = frontCounts.get(cluster.parentKey) ?? 0;
    const offset = FRONT_OFFSETS[count % FRONT_OFFSETS.length];
    frontCounts.set(cluster.parentKey, count + 1);
    positions.push({
      id: cluster.clusterKey, clusterKey: cluster.clusterKey,
      x: Math.max(2, Math.min(80, parent.x + offset.dx)), y: Math.max(3, Math.min(75, parent.y + offset.dy)),
      w: offset.w, h: offset.h,
    });
  }
  let outpostIndex = 0;
  for (const cluster of clusters.filter((item) => ['gateway', 'service'].includes(item.kind))) {
    const slot = OUTPOST_SLOTS[outpostIndex++ % OUTPOST_SLOTS.length];
    positions.push({ ...slot, id: cluster.clusterKey, clusterKey: cluster.clusterKey });
  }
  return positions;
}

function routeStyle(from, to) {
  const ax = from.x + from.w / 2, ay = from.y + from.h / 2, bx = to.x + to.w / 2, by = to.y + to.h / 2;
  const dx = bx - ax, dy = by - ay;
  return { left: `${ax}%`, top: `${ay}%`, width: `${Math.hypot(dx, dy)}%`, transform: `rotate(${Math.atan2(dy, dx) * 180 / Math.PI}deg)` };
}

function AgentEmblem({ identity, size = 'md', selected = false, className = '' }) {
  const [failed, setFailed] = useState(false);
  useEffect(() => setFailed(false), [identity.iconUrl]);
  const hue = identityHue(`${identity.servedModel}:${identity.endpointAlias}`);
  return <span className={`agent-emblem size-${size}${selected ? ' selected' : ''} ${className}`} style={{ '--identity-hue': hue }} title={`${identity.endpointAlias} · ${identity.servedModel}`}>
    {identity.iconUrl && !failed ? <img src={identity.iconUrl} alt="" onError={() => setFailed(true)} /> : <b>{initials(identity.endpointAlias || identity.servedModel)}</b>}
    <small>{identity.source === 'huggingface' ? 'HF' : identity.source === 'endpoint' ? 'EP' : identity.source === 'custom' ? 'UP' : ''}</small>
  </span>;
}

function Portrait({ identity, role, size = 'md' }) {
  return <span className={`senator-portrait size-${size}`} style={{ '--identity-hue': identityHue(identity.displayName) }}><UserRound aria-hidden="true" /><b>{initials(identity.displayName)}</b><small>{String(role || 'agent').slice(0, 1).toUpperCase()}</small></span>;
}

function IdentityMark({ session, identity, settings, selected = false, size = 'md', forMap = false }) {
  const mode = forMap ? settings.markerMode : settings.identityMode;
  const showPortrait = settings.theme === 'rome' && mode !== 'model';
  const showModel = settings.theme === 'atlas' || (mode !== 'person' && mode !== 'portrait');
  return <span className={`identity-mark mode-${mode}${selected ? ' selected' : ''}`}>{showPortrait && <Portrait identity={identity} role={session.role} size={size} />}{showModel && <AgentEmblem identity={identity} size={showPortrait ? 'xs' : size} selected={selected} className={showPortrait ? 'model-overlay' : ''} />}</span>;
}

function CitySymbol({ resource, slot, tier, isCapital, theme }) {
  const Icon = KIND_ICON[resource.kind] ?? Landmark;
  return <div className={`field-building kind-${resource.kind} tier-${tier} life-${resource.state}${isCapital ? ' capital-building' : ''}`} style={{ left: `${slot[0]}%`, top: `${slot[1]}%` }} title={`${resource.label} · ${stateLabel(resource.state)}`}><span className="building-shape" aria-hidden="true"><i /><i /><i /><i /><Icon /></span><b>{resource.label}</b><small>{theme === 'rome' ? 'civic work' : stateLabel(resource.state)}</small></div>;
}

function AgentMarker({ session, identity, slot, index = 0, settings, selected, related, onSelect }) {
  const pct = session.progress?.total ? Math.round((session.progress.done / session.progress.total) * 100) : 0;
  const allegiance = session.team || (session.role === 'challenger' ? 'red' : ['verifier', 'referee'].includes(session.role) ? 'blue' : 'operator');
  return <button type="button" className={`field-agent-marker team-${allegiance} state-${session.state}${selected ? ' selected' : ''}${related ? ' related' : ''}`} style={{ left: `${slot[0]}%`, top: `${slot[1]}%`, '--agent-progress': pct, '--agent-delay': `${index * 35}ms` }} onClick={(event) => { event.stopPropagation(); onSelect(session.id); }} aria-label={`${identity.displayName}, ${stateLabel(session.state)}`}><span className="unit-standard">{allegiance.slice(0, 1).toUpperCase()}</span><IdentityMark session={session} identity={identity} settings={settings} selected={selected} size="sm" forMap />{session.lastTool?.name && <span className="marker-equipment" title={`Using ${session.lastTool.name}`}><ToolIcon name={session.lastTool.name} /></span>}<span className="marker-label"><b>{identity.displayName}</b><small>{session.stateDetail || stateLabel(session.state)}</small></span></button>;
}

function ProceduralSettlement({ name, isCapital, tier }) {
  const plan = settlementPlan(name, isCapital, tier);
  const tierLabel = ['I', 'I', 'II', 'III', 'IV'][tier] ?? 'I';
  return <div
    className={`settlement-site terrain-${plan.terrain} tier-${tier}`}
    style={{ '--site-rotation': `${plan.rotation}deg`, '--site-counter-rotation': `${-plan.rotation}deg`, '--road-a': `${plan.roadA}deg`, '--road-b': `${plan.roadB}deg` }}
    data-seed={plan.seed.token}
    aria-label={`${name} settlement, generated from ${plan.seed.token}`}
  >
    <span className="settlement-ground" aria-hidden="true" />
    <span className="settlement-road road-a" aria-hidden="true" />
    <span className="settlement-road road-b" aria-hidden="true" />
    {isCapital && tier >= 3
      ? <img className="settlement-capital" src="/assets/living-rome/capital-tier-3.webp" alt="" aria-hidden="true" decoding="async" />
      : plan.modules.map((module, index) => <img
        key={module.id}
        className={`settlement-module module-${module.id}${index === 0 ? ' primary' : ' support'}`}
        src={module.src}
        alt=""
        aria-hidden="true"
        decoding="async"
        style={{ left: `${module.x}%`, top: `${module.y}%`, width: `${module.size}%`, zIndex: 9 + index, '--module-rotation': `${module.rotate}deg` }}
      />)}
    <span className="settlement-plaque" aria-hidden="true"><i>{isCapital ? '◆' : plan.seed.token.slice(0, 2)}</i><b>{name}</b><small>{tierLabel}</small></span>
  </div>;
}

function LegacyRegion({ position, cluster, agents, isCapital, selectedId, relatedIds, identities, settings, onAgent, onRegion, focused }) {
  const maturity = verifiedContribution(cluster, agents);
  const baseKind = cluster?.baseKind ?? cluster?.kind;
  const active = agents.some((agent) => !TERMINAL_STATES.has(agent.state)) || cluster?.resources?.some((item) => item.state === 'running');
  const useProceduralSettlement = baseKind === 'project' && settings.theme === 'rome';
  return <section className={`field-region${cluster ? ` settled kind-${cluster.kind}` : ' empty'}${active ? ' active' : ''}${isCapital ? ' capital' : ''}${focused ? ' focused' : ''}`} style={{ left: `${position.x}%`, top: `${position.y}%`, width: `${position.w}%`, height: `${position.h}%` }} onClick={(event) => { event.stopPropagation(); if (cluster) onRegion(position.id); }}><div className="region-influence" aria-hidden="true" />{useProceduralSettlement && <ProceduralSettlement name={cluster?.displayLabel || cluster?.label || 'Project'} isCapital={isCapital} tier={maturity.tier} />}{cluster && <header className="region-label"><b>{cluster.displayLabel}</b><small>{cluster.stateLabel || baseKind}</small></header>}{isCapital && <div className="capital-label"><i />CAPITAL · {cluster?.displayLabel}</div>}{baseKind === 'workfront' && <div className="front-standard" aria-label={cluster.stateLabel}><span className={cluster.teams?.includes('red') ? 'red on' : 'red'}>R</span><GitBranch /><span className={cluster.teams?.includes('blue') ? 'blue on' : 'blue'}>B</span>{cluster.defended && <ShieldCheck />}</div>}{cluster?.damaged && <div className="damage-signal" title={`${cluster.failureCount || 1} unresolved failure`}><i /><i /><i /></div>}{cluster && <div className="maturity-pips" aria-label={`Verified contribution ${maturity.score}%`}>{[1, 2, 3, 4].map((tier) => <i key={tier} className={tier <= maturity.tier ? 'on' : ''} />)}</div>}{(!useProceduralSettlement ? cluster?.resources ?? [] : []).slice(0, CITY_SLOTS.length).map((resource, index) => <CitySymbol key={resource.id} resource={resource} slot={CITY_SLOTS[index]} tier={maturity.tier} isCapital={isCapital && index === 0} theme={settings.theme} />)}{agents.slice(0, AGENT_SLOTS.length).map((session, index) => <AgentMarker key={session.id} session={session} identity={identities.get(session.id)} slot={AGENT_SLOTS[index]} settings={settings} selected={session.id === selectedId} related={relatedIds.has(session.id)} onSelect={onAgent} />)}</section>;
}

function Region({ position, cluster, agents, isCapital, selectedId, relatedIds, identities, settings, onAgent, onRegion, focused }) {
  const maturity = verifiedContribution(cluster, agents);
  const baseKind = cluster?.baseKind ?? cluster?.kind;
  const active = agents.some((agent) => !TERMINAL_STATES.has(agent.state)) || cluster?.resources?.some((item) => item.state === 'running');
  const useProceduralSettlement = baseKind === 'project' && settings.theme === 'rome';
  const actionLabel = baseKind === 'project'
    ? `Open ${cluster?.displayLabel ?? 'project'} City`
    : `Inspect ${cluster?.displayLabel ?? baseKind ?? 'region'}`;
  return <section
    className={`field-region${cluster ? ` settled kind-${cluster.kind}` : ' empty'}${active ? ' active' : ''}${isCapital ? ' capital' : ''}${focused ? ' focused' : ''}${agents.length > 6 ? ' agent-heavy' : ''}`}
    style={{ left: `${position.x}%`, top: `${position.y}%`, width: `${position.w}%`, height: `${position.h}%` }}
  >
    <div className="region-influence" aria-hidden="true" />
    {cluster && <button type="button" className="field-region-action" aria-label={actionLabel} onClick={(event) => { event.stopPropagation(); onRegion(position.id); }} />}
    {useProceduralSettlement && <ProceduralSettlement name={cluster?.displayLabel || cluster?.label || 'Project'} isCapital={isCapital} tier={maturity.tier} />}
    {cluster && <header className="region-label"><b>{cluster.displayLabel}</b><small>{cluster.stateLabel || baseKind}</small></header>}
    {isCapital && <div className="capital-label"><i />CAPITAL · {cluster?.displayLabel}</div>}
    {baseKind === 'workfront' && <div className="front-standard" aria-label={cluster.stateLabel}><span className={cluster.teams?.includes('red') ? 'red on' : 'red'}>R</span><GitBranch /><span className={cluster.teams?.includes('blue') ? 'blue on' : 'blue'}>B</span>{cluster.defended && <ShieldCheck />}</div>}
    {cluster?.damaged && <div className="damage-signal" title={`${cluster.failureCount || 1} unresolved failure`}><i /><i /><i /></div>}
    {cluster && <div className="maturity-pips" role="img" aria-label={`Verified contribution ${maturity.score}%`}>{[1, 2, 3, 4].map((tier) => <i key={tier} className={tier <= maturity.tier ? 'on' : ''} />)}</div>}
    {(!useProceduralSettlement ? cluster?.resources ?? [] : []).slice(0, CITY_SLOTS.length).map((resource, index) => <CitySymbol key={resource.id} resource={resource} slot={CITY_SLOTS[index]} tier={maturity.tier} isCapital={isCapital && index === 0} theme={settings.theme} />)}
    {agents.slice(0, AGENT_SLOTS.length).map((session, index) => <AgentMarker key={session.id} session={session} identity={identities.get(session.id)} slot={AGENT_SLOTS[index]} index={index} settings={settings} selected={session.id === selectedId} related={relatedIds.has(session.id)} onSelect={onAgent} />)}
  </section>;
}

function MaturityCard({ cluster, agents, onClose }) {
  if (!cluster) return null;
  const maturity = verifiedContribution(cluster, agents);
  const evidence = maturity.evidence ?? [];
  return <aside className="maturity-card" onClick={(event) => event.stopPropagation()}>
    <button type="button" onClick={onClose} aria-label="Close project details"><X /></button>
    <span>{cluster.kind}</span><h2>{cluster.label}</h2>
    <div className="maturity-score"><b>{maturity.score}%</b><small>evidence-backed maturity</small></div>
    <div className="maturity-bar" role="progressbar" aria-label="Evidence-backed maturity" aria-valuemin="0" aria-valuemax="100" aria-valuenow={maturity.score}><i style={{ width: `${maturity.score}%` }} /></div>
    <dl>
      <div><dt>Activity</dt><dd>{maturity.activity ?? 0}%</dd></div>
      <div><dt>Complete</dt><dd>{maturity.complete}%</dd></div>
      <div><dt>Verified</dt><dd>{maturity.verified}%</dd></div>
      <div><dt>Persisted</dt><dd>{maturity.persisted}%</dd></div>
      <div><dt>Reliable</dt><dd>{maturity.reliability ?? 0}%</dd></div>
    </dl>
    <p>Only criterion evidence, an independent verified verdict, and a revision-bound checkpoint grow the city. Activity is shown separately.</p>
    {evidence.length > 0 && <ul className="maturity-evidence" aria-label="Maturity evidence">{evidence.slice(-4).reverse().map((item, index) => <li key={`${item.type}:${item.seq ?? item.verdictId ?? item.checkpointId ?? index}`}><b>{item.type}</b><span>{item.criterion ?? item.verdictId ?? item.checkpointId ?? `event #${item.seq}`}</span></li>)}</ul>}
  </aside>;
}

function ToolIcon({ name }) {
  const lower = String(name).toLowerCase();
  const Icon = /git/.test(lower) ? GitBranch : /docker|container|podman/.test(lower) ? Container : /web|browser|fetch|search/.test(lower) ? Globe2 : /read|grep|glob/.test(lower) ? BookOpen : /edit|write/.test(lower) ? Wrench : TerminalSquare;
  return <Icon aria-hidden="true" />;
}

function LegacyAgentInspector({ session, identity, settings, role, agent, trace, collaborators, onClose, onSettings }) {
  if (!session) return null;
  const pct = session.progress?.total ? Math.round((session.progress.done / session.progress.total) * 100) : 0;
  const recent = trace.filter((event) => ['session.message', 'session.tool_use', 'session.tool_result'].includes(event.kind)).slice(-4).reverse();
  const tools = agent?.tools_allow ?? role?.tools_allow ?? [];
  return <aside className="field-side-panel agent-panel" onClick={(event) => event.stopPropagation()}><header><div className="inspector-identity"><IdentityMark session={session} identity={identity} settings={settings} selected size="lg" /><div><span>{settings.theme === 'rome' ? 'SENATOR' : 'AGENT'}</span><h2>{identity.displayName}</h2></div></div><div><button type="button" onClick={onSettings} aria-label="Open settings"><Settings /></button><button type="button" onClick={onClose} aria-label="Close agent"><X /></button></div></header><section><label>Objective</label><p>{session.target?.label ?? session.stateDetail ?? 'Awaiting a specific objective.'}</p></section><section><label>Progress <b>{pct}%</b></label><div className="inspector-progress"><i style={{ width: `${pct}%` }} /></div></section><section className="model-line"><label>Model / endpoint</label><div><AgentEmblem identity={identity} size="sm" /><p><b>{identity.endpointAlias}</b><span>{identity.servedModel}</span><small>{identity.hfRepo || identity.source}</small></p></div></section>{identity.distillationRole && <section className="teacher-line"><label>Distillation source</label><p><b>{identity.endpointAlias} is {identity.servedModel}</b><span>Teacher traces → {identity.studentTarget}</span></p></section>}<section><label>{settings.theme === 'rome' ? 'Authority' : 'Tools'}</label><div className="tool-loadout">{tools.length ? tools.map((tool) => <span key={tool}><ToolIcon name={tool} />{tool}</span>) : <em>No role tool allowlist</em>}</div></section><section><label>Working with</label><div className="collaboration-row">{collaborators.length ? collaborators.map((item) => <span key={item.id}><i className={`state-${item.state}`} />{item.name}</span>) : <em>No active links</em>}</div></section><section className="activity-ledger"><label>Current activity</label>{recent.length ? recent.map((event) => <div key={event.id ?? event.seq}><time>{new Date(event.ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</time><p>{event.data?.summary ?? event.data?.text ?? event.kind.replaceAll('.', ' / ')}</p></div>) : <em>No activity reported.</em>}</section></aside>;
}

function conversationText(event) {
  return event.data?.text ?? event.data?.content ?? event.data?.summary ?? '';
}

function AgentInspector({ session, identity, settings, role, agent, trace, collaborators, onClose, onSettings }) {
  if (!session) return null;
  const pct = session.progress?.total ? Math.round((session.progress.done / session.progress.total) * 100) : 0;
  const spawned = trace.find((event) => event.kind === 'session.spawned')?.data ?? {};
  const prompt = spawned.initialOrders ?? spawned.systemPrompt ?? session.target?.label ?? session.stateDetail;
  const messages = trace.filter((event) => event.kind === 'session.message' && conversationText(event)).slice(-8);
  const recent = trace.filter((event) => ['session.tool_use', 'session.tool_result'].includes(event.kind)).slice(-4).reverse();
  const tools = agent?.tools_allow ?? role?.tools_allow ?? [];
  return <aside className="field-side-panel agent-panel" onClick={(event) => event.stopPropagation()}>
    <header><div className="inspector-identity"><IdentityMark session={session} identity={identity} settings={settings} selected size="lg" /><div><span>{settings.theme === 'rome' ? 'SENATOR' : 'AGENT'}</span><h2>{identity.displayName}</h2></div></div><div><button type="button" onClick={onSettings} aria-label="Open settings"><Settings /></button><button type="button" onClick={onClose} aria-label="Close agent"><X /></button></div></header>
    <section><label>Objective</label><p>{session.target?.label ?? session.stateDetail ?? 'Awaiting a specific objective.'}</p></section>
    <section><label>Progress <b>{pct}%</b></label><div className="inspector-progress" role="progressbar" aria-label={`${identity.displayName} progress`} aria-valuemin="0" aria-valuemax="100" aria-valuenow={pct}><i style={{ width: `${pct}%` }} /></div></section>
    <section className="model-line"><label>Model / endpoint</label><div><AgentEmblem identity={identity} size="sm" /><p><b>{identity.endpointAlias}</b><span>{identity.servedModel}</span><small>{identity.hfRepo || identity.source}</small></p></div></section>
    <section><label>{settings.theme === 'rome' ? 'Authority' : 'Tools'}</label><div className="tool-loadout">{tools.length ? tools.map((tool) => <span key={tool}><ToolIcon name={tool} />{tool}</span>) : <em>No role tool allowlist</em>}</div></section>
    <section><label>Working with</label><div className="collaboration-row">{collaborators.length ? collaborators.map((item) => <span key={item.id}><i className={`state-${item.state}`} />{item.name}</span>) : <em>No active links</em>}</div></section>
    <section className="agent-prompt"><label>Prompt</label><p>{prompt || 'No prompt recorded for this session.'}</p>{spawned.systemPrompt && spawned.systemPrompt !== prompt && <details><summary>System instructions</summary><p>{spawned.systemPrompt}</p></details>}</section>
    <section className="conversation-ledger"><label>Conversation</label>{messages.length ? messages.map((event) => <div key={event.id ?? event.seq}><b>{event.data?.role ?? 'agent'}</b><p>{conversationText(event)}</p></div>) : <em>No conversation reported yet.</em>}</section>
    <section className="activity-ledger"><label>Current activity</label>{recent.length ? recent.map((event) => <div key={event.id ?? event.seq}><time>{new Date(event.ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</time><p>{event.data?.summary ?? event.data?.text ?? event.kind.replaceAll('.', ' / ')}</p></div>) : <em>No tool activity reported.</em>}</section>
  </aside>;
}

function ChoiceGroup({ value, options, onChange }) { return <div className="settings-choice">{options.map((option) => <button type="button" key={option.value} className={value === option.value ? 'on' : ''} onClick={() => onChange(option.value)}>{option.label}</button>)}</div>; }

function FieldSettings({ settings, setSettings, selected, config, onClose }) {
  const [tab, setTab] = useState('identity'), [saving, setSaving] = useState(false), [message, setMessage] = useState('');
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
  async function save() {
    setSaving(true); setMessage('');
    try {
      const persisted = saveFieldSettings(settings); setSettings(persisted);
      if (configuredAgent) await api.updateAgent({ agentId: configuredAgent.id, name: override.displayName || configuredAgent.name, toolsAllow: selectedTools });
      setMessage(configuredAgent ? 'Saved. Tool authority applies to future deployments.' : 'Presentation saved on this device.');
    } catch (error) { setMessage(error.message); } finally { setSaving(false); }
  }
  return <aside className="field-side-panel settings-panel" onClick={(event) => event.stopPropagation()}><header><div><span>FIELD</span><h2>Settings</h2></div><button type="button" onClick={onClose} aria-label="Close settings"><X /></button></header><nav>{['world', 'identity', 'models'].map((item) => <button type="button" key={item} className={tab === item ? 'on' : ''} onClick={() => setTab(item)}>{item}</button>)}</nav><div className="settings-scroll">
    {tab === 'world' && <><section><label>Theme</label><p className="settings-help">Atlas is the parallel agent board. Rome is the operations map.</p><ChoiceGroup value={settings.theme} options={[{ value: 'atlas', label: 'Atlas' }, { value: 'rome', label: 'Rome' }]} onChange={(theme) => patchSettings({ theme })} /></section><section><label>World density</label><ChoiceGroup value={settings.density} options={[{ value: 'quiet', label: 'Quiet' }, { value: 'balanced', label: 'Balanced' }, { value: 'dense', label: 'Dense' }]} onChange={(density) => patchSettings({ density })} /></section><section className="settings-toggle"><div><label>World motion</label><p>Animate active routes and status pulses.</p></div><button type="button" className={settings.motion ? 'on' : ''} onClick={() => patchSettings({ motion: !settings.motion })}><i /></button></section></>}
    {tab === 'identity' && <><section><label>Agent identity</label><ChoiceGroup value={settings.identityMode} options={[{ value: 'portrait', label: 'Portrait' }, { value: 'model', label: 'Model' }, { value: 'both', label: 'Both' }]} onChange={(identityMode) => patchSettings({ identityMode })} /></section><section><label>Map markers</label><ChoiceGroup value={settings.markerMode} options={[{ value: 'person', label: 'Person' }, { value: 'model', label: 'Model' }, { value: 'both', label: 'Both' }]} onChange={(markerMode) => patchSettings({ markerMode })} /></section><section><label>Emblem source</label><select value={settings.emblemSource} onChange={(event) => patchSettings({ emblemSource: event.target.value })}><option value="auto">Auto</option><option value="huggingface">Hugging Face</option><option value="endpoint">Endpoint</option><option value="upload">Upload</option><option value="initials">Initials</option></select><small>HF avatar › endpoint › upload › initials</small></section>{selected ? <section className="agent-settings"><label>Selected agent</label><h3>{override.displayName || selected.name}</h3><div className="settings-field"><span>Display name</span><input value={override.displayName ?? selected.name ?? ''} onChange={(event) => patchAgent({ displayName: event.target.value })} /></div><div className="settings-field"><span>Endpoint alias</span><input value={override.endpointAlias ?? ''} placeholder="Use endpoint name" onChange={(event) => patchAgent({ endpointAlias: event.target.value })} /></div><div className="settings-field"><span>HF repository</span><input value={override.hfRepo ?? ''} placeholder="owner/model" onChange={(event) => patchAgent({ hfRepo: event.target.value })} /></div><div className="settings-field"><span>Icon URL</span><input value={override.iconUrl ?? ''} placeholder="https://…" onChange={(event) => patchAgent({ iconUrl: event.target.value, iconDataUrl: '' })} /></div><label className="upload-control"><Upload />Upload icon<input type="file" accept="image/*" onChange={upload} /></label></section> : <section><p>Select an agent to edit its persistent name and emblem.</p></section>}{selected && <section><label>Tool authority</label><p className="settings-help">A configured agent may receive a subset of its role&apos;s real allowlist. Changes apply on its next deployment.</p><div className="authority-grid">{roleTools.map((tool) => <button type="button" key={tool} className={selectedTools.includes(tool) ? 'on' : ''} onClick={() => toggleTool(tool)}><ToolIcon name={tool} />{tool}<Check /></button>)}</div>{!configuredAgent && <small>Simulation and ad-hoc sessions cannot persist tool changes.</small>}</section>}</>}
    {tab === 'models' && <section className="model-registry-card"><label>Distillation teacher</label><h3>OX Alpha is GLM 5.3 Flash</h3><p>OX Alpha is the endpoint alias. It serves GLM 5.3 Flash and supplies teacher traces for the Ornith student.</p><div className="settings-field"><span>Teacher endpoint</span><input value={oxAlpha.endpointAlias} onChange={(event) => patchOx({ endpointAlias: event.target.value })} /></div><div className="settings-field"><span>Model served</span><input value={oxAlpha.servedModel} onChange={(event) => patchOx({ servedModel: event.target.value })} /></div><div className="settings-field"><span>HF repository</span><input value={oxAlpha.hfRepo} onChange={(event) => patchOx({ hfRepo: event.target.value })} /></div><div className="settings-field"><span>Student target</span><input value={oxAlpha.studentTarget} onChange={(event) => patchOx({ studentTarget: event.target.value })} /></div><div className="settings-toggle"><div><label>Collect teacher traces</label><p>Mark OX Alpha traces as distillation source data.</p></div><button type="button" className={oxAlpha.collectTeacherTraces ? 'on' : ''} onClick={() => patchOx({ collectTeacherTraces: !oxAlpha.collectTeacherTraces })}><i /></button></div></section>}
  </div><footer><button type="button" className="restore" onClick={() => setSettings(normalizeFieldSettings(DEFAULT_FIELD_SETTINGS))}>Restore defaults</button><span>{message}</span><button type="button" className="save" disabled={saving} onClick={save}>{saving ? 'Saving…' : 'Save settings'}</button></footer></aside>;
}

function RehearsalPanel({ simulation, workspaces, capitalId, theme, busy, error, onRun, onStop, onClose }) {
  useModalFocus(onClose, 'rehearsal-title');
  const [workspaceId, setWorkspaceId] = useState(capitalId || workspaces[0]?.id || '');
  const [speed, setSpeed] = useState(2);
  const active = simulation.active;
  const scenario = simulation.scenarios?.[0];
  return <div className="rehearsal-veil" onClick={(event) => event.stopPropagation()}><section className="rehearsal-panel" role="dialog" aria-modal="true" aria-labelledby="rehearsal-title"><header><div><span>{theme === 'rome' ? 'FIELD EXERCISE' : 'WORLD REHEARSAL'}</span><h2 id="rehearsal-title">Watch a codebase come alive</h2></div><button type="button" onClick={onClose} aria-label="Close rehearsal"><X /></button></header>{active ? <div className="rehearsal-live-state"><span className="live-orbit"><i /><Sparkles /></span><h3>{theme === 'rome' ? 'The cohort is deployed' : 'Agents are operating'}</h3><p>Twelve synthetic agents are surveying, building, challenging, and verifying the selected codebase. Every movement uses the production event schema inside an isolated rehearsal projection.</p><dl><div><dt>Codebase</dt><dd>{workspaces.find((item) => item.id === active.workspaceId)?.name ?? active.workspaceId}</dd></div><div><dt>Pace</dt><dd>{active.speed}×</dd></div><div><dt>Units</dt><dd>{active.sessionIds?.length ?? 0}</dd></div></dl><button type="button" className="stop-rehearsal" disabled={busy} onClick={onStop}><Square />End rehearsal</button></div> : <><div className="rehearsal-intro"><span className="rehearsal-sigil"><Sparkles /></span><div><h3>{scenario?.name ?? 'Long-horizon operations cycle'}</h3><p>{scenario?.description ?? 'Agents fan out across a codebase, coordinate through an incident, verify the work, and leave durable growth behind.'}</p></div></div><label className="rehearsal-field"><span>Codebase / capital</span><select value={workspaceId} onChange={(event) => setWorkspaceId(event.target.value)}>{workspaces.map((workspace) => <option key={workspace.id} value={workspace.id}>{workspace.name}</option>)}</select></label><div className="rehearsal-field"><span>Pace</span><div className="pace-choice">{[1, 2, 4].map((value) => <button type="button" key={value} className={speed === value ? 'on' : ''} onClick={() => setSpeed(value)}>{value}×<small>{value === 1 ? 'observe' : value === 2 ? 'lively' : 'rapid'}</small></button>)}</div></div><div className="rehearsal-safety"><ShieldCheck /><p><b>Projection only.</b> No provider calls, commands, file writes, or model cost. Synthetic events use the production event schema but remain outside production state, metrics, and traces.</p></div>{error && <p className="rehearsal-error">{error}</p>}<button type="button" className="begin-rehearsal" disabled={busy || !workspaceId || !simulation.enabled} onClick={() => onRun(workspaceId, speed)}><Play />{busy ? 'Mustering…' : theme === 'rome' ? 'Muster the cohort' : 'Begin rehearsal'}</button>{!simulation.enabled && <small className="rehearsal-disabled">Rehearsals are disabled on this Field server.</small>}</>}</section></div>;
}

function LegacyAgentRoster({ sessions, identities, settings, selectedId, onSelect }) {
  return <section className={`agent-roster${sessions.length ? '' : ' empty'}`} onClick={(event) => event.stopPropagation()}><header><i /><span>{settings.theme === 'rome' ? 'SENATE' : 'AGENTS'}</span><i /></header><div>{sessions.length ? sessions.slice(0, 8).map((session) => { const identity = identities.get(session.id); return <button type="button" key={session.id} className={`${session.id === selectedId ? 'selected' : ''} state-${session.state}`} onClick={() => onSelect(session.id)}><IdentityMark session={session} identity={identity} settings={settings} selected={session.id === selectedId} size="lg" /><b>{identity.displayName}</b><small>{identity.endpointAlias}</small><span>{[0, 1, 2, 3].map((n) => <i key={n} className={n < Math.max(1, Math.round(((session.progress?.done ?? 1) / Math.max(1, session.progress?.total ?? 4)) * 4)) ? 'on' : ''} />)}</span></button>; }) : <p>Quiet · muster when work begins</p>}</div></section>;
}

function AgentRoster({ sessions, identities, settings, selectedId, onSelect }) {
  return <section className={`agent-roster${sessions.length ? '' : ' empty'}`} onClick={(event) => event.stopPropagation()} aria-label={settings.theme === 'rome' ? 'Senate roster' : 'Agent roster'}>
    <header><i /><span>{settings.theme === 'rome' ? 'SENATE' : 'AGENTS'}{sessions.length ? ` · ${sessions.length}` : ''}</span><i /></header>
    <div>{sessions.length ? sessions.map((session) => {
      const identity = identities.get(session.id);
      const pct = session.progress?.total ? Math.round(((session.progress?.done ?? 0) / session.progress.total) * 100) : 0;
      return <button
        type="button"
        key={session.id}
        className={`${session.id === selectedId ? 'selected' : ''} state-${session.state}`}
        onClick={() => onSelect(session.id)}
        aria-pressed={session.id === selectedId}
        aria-label={`Open ${identity.displayName} in the Senate, ${stateLabel(session.state)}, ${pct}% complete`}
      >
        <IdentityMark session={session} identity={identity} settings={settings} selected={session.id === selectedId} size="lg" />
        <b>{identity.displayName}</b><small>{identity.endpointAlias}</small>
        <span role="img" aria-label={`${pct}% complete`}>{[0, 1, 2, 3].map((n) => <i key={n} className={n < Math.max(1, Math.round((pct / 100) * 4)) ? 'on' : ''} />)}</span>
      </button>;
    }) : <p>Quiet · muster when work begins</p>}</div>
  </section>;
}

function CapitalChooser({ workspaces, current, onChoose, onClose, pending, error, theme }) {
  useModalFocus(onClose, 'capital-title');
  return <div className="capital-veil" onClick={(event) => event.stopPropagation()}><section className="capital-choice" role="dialog" aria-modal="true" aria-labelledby="capital-title"><span className="choice-kicker">{theme === 'rome' ? 'FOUND THE CIVILIZATION' : 'ANCHOR THE WORLD'}</span><h2 id="capital-title">Choose the capital project</h2><p>The capital is the visual anchor and default coordination point. It does not move files, services, or permissions.</p><div className="capital-projects">{workspaces.map((workspace) => <button type="button" key={workspace.id} disabled={pending} onClick={() => onChoose(workspace.id)}><span className="project-sigil">{workspace.name.slice(0, 1).toUpperCase()}</span><span><b>{workspace.name}</b><small>{workspace.path}</small></span><i>{workspace.id === current ? 'CURRENT' : 'CHOOSE'}</i></button>)}</div>{error && <p className="capital-error">{error}</p>}{current && <button type="button" className="choice-cancel" onClick={onClose}>Keep current capital</button>}</section></div>;
}

export default function TheaterMode() {
  const st = useField();
  const selectedId = st.activeSessionId;
  const [settings, setSettings] = useState(loadFieldSettings), [selectedRegion, setSelectedRegion] = useState(null), [settingsOpen, setSettingsOpen] = useState(false), [choosingCapital, setChoosingCapital] = useState(false), [pendingCapital, setPendingCapital] = useState(false), [capitalError, setCapitalError] = useState(''), [trace, setTrace] = useState([]);
  const [rehearsalOpen, setRehearsalOpen] = useState(false), [rehearsalBusy, setRehearsalBusy] = useState(false), [rehearsalError, setRehearsalError] = useState('');
  const [simulation, setSimulation] = useState({ enabled: false, scenarios: [], active: null });
  const reconcileRef = useRef('');
  const view = useMemo(() => rehearsalSnapshot(st.snap, simulation.active), [st.snap, simulation.active]);
  const world = st.snap.world ?? { capitalWorkspaceId: null, assignments: {}, revision: 0 };
  const workspaces = st.snap.workspaces.filter((item) => item.mounted);
  useEffect(() => { api.simulations().then(setSimulation).catch(() => setSimulation({ enabled: false, scenarios: [], active: null })); }, []);
  const latestEvent = st.events.at(-1);
  useEffect(() => {
    if (!latestEvent?.kind?.startsWith('simulation.')) return;
    api.simulations().then(setSimulation).catch(() => {});
  }, [latestEvent?.seq]);
  const clusters = useMemo(() => infrastructureClusters(view, world.capitalWorkspaceId), [view, world.capitalWorkspaceId]);
  const clusterByKey = useMemo(() => new Map(clusters.map((item) => [item.clusterKey, item])), [clusters]);
  const positions = useMemo(() => livingWorldPositions(clusters, world.capitalWorkspaceId), [clusters, world.capitalWorkspaceId]);
  const positionByKey = useMemo(() => new Map(positions.map((item) => [item.clusterKey, item])), [positions]);
  const clusterByRegion = new Map(positions.map((position) => [position.id, clusterByKey.get(position.clusterKey)]));
  useEffect(() => {
    if (!world.capitalWorkspaceId || !clusters.length) return;
    const signature = `${world.capitalWorkspaceId}:${clusters.map((item) => item.clusterKey).sort().join('|')}`;
    const missing = clusters.some((item) => !world.assignments?.[item.clusterKey]);
    const currentKeys = new Set(clusters.map((item) => item.clusterKey));
    const stale = Object.keys(world.assignments ?? {}).some((key) => !currentKeys.has(key));
    const capitalWrong = world.assignments?.[`workspace:${world.capitalWorkspaceId}`]?.territoryId !== 'italia';
    if ((!missing && !stale && !capitalWrong) || reconcileRef.current === signature) return;
    reconcileRef.current = signature;
    api.reconcileWorld(clusters.map(({ clusterKey, label, kind, workspaceId }) => ({ clusterKey, label, kind, workspaceId }))).catch(() => { reconcileRef.current = ''; });
  }, [clusters, world.assignments, world.capitalWorkspaceId]);
  useEffect(() => { saveFieldSettings(settings); }, [settings]);
  const sessions = useMemo(() => view.sessions.filter((session) => !TERMINAL_STATES.has(session.state)).sort((a, b) => (b.startedAt ?? 0) - (a.startedAt ?? 0)).slice(0, 40), [view.sessions]);
  const selected = sessions.find((session) => session.id === selectedId) ?? null;
  useEffect(() => { let alive = true; if (!selectedId) { setTrace([]); return () => { alive = false; }; } api.trace(selectedId, 0, 2000).then((result) => { if (alive) setTrace(result.events ?? []); }).catch(() => { if (alive) setTrace([]); }); return () => { alive = false; }; }, [selectedId, selected?.messageCount, selected?.toolCount, selected?.state, selected?.progress?.done]);
  const endpoints = st.snap.endpoints?.length ? st.snap.endpoints : st.config?.endpoints ?? [];
  const identities = useMemo(() => new Map(sessions.map((session) => [session.id, identityFor(session, endpoints, settings)])), [sessions, endpoints, settings]);
  const sessionById = new Map(sessions.map((session) => [session.id, session]));
  const collaboratorIds = new Set();
  for (const edge of view.graph?.edges ?? []) { if (edge.type !== 'communicates_with') continue; const from = String(edge.from ?? '').replace(/^agent:/, ''), to = String(edge.to ?? '').replace(/^agent:/, ''); if (from === selectedId) collaboratorIds.add(to); if (to === selectedId) collaboratorIds.add(from); }
  const collaborators = [...collaboratorIds].map((id) => sessionById.get(id)).filter(Boolean);
  const agentsByRegion = new Map();
  for (const session of sessions) { const frontKey = frontKeyFor(session); const workspaceKey = session.workspaceId ? `workspace:${session.workspaceId}` : `workspace:${world.capitalWorkspaceId}`; const hasDeparted = !['spawning', 'ready'].includes(session.state); const regionId = hasDeparted && positionByKey.has(frontKey) ? frontKey : workspaceKey; if (!agentsByRegion.has(regionId)) agentsByRegion.set(regionId, []); agentsByRegion.get(regionId).push(session); }
  const capitalPosition = positionByKey.get(`workspace:${world.capitalWorkspaceId}`) ?? positions[0];
  const routeLimit = settings.density === 'quiet' ? 6 : settings.density === 'dense' ? 16 : 11;
  const routes = clusters.map((cluster) => ({ cluster, from: positionByKey.get(cluster.parentKey), to: positionByKey.get(cluster.clusterKey) })).filter((item) => item.from && item.to).slice(0, routeLimit);
  const active = sessions.filter((item) => !TERMINAL_STATES.has(item.state)), attention = active.filter((item) => ATTENTION_STATES.has(item.state));
  const selectedCluster = selectedRegion ? clusterByRegion.get(selectedRegion) : null, selectedRegionAgents = selectedRegion ? agentsByRegion.get(selectedRegion) ?? [] : [];
  const selectedRole = selected ? st.config?.roles?.find((item) => item.id === selected.role) : null;
  const selectedAgent = selected ? st.config?.agents?.find((item) => item.id === selected.agentId) : null;
  async function chooseCapital(workspaceId) { setPendingCapital(true); setCapitalError(''); try { await api.selectCapital(workspaceId); await api.reconcileWorld(clusters.map(({ clusterKey, label, kind, workspaceId: ws }) => ({ clusterKey, label, kind, workspaceId: ws }))); reconcileRef.current = ''; setChoosingCapital(false); } catch (error) { setCapitalError(error.message); } finally { setPendingCapital(false); } }
  async function runRehearsal(workspaceId, speed) { setRehearsalBusy(true); setRehearsalError(''); try { if (workspaceId !== world.capitalWorkspaceId) { await api.selectCapital(workspaceId); await api.reconcileWorld(clusters.map(({ clusterKey, label, kind, workspaceId: ws }) => ({ clusterKey, label, kind, workspaceId: ws }))); reconcileRef.current = ''; } const activeRun = await api.runSimulation('operations-cycle', speed, workspaceId); setSimulation((current) => ({ ...current, active: activeRun })); setRehearsalOpen(false); } catch (error) { setRehearsalError(error.message); } finally { setRehearsalBusy(false); } }
  async function stopRehearsal() { setRehearsalBusy(true); setRehearsalError(''); try { await api.stopSimulation(); const info = await api.simulations(); setSimulation(info); } catch (error) { setRehearsalError(error.message); } finally { setRehearsalBusy(false); } }
  const themeLabel = settings.theme === 'rome' ? 'ROME' : 'ATLAS';
  if (settings.theme === 'atlas') {
    return <AtlasMode settings={settings} setSettings={setSettings} />;
  }
  return <div className={`field-world-shell theme-${settings.theme} density-${settings.density}${settings.motion ? ' motion-on' : ' motion-off'}${selected || settingsOpen ? ' panel-open' : ''}${selected ? ' agent-selected' : ''}${sessions.length ? '' : ' senate-empty'}${simulation.active ? ' rehearsal-active' : ''}`} onClick={() => { clearActiveAgent(); setSelectedRegion(null); }}>
    {simulation.active && <div className="rehearsal-watermark" role="status"><Sparkles /><span><b>Rehearsal · synthetic events</b>Production state and metrics are isolated</span></div>}
    <header className="field-world-header" onClick={(event) => event.stopPropagation()}>
      <div><span>{simulation.active ? 'SYNTHETIC REHEARSAL' : settings.theme === 'rome' ? 'IMPERIUM OPERIS' : 'LIVING OPERATIONS'}</span><h1>{world.capitalWorkspaceId ? workspaces.find((item) => item.id === world.capitalWorkspaceId)?.name ?? 'Capital' : 'Choose a capital project'}</h1></div>
      <div className="field-status" aria-live="polite" aria-atomic="true"><span><b>{active.length}</b> active</span><span className={attention.length ? 'attention' : ''}><b>{attention.length}</b> attention</span><span><b>{clusters.filter((item) => (item.baseKind ?? item.kind) === 'workfront').length}</b> fronts</span></div>
      <div className="field-quick-settings"><button type="button" onClick={() => setSettings((current) => ({ ...current, theme: current.theme === 'rome' ? 'atlas' : 'rome' }))}>Theme <b>{themeLabel}</b></button><button type="button" onClick={() => setSettings((current) => ({ ...current, identityMode: current.identityMode === 'both' ? 'model' : current.identityMode === 'model' ? 'portrait' : 'both' }))}>Identity <b>{settings.identityMode}</b></button><button type="button" className={simulation.active ? 'rehearsal-live' : ''} onClick={() => { setRehearsalOpen(true); setRehearsalError(''); }}><Sparkles />{settings.theme === 'rome' ? 'Muster' : 'Rehearse'}{simulation.active && <b>LIVE</b>}</button><button type="button" onClick={() => setChoosingCapital(true)}>Capital</button><button type="button" className="settings-gear" onClick={() => setSettingsOpen(true)} aria-label="Open Field settings"><Settings /></button></div>
    </header>
    <main className="field-world-canvas living-world">
      <div className="world-map-base" aria-hidden="true" /><div className="world-contours" aria-hidden="true" />
      {settings.theme === 'rome' && <><div className="world-sea-label west">ORBIS OPERIS</div><div className="world-sea-label center">VIAE ET OPERA</div><div className="world-sea-label east">FINES ACTIVI</div></>}
      <div className="world-routes">{routes.map(({ cluster, from, to }) => <i key={cluster.clusterKey} className={`route-${cluster.baseKind ?? cluster.kind}`} style={routeStyle(from, to)} />)}</div>
      {positions.map((position) => <Region key={position.id} position={position} cluster={clusterByRegion.get(position.id)} agents={agentsByRegion.get(position.id) ?? []} isCapital={position.clusterKey === `workspace:${world.capitalWorkspaceId}`} selectedId={selectedId} relatedIds={collaboratorIds} identities={identities} settings={settings} onAgent={(id) => { selectAgent(id); setSettingsOpen(false); }} onRegion={(regionId) => { const cluster = clusterByRegion.get(regionId); if ((cluster?.baseKind ?? cluster?.kind) === 'project') openCity(cluster.workspaceId); else setSelectedRegion(regionId); }} focused={selectedRegion === position.id} />)}
      <MaturityCard cluster={selectedCluster} agents={selectedRegionAgents} position={positionByKey.get(selectedRegion)} onClose={() => setSelectedRegion(null)} />
    </main>
    <AgentRoster sessions={sessions} identities={identities} settings={settings} selectedId={selectedId} onSelect={(id) => { openSenate(id); setSettingsOpen(false); }} />
    {!settingsOpen && <AgentInspector session={selected} identity={selected ? identities.get(selected.id) : null} settings={settings} role={selectedRole} agent={selectedAgent} trace={trace} collaborators={collaborators} onClose={clearActiveAgent} onSettings={() => setSettingsOpen(true)} />}
    {trace.length >= 2000 && <div className="rehearsal-watermark" role="status">Showing the first 2,000 events. Open Traces to load the rest.</div>}
    {settingsOpen && <FieldSettings settings={settings} setSettings={setSettings} selected={selected} config={st.config} onClose={() => setSettingsOpen(false)} />}
    {rehearsalOpen && <RehearsalPanel simulation={simulation} workspaces={workspaces} capitalId={world.capitalWorkspaceId} theme={settings.theme} busy={rehearsalBusy} error={rehearsalError} onRun={runRehearsal} onStop={stopRehearsal} onClose={() => setRehearsalOpen(false)} />}
    {(!world.capitalWorkspaceId || choosingCapital) && <CapitalChooser workspaces={workspaces} current={world.capitalWorkspaceId} onChoose={chooseCapital} onClose={() => setChoosingCapital(false)} pending={pendingCapital} error={capitalError} theme={settings.theme} />}
  </div>;
}
