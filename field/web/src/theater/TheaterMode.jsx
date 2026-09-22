/* Rome — the operations map, folder by folder.

   The Board is every conversation at once, ordered by time. This is territory, ordered by
   space, and it answers one question: where is the work happening? Each mounted project is
   a city; the folders inside it are the districts around it, sized by real weight — how
   many files they hold and how much changed there — and toned by how recently anyone
   touched them, so the untouched corners stay visible. An agent stands in the folder that
   holds the file it is on, and moves as that changes; three markers in one district and
   none in the next is the thing this screen can show that a list cannot.

   Clicking a district opens the folder, and the folder is its conversations; expand that
   folder and it is also its files, its diffs and a shell in it. Along the bottom is time:
   drag it and the same map shows an earlier moment. Over it, when you ask for them, are
   the plans, drawn across the folders they cover.

   What used to be separate screens for each of those — the canvas Map, Project, History
   and Plans — each showed one slice of this map, so all four are gone. So are the city
   command hub, the senate roster, the region maturity card and the agent inspector. */

import { useEffect, useMemo, useRef, useState } from 'react';
import {
  BookOpen, Box, Container, Database, GitBranch, Globe2,
  Landmark, Network, Plus, RadioTower, ShieldCheck, Wrench,
} from 'lucide-react';
import { api } from '../net/client.js';
import { clearActiveAgent, romeRequestHandled, selectAgent, setChrome, useField } from '../state/store.js';
import { useModalFocus } from '../ui/useModalFocus.js';
import ToolIcon from '../ui/ToolIcon.jsx';
import PowerSources from '../setup/PowerSources.jsx';
import ContextMenu from '../hud/ContextMenu.jsx';
import PlansOverlay from '../campaigns/PlansOverlay.jsx';
import FieldSettings from './FieldSettings.jsx';
import FolderDetail from './FolderDetail.jsx';
import IslandPlate from './IslandPlate.jsx';
import TimeControl from './TimeControl.jsx';
import { sessionsInScope, TERMINAL_STATES } from '../ui/Conversation.jsx';
import { plainActivity } from '../ui/WorkCard.jsx';
import { identityFor, identityHue, initials, verifiedContribution } from './fieldPreferences.js';
import useFieldSettings from './useFieldSettings.js';
import {
  crumbsFor, districtsAt, folderRollup, markerRing, normalizeDir, parentOf,
  staleness, standingPlaces, weightLabel,
} from './districts.js';
import { frameFor, islandFor } from './island.js';

const ATTENTION_STATES = new Set(['blocked', 'error', 'waiting_permission']);
// A conversation that ended stays in a folder this long, as on the Board.
const RECENTLY_FINISHED_MS = 30 * 60 * 1000;
const MAX_DISTRICTS = 7;
const MAX_MARKERS = 8;

/* The settlement sits on the island the generator drew, at the quiet inland spot it
   picked; its footprint is a fraction of that island's frame so a crowded archipelago
   draws smaller towns. */
const settlementBox = (extent, site) => {
  const w = Math.min(18, extent.rx * 0.6);
  const h = Math.min(23, extent.ry * 0.8);
  return { x: site.x - w / 2, y: site.y - h / 2, w, h };
};
const LANDMARK_KIND = {
  model: 'model', runtime: 'runtime', knowledge: 'archive', verification: 'verification',
  interface: 'interface', storage: 'storage', core: 'core', general: 'project',
};
const KIND_ICON = {
  project: Landmark, workfront: GitBranch, gateway: RadioTower, service: Container,
  model: BookOpen, runtime: Wrench, archive: Database, verification: ShieldCheck,
  interface: Globe2, storage: Box, core: Network,
};
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

/* A folder's name still says what kind of place it is; the icon follows the name, as the
   landmarks always did. */
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

function stateLabel(value = 'unknown') { return String(value).replaceAll('_', ' '); }

const treeKey = (workspaceId, dir) => JSON.stringify([workspaceId, dir]);
const entriesOf = (trees, workspaceId, dir) => {
  const value = trees[treeKey(workspaceId, dir)];
  return Array.isArray(value) ? value : null;
};

function spokeStyle(from, to) {
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  return {
    left: `${from.x}%`, top: `${from.y}%`,
    width: `${Math.hypot(dx, dy)}%`,
    transform: `rotate(${(Math.atan2(dy, dx) * 180) / Math.PI}deg)`,
  };
}

/* One unit, one mark.

   A marker used to stack three glyphs that said the same thing: a portrait with the
   agent's two-letter initials, a role letter over it, and an emblem with the model's
   two-letter initials beside that — a two-letter code, a letter and the code again on a
   28px disc. It is one disc now: a single initial, coloured from whichever identity the
   operator chose to see, and a ring that carries the one fact the map is for, which is
   what state the unit is in. Everything else about it is one tap away in the folder. */
function UnitMark({ session, identity, settings, selected = false, size = 'md' }) {
  const [failed, setFailed] = useState(false);
  const model = settings.identity === 'model';
  const name = model ? (identity.endpointAlias || identity.servedModel) : identity.displayName;
  const icon = settings.identity === 'person' ? null : identity.iconUrl;
  useEffect(() => setFailed(false), [icon]);
  return <span
    className={`unit-mark size-${size} state-${session.state}${selected ? ' selected' : ''}`}
    style={{ '--identity-hue': identityHue(model ? `${identity.servedModel}:${identity.endpointAlias}` : identity.displayName) }}
  >
    {icon && !failed
      ? <img src={icon} alt="" onError={() => setFailed(true)} />
      : <b>{initials(name).slice(0, 1)}</b>}
  </span>;
}

function AgentMarker({ session, identity, x, y, index = 0, settings, selected, onSelect }) {
  const pct = session.progress?.total ? Math.round((session.progress.done / session.progress.total) * 100) : 0;
  return <button
    type="button"
    className={`field-agent-marker state-${session.state}${selected ? ' selected' : ''}`}
    style={{ left: `${x}%`, top: `${y}%`, '--agent-progress': pct, '--agent-delay': `${index * 35}ms` }}
    onClick={(event) => { event.stopPropagation(); onSelect(session); }}
    aria-label={`${identity.displayName}, ${stateLabel(session.state)}${session.focusPath ? `, on ${session.focusPath}` : ''}`}
  >
    <UnitMark session={session} identity={identity} settings={settings} selected={selected} size="sm" />
    {session.lastTool?.name && <span className="marker-equipment" title={`Using ${session.lastTool.name}`}><ToolIcon name={session.lastTool.name} /></span>}
    <span className="marker-label">
      <b>{identity.displayName}</b>
      {/* A sentence on hover, not an absolute path in uppercase mono. */}
      <small>{plainActivity(session)}</small>
    </span>
  </button>;
}

/* `.settlement-site` is inset -7% on each side of the region box, and -13% for the
   capital, so a module's width — a percentage of that site — is a percentage of 1.14 or
   1.26 region boxes. */
const SITE_OVERHANG = { town: 1.14, capital: 1.26 };

/* A capital should read as 14–18% of the island it stands on. It was 34.6%: the art was
   sized against the region box, and the region box is a small allowance inside a much
   larger island, so the town outgrew the ground. The cap is computed against the land's
   real bounding box and applied to every module. */
function moduleCapFor(landWidth, position, isCapital) {
  const site = (position?.w ?? 0) * (isCapital ? SITE_OVERHANG.capital : SITE_OVERHANG.town);
  if (!landWidth || !site) return 100;
  return Math.max(18, Math.min(100, (landWidth * 0.18) / site * 100));
}

function ProceduralSettlement({ name, isCapital, tier, moduleCap = 100 }) {
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
      ? <img className="settlement-capital" src="/assets/living-rome/capital-tier-3.webp" alt="" aria-hidden="true" decoding="async" style={{ maxWidth: `${moduleCap}%` }} />
      : plan.modules.map((module, index) => <img
        key={module.id}
        className={`settlement-module module-${module.id}${index === 0 ? ' primary' : ' support'}`}
        src={module.src}
        alt=""
        aria-hidden="true"
        decoding="async"
        style={{ left: `${module.x}%`, top: `${module.y}%`, width: `${Math.min(module.size, moduleCap)}%`, zIndex: 9 + index, '--module-rotation': `${module.rotate}deg` }}
      />)}
    <span className="settlement-plaque" aria-hidden="true"><i>{isCapital ? '◆' : plan.seed.token.slice(0, 2)}</i><b>{name}</b><small>{tierLabel}</small></span>
  </div>;
}

/** The city: one mounted project, drawn as the settlement it always was. */
function City({ position, workspace, isCapital, tier, active, focused, agents, landWidth, onOpen }) {
  const moduleCap = moduleCapFor(landWidth, position, isCapital);
  return <section
    className={`field-region settled kind-project${active ? ' active' : ''}${isCapital ? ' capital' : ''}${focused ? ' focused' : ''}`}
    style={{ left: `${position.x}%`, top: `${position.y}%`, width: `${position.w}%`, height: `${position.h}%` }}
  >
    <div className="region-influence" aria-hidden="true" />
    <button
      type="button"
      className="field-region-action"
      aria-label={`Open ${workspace.name} at its project root`}
      onClick={(event) => { event.stopPropagation(); onOpen(''); }}
    />
    {/* The plaque under the settlement is where the project is named, once. A second
        `capital-label` chip used to print the same name a few pixels away, and the page
        header a third time; both are gone. */}
    <ProceduralSettlement name={workspace.name} isCapital={isCapital} tier={tier} moduleCap={moduleCap} />
    {agents > 0 && <span className="city-crowd mono" aria-hidden="true">{agents}</span>}
  </section>;
}

/**
 * A district: one folder, drawn at a size that comes from its weight and a tone that
 * comes from how recently anyone was in it.
 */
function District({ district, heat, agents, attention, focused, onOpen, onStart }) {
  const Icon = KIND_ICON[LANDMARK_KIND[classify(district.dir)]] ?? Landmark;
  const facts = `${weightLabel(district)} · ${heat.label}`;
  return <div
    className={`district heat-${heat.id}${focused ? ' focused' : ''}${attention > 0 ? ' attention' : ''}`}
    style={{ left: `${district.x}%`, top: `${district.y}%`, '--district-weight': district.weight.toFixed(3) }}
  >
    {/* The plaque used to carry a file count and "n here" in mono under the name. The
        agents are already drawn standing in the district and the file count is on hover
        and in the folder sheet, so what the plaque says now is a badge when the place
        needs you — which is the only thing you must read from across the map. */}
    {attention > 0 && (
      <span className="district-badge mono" aria-hidden="true">{attention}</span>
    )}
    <button
      type="button"
      className="district-open"
      onClick={(event) => { event.stopPropagation(); onOpen(district.dir); }}
      title={`${district.dir} — ${facts}`}
      aria-label={attention > 0
        ? `Open ${district.dir}: ${attention} ${attention === 1 ? 'agent needs' : 'agents need'} you. ${facts}`
        : `Open ${district.dir}: ${facts}`}
    >
      <span className="building-shape" aria-hidden="true"><i /><i /><i /><i /><Icon /></span>
      <span className="district-plaque"><b>{district.name}</b></span>
    </button>
    <button
      type="button"
      className="district-add"
      title={`Start an agent in ${district.dir}`}
      aria-label={`Start an agent in ${district.dir}`}
      onClick={(event) => { event.stopPropagation(); onStart(district.dir, event); }}
    ><Plus aria-hidden="true" /></button>
  </div>;
}

function CapitalChooser({ workspaces, current, onChoose, onClose, pending, error }) {
  useModalFocus(onClose, 'capital-title');
  return <div className="capital-veil" onClick={(event) => event.stopPropagation()}><section className="capital-choice" role="dialog" aria-modal="true" aria-labelledby="capital-title"><span className="choice-kicker">FOUND THE CIVILIZATION</span><h2 id="capital-title">Choose the capital project</h2><p>The capital is the visual anchor and default coordination point. It does not move files, services, or permissions.</p><div className="capital-projects">{workspaces.map((workspace) => <button type="button" key={workspace.id} disabled={pending} onClick={() => onChoose(workspace.id)}><span className="project-sigil">{workspace.name.slice(0, 1).toUpperCase()}</span><span><b>{workspace.name}</b><small>{workspace.path}</small></span><i>{workspace.id === current ? 'CURRENT' : 'CHOOSE'}</i></button>)}</div>{error && <p className="capital-error">{error}</p>}{current && <button type="button" className="choice-cancel" onClick={onClose}>Keep current capital</button>}</section></div>;
}

export default function TheaterMode() {
  const st = useField();
  const [settings, setSettings] = useFieldSettings();
  const [place, setPlace] = useState(null);           // { workspaceId, dir } — the open folder
  const [expanded, setExpanded] = useState(false);    // that folder, filling the screen
  const [request, setRequest] = useState(null);       // what a caller asked the folder to show
  const [plansOpen, setPlansOpen] = useState(false);
  const [replay, setReplay] = useState(null);         // a past moment, or null for live
  const [starter, setStarter] = useState(null);       // { workspace, dir, agentId, screen }
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [powerOpen, setPowerOpen] = useState(false);
  const [choosingCapital, setChoosingCapital] = useState(false);
  const [pendingCapital, setPendingCapital] = useState(false);
  const [capitalError, setCapitalError] = useState('');
  const [trees, setTrees] = useState({});             // JSON [ws, dir] -> entries | 'error'
  const inFlight = useRef(new Set());
  const reconcileRef = useRef('');
  const defaultCapitalRef = useRef('');

  /* Live is the snapshot. A replay substitutes the three fields the map is drawn from —
     where the agents are, how stale each folder is, what changed — folded from the same
     event log up to the moment on the rail. Everything else, projects included, is what
     it is now: a project that is not mounted today was not a city yesterday either. */
  const view = useMemo(() => (replay
    ? { ...st.snap, now: replay.at || st.snap.now, sessions: replay.sessions, folders: replay.folders, files: replay.files }
    : st.snap), [st.snap, replay]);
  const replaying = Boolean(replay);
  const now = view.now ?? Date.now();
  const world = view.world ?? { capitalWorkspaceId: null, assignments: {}, revision: 0 };
  const workspaces = useMemo(() => view.workspaces.filter((item) => item.mounted), [view.workspaces]);
  const endpoints = view.endpoints?.length ? view.endpoints : st.config?.endpoints ?? [];

  // Anchor the world on the first project by default; the Capital row re-opens the chooser.
  useEffect(() => {
    if (world.capitalWorkspaceId || !workspaces.length || !st.connected) return;
    const first = workspaces[0].id;
    if (defaultCapitalRef.current === first) return;
    defaultCapitalRef.current = first;
    api.selectCapital(first).catch((error) => { defaultCapitalRef.current = ''; setCapitalError(error.message); });
  }, [world.capitalWorkspaceId, workspaces, st.connected]);

  // Territories are per project; the districts inside one are folders, not clusters.
  useEffect(() => {
    if (!world.capitalWorkspaceId || !workspaces.length) return;
    const clusters = workspaces.map((workspace) => ({
      clusterKey: `workspace:${workspace.id}`, label: workspace.name, kind: 'project', workspaceId: workspace.id,
    }));
    const signature = `${world.capitalWorkspaceId}:${clusters.map((item) => item.clusterKey).sort().join('|')}`;
    const currentKeys = new Set(clusters.map((item) => item.clusterKey));
    const missing = clusters.some((item) => !world.assignments?.[item.clusterKey]);
    const stale = Object.keys(world.assignments ?? {}).some((key) => !currentKeys.has(key));
    const capitalWrong = world.assignments?.[`workspace:${world.capitalWorkspaceId}`]?.territoryId !== 'italia';
    if ((!missing && !stale && !capitalWrong) || reconcileRef.current === signature) return;
    reconcileRef.current = signature;
    api.reconcileWorld(clusters).catch(() => { reconcileRef.current = ''; });
  }, [workspaces, world.assignments, world.capitalWorkspaceId]);

  // ---- the folder tree, read one level at a time and kept ------------------------
  // The map needs the level it is drawing plus each district's own listing for its file
  // count; the open folder needs its own children. Everything is cached by path.
  const wanted = useMemo(() => {
    const keys = new Set();
    const addLevel = (workspaceId, dir) => {
      keys.add(treeKey(workspaceId, dir));
      const entries = entriesOf(trees, workspaceId, dir);
      if (!entries) return;
      for (const entry of entries.filter((item) => item.dir).slice(0, 24)) {
        keys.add(treeKey(workspaceId, normalizeDir(entry.path)));
      }
    };
    for (const workspace of workspaces) {
      addLevel(workspace.id, place?.workspaceId === workspace.id ? parentOf(place.dir) : '');
    }
    if (place) addLevel(place.workspaceId, place.dir);
    return [...keys];
  }, [workspaces, place, trees]);

  // One GET per path, ever: the in-flight set is a ref rather than state so a re-render
  // while a listing is on the wire cannot cancel it or ask for it twice.
  useEffect(() => {
    for (const key of wanted) {
      if (trees[key] !== undefined || inFlight.current.has(key)) continue;
      inFlight.current.add(key);
      const [wsId, dir] = JSON.parse(key);
      api.tree(wsId, dir)
        .then((result) => setTrees((prev) => ({ ...prev, [key]: result.entries ?? [] })))
        .catch(() => setTrees((prev) => ({ ...prev, [key]: 'error' })))
        .finally(() => inFlight.current.delete(key));
    }
  }, [wanted, trees]);

  // ---- the map ------------------------------------------------------------------
  const liveSessions = useMemo(
    () => view.sessions.filter((session) => !TERMINAL_STATES.has(session.state)),
    [view.sessions],
  );
  const identities = useMemo(
    () => new Map(liveSessions.map((session) => [session.id, identityFor(session, endpoints, settings)])),
    [liveSessions, endpoints, settings],
  );

  /* The capital's island takes the first frame; the rest of the archipelago follows in a
     stable order, so a project does not swap seas because another one mounted. */
  const cities = useMemo(() => {
    const ordered = [...workspaces].sort(
      (a, b) => Number(b.id === world.capitalWorkspaceId) - Number(a.id === world.capitalWorkspaceId)
        || String(a.path ?? a.id).localeCompare(String(b.path ?? b.id)),
    );
    return ordered.map((workspace, index) => ({ workspace, frame: frameFor(ordered.length, index) }));
  }, [workspaces, world.capitalWorkspaceId]);

  const map = useMemo(() => cities.map(({ workspace, frame }) => {
    const level = place?.workspaceId === workspace.id ? parentOf(place.dir) : '';
    const entries = entriesOf(trees, workspace.id, level);
    const childTrees = new Map();
    for (const entry of entries?.filter((item) => item.dir) ?? []) {
      const child = normalizeDir(entry.path);
      const tree = entriesOf(trees, workspace.id, child);
      if (tree) childTrees.set(child, tree);
    }
    const districts = districtsAt({
      workspaceId: workspace.id, dir: level, entries, childTrees,
      folders: view.folders ?? [], files: view.files ?? [], now, limit: MAX_DISTRICTS,
    });
    const mine = liveSessions.filter((session) => session.workspaceId === workspace.id);
    const tier = verifiedContribution({ metrics: workspace.maturity }, mine).tier;

    /* The ground is generated from this folder and its children, seeded by the path: drill
       into `web` and you sail to `…/field/web`'s own island, the same one every time. It is
       memoized on that seed and on the folder list, never on the event log, so an arriving
       event moves the markers and leaves the coastline alone. */
    if (!districts) {
      const centre = { x: frame.cx, y: frame.cy };
      return {
        workspace, frame, island: null, heat: new Map(), level, tier, loading: true,
        position: settlementBox(frame, centre), cityCentre: centre, landWidth: frame.rx * 2,
        districts: [], standing: standingPlaces(mine, []),
      };
    }
    const island = islandFor({
      id: `${normalizeDir(workspace.path) || workspace.id}/${level}`,
      regions: districts.map((item) => ({ key: item.dir, name: item.name, weight: item.weight })),
      frame,
    });
    const ground = new Map(island.regions.map((region) => [region.key, region]));
    const laid = districts.map((district) => {
      const spot = ground.get(district.dir);
      return {
        ...district,
        x: spot?.x ?? island.site.x,
        y: spot?.y ?? island.site.y,
        cityX: island.site.x,
        cityY: island.site.y,
      };
    });
    const standing = standingPlaces(mine, laid.map((item) => item.dir));
    const heat = new Map(laid.map((district) => [
      district.dir, staleness(district.lastTs, now, (standing.get(district.dir) ?? []).length).id,
    ]));
    const position = settlementBox(island, island.site);
    return {
      workspace,
      frame,
      island,
      heat,
      position,
      landWidth: island.bbox?.w ?? island.rx * 2,
      cityCentre: { x: island.site.x, y: island.site.y },
      level,
      districts: laid,
      standing,
      tier,
      loading: false,
    };
  }), [cities, trees, place, view.folders, view.files, liveSessions, now]);

  const plates = useMemo(
    () => map.filter((entry) => entry.island).map((entry) => ({
      key: entry.workspace.id, island: entry.island, heat: entry.heat,
    })),
    [map],
  );

  // ---- the open folder ----------------------------------------------------------
  const open = useMemo(() => {
    if (!place) return null;
    const workspace = workspaces.find((item) => item.id === place.workspaceId);
    if (!workspace) return null;
    const entries = entriesOf(trees, workspace.id, place.dir);
    const childTrees = new Map();
    for (const entry of entries?.filter((item) => item.dir) ?? []) {
      const child = normalizeDir(entry.path);
      const tree = entriesOf(trees, workspace.id, child);
      if (tree) childTrees.set(child, tree);
    }
    const subfolders = districtsAt({
      workspaceId: workspace.id, dir: place.dir, entries, childTrees,
      folders: view.folders ?? [], files: view.files ?? [], now, limit: 40,
    });
    const mine = liveSessions.filter((session) => session.workspaceId === workspace.id);
    const standing = subfolders ? standingPlaces(mine, subfolders.map((item) => item.dir)) : null;
    const weight = {
      ...folderRollup({ workspaceId: workspace.id, dir: place.dir, folders: view.folders ?? [], files: view.files ?? [], now }),
      files: entries ? entries.filter((item) => !item.dir).length : null,
      subfolders: entries ? entries.filter((item) => item.dir).length : 0,
    };
    return {
      workspace,
      weight,
      subfolders: subfolders?.map((item) => ({ ...item, agents: standing?.get(item.dir)?.length ?? 0 })) ?? null,
      sessions: sessionsInScope(view.sessions, {
        workspaceId: workspace.id, dir: place.dir, finishedWithinMs: RECENTLY_FINISHED_MS, now,
      }),
    };
  }, [place, workspaces, trees, view.folders, view.files, view.sessions, liveSessions, now]);

  // ---- starting agents ----------------------------------------------------------
  const startAgent = (workspaceId, dir = '', event = null, agentId = null) => {
    // Nothing starts in the past: the rail has to be back at live first.
    if (replaying) return;
    const workspace = workspaces.find((item) => item.id === workspaceId) ?? workspaces[0];
    if (!workspace) return;
    setStarter({
      workspace,
      dir: normalizeDir(dir),
      agentId,
      screen: { x: event?.clientX ?? Math.max(8, window.innerWidth / 2 - 180), y: event?.clientY ?? 120 },
    });
  };

  // Ctrl+Alt+N reaches whichever Field screen is mounted; Rome answers it on the folder
  // you have open, or on the capital when you have none.
  useEffect(() => {
    const onStart = () => {
      const workspaceId = place?.workspaceId ?? world.capitalWorkspaceId ?? workspaces[0]?.id;
      startAgent(workspaceId, place?.dir ?? '');
    };
    window.addEventListener('field:start-agent', onStart);
    return () => window.removeEventListener('field:start-agent', onStart);
  }, [place, workspaces, world.capitalWorkspaceId]);

  async function chooseCapital(workspaceId) {
    setPendingCapital(true); setCapitalError('');
    try {
      await api.selectCapital(workspaceId);
      reconcileRef.current = '';
      setChoosingCapital(false);
    } catch (error) { setCapitalError(error.message); } finally { setPendingCapital(false); }
  }

  /* Opening a folder is a move in space, so it drops whichever agent was selected: only
     clicking a marker re-selects one, and only then does the sheet scroll to it. A folder
     opens compact; moving between folders while expanded stays expanded, because that is
     the same act of walking around with the files open. */
  const openFolder = (workspaceId, dir, { keepExpanded = false, keepAgent = false } = {}) => {
    if (!keepAgent) clearActiveAgent();
    setPlace({ workspaceId, dir: normalizeDir(dir) });
    if (!keepExpanded) { setExpanded(false); setRequest(null); }
    setSettingsOpen(false);
  };

  /* Anything anywhere can say "open this in Rome": a context menu, a conversation's
     "changed files", a plan's roster. The request arrives through the store and is
     consumed once, so re-rendering does not drag you back to it. */
  useEffect(() => {
    const ask = st.rome;
    if (!ask) return;
    if (ask.plans) setPlansOpen(true);
    if (ask.sessionId) selectAgent(ask.sessionId);
    if (ask.workspaceId) {
      setPlace({ workspaceId: ask.workspaceId, dir: normalizeDir(ask.dir ?? '') });
      setExpanded(Boolean(ask.expanded));
      setRequest(ask.path || ask.view || ask.pane || ask.url ? ask : null);
      setSettingsOpen(false);
    }
    romeRequestHandled(ask.seq);
  }, [st.rome]);

  // Escape backs out one layer at a time: the expanded folder, then the folder, then the
  // plans overlay. It never leaves Rome.
  useEffect(() => {
    const onKey = (event) => {
      if (event.key !== 'Escape' || event.defaultPrevented) return;
      if (expanded) { setExpanded(false); setRequest(null); return; }
      if (place) { setPlace(null); clearActiveAgent(); return; }
      if (plansOpen) setPlansOpen(false);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [expanded, place, plansOpen]);

  const capital = workspaces.find((item) => item.id === world.capitalWorkspaceId);
  const panelOpen = Boolean(open) || settingsOpen || plansOpen;

  /* The bar belongs to App now. Rome fills it with where you are — the project, then the
     open folder's path — and says when starting an agent is unavailable. There is no
     page title: the island's plaque already carries the project's name. */
  const here = workspaces.find((item) => item.id === place?.workspaceId) ?? capital;
  const crumbKey = `${here?.id ?? ''}|${place?.dir ?? ''}|${replaying}|${workspaces.length}|${plansOpen}`;
  useEffect(() => {
    const root = here?.name ?? (workspaces.length ? 'Opening the project…' : 'No project open');
    const steps = place ? crumbsFor(place.dir).slice(1) : [];
    setChrome({
      crumbs: [
        { key: 'root', label: root, dir: '' },
        ...steps.map((step) => ({ key: step.dir, label: step.name, dir: step.dir })),
      ],
      primaryDisabled: !workspaces.length || replaying,
      // One primary per screen, and it belongs to the innermost thing you opened.
      primaryQuiet: Boolean(place) || plansOpen,
      primaryHint: replaying ? 'You are looking at an earlier moment. Go back to live to start an agent.' : '',
    });
  }, [crumbKey]);

  // The breadcrumb and the gear are in the bar; both come back as events, the same way
  // Ctrl+Alt+N already did.
  useEffect(() => {
    const onNavigate = (event) => {
      const id = place?.workspaceId ?? world.capitalWorkspaceId ?? workspaces[0]?.id;
      if (id) openFolder(id, event.detail?.dir ?? '');
    };
    const onSettings = () => setSettingsOpen(true);
    window.addEventListener('field:navigate', onNavigate);
    window.addEventListener('field:open-settings', onSettings);
    return () => {
      window.removeEventListener('field:navigate', onNavigate);
      window.removeEventListener('field:open-settings', onSettings);
    };
  }, [place, workspaces, world.capitalWorkspaceId]);

  return <div
    className={`field-world-shell density-${settings.density}${settings.motion ? ' motion-on' : ' motion-off'}${panelOpen ? ' panel-open' : ''}${open ? ' folder-open' : ''}${expanded ? ' folder-expanded' : ''}${plansOpen ? ' plans-open' : ''}${replaying ? ' replaying' : ''}`}
    onClick={() => { setPlace(null); setExpanded(false); setRequest(null); clearActiveAgent(); }}
  >
    <main className="field-world-canvas living-world">
      <IslandPlate plates={plates} />

      <div className="world-routes">
        {map.flatMap(({ workspace, cityCentre, districts }) => districts.map((district) => (
          <i key={`${workspace.id}:${district.dir}`} className="route-district" style={spokeStyle(cityCentre, district)} />
        )))}
      </div>

      {map.map(({ workspace, position, cityCentre, level, districts, standing, tier, loading, landWidth }) => {
        const atCity = standing.get('') ?? [];
        return <div key={workspace.id} className="city-group">
          <City
            position={position}
            workspace={workspace}
            isCapital={workspace.id === world.capitalWorkspaceId}
            tier={tier}
            active={districts.some((district) => (standing.get(district.dir) ?? []).length > 0) || atCity.length > 0}
            focused={place?.workspaceId === workspace.id}
            agents={atCity.length}
            landWidth={landWidth}
            onOpen={(dir) => openFolder(workspace.id, dir)}
          />

          {level !== '' && (
            <button
              type="button"
              className="district-up"
              style={{ left: `${cityCentre.x}%`, top: `${position.y - 4}%` }}
              onClick={(event) => { event.stopPropagation(); openFolder(workspace.id, parentOf(level)); }}
            >↑ {level}</button>
          )}

          {districts.map((district) => {
            const here = standing.get(district.dir) ?? [];
            const heat = staleness(district.lastTs, now, here.length);
            return <div key={district.dir} className="district-group">
              <District
                district={district}
                heat={heat}
                agents={here.length}
                attention={here.filter((session) => ATTENTION_STATES.has(session.state) || session.pendingPermission).length}
                focused={place?.workspaceId === workspace.id && place.dir === district.dir}
                onOpen={(dir) => openFolder(workspace.id, dir)}
                onStart={(dir, event) => startAgent(workspace.id, dir, event)}
              />
              {here.slice(0, MAX_MARKERS).map((session, index) => {
                const offset = markerRing(here.length, index);
                return <AgentMarker
                  key={session.id}
                  session={session}
                  identity={identities.get(session.id)}
                  x={district.x + offset.dx}
                  y={district.y + offset.dy}
                  index={index}
                  settings={settings}
                  selected={session.id === st.activeSessionId}
                  onSelect={(picked) => {
                    openFolder(workspace.id, district.dir);
                    selectAgent(picked.id);
                  }}
                />;
              })}
              {here.length > MAX_MARKERS && (
                <span className="district-overflow mono" style={{ left: `${district.x}%`, top: `${district.y + 9}%` }}>
                  +{here.length - MAX_MARKERS}
                </span>
              )}
            </div>;
          })}

          {atCity.slice(0, MAX_MARKERS).map((session, index) => {
            const offset = markerRing(Math.max(2, atCity.length), index, { radius: 9 });
            return <AgentMarker
              key={session.id}
              session={session}
              identity={identities.get(session.id)}
              x={cityCentre.x + offset.dx}
              y={cityCentre.y + offset.dy}
              index={index}
              settings={settings}
              selected={session.id === st.activeSessionId}
              onSelect={(picked) => { openFolder(workspace.id, normalizeDir(picked.focusDir) || level); selectAgent(picked.id); }}
            />;
          })}

          {loading && <span className="district-reading mono" style={{ left: `${cityCentre.x}%`, top: `${position.y + position.h + 2}%` }}>reading the folders…</span>}
        </div>;
      })}

      {plansOpen && <PlansOverlay
        map={map}
        onClose={() => setPlansOpen(false)}
        onOpenAgent={(sessionId) => {
          const session = view.sessions.find((item) => item.id === sessionId);
          if (!session?.workspaceId) return;
          selectAgent(sessionId);
          openFolder(session.workspaceId, normalizeDir(session.focusDir), { keepAgent: true });
        }}
      />}
    </main>

    {/* Time, along the bottom of the territory. Live is the right edge. */}
    <TimeControl onReplay={setReplay} />

    {open && (
      <FolderDetail
        workspace={open.workspace}
        dir={place.dir}
        weight={open.weight}
        subfolders={open.subfolders}
        sessions={open.sessions}
        selectedId={st.activeSessionId}
        primary={!plansOpen}
        expanded={expanded}
        request={request}
        onNavigate={(dir) => openFolder(open.workspace.id, dir, { keepExpanded: expanded })}
        onStart={(dir, event, agentId) => startAgent(open.workspace.id, dir, event, agentId ?? null)}
        onToggleExpand={() => setExpanded((value) => { if (value) setRequest(null); return !value; })}
        onClose={() => { setPlace(null); setExpanded(false); setRequest(null); }}
      />
    )}

    {!workspaces.length && <div className="world-notice" role="status">No project is mounted. Add one under <code>workspaces</code> in <code>field/field.yaml</code> and restart Field.</div>}
    {capitalError && !choosingCapital && <div className="world-notice bad" role="alert">{capitalError}</div>}

    {settingsOpen && <FieldSettings
      settings={settings}
      setSettings={setSettings}
      selected={view.sessions.find((session) => session.id === st.activeSessionId) ?? null}
      config={st.config}
      onClose={() => setSettingsOpen(false)}
      onOpenModels={() => { setSettingsOpen(false); setPowerOpen(true); }}
      onChooseCapital={() => { setSettingsOpen(false); setChoosingCapital(true); }}
      onOpenPlans={() => { setSettingsOpen(false); setPlansOpen(true); }}
      planCount={view.campaigns?.length ?? 0}
    />}
    {powerOpen && <PowerSources onClose={() => setPowerOpen(false)} settings={settings} setSettings={setSettings} />}
    {starter && <ContextMenu
      fixed
      initialPane="spawn"
      initialAgentId={starter.agentId}
      screen={starter.screen}
      target={starter.dir
        ? { type: 'folder', id: starter.dir, label: starter.dir, workspaceId: starter.workspace.id }
        : { type: 'workspace', id: starter.workspace.id, workspaceId: starter.workspace.id, label: starter.workspace.name }}
      scopeNote={starter.dir ? `Folder in scope:\n- ${starter.dir}` : ''}
      onClose={() => setStarter(null)}
      onOpenModels={() => { setStarter(null); setPowerOpen(true); }}
    />}
    {choosingCapital && <CapitalChooser workspaces={workspaces} current={world.capitalWorkspaceId} onChoose={chooseCapital} onClose={() => setChoosingCapital(false)} pending={pendingCapital} error={capitalError} />}
  </div>;
}
