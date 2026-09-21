// Turns a server snapshot into world-space geometry.
// Every coordinate here is derived from something real: a mounted workspace, a folder
// an agent actually touched, a domain an agent actually opened. Nothing is placed to
// fill space, and nothing moves unless the underlying fact moved.

const FOLDER_W = 78;
const FOLDER_H = 20;
const FOLDER_GAP_X = 10;
const FOLDER_GAP_Y = 9;
const REGION_PAD_X = 18;
const REGION_PAD_TOP = 40;

const HEAT_HALFLIFE_MS = 120_000;
const PULSE_MS = 1400;

// The Field is a command surface, not a file browser. Cold folders disappear and busy
// regions cap their chips, so what is drawn is always what is worth looking at.
const MIN_FOLDER_HEAT = 0.08;
const MAX_FOLDERS_PER_REGION = 16;

export function heatOf(lastTs, hits, now) {
  if (!lastTs) return 0;
  const recency = Math.exp(-(now - lastTs) / HEAT_HALFLIFE_MS);
  const volume = Math.min(1, Math.log2(1 + (hits ?? 0)) / 6);
  return Math.max(0, Math.min(1, 0.65 * recency + 0.35 * volume));
}

export function pulseOf(lastTs, now) {
  if (!lastTs) return 0;
  const age = now - lastTs;
  return age < 0 || age > PULSE_MS ? 0 : 1 - age / PULSE_MS;
}

/** A wedge below the anchor. Readable at a glance as "these belong to that". */
function formationSlot(i, count) {
  const perRow = 6;
  const row = Math.floor(i / perRow);
  const inRow = i % perRow;
  const rowCount = Math.min(perRow, count - row * perRow);
  const spread = Math.min(64, 15 * (rowCount - 1));
  const t = rowCount === 1 ? 0 : (inRow / (rowCount - 1)) * 2 - 1;
  return { dx: t * spread, dy: 30 + row * 24 };
}

export function computeLayout(snap, positions, now, config) {
  const regions = [];
  const regionById = new Map();

  for (const w of snap.workspaces) {
    const override = positions[`workspace:${w.id}`];
    const r = {
      id: w.id, name: w.name, mounted: w.mounted, git: w.git, path: w.path,
      x: override?.x ?? w.region.x, y: override?.y ?? w.region.y,
      w: w.region.w, h: w.region.h,
      heat: heatOf(w.lastTs, w.changeCount, now),
      changeCount: w.changeCount,
    };
    regions.push(r);
    regionById.set(w.id, r);
  }

  // --- folders that have actually been touched -------------------------------
  const foldersByWs = new Map();
  for (const f of snap.folders) {
    const heat = heatOf(f.lastTs, f.hits, now);
    if (heat < MIN_FOLDER_HEAT) continue;    // gone cold; stop drawing it
    if (!foldersByWs.has(f.workspaceId)) foldersByWs.set(f.workspaceId, []);
    foldersByWs.get(f.workspaceId).push({ ...f, heat });
  }

  const folders = [];
  const folderByKey = new Map();
  for (const [wsId, all] of foldersByWs) {
    const region = regionById.get(wsId);
    if (!region) continue;

    // Keep the region readable: only the most active folders get a chip, and the
    // rest are counted rather than drawn. A wall of chips is not information.
    const list = all.length > MAX_FOLDERS_PER_REGION
      ? [...all].sort((a, b) => b.heat - a.heat).slice(0, MAX_FOLDERS_PER_REGION)
      : all;
    region.hiddenFolders = all.length - list.length;

    // Sorted by path, not by heat: a folder must not jump because it got busy.
    list.sort((a, b) => a.dir.localeCompare(b.dir));
    const cols = Math.max(1, Math.floor((region.w - REGION_PAD_X * 2 + FOLDER_GAP_X) / (FOLDER_W + FOLDER_GAP_X)));
    list.forEach((f, i) => {
      const col = i % cols;
      const row = Math.floor(i / cols);
      const node = {
        ...f,
        x: region.x + REGION_PAD_X + col * (FOLDER_W + FOLDER_GAP_X),
        y: region.y + REGION_PAD_TOP + row * (FOLDER_H + FOLDER_GAP_Y),
        w: FOLDER_W, h: FOLDER_H,
        label: f.dir === '' ? '/' : f.dir.split('/').slice(-1)[0],
        full: f.dir,
      };
      folders.push(node);
      folderByKey.set(f.key, node);
    });
  }

  // --- files as first-class objects, placed under their folder ---------------
  // A file is drawn only when its folder is (folders are already heat-gated), so the
  // Field never fills with cold files. Each file gets a real coordinate so a unit can
  // walk to it, it can pulse when touched, and it can be selected and ordered onto.
  const FILE_STEP = 12;
  const FILE_R = 3.2;
  const MAX_FILES_PER_FOLDER = 8;
  const files = [];
  const fileByKey = new Map();
  const filesByFolder = new Map();
  for (const file of snap.files) {
    const node = folderByKey.get(`${file.workspaceId}:${file.dir}`);
    if (!node) continue;
    if (!filesByFolder.has(node.key)) filesByFolder.set(node.key, []);
    filesByFolder.get(node.key).push(file);
  }
  for (const [folderKey, list] of filesByFolder) {
    const node = folderByKey.get(folderKey);
    // Most-recently-touched first so the ones worth seeing keep their slot.
    list.sort((a, b) => (b.lastTs ?? 0) - (a.lastTs ?? 0));
    const shown = list.slice(0, MAX_FILES_PER_FOLDER);
    const rowW = (shown.length - 1) * FILE_STEP;
    const cx = node.x + node.w / 2;
    const fy = node.y + node.h + 9;
    shown.forEach((file, i) => {
      const age = now - (file.lastTs ?? 0);
      const fileNode = {
        key: file.key, path: file.path, dir: file.dir, workspaceId: file.workspaceId,
        change: file.change, lastTs: file.lastTs,
        x: cx - rowW / 2 + i * FILE_STEP, y: fy, r: FILE_R,
        recency: age >= 0 && age < 60_000 ? 1 - age / 60_000 : 0,
        label: file.path.split('/').slice(-1)[0],
        folderKey,
      };
      files.push(fileNode);
      fileByKey.set(`${file.workspaceId}:${file.path}`, fileNode);
    });
  }

  // --- mission artifacts -----------------------------------------------------
  // A mission is a Markdown file in Git. It appears as a small contextual artifact on
  // the region it targets, near the bottom edge so it never competes with live work.
  const missions = [];
  let missionSlot = new Map();
  for (const m of config?.missions ?? []) {
    const region = regionById.get(m.workspace);
    if (!region) continue;
    const i = missionSlot.get(m.workspace) ?? 0;
    missionSlot.set(m.workspace, i + 1);
    const active = snap.assignments.some(
      (a) => a.targetType === 'mission' && a.targetId === m.id && a.status === 'active',
    );
    missions.push({
      id: m.id, name: m.name ?? m.id, workspaceId: m.workspace,
      target: m.target ?? null, active,
      x: region.x + REGION_PAD_X + i * 168,
      y: region.y + region.h - 30,
      w: 156, h: 18,
    });
  }

  // --- website zones, to the right of everything -----------------------------
  const rightEdge = regions.reduce((m, r) => Math.max(m, r.x + r.w), 0);
  const activeSites = snap.websites.filter((s) => s.hits > 0 || s.sessions.length > 0);
  const sites = activeSites.map((s, i) => ({
    ...s,
    x: rightEdge + 130,
    y: (regions[0]?.y ?? 0) + i * 56,
    w: 190, h: 40,
    heat: heatOf(s.lastTs, s.hits, now),
  }));
  const siteByDomain = new Map(sites.map((s) => [s.domain, s]));

  // --- agents ----------------------------------------------------------------
  const live = snap.sessions.filter((s) => s.state !== 'done' || (now - (s.endedAt ?? 0)) < 120_000);

  // Group by anchor so agents on the same work form up together. A unit anchors to the
  // exact file it is on when that file is drawn; otherwise it falls back to the folder,
  // then the workspace, then staging — the file case never regresses the fallback.
  const anchorKey = (s) => {
    if (s.workspaceId && s.focusPath != null && fileByKey.has(`${s.workspaceId}:${s.focusPath}`)) {
      return `file:${s.workspaceId}:${s.focusPath}`;
    }
    if (s.workspaceId && s.focusDir != null && folderByKey.has(`${s.workspaceId}:${s.focusDir}`)) {
      return `f:${s.workspaceId}:${s.focusDir}`;
    }
    if (s.workspaceId && regionById.has(s.workspaceId)) return `w:${s.workspaceId}`;
    return 'staging';
  };

  const groups = new Map();
  for (const s of live) {
    const k = anchorKey(s);
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(s);
  }

  const stagingX = (regions[0]?.x ?? 0) - 150;
  const stagingY = (regions[0]?.y ?? 0);

  const agents = [];
  for (const [key, list] of groups) {
    list.sort((a, b) => (a.name ?? '').localeCompare(b.name ?? ''));
    let ax; let ay; let onFile = false;
    if (key === 'staging') { ax = stagingX; ay = stagingY; }
    else if (key.startsWith('file:')) {
      const node = fileByKey.get(key.slice(5));
      ax = node.x; ay = node.y + node.r + 2; onFile = true;
    }
    else if (key.startsWith('f:')) {
      const node = folderByKey.get(key.slice(2));
      ax = node.x + node.w / 2; ay = node.y + node.h;
    } else {
      // Agents with no narrower focus muster in the lower half of their region.
      // The anchor sits high enough that the formation offset below it still lands
      // inside the region, and above the mission artifacts along the bottom edge.
      const r = regionById.get(key.slice(2));
      ax = r.x + r.w / 2;
      ay = r.y + r.h - 96;
    }

    list.forEach((s, i) => {
      const slot = formationSlot(i, list.length);
      agents.push({
        id: s.id, session: s,
        x: ax + slot.dx, y: ay + slot.dy,
        anchorX: ax, anchorY: ay,
        staged: key === 'staging', onFile,
      });
    });
  }

  const agentById = new Map(agents.map((a) => [a.id, a]));

  // --- routes: only for work that is actually happening ----------------------
  const routes = [];
  for (const a of agents) {
    const s = a.session;

    if (!a.staged && (a.anchorX !== a.x || a.anchorY !== a.y)) {
      routes.push({ kind: 'work', x1: a.x, y1: a.y, x2: a.anchorX, y2: a.anchorY, id: s.id });
    }
    if (s.browser?.domain) {
      const site = siteByDomain.get(s.browser.domain);
      if (site) {
        routes.push({
          kind: 'browser', id: s.id,
          x1: a.x, y1: a.y, x2: site.x, y2: site.y + site.h / 2,
        });
      }
    }
    for (const childId of s.children ?? []) {
      const child = agentById.get(childId);
      if (child) {
        routes.push({ kind: 'delegate', id: s.id, x1: a.x, y1: a.y, x2: child.x, y2: child.y });
      }
    }
  }

  // Coordination is drawn only when the harness has observed a real communication
  // edge. It gives the Field a living command graph without inventing geography.
  for (const edge of snap.graph?.edges ?? []) {
    if (edge.type !== 'communicates_with' || edge.active === false) continue;
    const fromId = String(edge.from ?? '').replace(/^agent:/, '');
    const toId = String(edge.to ?? '').replace(/^agent:/, '');
    const from = agentById.get(fromId);
    const to = agentById.get(toId);
    if (!from || !to) continue;
    routes.push({
      kind: 'communication', id: fromId,
      x1: from.x, y1: from.y, x2: to.x, y2: to.y, lastTs: edge.lastSeen,
    });
  }

  const all = [...regions.map((r) => ({ x: r.x, y: r.y, w: r.w, h: r.h })),
               ...sites.map((s) => ({ x: s.x, y: s.y, w: s.w, h: s.h })),
               ...agents.map((a) => ({ x: a.x - 20, y: a.y - 20, w: 40, h: 40 }))];
  const bounds = all.length ? {
    minX: Math.min(...all.map((b) => b.x)) - 80,
    minY: Math.min(...all.map((b) => b.y)) - 80,
    maxX: Math.max(...all.map((b) => b.x + b.w)) + 80,
    maxY: Math.max(...all.map((b) => b.y + b.h)) + 80,
  } : { minX: -400, minY: -300, maxX: 400, maxY: 300 };

  return {
    regions, folders, files, sites, missions, agents, routes,
    bounds, agentById, folderByKey, fileByKey,
  };
}
