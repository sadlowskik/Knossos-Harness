/* Rome is territory, and the unit of territory is a folder.

   The Board answers "what is every agent doing", ordered by time. Rome answers "where is
   the work happening", ordered by space: a mounted project is a city, the folders inside
   it are the districts you can see and click, and an agent stands in the folder holding
   the file it is on. Files are deliberately not places — thousands of nodes is not a map
   — and whole projects are already the Board's filter.

   Everything here is pure: the snapshot's `folders` records (key, workspaceId, dir, hits,
   lastTs, agents) and `files` records give the churn, `/api/fs/tree` gives the size, and
   the caller supplies both. No React, no fetching. */

const MINUTE = 60_000;
const HOUR = 60 * MINUTE;
const DAY = 24 * HOUR;

export const normalizeDir = (value) => (
  typeof value === 'string' ? value.replaceAll('\\', '/').replace(/^\/+|\/+$/g, '') : ''
);

export const parentOf = (dir) => (dir.includes('/') ? dir.slice(0, dir.lastIndexOf('/')) : '');

export const dirName = (dir) => (dir ? dir.split('/').filter(Boolean).at(-1) : 'project root');

/** Same prefix rule as the Board's `sessionsInScope`: a path is in a folder's subtree. */
export const under = (path, dir) => {
  const p = normalizeDir(path);
  return Boolean(p) && (dir === '' || p === dir || p.startsWith(`${dir}/`));
};

/** The crumbs from the project root down to `dir`, each one navigable. */
export function crumbsFor(dir) {
  const parts = normalizeDir(dir).split('/').filter(Boolean);
  const rows = [{ dir: '', name: 'project root' }];
  let acc = '';
  for (const part of parts) {
    acc = acc ? `${acc}/${part}` : part;
    rows.push({ dir: acc, name: part });
  }
  return rows;
}

/**
 * Everything the event log knows about one folder's subtree: how often agents have
 * worked in it, how many files changed, and when it was last touched at all.
 */
export function folderRollup({ workspaceId, dir, folders = [], files = [], now = Date.now() }) {
  let hits = 0;
  let lastTs = 0;
  let changes = 0;
  let recentChanges = 0;
  for (const folder of folders) {
    if (folder.workspaceId !== workspaceId) continue;
    if (!(dir === '' ? true : under(folder.dir, dir))) continue;
    hits += folder.hits ?? 0;
    lastTs = Math.max(lastTs, folder.lastTs ?? 0);
  }
  for (const file of files) {
    if (file.workspaceId !== workspaceId) continue;
    if (!(dir === '' ? true : under(file.path ?? file.dir, dir))) continue;
    changes += 1;
    if (now - (file.lastTs ?? 0) < DAY) recentChanges += 1;
    lastTs = Math.max(lastTs, file.lastTs ?? 0);
  }
  return { hits, changes, recentChanges, lastTs };
}

/**
 * How recently this folder changed, and whether anyone is standing in it — in the same
 * restrained set of semantic tones the rest of Field uses. `occupied` wins: a district
 * with agents in it is live whatever its timestamps say.
 */
export function staleness(lastTs, now = Date.now(), occupied = 0) {
  if (occupied > 0) {
    return { id: 'live', label: occupied === 1 ? '1 agent here now' : `${occupied} agents here now` };
  }
  if (!lastTs) return { id: 'untouched', label: 'no one has worked here' };
  const age = Math.max(0, now - lastTs);
  const when = age < MINUTE ? 'just now'
    : age < HOUR ? `${Math.max(1, Math.round(age / MINUTE))}m ago`
      : age < DAY ? `${Math.max(1, Math.round(age / HOUR))}h ago`
        : `${Math.max(1, Math.round(age / DAY))}d ago`;
  const id = age < 15 * MINUTE ? 'hot' : age < 2 * HOUR ? 'warm' : age < DAY ? 'cool' : 'cold';
  return { id, label: age < MINUTE ? 'touched just now' : `last touched ${when}`, when };
}

/**
 * Prominence from real weight, never decoration.
 *
 *   mass  = files directly in the folder + 4 per subfolder (how much territory it holds)
 *   churn = agent visits in its subtree + 2 per file changed there in the last day
 *   score = log1p(mass) + 0.6 · log1p(churn)
 *
 * The logs keep a 4,000-file folder from dwarfing a 40-file one off the map; the weight
 * returned is the score relative to the largest sibling, floored so the smallest district
 * is still a legible target.
 */
export function scoreDistricts(rows) {
  const scored = rows.map((row) => {
    const mass = (row.files ?? 0) + 4 * (row.subfolders ?? 0);
    const churn = (row.hits ?? 0) + 2 * (row.recentChanges ?? 0);
    return { ...row, score: Math.log1p(mass) + 0.6 * Math.log1p(churn) };
  });
  const max = scored.reduce((high, row) => Math.max(high, row.score), 0);
  return scored
    .map((row) => ({ ...row, weight: max > 0 ? Math.max(0.3, row.score / max) : 0.5 }))
    .sort((a, b) => b.score - a.score || a.dir.localeCompare(b.dir));
}

/** "128 files · 6 folders · 3 changed today" — the weight, spelled out. */
export function weightLabel(row) {
  if (row.files == null) return 'reading the folder…';
  const bits = [`${row.files} file${row.files === 1 ? '' : 's'}`];
  if (row.subfolders) bits.push(`${row.subfolders} folder${row.subfolders === 1 ? '' : 's'}`);
  if (row.recentChanges) bits.push(`${row.recentChanges} changed today`);
  return bits.join(' · ');
}

/**
 * Where each agent stands. A session's current file (`focusPath`, falling back to the
 * folder anchor `focusDir`) puts it in the deepest district that contains it; anything
 * outside every district on screen stands at the city itself, under the '' key.
 */
export function standingPlaces(sessions = [], dirs = []) {
  const byDir = new Map([['', []]]);
  for (const dir of dirs) byDir.set(dir, []);
  const deepestFirst = [...dirs].sort((a, b) => b.length - a.length);
  for (const session of sessions) {
    const path = normalizeDir(session.focusPath) || normalizeDir(session.focusDir);
    const home = deepestFirst.find((dir) => under(path, dir)) ?? '';
    byDir.get(home).push(session);
  }
  return byDir;
}

/**
 * The districts of one city at one level: the subfolders of `dir`, each with its own
 * size (from its tree), its churn (from the projection) and its weight.
 *
 * `entries` is the `/api/fs/tree` listing of `dir`; `childTrees` maps a child's path to
 * its own listing, or to null while it is still being read.
 */
export function districtsAt({
  workspaceId, dir = '', entries = null, childTrees = new Map(),
  folders = [], files = [], now = Date.now(), limit = 7,
}) {
  if (!entries) return null;
  const rows = entries.filter((entry) => entry.dir).map((entry) => {
    const childDir = normalizeDir(entry.path);
    const tree = childTrees.get(childDir) ?? null;
    const rollup = folderRollup({ workspaceId, dir: childDir, folders, files, now });
    return {
      workspaceId,
      dir: childDir,
      name: entry.name,
      files: tree ? tree.filter((item) => !item.dir).length : null,
      subfolders: tree ? tree.filter((item) => item.dir).length : 0,
      ...rollup,
    };
  });
  return scoreDistricts(rows).slice(0, limit);
}

/**
 * Districts laid out around their city, as percentages of the map. The ring keeps Rome's
 * shape — a settlement with its country around it — instead of a treemap; the radius
 * opens up as a city gains districts so labels do not collide.
 */
export function ringLayout(districts, city, { seed = 0 } = {}) {
  const count = districts.length;
  if (!count) return [];
  const cx = city.x + city.w / 2;
  const cy = city.y + city.h / 2;
  const rx = 18 + Math.min(9, count * 1.3);
  const ry = 14 + Math.min(7, count * 1.0);
  const start = -Math.PI / 2 + ((seed % 7) - 3) * 0.06;
  return districts.map((district, index) => {
    const angle = start + (index * 2 * Math.PI) / count;
    return {
      ...district,
      x: Math.max(5, Math.min(95, cx + rx * Math.cos(angle))),
      y: Math.max(9, Math.min(90, cy + ry * Math.sin(angle))),
      cityX: cx,
      cityY: cy,
    };
  });
}

/** Agent markers spread around the point they stand on, biggest crowds widest. */
export function markerRing(count, index, { radius = 5.4 } = {}) {
  if (count <= 1) return { dx: 0, dy: 3.4 };
  const spread = radius + Math.min(3.2, count * 0.35);
  const angle = -Math.PI / 2 + (index * 2 * Math.PI) / Math.min(count, 8) + (index >= 8 ? 0.4 : 0);
  return { dx: spread * Math.cos(angle), dy: spread * 0.78 * Math.sin(angle) + 3.4 };
}
