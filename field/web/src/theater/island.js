/* The ground under Rome: an island generated from the repository itself.

   A repository is not the Mediterranean. The plate a project sits on is drawn from the
   project: one region per top-level folder, sized by the weight districts.js already
   computes, with the coastline falling where the folders stop. Adding a folder adds a
   region; it does not redraw the island.

   Nothing here is random and nothing here is stateful. Every jitter, every noise field and
   every headland comes out of one seeded generator keyed on a string — the project's path,
   a folder's full path, a lattice cell's coordinates — so the same repository draws the
   same island on every machine, every launch and every window size.

   Pure geometry: no React, no DOM, no fetching. The caller supplies the folder rows and
   gets back paths in percentage-of-canvas coordinates. */

/* The plate is generated at one fixed aspect rather than the window's, so a wide monitor
   and a narrow one see the same island. Cells are round at this ratio and lean with it. */
const ASPECT = 1.9;

/* Below this many folders an island is a silly shape to draw: one region with a coastline
   around it is a blob, not a map. Fewer than six regions falls back to `isle` mode — the
   same pipeline at 0.58 scale with a calmer coast, so the settlement sits on a small isle
   in open water instead of pretending to be a continent. */
export const MIN_REGIONS = 6;

// ---- one seeded generator, and everything that leans on it ------------------------

/** FNV-1a. Strings in, a stable 32-bit seed out — the only entry point to randomness. */
export function hashString(value) {
  let hash = 2166136261;
  const text = String(value);
  for (let index = 0; index < text.length; index += 1) {
    hash ^= text.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

/** mulberry32: ten lines, one 32-bit word of state, uniform enough for terrain. */
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* A number in [0,1) from a name rather than from a sequence. This is what keeps the map
   stable: a folder's point comes from its own path, not from where it landed in a list, so
   inserting a folder ahead of it cannot move it. */
const unit = (...parts) => mulberry32(hashString(parts.map(String).join('/#/')))();

function valueNoise(seed) {
  const rand = mulberry32(seed);
  const size = 64;
  const grid = new Float64Array(size * size);
  for (let index = 0; index < grid.length; index += 1) grid[index] = rand();
  const at = (ix, iy) => grid[(((iy % size) + size) % size) * size + (((ix % size) + size) % size)];
  const ease = (t) => t * t * (3 - 2 * t);
  return (x, y) => {
    const x0 = Math.floor(x);
    const y0 = Math.floor(y);
    const fx = ease(x - x0);
    const fy = ease(y - y0);
    const top = at(x0, y0) + (at(x0 + 1, y0) - at(x0, y0)) * fx;
    const bottom = at(x0, y0 + 1) + (at(x0 + 1, y0 + 1) - at(x0, y0 + 1)) * fx;
    return top + (bottom - top) * fy;
  };
}

function fbm(noise, x, y, octaves = 3) {
  let sum = 0;
  let amplitude = 1;
  let norm = 0;
  let frequency = 1;
  for (let i = 0; i < octaves; i += 1) {
    sum += amplitude * noise(x * frequency, y * frequency);
    norm += amplitude;
    amplitude *= 0.5;
    frequency *= 2;
  }
  return sum / norm;
}

// ---- Voronoi, by clipping a box with every bisector -------------------------------
/* No Delaunay: with a couple of hundred sites, clipping the frame by each perpendicular
   bisector is exact, twenty lines, and fast enough to be invisible. */

function clipHalfPlane(poly, ax, ay, bx, by) {
  const mx = (ax + bx) / 2;
  const my = (ay + by) / 2;
  const dx = bx - ax;
  const dy = by - ay;
  const side = (p) => (p[0] - mx) * dx + (p[1] - my) * dy;
  const out = [];
  for (let i = 0; i < poly.length; i += 1) {
    const p = poly[i];
    const q = poly[(i + 1) % poly.length];
    const sp = side(p);
    const sq = side(q);
    if (sp <= 0) out.push(p);
    if ((sp < 0 && sq > 0) || (sp > 0 && sq < 0)) {
      const t = sp / (sp - sq);
      out.push([p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t]);
    }
  }
  return out;
}

const QUANT = 1e4;
const snap = (v) => Math.round(v * QUANT) / QUANT;

function voronoi(points, halfW, halfH) {
  return points.map((site, i) => {
    let poly = [[-halfW, -halfH], [halfW, -halfH], [halfW, halfH], [-halfW, halfH]];
    for (let j = 0; j < points.length && poly.length >= 3; j += 1) {
      if (j === i) continue;
      poly = clipHalfPlane(poly, site[0], site[1], points[j][0], points[j][1]);
    }
    // Snapped once, here, so every later step compares identical numbers.
    return poly.map(([x, y]) => [snap(x), snap(y)]);
  });
}

function areaCentroid(poly) {
  let twiceArea = 0;
  let cx = 0;
  let cy = 0;
  for (let i = 0; i < poly.length; i += 1) {
    const [x0, y0] = poly[i];
    const [x1, y1] = poly[(i + 1) % poly.length];
    const cross = x0 * y1 - x1 * y0;
    twiceArea += cross;
    cx += (x0 + x1) * cross;
    cy += (y0 + y1) * cross;
  }
  if (Math.abs(twiceArea) < 1e-9) {
    const n = poly.length || 1;
    return { area: 0, x: poly.reduce((s, p) => s + p[0], 0) / n, y: poly.reduce((s, p) => s + p[1], 0) / n };
  }
  return { area: Math.abs(twiceArea) / 2, x: cx / (3 * twiceArea), y: cy / (3 * twiceArea) };
}

// ---- midpoint displacement: the step that decides drawn or cheap ------------------
/* A Voronoi edge is a straight line and reads as one. Subdividing it four levels, with
   half the jitter each level, turns it into a bay or a headland. The displacement is keyed
   on the pair of stable site ids, not on positions or indices, so the two cells that share
   an edge generate the same polyline — watertight — and the same edge keeps its shape when
   a folder is added elsewhere. */
function displaceEdge(key, p, q, levels, amplitude) {
  let line = [p, q];
  for (let level = 0; level < levels; level += 1) {
    const next = [line[0]];
    const scale = amplitude / 2 ** level;
    for (let i = 0; i + 1 < line.length; i += 1) {
      const [x0, y0] = line[i];
      const [x1, y1] = line[i + 1];
      const dx = x1 - x0;
      const dy = y1 - y0;
      const len = Math.hypot(dx, dy) || 1e-6;
      const push = (unit(key, level, i) - 0.5) * 2 * scale * len;
      next.push([
        (x0 + x1) / 2 + (-dy / len) * push,
        (y0 + y1) / 2 + (dx / len) * push,
      ]);
      next.push(line[i + 1]);
    }
    line = next;
  }
  return line;
}

const pathOf = (points, close = true) => {
  if (!points.length) return '';
  let d = `M${points[0][0].toFixed(2)} ${points[0][1].toFixed(2)}`;
  for (let i = 1; i < points.length; i += 1) d += `L${points[i][0].toFixed(2)} ${points[i][1].toFixed(2)}`;
  return close ? `${d}Z` : d;
};

// ---- the plate --------------------------------------------------------------------

/**
 * Generate one island.
 *
 * `id`      the seed string — a project's path, or a folder's full path when you drill in.
 * `regions` [{ key, name, weight }], key being the folder's full path.
 * `frame`   { cx, cy, rx, ry } — where on the canvas this plate sits, in percent.
 */
export function generateIsland({ id, regions = [], frame }) {
  const started = (typeof performance === 'object' && performance.now) ? performance.now() : Date.now();
  const seed = hashString(id);
  const isle = regions.length < MIN_REGIONS;
  const scale = isle ? 0.58 : 1;
  const rx = frame.rx * scale;
  const ry = frame.ry * scale;

  /* Work in screen-isotropic units: x is stretched by the plate's aspect so a circle here
     is a circle on screen. Mapping back out is one divide. */
  const halfW = rx * ASPECT;
  const halfH = ry;
  const toPct = ([x, y]) => [frame.cx + x / ASPECT, frame.cy + y];

  const land = valueNoise(seed ^ 0x9e3779b9);
  const relief = valueNoise(seed ^ 0x85ebca6b);
  const coastAmp = isle ? 0.58 : 0.80;

  // --- seed points: one per folder, from its own path -------------------------------
  const target = Math.max(48, regions.length * 7 + 54);
  const step = Math.sqrt((4 * halfW * halfH) / target);

  const anchors = regions.map((region) => {
    const radius = Math.sqrt(unit(region.key, 'r')) * 0.70;
    const angle = unit(region.key, 'a') * Math.PI * 2;
    return [halfW * radius * Math.cos(angle), halfH * radius * Math.sin(angle)];
  });

  /* Relax them apart rather than re-rolling: a collision is resolved by pushing both
     points off each other a little, which moves a neighbour a few units and leaves the
     rest of the island alone. */
  const separation = step * 1.35;
  const seeds = anchors.map((p) => [p[0], p[1]]);
  for (let round = 0; round < 12; round += 1) {
    const push = seeds.map(() => [0, 0]);
    let touched = false;
    for (let i = 0; i < seeds.length; i += 1) {
      for (let j = i + 1; j < seeds.length; j += 1) {
        const dx = seeds[j][0] - seeds[i][0];
        const dy = (seeds[j][1] - seeds[i][1]);
        const dist = Math.hypot(dx, dy);
        if (dist >= separation) continue;
        touched = true;
        const over = (separation - dist) / 2;
        const ux = dist > 1e-6 ? dx / dist : 1;
        const uy = dist > 1e-6 ? dy / dist : 0;
        push[i][0] -= ux * over; push[i][1] -= uy * over;
        push[j][0] += ux * over; push[j][1] += uy * over;
      }
    }
    if (!touched) break;
    for (let i = 0; i < seeds.length; i += 1) {
      let x = seeds[i][0] + push[i][0] * 0.65;
      let y = seeds[i][1] + push[i][1] * 0.65;
      const r = Math.hypot(x / halfW, y / halfH);
      if (r > 0.76) { x = (x / r) * 0.76; y = (y / r) * 0.76; }
      seeds[i] = [x, y];
    }
  }

  /* Filler points come off a jittered lattice keyed on the lattice cell, never off a
     running sequence, so their count and their places do not shift when a folder lands. */
  const points = seeds.map((p) => [p[0], p[1]]);
  const ids = regions.map((region) => `f:${region.key}`);
  const cols = Math.max(2, Math.round((2 * halfW) / step));
  const rows = Math.max(2, Math.round((2 * halfH) / step));
  for (let iy = 0; iy < rows; iy += 1) {
    for (let ix = 0; ix < cols; ix += 1) {
      const x = -halfW + (ix + 0.5) * ((2 * halfW) / cols) + (unit(id, 'fx', ix, iy) - 0.5) * step * 0.8;
      const y = -halfH + (iy + 0.5) * ((2 * halfH) / rows) + (unit(id, 'fy', ix, iy) - 0.5) * step * 0.8;
      if (seeds.some((p) => Math.hypot(p[0] - x, p[1] - y) < separation * 0.8)) continue;
      points.push([x, y]);
      ids.push(`l:${ix},${iy}`);
    }
  }

  // --- Lloyd: even cells, with the folder points held near their own point ----------
  let cells = voronoi(points, halfW, halfH);
  for (let round = 0; round < 3; round += 1) {
    for (let i = 0; i < points.length; i += 1) {
      const c = areaCentroid(cells[i]);
      if (i < seeds.length) {
        points[i] = [c.x * 0.40 + seeds[i][0] * 0.60, c.y * 0.40 + seeds[i][1] * 0.60];
      } else {
        points[i] = [points[i][0] + (c.x - points[i][0]) * 0.75, points[i][1] + (c.y - points[i][1]) * 0.75];
      }
    }
    cells = voronoi(points, halfW, halfH);
  }

  const centroids = cells.map((poly) => areaCentroid(poly));

  // --- who owns what, and where the water starts ------------------------------------
  const owner = new Int32Array(cells.length).fill(-1);
  const isLand = new Uint8Array(cells.length);
  const noiseScale = 0.042;
  for (let i = 0; i < cells.length; i += 1) {
    let best = Infinity;
    for (let f = 0; f < seeds.length; f += 1) {
      const dx = centroids[i].x - points[f][0];
      const dy = centroids[i].y - points[f][1];
      const pull = 0.65 + 0.7 * (regions[f].weight ?? 0.5);
      const d = Math.hypot(dx, dy) / pull;
      if (d < best) { best = d; owner[i] = f; }
    }
    const r = Math.hypot(centroids[i].x / halfW, centroids[i].y / halfH);
    const edge = cells[i].some(([x, y]) => Math.abs(Math.abs(x) - halfW) < 1e-3 || Math.abs(Math.abs(y) - halfH) < 1e-3);
    const shape = 1 - r / 0.86
      + coastAmp * (fbm(land, centroids[i].x * noiseScale, centroids[i].y * noiseScale, 4) - 0.5);
    isLand[i] = edge ? 0 : (shape > 0 ? 1 : 0);
  }
  // A folder always has ground to stand on.
  for (let f = 0; f < seeds.length; f += 1) {
    let nearest = 0;
    let best = Infinity;
    for (let i = 0; i < cells.length; i += 1) {
      const d = Math.hypot(centroids[i].x - points[f][0], centroids[i].y - points[f][1]);
      if (d < best) { best = d; nearest = i; }
    }
    isLand[nearest] = 1;
  }

  // --- edges, displaced once and shared by both sides --------------------------------
  const cache = new Map();
  const neighbourAt = (self, mx, my) => {
    let found = -1;
    let best = Infinity;
    for (let j = 0; j < points.length; j += 1) {
      if (j === self) continue;
      const d = Math.hypot(points[j][0] - mx, points[j][1] - my);
      if (d < best) { best = d; found = j; }
    }
    return found;
  };
  const levels = isle ? 3 : 4;
  const amplitude = isle ? 0.14 : 0.19;
  const edgeLine = (a, b, p, q) => {
    // Canonical orientation, so both cells build the identical polyline.
    const flip = p[0] > q[0] || (p[0] === q[0] && p[1] > q[1]);
    const key = a < b ? `${ids[a]}|${ids[b]}` : `${ids[b]}|${ids[a]}`;
    let line = cache.get(key);
    if (!line) {
      line = displaceEdge(key, flip ? q : p, flip ? p : q, levels, amplitude);
      cache.set(key, line);
    }
    return flip ? [...line].reverse() : line;
  };

  const outlines = [];
  const coast = [];
  const borders = [];
  for (let i = 0; i < cells.length; i += 1) {
    const poly = cells[i];
    const ring = [];
    for (let e = 0; e < poly.length; e += 1) {
      const p = poly[e];
      const q = poly[(e + 1) % poly.length];
      const mx = (p[0] + q[0]) / 2;
      const my = (p[1] + q[1]) / 2;
      const other = neighbourAt(i, mx, my);
      const shared = other >= 0
        && Math.abs(Math.hypot(points[other][0] - mx, points[other][1] - my)
          - Math.hypot(points[i][0] - mx, points[i][1] - my)) < step * 0.35;
      const line = shared ? edgeLine(i, other, p, q) : [p, q];
      for (let k = 0; k + 1 < line.length; k += 1) ring.push(line[k]);
      if (!isLand[i] || !shared) continue;
      if (!isLand[other]) coast.push(line);
      else if (owner[other] !== owner[i]) borders.push(line);
    }
    outlines.push(ring);
  }

  // --- elevation: how far inland, plus a second field --------------------------------
  const depth = new Int32Array(cells.length).fill(-1);
  const adjacency = cells.map(() => new Set());
  for (let i = 0; i < cells.length; i += 1) {
    const poly = cells[i];
    for (let e = 0; e < poly.length; e += 1) {
      const p = poly[e];
      const q = poly[(e + 1) % poly.length];
      const other = neighbourAt(i, (p[0] + q[0]) / 2, (p[1] + q[1]) / 2);
      if (other >= 0) adjacency[i].add(other);
    }
  }
  let frontier = [];
  for (let i = 0; i < cells.length; i += 1) if (!isLand[i]) { depth[i] = 0; frontier.push(i); }
  let maxDepth = 1;
  while (frontier.length) {
    const next = [];
    for (const i of frontier) {
      for (const j of adjacency[i]) {
        if (depth[j] >= 0) continue;
        depth[j] = depth[i] + 1;
        maxDepth = Math.max(maxDepth, depth[j]);
        next.push(j);
      }
    }
    frontier = next;
  }

  // --- what the renderer draws -------------------------------------------------------
  const tiles = [];
  const hatches = [];
  const byRegion = regions.map(() => ({ area: 0, x: 0, y: 0, d: '' }));
  for (let i = 0; i < cells.length; i += 1) {
    if (!isLand[i]) continue;
    const elevation = Math.min(1, ((depth[i] < 0 ? 1 : depth[i]) / maxDepth) * 0.78
      + 0.42 * fbm(relief, centroids[i].x * 0.09, centroids[i].y * 0.09, 2));
    const region = regions[owner[i]];
    const d = pathOf(outlines[i].map(toPct));
    tiles.push({ d, region: region?.key ?? '', elevation });
    const bucket = byRegion[owner[i]];
    if (bucket) {
      bucket.area += centroids[i].area;
      bucket.x += centroids[i].x * centroids[i].area;
      bucket.y += centroids[i].y * centroids[i].area;
      /* One path per region rather than one per cell: adjacent cells of the same region
         then merge under the non-zero fill rule and no antialiasing seam shows between
         them. The seams that remain are exactly the borders we draw on purpose. */
      bucket.d += d;
    }
    if (elevation > 0.6 && unit(id, 'hill', i) < 0.4) {
      const [hx, hy] = toPct([centroids[i].x, centroids[i].y]);
      const w = 0.9 + elevation * 0.8;
      hatches.push(`M${(hx - w).toFixed(2)} ${(hy + w * 0.45).toFixed(2)}L${hx.toFixed(2)} ${(hy - w * 0.5).toFixed(2)}L${(hx + w).toFixed(2)} ${(hy + w * 0.45).toFixed(2)}`);
    }
  }

  /* A district stands on its own generating point, nudged toward the middle of the ground
     it actually owns. Standing on the centroid alone would mean a new neighbour — which
     legitimately takes a few cells — visibly shoves the old district sideways; standing on
     the point alone can drop the label in a bay. A quarter of the way is both. */
  const placed = regions.map((region, f) => {
    const bucket = byRegion[f];
    const anchor = points[f];
    const [x, y] = bucket.area > 0
      ? toPct([
        anchor[0] + (bucket.x / bucket.area - anchor[0]) * 0.25,
        anchor[1] + (bucket.y / bucket.area - anchor[1]) * 0.25,
      ])
      : toPct(anchor);
    return { key: region.key, name: region.name, x, y, d: bucket.d, area: bucket.area };
  });

  /* The settlement wants quiet inland ground, not a folder's doorstep. */
  let siteIndex = -1;
  let siteScore = -Infinity;
  for (let i = 0; i < cells.length; i += 1) {
    if (!isLand[i]) continue;
    let nearest = Infinity;
    for (let f = 0; f < seeds.length; f += 1) {
      nearest = Math.min(nearest, Math.hypot(centroids[i].x - points[f][0], centroids[i].y - points[f][1]));
    }
    const score = 1.7 * (depth[i] / maxDepth) + Math.min(1, nearest / (step * 2.4)) * 0.7;
    if (score > siteScore) { siteScore = score; siteIndex = i; }
  }
  const site = siteIndex >= 0 ? toPct([centroids[siteIndex].x, centroids[siteIndex].y]) : [frame.cx, frame.cy];

  const ms = ((typeof performance === 'object' && performance.now) ? performance.now() : Date.now()) - started;
  return {
    id,
    isle,
    ms,
    // The frame the island actually filled — `isle` mode shrinks it, and the settlement
    // drawn on top has to shrink with it or the town is larger than the island.
    rx,
    ry,
    cellCount: cells.length,
    landCount: tiles.length,
    tiles,
    land: tiles.map((tile) => tile.d).join(''),
    coast: coast.map((line) => pathOf(line.map(toPct), false)),
    borders: borders.map((line) => pathOf(line.map(toPct), false)),
    hatches,
    regions: placed,
    site: { x: site[0], y: site[1] },
  };
}

/* Where each project's plate sits on the shared sea. One project gets the whole plate;
   more share it, because a mounted project is an island and several are an archipelago. */
const FRAMES = [
  [{ cx: 50, cy: 52, rx: 33, ry: 31 }],
  [{ cx: 27, cy: 52, rx: 21, ry: 29 }, { cx: 73, cy: 52, rx: 21, ry: 29 }],
  [{ cx: 28, cy: 35, rx: 18, ry: 21 }, { cx: 72, cy: 34, rx: 18, ry: 21 }, { cx: 50, cy: 74, rx: 18, ry: 20 }],
  [{ cx: 27, cy: 33, rx: 16, ry: 19 }, { cx: 71, cy: 31, rx: 16, ry: 19 },
    { cx: 26, cy: 74, rx: 16, ry: 18 }, { cx: 73, cy: 73, rx: 16, ry: 18 }],
  [{ cx: 50, cy: 30, rx: 15, ry: 18 }, { cx: 17, cy: 40, rx: 13, ry: 16 }, { cx: 84, cy: 38, rx: 13, ry: 16 },
    { cx: 28, cy: 77, rx: 13, ry: 15 }, { cx: 73, cy: 76, rx: 13, ry: 15 }],
];

export function frameFor(count, index) {
  const row = FRAMES[Math.min(Math.max(count, 1), FRAMES.length) - 1];
  return row[Math.min(index, row.length - 1)];
}

/* Generating a plate is cheap but not free, and nothing about it changes when an event
   arrives. It is cached on the seed and the shape of the folder list — not on weights
   rounded to nothing, which would redraw the island every time a file changed. */
const cache = new Map();
const CACHE_LIMIT = 24;

export function islandFor({ id, regions, frame }) {
  const signature = `${id}|${frame.cx},${frame.cy},${frame.rx},${frame.ry}|${
    regions.map((r) => `${r.key}:${(r.weight ?? 0.5).toFixed(1)}`).join(',')}`;
  const hit = cache.get(signature);
  if (hit) return hit;
  const made = generateIsland({ id, regions, frame });
  cache.set(signature, made);
  if (cache.size > CACHE_LIMIT) cache.delete(cache.keys().next().value);
  return made;
}
