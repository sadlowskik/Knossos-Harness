// Canvas renderer for the Field.
// Motion in this file comes from exactly two sources: the operator moving the camera,
// and real events arriving. There is no idle animation.

const C = {
  bg: '#131210',
  grid: 'rgba(255,255,255,0.030)',
  regionFill: 'rgba(255,255,255,0.013)',
  regionStroke: '#2E2B26',
  regionStrokeHot: '#4A463E',
  line: '#302D28',
  line2: '#45413A',
  ink: '#F6F4F2',
  body: '#D8D2C8',
  muted: '#A39C91',
  faint: '#7C776D',
  ghost: '#5A554B',
  ember: '#FF7A1A',
  emberDim: '#B4571A',
  amber: '#FFD08A',
  rust: '#B4472F',
  verify: '#6FA88A',
  steel: '#7E8B94',
};

const STATE_COLOR = {
  spawning: C.ghost,
  ready: C.steel,
  idle: C.faint,
  thinking: C.amber,
  working: C.ember,
  waiting_permission: C.amber,
  blocked: C.rust,
  error: C.rust,
  interrupted: C.ghost,
  paused: C.steel,
  done: C.verify,
  cancelled: C.ghost,
};

export const MINIMAP = { w: 176, h: 118, margin: 14 };

export function makeCamera() { return { x: 0, y: 0, z: 0.85 }; }

export function toScreen(cam, view, wx, wy) {
  return [(wx - cam.x) * cam.z + view.w / 2, (wy - cam.y) * cam.z + view.h / 2];
}

export function toWorld(cam, view, sx, sy) {
  return [(sx - view.w / 2) / cam.z + cam.x, (sy - view.h / 2) / cam.z + cam.y];
}

export function minimapRect(view) {
  return {
    x: view.w - MINIMAP.w - MINIMAP.margin,
    y: view.h - MINIMAP.h - MINIMAP.margin,
    w: MINIMAP.w, h: MINIMAP.h,
  };
}

function mix(a, b, t) {
  const pa = parseInt(a.slice(1), 16); const pb = parseInt(b.slice(1), 16);
  const r = Math.round((pa >> 16) * (1 - t) + (pb >> 16) * t);
  const g = Math.round(((pa >> 8) & 255) * (1 - t) + ((pb >> 8) & 255) * t);
  const bl = Math.round((pa & 255) * (1 - t) + (pb & 255) * t);
  return `rgb(${r},${g},${bl})`;
}

function rr(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, r);
}

/**
 * Fit a label to the space available, ending it with an ellipsis rather than letting it
 * run off the edge of its chip. Returns null when there is not enough room to say
 * anything meaningful, in which case the caller should draw no text at all.
 */
function fitText(ctx, text, maxWidth) {
  if (maxWidth < 14) return null;
  if (ctx.measureText(text).width <= maxWidth) return text;
  let lo = 0; let hi = text.length;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (ctx.measureText(text.slice(0, mid) + '…').width <= maxWidth) lo = mid;
    else hi = mid - 1;
  }
  return lo >= 2 ? text.slice(0, lo) + '…' : null;
}

// ---------------------------------------------------------------- unit glyphs

const SHAPES = {
  architect: (ctx, x, y, s) => {           // diamond
    ctx.beginPath();
    ctx.moveTo(x, y - s); ctx.lineTo(x + s, y); ctx.lineTo(x, y + s); ctx.lineTo(x - s, y);
    ctx.closePath();
  },
  builder: (ctx, x, y, s) => {             // square
    ctx.beginPath();
    ctx.roundRect(x - s * 0.82, y - s * 0.82, s * 1.64, s * 1.64, 1.5);
  },
  scout: (ctx, x, y, s) => {               // forward triangle
    ctx.beginPath();
    ctx.moveTo(x, y - s); ctx.lineTo(x + s * 0.92, y + s * 0.72); ctx.lineTo(x - s * 0.92, y + s * 0.72);
    ctx.closePath();
  },
  verifier: (ctx, x, y, s) => {            // hexagon
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
      const a = (Math.PI / 3) * i - Math.PI / 2;
      const px = x + Math.cos(a) * s; const py = y + Math.sin(a) * s;
      i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    }
    ctx.closePath();
  },
  archivist: (ctx, x, y, s) => {           // circle
    ctx.beginPath();
    ctx.arc(x, y, s * 0.88, 0, Math.PI * 2);
  },
};

function shapeFor(role) { return SHAPES[role] ?? SHAPES.builder; }

// ---------------------------------------------------------------- main draw

export function draw(ctx, opts) {
  const {
    layout, cam, view, selection, hover, marquee, now,
    pulses, showLabels, contextTarget,
  } = opts;

  const sel = new Set(selection);

  ctx.save();
  ctx.fillStyle = C.bg;
  ctx.fillRect(0, 0, view.w, view.h);

  drawGrid(ctx, cam, view);

  // regions -------------------------------------------------------------
  for (const r of layout.regions) {
    const [sx, sy] = toScreen(cam, view, r.x, r.y);
    const w = r.w * cam.z; const h = r.h * cam.z;

    rr(ctx, sx, sy, w, h, 6);
    ctx.fillStyle = C.regionFill;
    ctx.fill();
    ctx.lineWidth = 1;
    if (!r.mounted) {
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = C.rust;
    } else {
      ctx.strokeStyle = r.heat > 0.15 ? C.regionStrokeHot : C.regionStroke;
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // A hot workspace gets an ember seam along its top edge, scaled to real churn.
    if (r.heat > 0.02 && r.mounted) {
      ctx.beginPath();
      ctx.moveTo(sx + 6, sy + 0.5);
      ctx.lineTo(sx + 6 + (w - 12) * Math.min(1, r.heat), sy + 0.5);
      ctx.strokeStyle = C.ember;
      ctx.globalAlpha = 0.25 + r.heat * 0.5;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    if (cam.z > 0.28) {
      ctx.font = '600 10px "Space Grotesk", sans-serif';
      ctx.fillStyle = r.mounted ? C.muted : C.rust;
      ctx.textBaseline = 'alphabetic';
      ctx.letterSpacing = '0.11em';
      ctx.fillText(r.name.toUpperCase(), sx + 14, sy + 21);
      ctx.letterSpacing = '0px';

      ctx.font = '400 9px "IBM Plex Mono", monospace';
      ctx.fillStyle = C.ghost;
      const bits = [];
      if (!r.mounted) bits.push('NOT MOUNTED');
      else if (r.git) bits.push(`${r.git.branch ?? '—'} · ${r.git.files.length} changed`);
      if (r.hiddenFolders > 0) bits.push(`+${r.hiddenFolders} quieter`);
      const meta = bits.join('  ') || '—';
      const mw = ctx.measureText(meta).width;
      ctx.fillText(meta, sx + w - 14 - mw, sy + 21);
    }
  }

  // folders -------------------------------------------------------------
  for (const f of layout.folders) {
    const [sx, sy] = toScreen(cam, view, f.x, f.y);
    const w = f.w * cam.z; const h = f.h * cam.z;
    if (w < 6) continue;

    rr(ctx, sx, sy, w, h, 2.5);
    ctx.fillStyle = f.heat > 0.5
      ? `rgba(255,122,26,${0.05 + f.heat * 0.10})`
      : `rgba(255,255,255,${0.020 + f.heat * 0.045})`;
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.strokeStyle = f.heat > 0.45 ? mix(C.line2, C.emberDim, f.heat) : C.line;
    ctx.stroke();

    if (cam.z > 0.34) {
      ctx.font = '400 9px "IBM Plex Mono", monospace';
      const label = fitText(ctx, f.label, w - 11);
      if (label) {
        ctx.fillStyle = f.heat > 0.5 ? C.body : C.faint;
        ctx.textBaseline = 'middle';
        ctx.fillText(label, sx + 6, sy + h / 2 + 0.5);
      }
    }

    // heat bar along the bottom edge
    if (f.heat > 0.04) {
      ctx.beginPath();
      ctx.moveTo(sx + 1, sy + h - 0.5);
      ctx.lineTo(sx + 1 + (w - 2) * f.heat, sy + h - 0.5);
      ctx.strokeStyle = C.ember;
      ctx.globalAlpha = 0.35 + f.heat * 0.45;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
  }

  // file change ticks ---------------------------------------------------
  for (const t of layout.fileTicks) {
    const [sx, sy] = toScreen(cam, view, t.x, t.y);
    ctx.beginPath();
    ctx.arc(sx, sy, 1.9, 0, Math.PI * 2);
    ctx.fillStyle = t.change === 'unlink' ? C.rust : t.change === 'add' ? C.verify : C.amber;
    ctx.globalAlpha = 0.25 + t.strength * 0.75;
    ctx.fill();
    ctx.globalAlpha = 1;
  }

  // mission artifacts ---------------------------------------------------
  for (const m of layout.missions ?? []) {
    const [sx, sy] = toScreen(cam, view, m.x, m.y);
    const w = m.w * cam.z; const h = m.h * cam.z;
    if (w < 20) continue;

    rr(ctx, sx, sy, w, h, 2);
    ctx.fillStyle = m.active ? 'rgba(255,122,26,0.09)' : 'rgba(255,255,255,0.016)';
    ctx.fill();
    ctx.strokeStyle = m.active ? 'rgba(255,122,26,0.42)' : C.line;
    ctx.lineWidth = 1;
    ctx.stroke();

    // The tick marks it as an instruction document rather than a place.
    ctx.beginPath();
    ctx.moveTo(sx + 5, sy + 4);
    ctx.lineTo(sx + 5, sy + h - 4);
    ctx.strokeStyle = m.active ? C.ember : C.line2;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    if (cam.z > 0.34) {
      ctx.font = '400 9px "IBM Plex Mono", monospace';
      const label = fitText(ctx, m.name, w - 16);
      if (label) {
        ctx.fillStyle = m.active ? C.amber : C.ghost;
        ctx.textBaseline = 'middle';
        ctx.fillText(label, sx + 11, sy + h / 2 + 0.5);
      }
    }
  }

  // websites ------------------------------------------------------------
  for (const s of layout.sites) {
    const [sx, sy] = toScreen(cam, view, s.x, s.y);
    const w = s.w * cam.z; const h = s.h * cam.z;
    rr(ctx, sx, sy, w, h, 4);
    ctx.fillStyle = 'rgba(126,139,148,0.045)';
    ctx.fill();
    ctx.strokeStyle = s.sessions.length ? C.steel : C.line;
    ctx.lineWidth = 1;
    ctx.stroke();

    if (cam.z > 0.34) {
      ctx.font = '400 9.5px "IBM Plex Mono", monospace';
      ctx.fillStyle = s.sessions.length ? C.body : C.faint;
      ctx.textBaseline = 'middle';
      const label = fitText(ctx, s.domain, w - 26);
      if (label) ctx.fillText(label, sx + 10, sy + h / 2);
      if (s.sessions.length) {
        ctx.beginPath();
        ctx.arc(sx + w - 10, sy + h / 2, 2.5, 0, Math.PI * 2);
        ctx.fillStyle = C.steel;
        ctx.fill();
      }
    }
  }

  // routes --------------------------------------------------------------
  for (const rt of layout.routes) {
    const [x1, y1] = toScreen(cam, view, rt.x1, rt.y1);
    const [x2, y2] = toScreen(cam, view, rt.x2, rt.y2);
    const pulse = pulses[rt.id] ?? 0;
    const selected = sel.has(rt.id);

    const mx = (x1 + x2) / 2;
    const my = (y1 + y2) / 2 - Math.min(40, Math.abs(x2 - x1) * 0.16);

    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.quadraticCurveTo(mx, my, x2, y2);

    if (rt.kind === 'delegate') { ctx.setLineDash([3, 3]); ctx.strokeStyle = C.steel; }
    else if (rt.kind === 'communication') { ctx.setLineDash([5, 5]); ctx.strokeStyle = C.amber; }
    else if (rt.kind === 'browser') { ctx.setLineDash([1.5, 3.5]); ctx.strokeStyle = C.steel; }
    else { ctx.setLineDash([]); ctx.strokeStyle = C.ember; }

    ctx.globalAlpha = (selected ? 0.34 : 0.13) + pulse * 0.40;
    ctx.lineWidth = selected ? 1.3 : 1;
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }

  // agents --------------------------------------------------------------
  for (const a of layout.agents) {
    drawAgent(ctx, a, {
      cam, view, selected: sel.has(a.id), hovered: hover === a.id,
      pulse: pulses[a.id] ?? 0, now, showLabels,
    });
  }

  // marquee -------------------------------------------------------------
  if (marquee) {
    const x = Math.min(marquee.x1, marquee.x2);
    const y = Math.min(marquee.y1, marquee.y2);
    const w = Math.abs(marquee.x2 - marquee.x1);
    const h = Math.abs(marquee.y2 - marquee.y1);
    ctx.fillStyle = 'rgba(255,122,26,0.07)';
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = C.ember;
    ctx.globalAlpha = 0.7;
    ctx.lineWidth = 1;
    ctx.strokeRect(x + 0.5, y + 0.5, w, h);
    ctx.globalAlpha = 1;
  }

  // right-click target highlight ----------------------------------------
  if (contextTarget) {
    const [sx, sy] = toScreen(cam, view, contextTarget.x, contextTarget.y);
    ctx.beginPath();
    ctx.arc(sx, sy, 15, 0, Math.PI * 2);
    ctx.strokeStyle = C.ember;
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 3]);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  drawMinimap(ctx, layout, cam, view, sel);

  ctx.restore();
}

function drawGrid(ctx, cam, view) {
  if (cam.z < 0.35) return;
  const step = 80 * cam.z;
  const ox = ((-cam.x * cam.z + view.w / 2) % step + step) % step;
  const oy = ((-cam.y * cam.z + view.h / 2) % step + step) % step;
  ctx.strokeStyle = C.grid;
  ctx.lineWidth = 1;
  ctx.globalAlpha = Math.min(1, (cam.z - 0.35) * 2.2);
  ctx.beginPath();
  for (let x = ox; x < view.w; x += step) { ctx.moveTo(Math.round(x) + 0.5, 0); ctx.lineTo(Math.round(x) + 0.5, view.h); }
  for (let y = oy; y < view.h; y += step) { ctx.moveTo(0, Math.round(y) + 0.5); ctx.lineTo(view.w, Math.round(y) + 0.5); }
  ctx.stroke();
  ctx.globalAlpha = 1;
}

function drawAgent(ctx, a, o) {
  const { cam, view, selected, hovered, pulse, now, showLabels } = o;
  const s = a.session;
  const [x, y] = toScreen(cam, view, a.x, a.y);
  const size = Math.max(4.5, 7.5 * Math.min(1.35, cam.z));
  const color = STATE_COLOR[s.state] ?? C.faint;
  const shape = shapeFor(s.role);

  // A real pending permission is the one thing that pulses on its own, because it
  // is a live request blocking a real process.
  if (s.state === 'waiting_permission') {
    const t = (now % 1200) / 1200;
    ctx.beginPath();
    ctx.arc(x, y, size + 4 + t * 9, 0, Math.PI * 2);
    ctx.strokeStyle = C.amber;
    ctx.globalAlpha = 0.5 * (1 - t);
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  if (pulse > 0) {
    ctx.beginPath();
    ctx.arc(x, y, size + 3 + (1 - pulse) * 7, 0, Math.PI * 2);
    ctx.strokeStyle = color;
    ctx.globalAlpha = pulse * 0.42;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  if (selected || hovered) {
    ctx.beginPath();
    ctx.arc(x, y, size + 6, 0, Math.PI * 2);
    ctx.strokeStyle = C.ember;
    ctx.globalAlpha = selected ? 0.95 : 0.4;
    ctx.lineWidth = selected ? 1.4 : 1;
    ctx.stroke();
    ctx.globalAlpha = 1;

    if (selected) {
      const r = size + 6;
      ctx.strokeStyle = C.ember;
      ctx.lineWidth = 1.4;
      for (const [dx, dy] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
        ctx.beginPath();
        ctx.moveTo(x + dx * r, y + dy * r - dy * 3.5);
        ctx.lineTo(x + dx * r, y + dy * r);
        ctx.lineTo(x + dx * r - dx * 3.5, y + dy * r);
        ctx.stroke();
      }
    }
  }

  shape(ctx, x, y, size);
  ctx.fillStyle = s.state === 'interrupted' || s.state === 'spawning' ? 'transparent' : color;
  ctx.fill();
  ctx.lineWidth = 1.2;
  ctx.strokeStyle = s.state === 'interrupted' ? C.ghost : mix(color, '#000000', 0.42);
  ctx.stroke();

  if (s.simulated) {
    ctx.beginPath();
    ctx.arc(x, y, size + 3.5, 0, Math.PI * 2);
    ctx.setLineDash([2, 2]);
    ctx.strokeStyle = C.steel;
    ctx.globalAlpha = 0.72;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.globalAlpha = 1;
  }

  // Subagents the agent created itself. Field knows they exist and what they were asked
  // for, but not their internals, so they are satellites rather than units.
  const delegations = s.delegations ?? [];
  if (delegations.length && cam.z > 0.4) {
    const shown = Math.min(delegations.length, 4);
    for (let i = 0; i < shown; i++) {
      const a = -Math.PI / 2 + (i - (shown - 1) / 2) * 0.5;
      const dx = x + Math.cos(a) * (size + 11);
      const dy = y + Math.sin(a) * (size + 11);
      ctx.beginPath();
      ctx.moveTo(x + Math.cos(a) * (size + 2), y + Math.sin(a) * (size + 2));
      ctx.lineTo(dx, dy);
      ctx.strokeStyle = C.steel;
      ctx.globalAlpha = 0.45;
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.globalAlpha = 1;

      ctx.beginPath();
      ctx.arc(dx, dy, 2.1, 0, Math.PI * 2);
      ctx.fillStyle = C.steel;
      ctx.fill();
    }
  }

  if (s.verified === 'verified') {
    ctx.beginPath();
    ctx.arc(x + size + 3, y - size - 1, 2.4, 0, Math.PI * 2);
    ctx.fillStyle = C.verify;
    ctx.fill();
  } else if (s.verified === 'rejected') {
    ctx.beginPath();
    ctx.arc(x + size + 3, y - size - 1, 2.4, 0, Math.PI * 2);
    ctx.fillStyle = C.rust;
    ctx.fill();
  }

  if (cam.z < 0.45) return;

  // status bars above the unit: context, then progress when the agent has a plan
  const bw = 24; const bx = x - bw / 2; let by = y - size - 9;

  ctx.fillStyle = 'rgba(255,255,255,0.10)';
  ctx.fillRect(bx, by, bw, 2);
  const ctxPct = Math.min(1, (s.contextPct ?? 0) / 100);
  ctx.fillStyle = ctxPct > 0.85 ? C.rust : ctxPct > 0.6 ? C.amber : C.steel;
  ctx.fillRect(bx, by, bw * ctxPct, 2);

  if (s.progress?.total) {
    by -= 3.5;
    ctx.fillStyle = 'rgba(255,255,255,0.10)';
    ctx.fillRect(bx, by, bw, 2);
    ctx.fillStyle = C.ember;
    ctx.fillRect(bx, by, bw * (s.progress.done / s.progress.total), 2);
  }

  if (showLabels || selected || hovered || s.simulated) {
    ctx.font = '500 9px "IBM Plex Mono", monospace';
    ctx.textBaseline = 'top';
    ctx.fillStyle = selected ? C.ink : C.muted;
    const label = s.name ?? s.id.slice(0, 6);
    const lw = ctx.measureText(label).width;
    ctx.fillText(label, x - lw / 2, y + size + 5);
  }
}

function drawMinimap(ctx, layout, cam, view, sel) {
  const box = minimapRect(view);
  const b = layout.bounds;
  const bw = b.maxX - b.minX; const bh = b.maxY - b.minY;
  const scale = Math.min((box.w - 12) / bw, (box.h - 12) / bh);
  const ox = box.x + 6 + ((box.w - 12) - bw * scale) / 2;
  const oy = box.y + 6 + ((box.h - 12) - bh * scale) / 2;
  const mx = (wx) => ox + (wx - b.minX) * scale;
  const my = (wy) => oy + (wy - b.minY) * scale;

  rr(ctx, box.x, box.y, box.w, box.h, 5);
  ctx.fillStyle = 'rgba(12,11,9,0.86)';
  ctx.fill();
  ctx.strokeStyle = '#302D28';
  ctx.lineWidth = 1;
  ctx.stroke();

  for (const r of layout.regions) {
    ctx.fillStyle = 'rgba(255,255,255,0.05)';
    ctx.fillRect(mx(r.x), my(r.y), r.w * scale, r.h * scale);
    ctx.strokeStyle = r.heat > 0.15 ? 'rgba(255,122,26,0.5)' : '#3A362F';
    ctx.lineWidth = 0.8;
    ctx.strokeRect(mx(r.x), my(r.y), r.w * scale, r.h * scale);
  }
  for (const s of layout.sites) {
    ctx.fillStyle = 'rgba(126,139,148,0.22)';
    ctx.fillRect(mx(s.x), my(s.y), s.w * scale, s.h * scale);
  }
  for (const a of layout.agents) {
    ctx.beginPath();
    ctx.arc(mx(a.x), my(a.y), sel.has(a.id) ? 2.2 : 1.5, 0, Math.PI * 2);
    ctx.fillStyle = sel.has(a.id) ? C.ember : (STATE_COLOR[a.session.state] ?? C.faint);
    ctx.fill();
  }

  // viewport rectangle
  const [wx1, wy1] = toWorld(cam, view, 0, 0);
  const [wx2, wy2] = toWorld(cam, view, view.w, view.h);
  ctx.strokeStyle = 'rgba(255,255,255,0.32)';
  ctx.lineWidth = 1;
  ctx.strokeRect(mx(wx1), my(wy1), (wx2 - wx1) * scale, (wy2 - wy1) * scale);

  return box;
}

export function minimapToWorld(layout, view, sx, sy) {
  const box = minimapRect(view);
  const b = layout.bounds;
  const bw = b.maxX - b.minX; const bh = b.maxY - b.minY;
  const scale = Math.min((box.w - 12) / bw, (box.h - 12) / bh);
  const ox = box.x + 6 + ((box.w - 12) - bw * scale) / 2;
  const oy = box.y + 6 + ((box.h - 12) - bh * scale) / 2;
  return [b.minX + (sx - ox) / scale, b.minY + (sy - oy) / scale];
}
