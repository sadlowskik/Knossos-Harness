import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { computeLayout, pulseOf } from './layout.js';
import { draw, minimapRect, minimapToWorld, toScreen, toWorld } from './renderer.js';
import {
  addSelection, clearSelection, getState, openChanges, openInWorkspace, selectOnly, setState, toggleSelection, useField,
} from '../state/store.js';
import { api } from '../net/client.js';
import ContextMenu from '../hud/ContextMenu.jsx';
import SelectionHUD from '../hud/SelectionHUD.jsx';
import UnitInspector from './UnitInspector.jsx';
import EndpointRail from '../hud/EndpointRail.jsx';
import PermissionRequests from '../hud/PermissionRequests.jsx';
import PowerSources from '../setup/PowerSources.jsx';

const HIT_RADIUS = 13;

export default function FieldMode() {
  const st = useField();
  const canvasRef = useRef(null);
  const wrapRef = useRef(null);
  const [view, setView] = useState({ w: 800, h: 600 });
  const [marquee, setMarquee] = useState(null);
  const [menu, setMenu] = useState(null);
  const [hover, setHover] = useState(null);
  const [showLabels, setShowLabels] = useState(false);
  const [inspectId, setInspectId] = useState(null);

  const camRef = useRef(st.camera);
  const dirtyRef = useRef(true);
  const dragRef = useRef(null);
  const nowRef = useRef(Date.now());
  // Rendered unit positions, eased toward their posted layout coordinate so a unit
  // visibly walks to a new file/folder rather than teleporting.
  const posRef = useRef({});
  const animLayoutRef = useRef(null);

  const reservedUsd = (st.snap.budgetReservations ?? [])
    .reduce((sum, row) => sum + (row.reservedUsd ?? 0), 0);

  const layout = useMemo(
    () => computeLayout(st.snap, st.snap.positions ?? {}, Date.now(), st.config),
    [st.snap, st.config],
  );
  const inspectSession = inspectId
    ? st.snap.sessions.find((s) => s.id === inspectId) ?? null
    : null;
  const [modelsOpen, setModelsOpen] = useState(false);
  const workspaces = st.snap.workspaces.filter((w) => w.mounted);
  const liveSessions = st.snap.sessions.filter((s) => !TERMINAL.has(s.state));
  const hasModel = (st.snap.endpoints?.length || st.config?.endpoints?.length) > 0;

  // Open the start-an-agent dialog for a project, anchored to the control that asked.
  const startAgent = (workspace, e) => {
    const rect = wrapRef.current?.getBoundingClientRect();
    const x = e?.clientX != null ? e.clientX - (rect?.left ?? 0) : view.w / 2 - 140;
    const y = e?.clientY != null ? e.clientY - (rect?.top ?? 0) + 8 : view.h / 2 - 160;
    setMenu({
      screen: { x, y },
      pane: 'spawn',
      target: { type: 'workspace', id: workspace.id, workspaceId: workspace.id, label: workspace.name },
      world: null,
    });
  };
  const layoutRef = useRef(layout);
  layoutRef.current = layout;
  dirtyRef.current = true;

  // ---------------------------------------------------------------- sizing

  useLayoutEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect();
      setView({ w: Math.max(1, r.width), h: Math.max(1, r.height) });
      dirtyRef.current = true;
    });
    ro.observe(el);
    const r = el.getBoundingClientRect();
    setView({ w: Math.max(1, r.width), h: Math.max(1, r.height) });
    return () => ro.disconnect();
  }, []);

  // Keep the whole field framed until the operator takes the camera. Framing on the
  // first layout alone gets it wrong, because the viewport has not been measured yet.
  const userMovedRef = useRef(false);
  useEffect(() => {
    if (userMovedRef.current || !layout.regions.length || view.w < 300) return;
    const b = layout.bounds;
    const z = Math.min(1.1, Math.min(view.w / (b.maxX - b.minX), view.h / (b.maxY - b.minY)) * 0.92);
    camRef.current = { x: (b.minX + b.maxX) / 2, y: (b.minY + b.maxY) / 2, z: Math.max(0.2, z) };
    dirtyRef.current = true;
  }, [layout, view]);

  // ---------------------------------------------------------------- render loop

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let raf = 0;
    let lastDraw = 0;

    const frame = () => {
      raf = requestAnimationFrame(frame);
      const now = Date.now();
      nowRef.current = now;
      const s = getState();
      const lay = layoutRef.current;

      const pulses = {};
      let animating = false;
      for (const [id, ts] of Object.entries(s.lastEventBySession)) {
        const p = pulseOf(ts, now);
        if (p > 0) { pulses[id] = p; animating = true; }
      }
      if (lay.agents.some((a) => a.session.state === 'waiting_permission')) animating = true;

      // --- movement -----------------------------------------------------
      // Each layout carries the true target coordinates. Capture them once per layout,
      // then ease the rendered position toward the target every frame while it differs.
      // A unit that has arrived is pinned exactly, so a settled Field never animates.
      const pos = posRef.current;
      if (animLayoutRef.current !== lay) {
        animLayoutRef.current = lay;
        const liveIds = new Set(lay.agents.map((a) => a.id));
        for (const id of Object.keys(pos)) if (!liveIds.has(id)) delete pos[id];
        for (const a of lay.agents) {
          const cur = pos[a.id];
          if (cur) { cur.tx = a.x; cur.ty = a.y; }
          else pos[a.id] = { x: a.x, y: a.y, tx: a.x, ty: a.y };
        }
      }
      const EASE = 0.2;
      for (const a of lay.agents) {
        const cur = pos[a.id];
        if (!cur) continue;
        const dx = cur.tx - cur.x; const dy = cur.ty - cur.y;
        if (dx * dx + dy * dy < 0.25) { cur.x = cur.tx; cur.y = cur.ty; }
        else { cur.x += dx * EASE; cur.y += dy * EASE; animating = true; }
        a.x = cur.x; a.y = cur.y;   // draw and hit-test follow the visible position
      }

      // A file flashes when the unit posted on it takes an action there.
      const filePulses = {};
      const fbk = lay.fileByKey;
      if (fbk) {
        for (const a of lay.agents) {
          const p = pulses[a.id];
          const sess = a.session;
          if (!p || sess.workspaceId == null || sess.focusPath == null) continue;
          const node = fbk.get(`${sess.workspaceId}:${sess.focusPath}`);
          if (node) filePulses[node.key] = Math.max(filePulses[node.key] ?? 0, p);
        }
      }

      // Heat decays with real time, so refresh it at a low rate rather than per frame.
      const heatPresent = lay.folders.length > 0 || lay.regions.some((r) => r.heat > 0.02);
      const heatDue = heatPresent && now - lastDraw > 250;

      if (!dirtyRef.current && !animating && !heatDue) return;
      dirtyRef.current = false;
      lastDraw = now;

      const dpr = Math.min(2, window.devicePixelRatio || 1);
      if (canvas.width !== Math.round(view.w * dpr) || canvas.height !== Math.round(view.h * dpr)) {
        canvas.width = Math.round(view.w * dpr);
        canvas.height = Math.round(view.h * dpr);
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      draw(ctx, {
        layout: lay,
        cam: camRef.current,
        view,
        selection: s.selection,
        hover,
        marquee,
        now,
        pulses,
        filePulses,
        showLabels,
        contextTarget: menu?.world ?? null,
      });
    };

    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [view, hover, marquee, showLabels, menu]);

  // ---------------------------------------------------------------- hit testing

  const hitTest = useCallback((sx, sy) => {
    const cam = camRef.current;
    const lay = layoutRef.current;

    for (const a of lay.agents) {
      const [x, y] = toScreen(cam, view, a.x, a.y);
      if ((x - sx) ** 2 + (y - sy) ** 2 < HIT_RADIUS ** 2) {
        return { type: 'agent', id: a.id, session: a.session, world: { x: a.x, y: a.y } };
      }
    }
    // Files sit under their folders and are smaller, so they are tested first.
    for (const f of lay.files ?? []) {
      const [x, y] = toScreen(cam, view, f.x, f.y);
      const rr = f.r * cam.z + 5;
      if ((x - sx) ** 2 + (y - sy) ** 2 < rr ** 2) {
        return {
          type: 'file', id: f.path, workspaceId: f.workspaceId, label: f.path,
          world: { x: f.x, y: f.y },
        };
      }
    }
    for (const f of lay.folders) {
      const [x, y] = toScreen(cam, view, f.x, f.y);
      if (sx >= x && sx <= x + f.w * cam.z && sy >= y && sy <= y + f.h * cam.z) {
        return {
          type: 'folder', id: f.full, workspaceId: f.workspaceId, label: f.full || '/',
          world: { x: f.x + f.w / 2, y: f.y + f.h / 2 },
        };
      }
    }
    for (const m of lay.missions ?? []) {
      const [x, y] = toScreen(cam, view, m.x, m.y);
      if (sx >= x && sx <= x + m.w * cam.z && sy >= y && sy <= y + m.h * cam.z) {
        return {
          type: 'mission', id: m.id, workspaceId: m.workspaceId, label: m.name,
          world: { x: m.x + m.w / 2, y: m.y + m.h / 2 },
        };
      }
    }
    for (const s of lay.sites) {
      const [x, y] = toScreen(cam, view, s.x, s.y);
      if (sx >= x && sx <= x + s.w * cam.z && sy >= y && sy <= y + s.h * cam.z) {
        return {
          type: 'website', id: s.domain, label: s.domain, url: s.lastUrl,
          world: { x: s.x + s.w / 2, y: s.y + s.h / 2 },
        };
      }
    }
    for (const r of lay.regions) {
      const [x, y] = toScreen(cam, view, r.x, r.y);
      if (sx >= x && sx <= x + r.w * cam.z && sy >= y && sy <= y + r.h * cam.z) {
        return {
          type: 'workspace', id: r.id, workspaceId: r.id, label: r.name,
          world: { x: r.x + r.w / 2, y: r.y + r.h / 2 },
        };
      }
    }
    return { type: 'empty', world: { x: 0, y: 0 } };
  }, [view]);

  const localPoint = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    return [e.clientX - rect.left, e.clientY - rect.top];
  };

  // ---------------------------------------------------------------- pointer

  const onPointerDown = (e) => {
    const [sx, sy] = localPoint(e);
    canvasRef.current.setPointerCapture(e.pointerId);
    setMenu(null);

    const inMinimap = (() => {
      const b = minimapRect(view);
      return sx >= b.x && sx <= b.x + b.w && sy >= b.y && sy <= b.y + b.h;
    })();

    if (inMinimap && e.button === 0) {
      const [wx, wy] = minimapToWorld(layoutRef.current, view, sx, sy);
      camRef.current = { ...camRef.current, x: wx, y: wy };
      dirtyRef.current = true;
      userMovedRef.current = true;
      dragRef.current = { kind: 'minimap' };
      return;
    }

    if (e.button === 1 || e.button === 0 && e.altKey) {
      userMovedRef.current = true;
      dragRef.current = { kind: 'pan', sx, sy, cam: { ...camRef.current } };
      return;
    }

    // The secondary button is handled in onContextMenu, which is the canonical event
    // for it and fires in environments that never synthesize a button-2 pointerdown.
    if (e.button !== 0) return;

    const hit = hitTest(sx, sy);
    if (hit.type === 'agent') {
      if (e.shiftKey) toggleSelection(hit.id);
      else selectOnly([hit.id]);
      dragRef.current = { kind: 'click' };
      return;
    }
    dragRef.current = { kind: 'marquee', x1: sx, y1: sy, additive: e.shiftKey };
    setMarquee({ x1: sx, y1: sy, x2: sx, y2: sy });
  };

  const onPointerMove = (e) => {
    const [sx, sy] = localPoint(e);
    const d = dragRef.current;

    if (!d) {
      const hit = hitTest(sx, sy);
      const id = hit.type === 'agent' ? hit.id : null;
      if (id !== hover) { setHover(id); dirtyRef.current = true; }
      return;
    }

    if (d.kind === 'pan') {
      camRef.current = {
        ...d.cam,
        x: d.cam.x - (sx - d.sx) / d.cam.z,
        y: d.cam.y - (sy - d.sy) / d.cam.z,
      };
      dirtyRef.current = true;
    } else if (d.kind === 'minimap') {
      const [wx, wy] = minimapToWorld(layoutRef.current, view, sx, sy);
      camRef.current = { ...camRef.current, x: wx, y: wy };
      dirtyRef.current = true;
    } else if (d.kind === 'marquee') {
      setMarquee({ x1: d.x1, y1: d.y1, x2: sx, y2: sy });
    }
  };

  const onPointerUp = (e) => {
    const d = dragRef.current;
    dragRef.current = null;
    if (!d) return;

    if (d.kind === 'marquee') {
      const [sx, sy] = localPoint(e);
      const moved = Math.abs(sx - d.x1) > 3 || Math.abs(sy - d.y1) > 3;
      if (moved) {
        const cam = camRef.current;
        const x1 = Math.min(d.x1, sx); const x2 = Math.max(d.x1, sx);
        const y1 = Math.min(d.y1, sy); const y2 = Math.max(d.y1, sy);
        const picked = layoutRef.current.agents.filter((a) => {
          const [x, y] = toScreen(cam, view, a.x, a.y);
          return x >= x1 && x <= x2 && y >= y1 && y <= y2;
        }).map((a) => a.id);
        if (d.additive) addSelection(picked);
        else selectOnly(picked);
      } else if (!d.additive) {
        clearSelection();
      }
      setMarquee(null);
    }
    setState({ camera: camRef.current });
  };

  // React attaches onWheel passively, so preventDefault there is ignored and the page
  // scrolls instead of the Field zooming. Bind it natively.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const handler = (e) => {
      e.preventDefault();
      userMovedRef.current = true;
      const rect = canvas.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      const cam = camRef.current;
      const [wx, wy] = toWorld(cam, view, sx, sy);
      const z = Math.max(0.18, Math.min(2.6, cam.z * (e.deltaY > 0 ? 0.9 : 1.11)));
      // Keep the world point under the cursor fixed while zooming.
      camRef.current = { z, x: wx - (sx - view.w / 2) / z, y: wy - (sy - view.h / 2) / z };
      dirtyRef.current = true;
    };
    canvas.addEventListener('wheel', handler, { passive: false });
    return () => canvas.removeEventListener('wheel', handler);
  }, [view]);

  // ---------------------------------------------------------------- keys

  useEffect(() => {
    const onKey = (e) => {
      const tag = document.activeElement?.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || document.activeElement?.isContentEditable) return;

      if (e.key === 'Alt') { setShowLabels(true); return; }

      if (/^[1-9]$/.test(e.key)) {
        const group = e.key;
        const s = getState();
        if (e.ctrlKey || e.metaKey) {
          e.preventDefault();
          if (s.selection.length) api.controlGroup(group, s.selection).catch(console.error);
        } else {
          const ids = s.snap.controlGroups?.[group] ?? [];
          const alive = new Set(s.snap.sessions.map((x) => x.id));
          selectOnly(ids.filter((id) => alive.has(id)));
        }
        return;
      }

      if (e.key === 'Escape') { setMenu(null); clearSelection(); return; }

      if (e.key === 'a' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        selectOnly(layoutRef.current.agents.map((a) => a.id));
        return;
      }

      if (e.key === 'f' && !e.ctrlKey) {
        // Frame the selection, or the whole field when nothing is selected.
        const s = getState();
        const pts = s.selection.length
          ? layoutRef.current.agents.filter((a) => s.selection.includes(a.id))
          : layoutRef.current.agents;
        if (pts.length) {
          const cx = pts.reduce((m, p) => m + p.x, 0) / pts.length;
          const cy = pts.reduce((m, p) => m + p.y, 0) / pts.length;
          camRef.current = { ...camRef.current, x: cx, y: cy };
          dirtyRef.current = true;
        }
      }
    };
    const onKeyUp = (e) => { if (e.key === 'Alt') setShowLabels(false); };

    window.addEventListener('keydown', onKey);
    window.addEventListener('keyup', onKeyUp);
    return () => {
      window.removeEventListener('keydown', onKey);
      window.removeEventListener('keyup', onKeyUp);
    };
  }, []);

  return (
    <div className="field-wrap" ref={wrapRef}>
      <canvas
        ref={canvasRef}
        className="field-canvas"
        style={{ width: view.w, height: view.h }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onDoubleClick={(e) => {
          const [sx, sy] = localPoint(e);
          const hit = hitTest(sx, sy);
          if (hit.type === 'agent') { selectOnly([hit.id]); setInspectId(hit.id); }
        }}
        onContextMenu={(e) => {
          e.preventDefault();
          const [sx, sy] = localPoint(e);
          const hit = hitTest(sx, sy);
          setMenu({ screen: { x: sx, y: sy }, target: hit, world: hit.world });
        }}
      />

      <div className="field-strip" onPointerDown={(e) => e.stopPropagation()}>
        {workspaces.map((w) => (
          <ProjectCard
            key={w.id}
            workspace={w}
            sessions={liveSessions.filter((x) => x.workspaceId === w.id)}
            stamps={st.lastEventBySession}
            now={st.snap.now}
            onStart={(e) => startAgent(w, e)}
            onOpen={() => openInWorkspace({ type: 'workspace', workspaceId: w.id, path: '' })}
            onSelect={(ids) => selectOnly(ids)}
          />
        ))}
        <div className="field-cost mono" aria-live="polite">
          <span className="field-cost-amount">${(st.snap.totals?.costUsd ?? 0).toFixed(3)}</span>
          <span className="field-cost-label">spent</span>
          {reservedUsd > 0 && <><span className="field-cost-amount dim">${reservedUsd.toFixed(2)}</span><span className="field-cost-label">reserved</span></>}
        </div>
      </div>

      {!liveSessions.length && !menu && (
        <div className="field-empty" onPointerDown={(e) => e.stopPropagation()}>
          <span className="atlas-empty-mark" aria-hidden="true" />
          <b>{workspaces.length ? 'No agents on the field yet' : 'No project open'}</b>
          {workspaces.length ? (
            <p>Start an agent on a project and it appears here, walking to the files it touches. Right-click any project or folder for more.</p>
          ) : (
            <p>Add a project folder under <code>workspaces</code> in <code>field/field.yaml</code>, then restart Field.</p>
          )}
          <div className="atlas-empty-actions">
            {workspaces.length > 0 && <button type="button" className="btn primary" onClick={(e) => startAgent(workspaces[0], e)}>Start an agent</button>}
            {!hasModel && <button type="button" className="btn ghost" onClick={() => setModelsOpen(true)}>Set up a model</button>}
          </div>
        </div>
      )}

      <div className="field-hint label">
        drag select · shift add · double-click a unit to talk to it · ctrl+digit group · right-click to assign · alt-drag pan · scroll zoom
      </div>

      <PermissionRequests />
      <SelectionHUD />
      <EndpointRail />

      {inspectSession && (
        <UnitInspector session={inspectSession} onClose={() => setInspectId(null)} />
      )}

      {menu && (
        <ContextMenu
          screen={menu.screen}
          target={menu.target}
          initialPane={menu.pane ?? null}
          onClose={() => setMenu(null)}
        />
      )}
      {modelsOpen && <PowerSources onClose={() => setModelsOpen(false)} />}
    </div>
  );
}

const TERMINAL = new Set(['done', 'cancelled', 'interrupted', 'error']);
const STATE_WORD = {
  waiting_permission: 'needs approval', blocked: 'blocked', error: 'failed', thinking: 'thinking',
  working: 'working', paused: 'paused', spawning: 'starting', ready: 'ready', idle: 'idle',
};

function ago(ts, now) {
  if (!ts) return 'no activity yet';
  const s = Math.max(0, Math.round(((now ?? Date.now()) - ts) / 1000));
  if (s < 5) return 'just now';
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${Math.round(s / 3600)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}

// A project card over the map: what the canvas cannot say in nine-pixel type.
function ProjectCard({ workspace, sessions, stamps, now, onStart, onOpen, onSelect }) {
  const last = Math.max(
    workspace.lastTs ?? 0,
    ...sessions.map((s) => stamps[s.id] ?? s.startedAt ?? 0),
  );
  const attention = sessions.filter((s) => ['waiting_permission', 'blocked', 'error'].includes(s.state)).length;
  const changed = workspace.git?.files?.length ?? workspace.changeCount ?? 0;
  return (
    <section className={`field-card${sessions.length ? '' : ' idle'}${attention ? ' attention' : ''}`} aria-label={`${workspace.name} project`}>
      <header>
        <button type="button" className="field-card-name" onClick={onOpen} title={`${workspace.path} — open files`}>{workspace.name}</button>
        <span className="field-card-git mono" title={workspace.git?.branch ?? ''}>
          {workspace.git?.branch ?? 'no git'}
          {changed
            ? <>{' · '}<button type="button" className="field-card-changes" onClick={() => openChanges(workspace.id)} title="Review, accept or revert the changes">{changed} changed</button></>
            : ' · clean'}
        </span>
      </header>
      <div className="field-card-agents">
        {sessions.length ? sessions.slice(0, 4).map((s) => (
          <button type="button" key={s.id} className="field-card-agent" onClick={() => onSelect([s.id])} title={s.lastTool?.summary ?? s.stateDetail ?? ''}>
            <span className={`dot ${s.state}`} aria-hidden="true" />
            <b>{s.name ?? s.id.slice(0, 6)}</b>
            <small>{STATE_WORD[s.state] ?? s.state}{s.lastTool?.name ? ` · ${s.lastTool.name}` : ''}</small>
          </button>
        )) : <span className="field-card-none">No agent on this project</span>}
        {sessions.length > 4 && <button type="button" className="field-card-more" onClick={() => onSelect(sessions.map((s) => s.id))}>+{sessions.length - 4} more</button>}
      </div>
      <footer>
        <span className="field-card-when">{sessions.length ? `${sessions.length} working` : 'idle'} · {ago(last, now)}</span>
        <button type="button" className={`btn sm ${sessions.length ? 'ghost' : 'primary'}`} onClick={onStart}>Start an agent</button>
      </footer>
    </section>
  );
}
