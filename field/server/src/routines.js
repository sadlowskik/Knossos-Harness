// Persistent and scheduled agent work.
// A routine fires on a real clock tick or a real filesystem change, and when it fires
// it spawns a real session. Disabled routines are inert.
import path from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import { globToRegExp } from './glob.js';

// --- minimal 5-field cron: minute hour day-of-month month day-of-week ---------

function matchField(spec, value, min, max) {
  for (const part of String(spec).split(',')) {
    if (part === '*') return true;
    const step = part.includes('/') ? Number(part.split('/')[1]) : 1;
    const range = part.split('/')[0];
    if (range === '*') {
      if ((value - min) % step === 0) return true;
      continue;
    }
    if (range.includes('-')) {
      const [a, b] = range.split('-').map(Number);
      if (value >= a && value <= b && (value - a) % step === 0) return true;
      continue;
    }
    const start = Number(range);
    if (part.includes('/') ? value >= start && value <= max && (value - start) % step === 0 : start === value) return true;
  }
  return false;
}

export function cronMatches(expr, date = new Date()) {
  validateCron(expr);
  const parts = String(expr).trim().split(/\s+/);
  const [mi, ho, dom, mo, dow] = parts;
  const dayMatches = matchField(dom, date.getUTCDate(), 1, 31);
  const weekdayMatches = matchField(dow, date.getUTCDay(), 0, 6);
  const dayGate = dom !== '*' && dow !== '*' ? dayMatches || weekdayMatches : dayMatches && weekdayMatches;
  return matchField(mi, date.getUTCMinutes(), 0, 59)
    && matchField(ho, date.getUTCHours(), 0, 23)
    && dayGate
    && matchField(mo, date.getUTCMonth() + 1, 1, 12);
}

export function validateCron(expr) {
  const parts = String(expr ?? '').trim().split(/\s+/);
  if (parts.length !== 5) throw new Error('cron must contain exactly five fields');
  const limits = [[0, 59], [0, 23], [1, 31], [1, 12], [0, 6]];
  parts.forEach((field, index) => validateCronField(field, ...limits[index]));
  return true;
}

function validateCronField(field, min, max) {
  if (!field) throw new Error('cron field is empty');
  for (const part of field.split(',')) {
    if (!/^\*(?:\/\d+)?$|^\d+(?:-\d+)?(?:\/\d+)?$/.test(part)) {
      throw new Error(`unsupported cron field: ${part}`);
    }
    const [range, stepText] = part.split('/');
    const step = stepText == null ? 1 : Number(stepText);
    if (!Number.isInteger(step) || step < 1 || step > max - min + 1) throw new Error(`invalid cron step: ${part}`);
    if (range === '*') continue;
    const [start, end = start] = range.split('-').map(Number);
    if (!Number.isInteger(start) || !Number.isInteger(end) || start < min || end > max || start > end) {
      throw new Error(`cron value is outside ${min}-${max}: ${part}`);
    }
  }
}

export function startRoutines({
  cfg, registry, log, emit, projection,
  timers = globalThis,
  now = () => new Date(),
  createId = randomUUID,
}) {
  for (const routine of cfg.routines) validateRoutine(routine, cfg);
  const pending = new Map();   // routineId -> { paths:Set, timer }
  const enabled = new Map(cfg.routines.map((r) => [
    r.id,
    projection?.routines?.get(r.id)?.enabled ?? !!r.enabled,
  ]));
  const activeRuns = new Map(); // sessionId -> { runId, routineId }
  const activeByRoutine = new Map();
  const queuedByRoutine = new Map();
  const cooldownUntil = new Map();
  const cooldownTimers = new Map();
  const lastScheduleSlot = new Map(cfg.routines.map((r) => [r.id, projection?.routines?.get(r.id)?.lastScheduleSlot ?? '']));
  const nextRunCache = new Map();
  let clockTimer = null;
  let started = false;
  let stopped = false;

  function queueAuthority(routine) {
    const canonical = (value) => Array.isArray(value) ? value.map(canonical)
      : value && typeof value === 'object' ? Object.fromEntries(Object.keys(value).sort().map(key => [key, canonical(value[key])])) : value;
    return createHash('sha256').update(JSON.stringify(canonical({
      routine, defaults: cfg.defaults,
      role: cfg.roles.find(item => item.id === routine.role),
      agent: cfg.agents.find(item => item.role === routine.role),
      endpoint: cfg.endpoints.find(item => item.id === routine.endpoint),
      workspace: cfg.workspaces.find(item => item.id === routine.workspace),
    }))).digest('hex');
  }

  for (const routine of cfg.routines) {
    const previous = projection?.routines?.get(routine.id);
    cooldownUntil.set(routine.id, previous?.cooldownUntil ?? 0);
    if (previous?.queuedRuns?.length) {
      for (const queued of [...previous.queuedRuns]) {
        const reason = queued.claimed ? 'interrupted_after_queue_claim'
          : !enabled.get(routine.id) ? 'disabled'
          : queued.authorityHash !== queueAuthority(routine) ? 'queued_configuration_changed'
          : queuedByRoutine.has(routine.id) ? 'queue_full' : null;
        if (reason) emit(queued.claimed ? 'routine.failed' : 'routine.skipped', {
          routineId: routine.id, runId: queued.runId, reason,
        }, { subject: routine.id, source: 'derived' });
        else queuedByRoutine.set(routine.id, { ...queued, routineId: routine.id });
      }
    }
    const interrupted = previous?.activeRunIds?.length
      ? previous.activeRunIds
      : (previous?.activeRunId ? [previous.activeRunId] : []);
    for (const runId of interrupted) emit('routine.failed', {
      routineId: routine.id, runId,
      reason: 'Field restarted while this run was active; outcome is interrupted, not completed',
    }, { subject: routine.id, source: 'derived' });
  }

  function runsFor(id) {
    return [...activeRuns.values()].filter((run) => run.routineId === id);
  }

  function interruptRuns(routine, reason) {
    for (const run of runsFor(routine.id)) {
      emit('routine.failed', {
        routineId: routine.id, runId: run.runId, sessionId: run.sessionId, reason,
      }, { subject: routine.id, source: 'derived' });
      activeRuns.delete(run.sessionId);
      try { registry.command?.('cancel', { sessionIds: [run.sessionId] }); } catch { /* mock registries may omit cancel */ }
    }
    activeByRoutine.delete(routine.id);
  }

  function fire(routine, reason, changedPaths, runIdOverride = null) {
    if (stopped) return { admitted: false, reason: 'stopped' };
    if (!enabled.get(routine.id)) return { admitted: false, reason: 'disabled' };
    const cooldown = cooldownUntil.get(routine.id) ?? 0;
    const active = runsFor(routine.id).length > 0;
    const cooling = now().getTime() < cooldown;
    const queued = !runIdOverride && queuedByRoutine.has(routine.id);
    if (active || cooling || queued) {
      if (routine.overlap === 'queue-one') return enqueue(routine, reason, changedPaths);
      if (routine.overlap === 'replace' && active && !cooling && !queued) interruptRuns(routine, 'replaced');
      else if (!(routine.overlap === 'parallel' && active && !cooling && !queued)) {
        emit('routine.skipped', {
          routineId: routine.id, reason: active ? 'overlap' : 'cooldown',
          activeRunId: activeByRoutine.get(routine.id)?.runId ?? null,
        }, { subject: routine.id, source: 'derived' });
        return { admitted: false, reason: 'overlap' };
      }
    }
    const agent = cfg.agents.find((a) => a.role === routine.role);
    if (!agent) {
      if (runIdOverride) emit('routine.failed', {
        routineId: routine.id, runId: runIdOverride, reason: `no agent has role "${routine.role}"`,
      }, { subject: routine.id, source: 'derived' });
      return { admitted: false, reason: `no agent has role "${routine.role}"` };
    }
    let orders = routine.orders ?? '';
    if (changedPaths?.length) {
      orders += `\n\nFiles that changed since the last run:\n` +
        changedPaths.slice(0, 40).map((p) => `- ${p}`).join('\n');
    }
    const runId = runIdOverride ?? `${routine.id}:${createId()}`;
    try {
      const s = registry.spawn({
        agentId: agent.id,
        workspaceId: routine.workspace,
        orders,
        endpointId: routine.endpoint,
        thinking: routine.thinking,
        budgetUsd: routine.budget_usd,
        target: { type: 'workspace', id: routine.workspace, workspaceId: routine.workspace },
      });
      const startedAt = now().getTime();
      const run = { runId, routineId: routine.id, sessionId: s.id, startedAt };
      activeRuns.set(s.id, run);
      activeByRoutine.set(routine.id, run);
      cooldownUntil.set(routine.id, 0);
      emit('routine.triggered', {
        routineId: routine.id, runId, sessionIds: [s.id], reason,
        budgetUsd: routine.budget_usd ?? null, startedAt,
      }, { subject: routine.id, source: 'derived' });
      return { admitted: true, ...run };
    } catch (e) {
      emit('routine.failed', {
        routineId: routine.id, runId, reason: `spawn failed: ${e.message}`,
      }, { subject: routine.id, source: 'derived' });
      return { admitted: false, reason: e.message };
    }
  }

  function enqueue(routine, reason, paths = []) {
    if (queuedByRoutine.has(routine.id)) {
      emit('routine.skipped', { routineId: routine.id, reason: 'queue_full' }, { subject: routine.id, source: 'derived' });
      return { admitted: false, reason: 'queue_full' };
    }
    const queued = { runId: `${routine.id}:${createId()}`, routineId: routine.id, paths: paths.slice(0, 40), queuedAt: now().getTime(), reason, authorityHash: queueAuthority(routine) };
    emit('routine.queued', queued, { subject: routine.id, source: 'derived' });
    queuedByRoutine.set(routine.id, queued);
    pump(routine);
    return { admitted: true, queued: true, ...queued };
  }

  function pump(routine) {
    const queued = queuedByRoutine.get(routine.id);
    if (stopped || !queued || queued.claimed || activeByRoutine.has(routine.id) || !enabled.get(routine.id)) return;
    if (queued.authorityHash !== queueAuthority(routine)) {
      emit('routine.skipped', { routineId: routine.id, runId: queued.runId, reason: 'queued_configuration_changed' }, { subject: routine.id, source: 'derived' });
      queuedByRoutine.delete(routine.id);
      return;
    }
    const wait = Math.max(0, (cooldownUntil.get(routine.id) ?? 0) - now().getTime());
    if (wait > 0) {
      if (!cooldownTimers.has(routine.id)) {
        const timer = timers.setTimeout(() => { cooldownTimers.delete(routine.id); pump(routine); }, wait);
        timer.unref?.(); cooldownTimers.set(routine.id, timer);
      }
      return;
    }
    if (cooldownTimers.has(routine.id)) { timers.clearTimeout(cooldownTimers.get(routine.id)); cooldownTimers.delete(routine.id); }
    emit('routine.queue_claimed', { routineId: routine.id, runId: queued.runId }, { subject: routine.id, source: 'derived' });
    queued.claimed = true;
    try { fire(routine, `queued after ${queued.reason ?? 'run'}`, queued.paths, queued.runId); }
    finally { queuedByRoutine.delete(routine.id); }
  }

  // --- schedule triggers: checked once a minute on the real clock ---
  function clockTick() {
    const current = now();
    const slot = current.toISOString().slice(0, 16);
    for (const r of cfg.routines) {
      if (!enabled.get(r.id) || r.trigger?.kind !== 'schedule') continue;
      if (!cronMatches(r.trigger.cron, current) || (lastScheduleSlot.get(r.id) ?? '') >= slot) continue;
      // Persist the watermark before spawning. Restart/backwards clock movement
      // cannot duplicate a consequential scheduled run. Missed slots are skipped.
      emit('routine.schedule_claimed', { routineId: r.id, slot }, { subject: r.id, source: 'derived' });
      lastScheduleSlot.set(r.id, slot);
      fire(r, `cron ${r.trigger.cron} UTC`);
    }
    if (!stopped) {
      clockTimer = timers.setTimeout(clockTick, 5000);
      clockTimer.unref?.();
    }
  }
  // --- filesystem triggers: debounced against real change events ---
  const unsubscribe = log.subscribe((evt) => {
    const run = evt.data?.sessionId ? activeRuns.get(evt.data.sessionId) : null;
      if (run && ['session.turn_complete', 'session.ended'].includes(evt.kind)) {
      const routine = cfg.routines.find((item) => item.id === run.routineId);
      const until = now().getTime() + Number(routine?.cooldown_seconds ?? 0) * 1000;
      const failed = evt.kind === 'session.ended'
        ? ['error', 'cancelled'].includes(evt.data.reason)
        : evt.data.isError === true;
      emit(failed ? 'routine.failed' : 'routine.completed', {
        routineId: run.routineId, runId: run.runId, sessionId: run.sessionId,
        cooldownUntil: until,
        reason: failed ? (evt.data.error ?? evt.data.reason ?? 'run failed') : 'run completed',
      }, { subject: run.routineId, source: 'derived' });
      activeRuns.delete(evt.data.sessionId);
      const remaining = runsFor(run.routineId);
      if (remaining.length) activeByRoutine.set(run.routineId, remaining[remaining.length - 1]);
      else activeByRoutine.delete(run.routineId);
      cooldownUntil.set(run.routineId, until);
      const queued = queuedByRoutine.get(run.routineId);
      if (queued) {
        pump(routine);
      }
      return;
    }
    if (evt.kind !== 'fs.changed' || evt.source === 'synthetic') return;
    for (const r of cfg.routines) {
      if (!enabled.get(r.id)) continue;
      const t = r.trigger;
      if (t?.kind !== 'fs_change') continue;
      if (t.workspace && t.workspace !== evt.data.workspaceId) continue;
      const patterns = (t.paths ?? ['**']).map(globToRegExp);
      if (!patterns.some((re) => re.test(evt.data.path))) continue;
      if (run?.routineId === r.id) {
        emit('routine.skipped', {
          routineId: r.id, runId: run.runId, reason: 'self_trigger', path: evt.data.path,
        }, { subject: r.id, source: 'derived' });
        continue;
      }

      if (!pending.has(r.id)) pending.set(r.id, { paths: new Set(), timer: null });
      const p = pending.get(r.id);
      p.paths.add(evt.data.path);
      if (p.timer) timers.clearTimeout(p.timer);
      p.timer = timers.setTimeout(() => {
        const paths = [...p.paths];
        pending.delete(r.id);
        if (enabled.get(r.id)) fire(r, `${paths.length} file(s) changed`, paths);
      }, (t.debounce_seconds ?? 60) * 1000);
      p.timer.unref?.();
    }
  });

  return {
    start() {
      if (started || stopped) return false;
      started = true;
      for (const routine of cfg.routines) pump(routine);
      clockTick();
      return true;
    },
    setEnabled(id, on, { confirmRisk = false } = {}) {
      if (!enabled.has(id)) return false;
      if (on && !confirmRisk) throw new Error('enabling a routine requires explicit operator confirmation');
      emit('routine.enabled', { routineId: id, enabled: on }, { subject: id });
      enabled.set(id, on);
      if (!on && pending.has(id)) {
        timers.clearTimeout(pending.get(id).timer);
        pending.delete(id);
      }
      if (!on && queuedByRoutine.has(id)) {
        const queued = queuedByRoutine.get(id);
        emit('routine.skipped', { routineId: id, runId: queued.runId, reason: 'disabled' }, { subject: id, source: 'derived' });
        queuedByRoutine.delete(id);
      }
      if (!on && cooldownTimers.has(id)) { timers.clearTimeout(cooldownTimers.get(id)); cooldownTimers.delete(id); }
      return true;
    },
    run(id) {
      const routine = cfg.routines.find((item) => item.id === id);
      if (!routine) return { admitted: false, reason: 'unknown routine' };
      const wasEnabled = enabled.get(id);
      if (!wasEnabled) enabled.set(id, true);
      try { return fire(routine, 'operator'); }
      finally { if (!wasEnabled) enabled.set(id, false); }
    },
    isEnabled(id) { return !!enabled.get(id); },
    states() { return Object.fromEntries(enabled); },
    details() {
      return Object.fromEntries(cfg.routines.map((routine) => {
        const state = projection?.routines?.get(routine.id) ?? {};
        const minute = Math.floor(now().getTime() / 60_000);
        let nextRunAt = null;
        if (enabled.get(routine.id) && routine.trigger?.kind === 'schedule') {
          const cached = nextRunCache.get(routine.id);
          if (cached?.minute === minute) nextRunAt = cached.at;
          else {
            const watermark = state.lastScheduleSlot ? Date.parse(`${state.lastScheduleSlot}:00Z`) : 0;
            nextRunAt = nextCronRun(routine.trigger.cron, new Date(Math.max(now().getTime(), watermark)));
            nextRunCache.set(routine.id, { minute, at: nextRunAt });
          }
        }
        return [routine.id, {
          enabled: !!enabled.get(routine.id), timezone: 'UTC',
          activeRunId: runsFor(routine.id).slice(-1)[0]?.runId ?? null,
          activeRunIds: runsFor(routine.id).map((run) => run.runId),
          queuedRuns: queuedByRoutine.has(routine.id) ? [queuedByRoutine.get(routine.id)] : [],
          lastRunAt: state.lastRunAt ?? null, lastOutcome: state.lastOutcome ?? null,
          nextRunAt, currentOwner: state.currentOwner ?? null,
          history: (state.history ?? []).slice(-50),
        }];
      }));
    },
    stop() {
      stopped = true;
      unsubscribe();
      if (clockTimer) timers.clearTimeout(clockTimer);
      for (const p of pending.values()) if (p.timer) timers.clearTimeout(p.timer);
      pending.clear();
      for (const timer of cooldownTimers.values()) timers.clearTimeout(timer);
      cooldownTimers.clear();
      // Unclaimed work remains durable for the next process. Only an explicit
      // disable cancels it; a claimed run is reconciled as interrupted on restart.
      queuedByRoutine.clear();
    },
  };
}

export function validateRoutine(routine, cfg) {
  if (!routine?.id || typeof routine.id !== 'string') throw new Error('routine id is required');
  if (!cfg.roles.some((role) => role.id === routine.role)) throw new Error(`routine ${routine.id} has unknown role ${routine.role}`);
  if (!cfg.endpoints.some((endpoint) => endpoint.id === routine.endpoint)) throw new Error(`routine ${routine.id} has unknown endpoint ${routine.endpoint}`);
  if (!cfg.workspaces.some((workspace) => workspace.id === routine.workspace && workspace.mounted)) {
    throw new Error(`routine ${routine.id} has unavailable workspace ${routine.workspace}`);
  }
  if (!['schedule', 'fs_change'].includes(routine.trigger?.kind)) throw new Error(`routine ${routine.id} has unsupported trigger`);
  if (routine.trigger.kind === 'schedule') validateCron(routine.trigger.cron);
  if (routine.trigger.timezone != null && routine.trigger.timezone !== 'UTC') throw new Error('routine schedules support UTC only');
  if (routine.overlap != null && !['skip', 'queue-one', 'replace', 'parallel'].includes(routine.overlap)) throw new Error('routine overlap policy supports skip, queue-one, replace, or parallel');
  if (routine.max_queued_runs != null && routine.max_queued_runs !== 1) throw new Error('routine max_queued_runs currently supports only 1');
  if (routine.cooldown_seconds != null && (!Number.isFinite(Number(routine.cooldown_seconds)) || Number(routine.cooldown_seconds) < 0 || Number(routine.cooldown_seconds) > 86_400)) throw new Error(`routine ${routine.id} has invalid cooldown`);
  if (routine.trigger.kind === 'fs_change') {
    const seconds = Number(routine.trigger.debounce_seconds ?? 60);
    if (!Number.isFinite(seconds) || seconds < 1 || seconds > 86_400) throw new Error(`routine ${routine.id} has invalid debounce`);
  }
  if (!routine.completion?.kind) throw new Error(`routine ${routine.id} requires a completion policy`);
  if (routine.budget_usd != null && (!Number.isFinite(routine.budget_usd) || routine.budget_usd <= 0)) {
    throw new Error(`routine ${routine.id} has invalid dollar budget`);
  }
  return true;
}

/** Strictly future UTC occurrence, bounded to eight years (including leap gaps). */
export function nextCronRun(expr, after = new Date()) {
  validateCron(expr);
  const [mi, ho, dom, mo, dow] = expr.trim().split(/\s+/);
  const hours = Array.from({ length: 24 }, (_, i) => i).filter(h => matchField(ho, h, 0, 23));
  const minutes = Array.from({ length: 60 }, (_, i) => i).filter(m => matchField(mi, m, 0, 59));
  const day = new Date(after);
  day.setUTCHours(0, 0, 0, 0);
  for (let i = 0; i < 366 * 8; i += 1, day.setUTCDate(day.getUTCDate() + 1)) {
    if (!matchField(mo, day.getUTCMonth() + 1, 1, 12)) continue;
    const a = matchField(dom, day.getUTCDate(), 1, 31);
    const b = matchField(dow, day.getUTCDay(), 0, 6);
    if (!(dom !== '*' && dow !== '*' ? a || b : a && b)) continue;
    for (const hour of hours) for (const minute of minutes) {
      const candidate = new Date(day);
      candidate.setUTCHours(hour, minute, 0, 0);
      if (candidate > after) return candidate.toISOString();
    }
  }
  return null;
}
