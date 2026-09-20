import assert from 'node:assert/strict';
import { Projection } from '../src/store/projection.js';
import { cronMatches, nextCronRun, startRoutines, validateCron, validateRoutine } from '../src/routines.js';

assert.equal(cronMatches('0 5 * * *', new Date('2024-01-01T00:00:00-05:00')), true, 'cron uses UTC');
assert.equal(cronMatches('0 0 1 * 0', new Date('2024-01-01T00:00:00Z')), true, 'restricted day fields use standard OR semantics');
assert.equal(cronMatches('0 0 2 * 1', new Date('2024-01-01T00:00:00Z')), true, 'weekday may satisfy the day gate');
assert.throws(() => validateCron('* * * *'), /five fields/);
assert.throws(() => validateCron('*/0 * * * *'), /invalid cron step/);
assert.throws(() => validateCron('60 * * * *'), /outside 0-59/);
assert.throws(() => validateCron('* * 8-2 * *'), /outside 1-31/);
assert.equal(cronMatches('5/15 * * * *', new Date('2024-01-01T00:20:00Z')), true);
assert.equal(nextCronRun('0 0 29 2 *', new Date('2025-03-01T00:00:00Z')), '2028-02-29T00:00:00.000Z');
assert.equal(nextCronRun('0 0 31 2 *', new Date('2025-03-01T00:00:00Z')), null);
assert.equal(nextCronRun('0 5 * * *', new Date('2024-03-10T05:00:00Z')), '2024-03-11T05:00:00.000Z');

const routine = {
  id: 'watch-source', name: 'Watch source', enabled: true,
  role: 'builder', endpoint: 'local', workspace: 'cameo', thinking: 'medium',
  budget_usd: 0.5, orders: 'Inspect changed source files.',
  trigger: { kind: 'fs_change', workspace: 'cameo', paths: ['src/**'], debounce_seconds: 1 },
  completion: { kind: 'report' },
};
const cfg = {
  defaults: {},
  routines: [routine],
  roles: [{ id: 'builder' }],
  agents: [{ id: 'builder-1', role: 'builder' }],
  endpoints: [{ id: 'local', name: 'Local', kind: 'openai-compatible', model: 'test' }],
  workspaces: [{ id: 'cameo', name: 'Cameo', path: process.cwd(), mounted: true }],
  websites: [],
};

assert.equal(validateRoutine(routine, cfg), true);
assert.throws(() => validateRoutine({ ...routine, role: 'missing' }, cfg), /unknown role/);
assert.throws(() => validateRoutine({ ...routine, endpoint: 'missing' }, cfg), /unknown endpoint/);
assert.throws(() => validateRoutine({ ...routine, workspace: 'missing' }, cfg), /unavailable workspace/);
assert.throws(() => validateRoutine({ ...routine, trigger: { kind: 'manual' } }, cfg), /unsupported trigger/);
assert.throws(() => validateRoutine({ ...routine, budget_usd: 0 }, cfg), /invalid dollar budget/);

const projection = new Projection(cfg);
projection.apply({ seq: 1, ts: 10, kind: 'routine.enabled', data: { routineId: routine.id, enabled: false } });

let sequence = 1;
const listeners = new Set();
const events = [];
const log = {
  subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); },
  publish(event) { for (const fn of [...listeners]) fn(event); },
};
const emit = (kind, data, meta = {}) => {
  const event = { seq: ++sequence, ts: 1_700_000_000_000 + sequence, kind, data, ...meta };
  events.push(event);
  projection.apply(event);
  log.publish(event);
  return event;
};

let timerSequence = 0;
const timerJobs = new Map();
const timers = {
  setTimeout(fn, ms) {
    const timer = { id: ++timerSequence, ms, active: true, unref() {} };
    timerJobs.set(timer.id, { timer, fn });
    return timer;
  },
  clearTimeout(timer) { if (timer) timer.active = false; },
};
const runTimers = (ms) => {
  for (const { timer, fn } of [...timerJobs.values()]) {
    if (!timer.active || timer.ms !== ms) continue;
    timer.active = false;
    fn();
  }
};

let spawnCount = 0;
const spawns = [];
const registry = {
  spawn(options) {
    spawns.push(options);
    return { id: `session-${++spawnCount}` };
  },
};
const controller = startRoutines({
  cfg, registry, log, emit, projection, timers,
  now: () => new Date('2024-01-01T00:00:00Z'),
  createId: () => `run-${spawnCount + 1}`,
});
assert.equal(controller.start(), true);
assert.equal(controller.start(), false, 'the scheduler clock starts once');

assert.equal(controller.isEnabled(routine.id), false, 'durable replayed state overrides the Git default');
assert.throws(() => controller.setEnabled(routine.id, true), /explicit operator confirmation/);
assert.equal(controller.setEnabled(routine.id, true, { confirmRisk: true }), true);

log.publish({
  seq: ++sequence, ts: 20, kind: 'fs.changed', source: 'observed',
  data: { workspaceId: 'cameo', path: 'src/changed.js' },
});
controller.setEnabled(routine.id, false);
runTimers(1000);
assert.equal(spawnCount, 0, 'disabling cancels pending debounced work');

controller.setEnabled(routine.id, true, { confirmRisk: true });
const first = controller.run(routine.id);
assert.equal(first.admitted, true);
assert.equal(spawns[0].budgetUsd, 0.5, 'routine budget reaches the session registry');
assert.equal(controller.run(routine.id).reason, 'overlap');
assert.equal(events.some((event) => event.kind === 'routine.skipped' && event.data.reason === 'overlap'), true);

log.publish({
  seq: ++sequence, ts: 30, kind: 'fs.changed', source: 'observed',
  data: { sessionId: first.sessionId, workspaceId: 'cameo', path: 'src/self.js' },
});
runTimers(1000);
assert.equal(spawnCount, 1, 'a routine cannot trigger itself from its own filesystem writes');
assert.equal(events.some((event) => event.kind === 'routine.skipped' && event.data.reason === 'self_trigger'), true);

log.publish({
  seq: ++sequence, ts: 40, kind: 'session.turn_complete', source: 'observed',
  data: { sessionId: first.sessionId, isError: false },
});
assert.equal(controller.details()[routine.id].lastOutcome, 'completed');
assert.equal(controller.run(routine.id).admitted, true, 'completion releases the overlap lock');

controller.stop();
assert.equal([...timerJobs.values()].every(({ timer }) => !timer.active), true, 'shutdown clears clock and debounce timers');

const scheduled = { ...routine, id: 'scheduled', trigger: { kind: 'schedule', cron: '* * * * *' } };
const scheduledCfg = { ...cfg, routines: [scheduled] };
const scheduleProjection = new Projection(scheduledCfg);
let scheduleSpawns = 0;
const scheduleLog = { subscribe() { return () => {}; } };
const scheduleEmit = (kind, data, meta) => scheduleProjection.apply({ seq: ++sequence, ts: Date.now(), kind, data, ...meta });
const createScheduler = () => startRoutines({ cfg: scheduledCfg, registry: { spawn() { return { id: `scheduled-${++scheduleSpawns}` }; } }, log: scheduleLog,
  emit: scheduleEmit, projection: scheduleProjection, timers, now: () => new Date('2024-01-01T00:00:00Z') });
const beforeRestart = createScheduler();
beforeRestart.start();
assert.equal(scheduleSpawns, 1);
beforeRestart.stop();
const afterRestart = createScheduler();
afterRestart.start();
assert.equal(scheduleSpawns, 1, 'persisted schedule watermark prevents same-minute duplicate after restart');
assert.equal(afterRestart.details().scheduled.lastOutcome, 'failed', 'interrupted run is never left running');
assert.equal(afterRestart.details().scheduled.nextRunAt, '2024-01-01T00:01:00.000Z');
afterRestart.stop();

const queueRoutine = { ...routine, id: 'queue-one', overlap: 'queue-one', max_queued_runs: 1, cooldown_seconds: 2 };
assert.equal(validateRoutine(queueRoutine, { ...cfg, routines: [queueRoutine] }), true);
const queueProjection = new Projection({ ...cfg, routines: [queueRoutine] });
const queueListeners = new Set();
const queueEvents = [];
const queueLog = { subscribe(fn) { queueListeners.add(fn); return () => queueListeners.delete(fn); }, publish(event) { for (const fn of queueListeners) fn(event); } };
let queueSeq = 0;
const queueEmit = (kind, data, meta = {}) => {
  const event = { seq: ++queueSeq, ts: 1_700_000_000_000 + queueSeq, kind, data, ...meta };
  queueEvents.push(event); queueProjection.apply(event); queueLog.publish(event); return event;
};
let queueSpawns = 0;
let queueNow = new Date('2024-01-01T00:00:00Z');
const queueController = startRoutines({
  cfg: { ...cfg, routines: [queueRoutine] },
  registry: { spawn() { return { id: `queue-session-${++queueSpawns}` }; } },
  log: queueLog, emit: queueEmit, projection: queueProjection, timers,
  now: () => queueNow, createId: () => `queue-${queueSpawns + 1}`,
});
const activeQueue = queueController.run(queueRoutine.id);
assert.equal(activeQueue.admitted, true);
assert.equal(queueController.run(queueRoutine.id).queued, true, 'queue-one retains one trigger during overlap');
assert.equal(queueController.run(queueRoutine.id).reason, 'queue_full', 'second queued trigger is bounded');
queueEmit('session.turn_complete', { sessionId: activeQueue.sessionId, isError: false });
queueNow = new Date(queueNow.getTime() + 2000);
runTimers(2000);
assert.equal(queueSpawns, 2, 'queued run dispatches after cooldown');
assert.equal(queueEvents.filter((event) => event.kind === 'routine.triggered').length, 2);
const queuedActive = [...queueEvents].reverse().find((event) => event.kind === 'routine.triggered').data.sessionIds[0];
queueController.run(queueRoutine.id);
queueController.setEnabled(queueRoutine.id, false);
assert.equal(queueEvents.some((event) => event.kind === 'routine.skipped' && event.data.reason === 'disabled'), true, 'disable drops queued work truthfully');
queueEmit('session.ended', { sessionId: queuedActive, reason: 'error', error: 'budget exhausted' });
assert.equal(queueProjection.routines.get(queueRoutine.id).lastOutcome, 'failed');
queueController.stop();

const replaceRoutine = { ...routine, id: 'replace', overlap: 'replace' };
assert.equal(validateRoutine(replaceRoutine, { ...cfg, routines: [replaceRoutine] }), true);
const replaceProjection = new Projection({ ...cfg, routines: [replaceRoutine] });
const replaceEvents = [];
let replaceSeq = 0;
const replaceEmit = (kind, data, meta = {}) => {
  const event = { seq: ++replaceSeq, ts: 1_700_000_100_000 + replaceSeq, kind, data, ...meta };
  replaceEvents.push(event); replaceProjection.apply(event); return event;
};
let replaceSpawns = 0;
const cancelled = [];
const replaceController = startRoutines({
  cfg: { ...cfg, routines: [replaceRoutine] },
  registry: {
    spawn() { return { id: `replace-session-${++replaceSpawns}` }; },
    command(kind, payload) { if (kind === 'cancel') cancelled.push(...(payload.sessionIds ?? [])); return { cancelled: (payload.sessionIds ?? []).length }; },
  },
  log: { subscribe() { return () => {}; } },
  emit: replaceEmit, projection: replaceProjection, timers,
  now: () => new Date('2024-01-01T00:00:00Z'), createId: () => `replace-${replaceSpawns + 1}`,
});
replaceController.setEnabled(replaceRoutine.id, true, { confirmRisk: true });
const firstReplace = replaceController.run(replaceRoutine.id);
assert.equal(firstReplace.admitted, true);
const secondReplace = replaceController.run(replaceRoutine.id);
assert.equal(secondReplace.admitted, true, 'replace starts a new run while one is active');
assert.equal(replaceSpawns, 2);
assert.deepEqual(cancelled, [firstReplace.sessionId]);
assert.equal(replaceEvents.some((event) => event.kind === 'routine.failed' && event.data.reason === 'replaced'), true);
assert.equal(replaceProjection.routines.get(replaceRoutine.id).activeRunId, secondReplace.runId);
replaceController.stop();

const parallelRoutine = { ...routine, id: 'parallel', overlap: 'parallel' };
assert.equal(validateRoutine(parallelRoutine, { ...cfg, routines: [parallelRoutine] }), true);
const parallelProjection = new Projection({ ...cfg, routines: [parallelRoutine] });
const parallelListeners = new Set();
const parallelEvents = [];
let parallelSeq = 0;
const parallelLog = { subscribe(fn) { parallelListeners.add(fn); return () => parallelListeners.delete(fn); }, publish(event) { for (const fn of parallelListeners) fn(event); } };
const parallelEmit = (kind, data, meta = {}) => {
  const event = { seq: ++parallelSeq, ts: 1_700_000_200_000 + parallelSeq, kind, data, ...meta };
  parallelEvents.push(event); parallelProjection.apply(event); parallelLog.publish(event); return event;
};
let parallelSpawns = 0;
const parallelController = startRoutines({
  cfg: { ...cfg, routines: [parallelRoutine] },
  registry: { spawn() { return { id: `parallel-session-${++parallelSpawns}` }; } },
  log: parallelLog, emit: parallelEmit, projection: parallelProjection, timers,
  now: () => new Date('2024-01-01T00:00:00Z'), createId: () => `parallel-${parallelSpawns + 1}`,
});
parallelController.setEnabled(parallelRoutine.id, true, { confirmRisk: true });
const firstParallel = parallelController.run(parallelRoutine.id);
const secondParallel = parallelController.run(parallelRoutine.id);
assert.equal(firstParallel.admitted && secondParallel.admitted, true, 'parallel admits concurrent runs');
assert.equal(parallelSpawns, 2);
assert.deepEqual(parallelController.details().parallel.activeRunIds.sort(), [firstParallel.runId, secondParallel.runId].sort());
parallelEmit('session.turn_complete', { sessionId: firstParallel.sessionId, isError: false });
assert.equal(parallelProjection.routines.get(parallelRoutine.id).activeRunId, secondParallel.runId);
assert.equal(parallelController.details().parallel.activeRunIds.length, 1);
parallelController.stop();

console.log('routine tests passed');
