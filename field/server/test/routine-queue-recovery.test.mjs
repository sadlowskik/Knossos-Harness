import assert from 'node:assert/strict';
import { Projection } from '../src/store/projection.js';
import { startRoutines } from '../src/routines.js';

const config = {
  defaults: {}, roles: [{ id: 'builder' }], agents: [{ id: 'builder', role: 'builder' }],
  endpoints: [{ id: 'local', model: 'fixture' }],
  workspaces: [{ id: 'workspace', path: '/fixture', mounted: true }], websites: [],
  routines: [{ id: 'r', enabled: true, role: 'builder', endpoint: 'local', workspace: 'workspace',
    orders: 'Review', budget_usd: 1, overlap: 'queue-one', cooldown_seconds: 2,
    trigger: { kind: 'fs_change', paths: ['src/**'] }, completion: { kind: 'report' } }],
};

function fixture() {
  const events = [], listeners = new Set(), jobs = new Set();
  let clock = 1700000000000, ids = 0, starts = 0, projection;
  const test = { failEmit: null, failSpawn: false, events, get starts() { return starts; } };
  const emit = (kind, data, meta = {}) => {
    if (test.failEmit === kind) throw new Error('disk full');
    const event = { seq: events.length + 1, ts: clock, kind, data: structuredClone(data), ...meta };
    events.push(event); projection.apply(event);
    for (const listener of [...listeners]) listener(event);
    return event;
  };
  const timers = {
    setTimeout(fn, ms) { const job = { fn, due: clock + ms, unref() {} }; jobs.add(job); return job; },
    clearTimeout(job) { jobs.delete(job); },
  };
  test.advance = (ms) => {
    clock += ms;
    for (const job of [...jobs]) if (job.due <= clock && jobs.delete(job)) job.fn();
  };
  test.boot = (cfg = config) => {
    projection = new Projection(cfg);
    for (const event of events) projection.apply(event);
    return startRoutines({ cfg, projection, emit, timers, now: () => new Date(clock),
      log: { subscribe(fn) { listeners.add(fn); return () => listeners.delete(fn); } },
      registry: { spawn() { if (test.failSpawn) throw new Error('budget exhausted'); return { id: `s${++starts}` }; } },
      createId: () => `run${++ids}` });
  };
  test.settle = (run) => emit('session.turn_complete', { sessionId: run.sessionId });
  test.emit = emit;
  test.state = () => projection.routines.get('r');
  return test;
}

{
  const f = fixture(); let controller = f.boot();
  const active = controller.run('r'); f.settle(active);
  assert.equal(controller.run('r').queued, true, 'enqueue during idle cooldown schedules a pump');
  assert.equal(controller.run('r').reason, 'queue_full');
  controller.stop();
  controller = f.boot(); controller.start();
  assert.equal(f.starts, 1, 'restart respects persisted cooldown');
  f.advance(1999); assert.equal(f.starts, 1);
  f.advance(1); assert.equal(f.starts, 2, 'unclaimed queued work resumes once');
  assert.equal(f.state().queuedRuns.length, 0);
  const claim = f.events.findIndex(e => e.kind === 'routine.queue_claimed');
  const triggered = f.events.findIndex((e, i) => i > claim && e.kind === 'routine.triggered');
  assert.ok(claim >= 0 && triggered > claim, 'durable claim precedes dispatch result');
  controller.stop(); f.advance(10000); assert.equal(f.starts, 2);
  assert.equal(controller.run('r').reason, 'stopped');
}

{
  const f = fixture(); const controller = f.boot();
  const active = controller.run('r');
  f.failEmit = 'routine.queued';
  assert.throws(() => controller.run('r'), /disk full/);
  assert.equal(controller.details().r.queuedRuns.length, 0, 'failed persistence creates no phantom queue');
  f.failEmit = null;
  assert.equal(controller.run('r').queued, true);
  f.settle(active);
  assert.equal(controller.run('r').reason, 'queue_full', 'waiting timer retains the single queue slot');
  controller.setEnabled('r', false);
  f.advance(2000); assert.equal(f.starts, 1, 'disable cancels cooldown dispatch');
  controller.stop();
}

{
  const f = fixture(); let controller = f.boot();
  const active = controller.run('r'); controller.run('r'); f.settle(active);
  const queued = f.state().queuedRuns[0];
  f.emit('routine.queue_claimed', { routineId: 'r', runId: queued.runId });
  controller.stop(); controller = f.boot(); controller.start(); f.advance(2000);
  assert.equal(f.starts, 1, 'crash after claim never blindly replays consequential work');
  assert.ok(f.events.some(e => e.data.reason === 'interrupted_after_queue_claim'));
  assert.equal(f.state().queuedRuns.length, 0);
  controller.stop();
}

{
  const f = fixture(); let controller = f.boot();
  const active = controller.run('r'); controller.run('r'); f.settle(active); controller.stop();
  const changed = structuredClone(config); changed.routines[0].orders = 'Different authority';
  controller = f.boot(changed); controller.start(); f.advance(2000);
  assert.equal(f.starts, 1, 'restart cannot execute queued work under changed configuration');
  assert.ok(f.events.some(e => e.data.reason === 'queued_configuration_changed'));
  controller.stop();
}

{
  const f = fixture(); const controller = f.boot();
  const active = controller.run('r'); controller.run('r'); f.settle(active);
  f.failSpawn = true; f.advance(2000);
  assert.equal(f.state().queuedRuns.length, 0, 'failed admission settles the claimed queue');
  assert.equal(f.state().lastOutcome, 'failed');
  f.failSpawn = false; controller.stop();
  const restarted = f.boot(); restarted.start(); f.advance(5000);
  assert.equal(f.starts, 1, 'failed admission is not silently retried after restart');
  restarted.stop();
}

{
  const f = fixture(); const liveConfig = structuredClone(config);
  const controller = f.boot(liveConfig);
  const active = controller.run('r'); controller.run('r'); f.settle(active);
  liveConfig.routines[0].budget_usd = 20;
  f.advance(2000);
  assert.equal(f.starts, 1, 'configuration changes during cooldown invalidate queued authority');
  assert.equal(f.state().queuedRuns.length, 0);
  controller.stop();
}

console.log('routine queue recovery: cooldown, replay, authority, claim, disable and persistence failures pass');
