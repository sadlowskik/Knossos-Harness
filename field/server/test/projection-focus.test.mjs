import assert from 'node:assert/strict';
import { Projection } from '../src/store/projection.js';

const projection = new Projection({
  workspaces: [{ id: 'alpha', name: 'Alpha', mounted: true }],
  endpoints: [],
  websites: [],
});

// A session that has done nothing has no file-level position.
projection.apply({
  seq: 1, ts: 1000, kind: 'session.spawned', source: 'observed', subject: 'unit-1',
  data: { sessionId: 'unit-1', workspaceId: 'alpha', role: 'builder' },
});
let snap = projection.snapshot(2000);
let unit = snap.sessions.find((s) => s.id === 'unit-1');
assert.equal(unit.focusPath, undefined, 'a fresh session has no focusPath');
assert.equal(unit.focusDir, undefined, 'a fresh session has no focusDir');

// A tool_use carrying a path sets focusPath and keeps focusDir as the folder anchor.
projection.apply({
  seq: 2, ts: 1100, kind: 'session.tool_use', source: 'observed', subject: 'unit-1',
  data: { sessionId: 'unit-1', name: 'Edit', workspaceId: 'alpha', dir: 'src/core', path: 'src/core/service.ts', summary: 'Editing service' },
});
snap = projection.snapshot(2000);
unit = snap.sessions.find((s) => s.id === 'unit-1');
assert.equal(unit.focusPath, 'src/core/service.ts', 'a tool_use with a path sets focusPath');
assert.equal(unit.focusDir, 'src/core', 'focusDir remains the folder anchor');

// A tool_use without a path leaves focusPath at its prior value (never clobbered to undefined).
projection.apply({
  seq: 3, ts: 1200, kind: 'session.tool_use', source: 'observed', subject: 'unit-1',
  data: { sessionId: 'unit-1', name: 'Bash', workspaceId: 'alpha', dir: 'src/core', summary: 'Running tests' },
});
snap = projection.snapshot(2000);
unit = snap.sessions.find((s) => s.id === 'unit-1');
assert.equal(unit.focusPath, 'src/core/service.ts', 'a pathless tool_use does not erase focusPath');

// A second session that only ran a pathless tool never gains a focusPath.
projection.apply({
  seq: 4, ts: 1300, kind: 'session.spawned', source: 'observed', subject: 'unit-2',
  data: { sessionId: 'unit-2', workspaceId: 'alpha', role: 'verifier' },
});
projection.apply({
  seq: 5, ts: 1400, kind: 'session.tool_use', source: 'observed', subject: 'unit-2',
  data: { sessionId: 'unit-2', name: 'Bash', workspaceId: 'alpha', dir: 'tests', summary: 'Running suite' },
});
snap = projection.snapshot(2000);
unit = snap.sessions.find((s) => s.id === 'unit-2');
assert.equal(unit.focusPath, undefined, 'absence of a path leaves focusPath undefined');
assert.equal(unit.focusDir, 'tests', 'focusDir still tracks the folder');

console.log('projection focus: tool_use path sets focusPath additively, absence leaves it undefined');
