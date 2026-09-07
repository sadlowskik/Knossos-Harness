import assert from 'node:assert/strict';
import { assertSafeTimeline, operationsTimeline } from '../src/simulation/field-simulator.js';

const timeline = operationsTimeline('test-run', { workspaceId: 'any-codebase', workspaceName: 'Any Codebase' });
assert.equal(assertSafeTimeline(timeline), true);
assert.ok(timeline.length > 100, 'scenario should be event-rich');
assert.equal(timeline.filter((e) => e.kind === 'session.spawned').length, 12);
assert.ok(timeline.at(-1).at >= 45_000 && timeline.at(-1).at <= 90_000, '1x rehearsal should last 45-90 seconds');
assert.ok(timeline.some((e) => e.kind === 'session.state' && e.data.state === 'moving'));
assert.ok(timeline.filter((e) => e.kind === 'session.spawned').every((e) => e.data.initialOrders && e.data.systemPrompt));
assert.ok(timeline.some((e) => e.kind === 'agent.communication'));
assert.ok(timeline.some((e) => e.kind === 'browser.navigated'));
assert.ok(timeline.some((e) => e.kind === 'fs.changed'));
assert.ok(timeline.some((e) => e.kind === 'session.progress'));
assert.ok(timeline.some((e) => e.kind === 'session.state' && e.data.state === 'blocked'));
assert.ok(timeline.filter((e) => e.data.workspaceId).every((e) => e.data.workspaceId === 'any-codebase'));
assert.ok(timeline.every((e) => e.data.model !== 'GLM-5.3-Flash'));
console.log(`simulation: ${timeline.length} safe synthetic events`);
