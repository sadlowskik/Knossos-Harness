import assert from 'node:assert/strict';
import { once } from 'node:events';

process.env.FIELD_CLAUDE_BIN = process.execPath;
process.env.FIELD_KNOSSOS_BIN = process.execPath;
const { HarnessSession } = await import('../src/harness/session.js');
const { KnossosSession } = await import('../src/harness/knossos-session.js');

for (const Adapter of [HarnessSession, KnossosSession]) {
  const session = new Adapter({ id: 'generation', cwd: process.cwd() });
  session.buildArgs = () => ['-e', 'process.stdin.resume(); setInterval(() => {}, 1000);'];
  const events = [];
  session.on('event', (kind, data) => events.push({ kind, data }));
  const timeout = setTimeout(() => {
    for (const child of session.ownedProcesses) child.kill();
    throw new Error('process generation fixture timed out');
  }, 5000);
  try {
    session.start();
    const old = session.proc;
    await once(old, 'spawn');
    const oldClosed = once(old, 'close');
    session.pause();
    assert.ok(session.ownedProcesses.has(old), 'terminating process retains capacity until close');
    session.resume();
    const replacement = session.proc;
    assert.notEqual(replacement, old);
    await oldClosed;
    assert.equal(session.proc, replacement, 'old close cannot clear the replacement');
    assert.ok(!session.ownedProcesses.has(old));
    assert.ok(session.ownedProcesses.has(replacement));
    assert.equal(events.filter(e => e.kind === 'session.ended').length, 0, 'old close cannot terminate the replacement session');
    const replacementClosed = once(replacement, 'close');
    session.cancel();
    await replacementClosed;
    assert.equal(session.ownedProcesses.size, 0);
  } finally {
    clearTimeout(timeout);
    for (const child of session.ownedProcesses) child.kill();
  }
}
console.log('process generation: immediate pause/resume preserves replacement ownership for both adapters');
