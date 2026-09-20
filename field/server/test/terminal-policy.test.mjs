import assert from 'node:assert/strict';
import {
  OutputBudget,
  redactCommand,
  TERMINAL_LIMITS,
  validateTerminalCommand,
} from '../src/terminal-policy.js';

assert.equal(validateTerminalCommand('npm test'), 'npm test');
assert.throws(() => validateTerminalCommand(''), /required/);
assert.throws(() => validateTerminalCommand('bad\0command'), /null byte/);
assert.throws(() => validateTerminalCommand('x'.repeat(TERMINAL_LIMITS.commandChars + 1)), /too long/);

const redacted = redactCommand(
  'curl -H "Authorization: Bearer abcdefghijklmnop" x; TOKEN=secretvalue API_KEY=anothersecret sk-testcredential',
);
assert.equal(redacted.includes('abcdefghijklmnop'), false);
assert.equal(redacted.includes('secretvalue'), false);
assert.equal(redacted.includes('anothersecret'), false);
assert.equal(redacted.includes('sk-testcredential'), false);

const bytes = new OutputBudget({ bytes: 5, lines: 10 });
assert.deepEqual(bytes.accept(Buffer.from('1234')), { text: '1234', exceeded: false });
assert.deepEqual(bytes.accept(Buffer.from('567')), { text: '5', exceeded: true });
assert.deepEqual(bytes.accept(Buffer.from('ignored')), { text: '', exceeded: true });

const lines = new OutputBudget({ bytes: 100, lines: 2 });
assert.deepEqual(lines.accept('one\ntwo\nthree\n'), { text: 'one\ntwo\n', exceeded: true });

console.log('terminal policy: validation, credential redaction, byte cap, and line cap passed');
