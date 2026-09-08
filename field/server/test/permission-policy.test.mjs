import assert from 'node:assert/strict';
import path from 'node:path';
import { denyListFor, evaluatePermissionPolicy, validateEgressUrl } from '../src/harness/registry.js';
import { HarnessSession } from '../src/harness/session.js';

const root = path.resolve('C:/workspace/project');
const check = (overrides) => evaluatePermissionPolicy({
  toolName: 'Read', input: {}, workspacePath: root,
  readOnly: false, environmentScope: 'sandbox', ...overrides,
});

assert.match(check({
  toolName: 'Write', input: { file_path: '../outside.txt' },
}).message, /escapes/);
assert.match(check({
  toolName: 'Write', input: { file_path: 'src/inside.txt' }, environmentScope: 'production-readonly',
}).message, /read-only/);
assert.match(check({
  toolName: 'Edit', input: { file_path: 'src/inside.txt' }, readOnly: true,
}).message, /role is read-only/);
assert.match(check({
  toolName: 'Bash', input: { command: 'rm -rf build' }, readOnly: true,
}).message, /may not mutate/);
assert.match(check({
  toolName: 'Bash', input: { command: 'Get-Content ..\\secrets.env' }, readOnly: true,
}).message, /escape scope/);
assert.equal(check({
  toolName: 'Bash', input: { command: 'npm test' }, readOnly: true,
}), null);
assert.equal(check({
  toolName: 'Write', input: { file_path: 'src/inside.txt' }, readOnly: false,
}), null);
assert.match(check({
  toolName: 'write_file', input: { path: 'src/code.rs' },
  allowedTools: ['Read', 'Grep', 'Glob'], readOnly: true,
}).message, /not in this role's tool capability set/);
assert.match(check({
  toolName: 'WebFetch', input: { url: 'https://example.com' },
  allowedTools: ['Read', 'Grep', 'Glob'],
}).message, /not in this role's tool capability set/);
assert.equal(check({
  toolName: 'search', input: { query: 'needle' },
  allowedTools: ['Read', 'Grep', 'Glob'], readOnly: true,
}), null, 'Knossos search maps to the declared Grep capability');
assert.equal(check({
  toolName: 'write_file', input: { path: 'docs/plan.md' },
  allowedTools: ['Read', 'Write'], writeScope: ['**/*.md'],
}), null);
assert.match(check({
  toolName: 'write_file', input: { path: 'src/code.rs' },
  allowedTools: ['Read', 'Write'], writeScope: ['**/*.md'],
}).message, /outside this role's write scope/);
assert.match(check({
  toolName: 'write_file', input: {},
  allowedTools: ['Read', 'Write'], writeScope: ['**/*.md'],
}).message, /no verifiable workspace path/);
assert.match(check({
  toolName: 'write_file', input: { path: 'docs/plan.md' },
  allowedTools: ['Read', 'Write'],
  validateWritePath: () => { throw new Error('junction'); },
}).message, /workspace path or sensitive-file policy/);
assert.equal(validateEgressUrl('https://docs.claude.com/en/docs', ['docs.claude.com']), null);
assert.equal(validateEgressUrl('https://sub.docs.claude.com/page', ['docs.claude.com']), null);
assert.match(validateEgressUrl('http://127.0.0.1:8080/admin', ['127.0.0.1']), /blocked/);
assert.match(validateEgressUrl('http://[::1]:8080/admin', ['::1']), /blocked/);
assert.match(validateEgressUrl('http://169.254.169.254/latest/meta-data', ['169.254.169.254']), /blocked/);
assert.match(validateEgressUrl('https://user:pass@docs.claude.com/', ['docs.claude.com']), /credential-free/);
assert.match(validateEgressUrl('file:///etc/passwd', ['docs.claude.com']), /credential-free/);
assert.match(validateEgressUrl('https://evil.example/', ['docs.claude.com']), /not declared/);
assert.equal(check({
  toolName: 'WebFetch', input: { url: 'https://docs.claude.com/en/docs' },
  allowedTools: ['WebFetch'], allowedDomains: ['docs.claude.com'],
}), null);
assert.match(check({
  toolName: 'WebFetch', input: { url: 'http://localhost:8080/secrets' },
  allowedTools: ['WebFetch'], allowedDomains: ['localhost'],
}).message, /blocked/);
assert.match(check({
  toolName: 'Bash', input: { command: 'curl https://docs.claude.com' },
  allowedTools: ['Bash'], allowedDomains: ['docs.claude.com'],
}).message, /network-capable shell/);
assert.match(check({
  toolName: 'Read', input: { file_path: '../outside.txt' }, readOnly: true,
}).message, /escapes/);

assert.deepEqual(
  denyListFor(
    { tools_allow: ['Read', 'Grep', 'Edit', 'Bash'] },
    { tools_allow: ['Read', 'Grep'] },
  ),
  ['Edit', 'Bash', 'Task', 'Agent'],
);
assert.deepEqual(
  denyListFor(
    { read_only: true, tools_allow: ['Read', 'Grep', 'WebSearch'] },
    { tools_allow: ['Read'] },
  ),
  ['Edit', 'Write', 'NotebookEdit', 'PowerShell', 'Grep', 'WebSearch', 'Task', 'Agent'],
);

const boundedSession = new HarnessSession({
  id: '00000000-0000-4000-8000-000000000001',
  effort: 'medium', cwd: root, allowedTools: ['Read', 'Grep'],
  disallowedTools: ['Task', 'Agent'],
});
const args = boundedSession.buildArgs();
assert.equal(args[args.indexOf('--tools') + 1], 'Read,Grep', 'role allowlist limits available Claude built-ins');
assert.equal(args.includes('--allowedTools'), false, 'consequential tools still reach Field approval');
assert.equal(args[args.indexOf('--disallowedTools') + 1], 'Task,Agent');

console.log('permissions: read-only roles, per-agent loadouts, environment scope, path escape, and shell mutation policy passed');
