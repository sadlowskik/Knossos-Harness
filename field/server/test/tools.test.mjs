// Tool calls are how an agent gets its position on the Field, so the translation from
// a real tool call to a workspace and directory has to be exact.
//
// Regression: Grep/Glob/Bash accept a file *or* a directory, and the first version used
// the resolved path as-is. A Grep against one file put that file on the Field as though
// it were a folder.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { describeTool, locate } from '../src/harness/tools.js';

// A real tree on disk: locate() stats paths, so this cannot be faked with strings.
const root = fs.mkdtempSync(path.join(os.tmpdir(), 'field-tools-'));
const nested = path.join(root, 'inner');
fs.mkdirSync(path.join(root, 'core', 'placement', 'src'), { recursive: true });
fs.mkdirSync(path.join(nested, 'field', 'roles'), { recursive: true });
fs.writeFileSync(path.join(root, 'core', 'placement', 'src', 'lib.rs'), '// x');
fs.writeFileSync(path.join(nested, 'field', 'roles', 'scout.md'), '# scout');

const workspaces = [
  { id: 'outer', path: root, canonicalPath: fs.realpathSync(root) },
  { id: 'inner', path: nested, canonicalPath: fs.realpathSync(nested) },
];
const ctx = { workspaces, cwd: root };
const p = (...parts) => path.join(root, ...parts);

let failures = 0;
const check = (ok, msg) => { if (!ok) { failures++; console.error('FAIL ' + msg); } };

const cases = [
  ['Read', { file_path: p('core', 'placement', 'src', 'lib.rs') }, 'outer', 'core/placement/src'],
  ['Grep', { pattern: 'fn', path: p('core', 'placement', 'src', 'lib.rs') }, 'outer', 'core/placement/src'],
  ['Grep', { pattern: 'fn', path: p('core', 'placement', 'src') }, 'outer', 'core/placement/src'],
  ['Glob', { pattern: '*.rs', path: p('core') }, 'outer', 'core'],
  ['Bash', { command: 'ls -la' }, 'outer', ''],
];

for (const [name, input, wantWs, wantDir] of cases) {
  const r = describeTool(name, input, ctx);
  check(r.workspaceId === wantWs && r.dir === wantDir,
    `${name} => ws=${r.workspaceId} dir="${r.dir}" (want ${wantWs} "${wantDir}")`);
}

// The deepest matching workspace wins, so a nested workspace is never attributed to its parent.
const inner = describeTool('Read', { file_path: path.join(nested, 'field', 'roles', 'scout.md') }, ctx);
check(inner.workspaceId === 'inner' && inner.dir === 'field/roles',
  `nested => ws=${inner.workspaceId} dir="${inner.dir}"`);

// Paths outside every workspace are not placed at all.
check(locate(path.join(os.tmpdir(), 'somewhere-else', 'x.txt'), workspaces) === null,
  'a path outside every workspace should not resolve');

// Browser and delegation calls are recognised as such.
const web = describeTool('WebFetch', { url: 'https://docs.claude.com/en/api' }, ctx);
check(web.browser?.domain === 'docs.claude.com', `WebFetch domain => ${web.browser?.domain}`);

const task = describeTool('Task', { description: 'audit', subagent_type: 'Explore' }, ctx);
check(task.delegation?.type === 'Explore', `Task delegation => ${task.delegation?.type}`);

const todo = describeTool('TodoWrite', {
  todos: [{ status: 'completed' }, { status: 'pending' }, { status: 'completed' }],
}, ctx);
check(todo.progress?.done === 2 && todo.progress?.total === 3,
  `TodoWrite progress => ${JSON.stringify(todo.progress)}`);

fs.rmSync(root, { recursive: true, force: true });

if (failures) {
  console.error(`\ntools: ${failures} failed`);
  process.exit(1);
}
console.log('tools: 10 assertions passed');
