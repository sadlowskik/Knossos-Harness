import assert from 'node:assert/strict';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {
  canonicalizeWorkspace,
  readWorkspaceFile,
  resolveWorkspacePath,
  writeWorkspaceFile,
} from '../src/workspace-path.js';

const fixture = fs.mkdtempSync(path.join(os.tmpdir(), 'field-paths-'));
const root = path.join(fixture, 'workspace');
const outside = path.join(fixture, 'outside');
fs.mkdirSync(path.join(root, 'src'), { recursive: true });
fs.mkdirSync(outside);
fs.writeFileSync(path.join(root, 'src', 'visible.txt'), 'visible');
fs.writeFileSync(path.join(root, '.env.local'), 'SECRET=canary');
fs.writeFileSync(path.join(root, 'private.pem'), 'private');
fs.writeFileSync(path.join(root, 'ignored.txt'), 'ignored');
fs.writeFileSync(path.join(root, '.gitignore'), 'ignored.txt\nbuild/\n!build/keep.txt\n');
fs.mkdirSync(path.join(root, 'build'));
fs.writeFileSync(path.join(root, 'build', 'drop.txt'), 'drop');
fs.writeFileSync(path.join(root, 'build', 'keep.txt'), 'keep');
fs.writeFileSync(path.join(outside, 'escape.txt'), 'escape');

const workspace = canonicalizeWorkspace({ id: 'fixture', path: root }, fixture);
const cfg = { workspaces: [workspace] };
assert.equal(workspace.mounted, true);
assert.equal(workspace.path, fs.realpathSync(root));

const visible = resolveWorkspacePath(cfg, 'fixture', 'src/visible.txt', { type: 'file' });
assert.equal(readWorkspaceFile(visible), 'visible');
assert.equal(workspace.isVisible('src/visible.txt'), true);
assert.equal(workspace.isVisible('ignored.txt'), false);
assert.equal(workspace.isVisible('build/drop.txt'), false);
assert.equal(workspace.isVisible('.env.local'), false);
assert.equal(workspace.isVisible('private.pem'), false);
assert.equal(workspace.isVisible('.envrc'), false);
assert.equal(workspace.isVisible('build/keep.txt'), false, 'excluded parent cannot be resurrected');
fs.writeFileSync(path.join(root, 'src', '.gitignore'), '/local.txt\n*.secret\n!keep.secret\n');
assert.equal(workspace.isVisible('src/local.txt'), false);
assert.equal(workspace.isVisible('local.txt'), true, 'nested policy stays scoped');
assert.equal(workspace.isVisible('src/hidden.secret'), false);
assert.equal(workspace.isVisible('src/keep.secret'), true);
fs.mkdirSync(path.join(root, 'src', 'nested'));
assert.equal(workspace.isVisible('src/nested/local.txt'), true, 'anchored rule is directory relative');
fs.writeFileSync(path.join(root, 'src', 'nested', '.gitignore'), '!keep.secret\n');
assert.equal(workspace.isVisible('src/nested/keep.secret'), true);
fs.appendFileSync(path.join(root, 'src', '.gitignore'), 'visible.txt\n');
assert.equal(workspace.isVisible('src/visible.txt'), false, 'policy edits apply immediately');
fs.writeFileSync(path.join(root, 'src', '.gitignore'), '');
assert.equal(workspace.isVisible('src/visible.txt'), true);
assert.throws(() => resolveWorkspacePath(cfg, 'fixture', '../outside/escape.txt'), /escapes|relative/);
assert.throws(() => resolveWorkspacePath(cfg, 'fixture', '.env.local'), /security policy/);
assert.throws(() => resolveWorkspacePath(cfg, 'fixture', 'ignored.txt'), /security policy/);

const created = resolveWorkspacePath(cfg, 'fixture', 'src/new.txt', { operation: 'write', type: 'file' });
assert.equal(created.exists, false);
writeWorkspaceFile(created, 'new');
assert.equal(fs.readFileSync(path.join(root, 'src', 'new.txt'), 'utf8'), 'new');

let linked = false;
try {
  fs.symlinkSync(outside, path.join(root, 'linked-outside'), process.platform === 'win32' ? 'junction' : 'dir');
  linked = true;
} catch (error) {
  if (error.code !== 'EPERM') throw error;
}
if (linked) {
  assert.throws(
    () => resolveWorkspacePath(cfg, 'fixture', 'linked-outside/escape.txt', { type: 'file' }),
    /symlink|junction|security policy/,
  );
  assert.throws(
    () => resolveWorkspacePath(cfg, 'fixture', 'linked-outside/new.txt', { operation: 'write', type: 'file' }),
    /symlink|junction|security policy/,
  );
}

fs.rmSync(fixture, { recursive: true, force: true });
console.log(`workspace paths: canonical roots, secrets, gitignore, writes, and ${linked ? 'link/junction' : 'link-skip'} containment passed`);
