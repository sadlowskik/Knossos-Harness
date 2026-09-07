import assert from 'node:assert/strict';
import { readCleanRevision } from '../src/watch/git.js';

const revision = '0123456789abcdef0123456789abcdef01234567';
const clean = await readCleanRevision('/workspace', {
  read: async () => ({ branch: 'main', files: [] }),
  runGit: async (_cwd, args) => {
    assert.deepEqual(args, ['rev-parse', '--verify', 'HEAD']);
    return `${revision}\n`;
  },
});
assert.deepEqual(clean, { revision, branch: 'main' });

await assert.rejects(() => readCleanRevision('/workspace', {
  read: async () => ({ branch: 'main', files: [
    { path: 'tracked.txt' }, { path: 'untracked.txt' },
  ] }),
  runGit: async () => { throw new Error('must not inspect HEAD after dirty status'); },
}), /uncommitted changes \(tracked.txt, untracked.txt\)/);

await assert.rejects(() => readCleanRevision('/workspace', {
  read: async () => ({ branch: 'main', files: [] }),
  runGit: async () => 'not-a-revision',
}), /not a valid Git revision/);

console.log('git state: clean revision binding and dirty-worktree refusal passed');
