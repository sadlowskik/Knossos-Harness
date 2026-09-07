// Real Git state per workspace. Polled, compared, and only emitted when it actually changes.
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { buildChildEnvironment } from '../child-env.js';

const run = promisify(execFile);

async function git(cwd, args, maxBuffer = 4 * 1024 * 1024) {
  const safeArgs = [
    '-c', `core.hooksPath=${process.platform === 'win32' ? 'NUL' : '/dev/null'}`,
    '-c', 'core.fsmonitor=false',
    '-c', 'credential.helper=',
    ...args,
  ];
  const { stdout } = await run('git', safeArgs, {
    cwd, maxBuffer, windowsHide: true,
    env: buildChildEnvironment({ overrides: { GIT_OPTIONAL_LOCKS: '0', GIT_TERMINAL_PROMPT: '0' } }),
  });
  return stdout;
}

const STATUS_LABEL = {
  M: 'modified', A: 'added', D: 'deleted', R: 'renamed',
  C: 'copied', U: 'conflicted', '?': 'untracked', '!': 'ignored',
};

export async function readStatus(cwd) {
  // A workspace can sit inside a larger repository. Scope the status to this subtree,
  // and report paths relative to the workspace rather than to the repository root.
  let prefix = '';
  try {
    prefix = (await git(cwd, ['rev-parse', '--show-prefix'])).trim().replace(/\\/g, '/');
  } catch { prefix = ''; }

  const out = await git(cwd, ['status', '--porcelain=v1', '-b', '--', '.']);
  const lines = out.split('\n').filter(Boolean);
  let branch = null; let ahead = 0; let behind = 0;
  const files = [];

  for (const line of lines) {
    if (line.startsWith('## ')) {
      const head = line.slice(3);
      branch = head.split('...')[0].trim();
      const a = /ahead (\d+)/.exec(head); if (a) ahead = Number(a[1]);
      const b = /behind (\d+)/.exec(head); if (b) behind = Number(b[1]);
      continue;
    }
    const x = line[0]; const y = line[1];
    const file = line.slice(3).trim();
    const code = x !== ' ' && x !== '?' ? x : y !== ' ' ? y : x;
    const norm = file.replace(/\\/g, '/');
    const scoped = prefix && norm.startsWith(prefix) ? norm.slice(prefix.length) : norm;
    files.push({
      // An untracked workspace is reported by git as its own directory, which strips
      // to an empty string. Show it as the workspace root rather than as a blank row.
      path: scoped === '' || scoped === '/' ? '.' : scoped,
      status: STATUS_LABEL[code] ?? 'changed',
      staged: x !== ' ' && x !== '?',
    });
  }
  return { branch, ahead, behind, files };
}

export async function diffFile(cwd, file) {
  try {
    const staged = await git(cwd, ['diff', '--no-ext-diff', '--cached', '--', file]);
    const unstaged = await git(cwd, ['diff', '--no-ext-diff', '--', file]);
    const text = (staged + unstaged).trim();
    if (text) return text;
    // A brand new untracked file has no diff; show it as an addition.
    const show = await git(cwd, ['status', '--porcelain=v1', '--', file]);
    return show.trim().startsWith('??') ? '(untracked — no diff yet)' : '(no changes)';
  } catch (e) {
    return `(git diff failed: ${e.message})`;
  }
}

export async function readCleanRevision(cwd, { read = readStatus, runGit = git } = {}) {
  const status = await read(cwd);
  if (status.files.length) {
    const sample = status.files.slice(0, 5).map((file) => file.path).join(', ');
    const remainder = status.files.length > 5 ? ` and ${status.files.length - 5} more` : '';
    throw new Error(`workspace has uncommitted changes (${sample}${remainder})`);
  }
  const revision = (await runGit(cwd, ['rev-parse', '--verify', 'HEAD'])).trim();
  if (!/^[a-f0-9]{40,64}$/i.test(revision)) throw new Error('workspace HEAD is not a valid Git revision');
  return { revision, branch: status.branch };
}

export async function log(cwd, limit = 40) {
  try {
    // %x1f is a literal unit-separator byte, which cannot appear in a commit subject.
    const out = await git(cwd, ['log', `-${limit}`, '--pretty=format:%h%x1f%an%x1f%ar%x1f%s']);
    return out.split('\n').filter(Boolean).map((l) => {
      const [hash, author, when, subject] = l.split('\x1f');
      return { hash, author, when, subject };
    });
  } catch { return []; }
}

export function startGitWatchers(cfg, emit, intervalMs = 5000) {
  const last = new Map();
  let stopped = false;

  async function tick() {
    for (const w of cfg.workspaces) {
      if (!w.mounted || w.git === false) continue;
      try {
        const st = await readStatus(w.path);
        const fingerprint = JSON.stringify(st);
        if (last.get(w.id) === fingerprint) continue;
        last.set(w.id, fingerprint);
        emit('git.status', { workspaceId: w.id, ...st }, { subject: w.id });
      } catch {
        // Not a repository, or git is unavailable. Report once, then stay quiet.
        if (!last.has(w.id)) {
          last.set(w.id, 'nogit');
          emit('git.status', { workspaceId: w.id, branch: null, ahead: 0, behind: 0, files: [] }, { subject: w.id });
        }
      }
    }
    if (!stopped) setTimeout(tick, intervalMs);
  }

  tick();
  return () => { stopped = true; };
}
