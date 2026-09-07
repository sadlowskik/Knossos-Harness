// Real filesystem activity for every mounted workspace.
// chokidar v4 dropped glob support, so the ignore list from field.yaml is compiled
// into a single matcher function here.
import chokidar from 'chokidar';
import path from 'node:path';
import { buildMatcher } from '../glob.js';

// `.field-state` matters more than it looks: the event store writes its SQLite WAL
// there on every append. Watching it would turn each event into a filesystem event
// and feed the log back into itself.
const ALWAYS_IGNORE = [
  '**/.git/**', '**/node_modules/**', '**/target/**', '**/dist/**',
  '**/.field-state/**', '**/.pytest_cache/**', '**/__pycache__/**', '**/.venv/**',
  '**/*.pyc', '**/*.swp', '**/*.tmp', '**/.DS_Store',
];

export function startFsWatchers(cfg, emit) {
  const watchers = [];

  for (const w of cfg.workspaces) {
    if (!w.mounted) continue;
    const ignore = buildMatcher([...ALWAYS_IGNORE, ...(w.watch?.ignore ?? [])]);

    // Patterns may be absolute-ish (`**/node_modules/**`) or workspace-relative
    // (`RTS/**`). Test both forms so field.yaml can use whichever reads clearer.
    const root = path.resolve(w.path);
    const ignores = (abs) => {
      if (ignore(abs)) return true;
      const rel = path.relative(root, abs).replace(/\\/g, '/');
      return rel !== '' && !rel.startsWith('..') && ignore(rel);
    };

    const watcher = chokidar.watch(w.path, {
      ignored: ignores,
      ignoreInitial: true,
      persistent: true,
      followSymlinks: false,     // npm workspaces symlink packages back into node_modules
      depth: 14,
      awaitWriteFinish: { stabilityThreshold: 250, pollInterval: 50 },
    });

    const on = (change) => (abs) => {
      const rel = path.relative(w.path, abs).replace(/\\/g, '/');
      if (!rel || rel.startsWith('..')) return;
      const dir = rel.includes('/') ? rel.slice(0, rel.lastIndexOf('/')) : '';
      emit('fs.changed', { workspaceId: w.id, path: rel, dir, change }, { subject: w.id });
    };

    watcher
      .on('add', on('add'))
      .on('change', on('change'))
      .on('unlink', on('unlink'))
      .on('error', (e) => console.error(`[fs:${w.id}]`, e.message));

    watchers.push(watcher);
  }

  return () => Promise.all(watchers.map((w) => w.close()));
}
