// Regression test for the watch-ignore matcher.
//
// The original implementation chained .replace() calls, and each replacement rewrote
// regex syntax inserted by the previous one, so every pattern silently matched nothing.
// The visible cost was a cluttered Field; the real cost was that the event store's own
// SQLite WAL writes were watched, so each appended event produced a filesystem event
// that appended another. Keep these assertions.
import { buildMatcher, globToRegExp } from '../src/glob.js';

const ALWAYS = [
  '**/.git/**', '**/node_modules/**', '**/target/**', '**/dist/**',
  '**/.field-state/**', '**/.pytest_cache/**', '**/__pycache__/**', '**/.venv/**',
  '**/*.pyc', '**/*.swp', '**/*.tmp',
];
const ignored = buildMatcher(ALWAYS);
const ROOT = String.raw`C:\Users\dev\project`;

const MUST_IGNORE = [
  ROOT + String.raw`\.field-state\field.db-wal`,
  ROOT + String.raw`\.field-state`,
  ROOT + String.raw`\daedalus\.pytest_cache\README.md`,
  ROOT + String.raw`\model\__pycache__\pipeline.cpython-312.pyc`,
  ROOT + String.raw`\.venv\Lib\site-packages\x.py`,
  ROOT + String.raw`\.git\index`,
  ROOT + String.raw`\.git`,
  ROOT + String.raw`\node_modules\pkg\src\x.js`,
  ROOT + String.raw`\node_modules`,
  ROOT + String.raw`\web\dist\assets\index.js`,
  ROOT + String.raw`\core\target\debug\x`,
  ROOT + String.raw`\notes.swp`,
  '/home/dev/project/node_modules/pkg/index.js',
];

const MUST_WATCH = [
  ROOT + String.raw`\server\src\watch\fs.js`,
  ROOT + String.raw`\archiso\airootfs\usr\local\bin\cameo-install`,
  ROOT + String.raw`\field\roles\builder.md`,
  '/home/dev/project/src/main.rs',
];

// Routine triggers match workspace-relative posix paths.
const TRIGGERS = [
  ['core/**/*.rs', 'core/placement/src/lib.rs', true],
  ['core/**/*.rs', 'cameod/src/main.rs', false],
  ['core/**/*.rs', 'core/lib.rs', true],
  ['cameod/**/*.rs', 'cameod/src/hub.rs', true],
  ['docs/*.md', 'docs/architecture.md', true],
  ['docs/*.md', 'docs/deep/nested.md', false],
  ['**', 'anything/at/all.txt', true],
];

let failures = 0;
const check = (ok, msg) => { if (!ok) { failures++; console.error('FAIL ' + msg); } };

for (const p of MUST_IGNORE) check(ignored(p), `should ignore ${p}`);
for (const p of MUST_WATCH) check(!ignored(p), `should watch ${p}`);
for (const [pattern, path, want] of TRIGGERS) {
  check(globToRegExp(pattern).test(path) === want, `${pattern} vs ${path} should be ${want}`);
}

const total = MUST_IGNORE.length + MUST_WATCH.length + TRIGGERS.length;
if (failures) {
  console.error(`\nglob: ${failures}/${total} failed`);
  process.exit(1);
}
console.log(`glob: ${total} assertions passed`);
