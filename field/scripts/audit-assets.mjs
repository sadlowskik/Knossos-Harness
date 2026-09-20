import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const publicRoot = path.join(root, 'web', 'public');
const manifest = JSON.parse(fs.readFileSync(path.join(root, 'asset-manifest.json'), 'utf8'));
const release = process.argv.includes('--release');
const errors = [];

function walk(dir) {
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const absolute = path.join(dir, entry.name);
    return entry.isDirectory() ? walk(absolute) : [absolute];
  });
}

const files = walk(path.join(publicRoot, 'assets'));
const actual = new Map(files.map((absolute) => [
  path.relative(publicRoot, absolute).replace(/\\/g, '/'), absolute,
]));
const declared = new Map(manifest.assets.map((asset) => [asset.path, asset]));
const sourceFiles = [path.join(root, 'web', 'index.html'), ...walk(path.join(root, 'web', 'src'))]
  .filter((file) => /\.(?:html|css|js|jsx)$/.test(file));
const source = sourceFiles.map((file) => fs.readFileSync(file, 'utf8')).join('\n');

for (const [relative, absolute] of actual) {
  const item = declared.get(relative);
  if (!item) errors.push(`${relative}: missing manifest entry`);
  if (!source.includes(`/${relative}`)) errors.push(`${relative}: shipped but unreferenced`);
  const digest = crypto.createHash('sha256').update(fs.readFileSync(absolute)).digest('hex');
  if (item && digest !== item.sha256) errors.push(`${relative}: SHA-256 differs from manifest`);
}
for (const relative of declared.keys()) {
  if (!actual.has(relative)) errors.push(`${relative}: declared but missing`);
}
const total = files.reduce((sum, file) => sum + fs.statSync(file).size, 0);
if (total > manifest.bundle_budget_bytes) {
  errors.push(`asset payload ${total} bytes exceeds budget ${manifest.bundle_budget_bytes}`);
}
if (release) {
  for (const item of manifest.assets) {
    if (item.review_status !== 'APPROVED' || item.license === 'UNVERIFIED' || item.creator === 'unknown') {
      errors.push(`${item.path}: release provenance is ${item.review_status}`);
    }
  }
}

if (errors.length) {
  console.error(`asset audit failed (${errors.length})\n- ${errors.join('\n- ')}`);
  process.exitCode = 1;
} else {
  console.log(`asset audit passed: ${files.length} referenced files, ${total} bytes, hashes verified${release ? ', release provenance approved' : ''}`);
}
