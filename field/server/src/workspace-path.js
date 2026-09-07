import fs from 'node:fs';
import path from 'node:path';
import ignore from 'ignore';
import { buildMatcher } from './glob.js';

const PRIVATE_BASENAMES = new Set([
  '.git', '.field-state', '.ssh', '.gnupg', '.aws', '.azure', '.kube',
  '.npmrc', '.yarnrc', '.pypirc', '.netrc', '_netrc',
  'credentials', 'credentials.json', 'service-account.json',
  'id_rsa', 'id_dsa', 'id_ecdsa', 'id_ed25519',
]);
const PRIVATE_EXTENSIONS = new Set([
  '.pem', '.key', '.p12', '.pfx', '.jks', '.keystore', '.kdbx',
]);

function realpath(value) {
  return (fs.realpathSync.native ?? fs.realpathSync)(value);
}

function pathKey(value) {
  const resolved = path.resolve(value);
  return process.platform === 'win32' ? resolved.toLowerCase() : resolved;
}

function isWithin(root, candidate) {
  const rootKey = pathKey(root);
  const candidateKey = pathKey(candidate);
  return candidateKey === rootKey || candidateKey.startsWith(rootKey + path.sep);
}

function normalizeRelative(value) {
  if (value == null || value === '') return '';
  if (typeof value !== 'string' || value.includes('\0') || path.isAbsolute(value)) {
    throw new Error('workspace paths must be relative');
  }
  const normalized = path.normalize(value);
  if (normalized === '..' || normalized.startsWith(`..${path.sep}`)) {
    throw new Error('path escapes the workspace root');
  }
  return normalized === '.' ? '' : normalized;
}

function sensitiveDefault(relativePath) {
  const parts = relativePath.replace(/\\/g, '/').split('/').filter(Boolean);
  for (let index = 0; index < parts.length; index += 1) {
    const name = parts[index].toLowerCase();
    if (PRIVATE_BASENAMES.has(name)) return true;
    if (name === '.config' && parts[index + 1]?.toLowerCase() === 'gcloud') return true;
    if (name.startsWith('.env')) return true;
    if (PRIVATE_EXTENSIONS.has(path.extname(name))) return true;
  }
  return false;
}

function gitIgnoreRules(root) {
  const file = path.join(root, '.gitignore');
  const rules = ignore();
  let fd;
  try {
    if (!fs.lstatSync(file).isFile()) throw new Error('invalid ignore file');
    fd = fs.openSync(file, fs.constants.O_RDONLY | (fs.constants.O_NOFOLLOW ?? 0));
    const stat = fs.fstatSync(fd);
    if (!stat.isFile() || stat.size > 1024 * 1024) throw new Error('invalid ignore file');
    rules.add(fs.readFileSync(fd, 'utf8'));
  } catch (error) {
    if (error.code !== 'ENOENT') return null;
  } finally {
    if (fd !== undefined) fs.closeSync(fd);
  }
  return rules;
}

export function createWorkspaceVisibility(root, customDeny = []) {
  const custom = buildMatcher(customDeny.map((item) => String(item).replace(/\\/g, '/')));
  return function isVisible(relativePath) {
    const normalized = String(relativePath ?? '').replace(/\\/g, '/').replace(/^\.\//, '');
    if (!normalized || !ignore.isPathValid(normalized) || sensitiveDefault(normalized) || custom(normalized)) return false;
    const parts = normalized.replace(/\/$/, '').split('/');
    const policies = [];
    // Parent exclusions cannot be undone by descendant negations. Reload policy
    // so a newly ignored credential is hidden without restarting Field.
    for (let depth = 0; depth < parts.length; depth += 1) {
      const rules = gitIgnoreRules(path.join(root, ...parts.slice(0, depth)));
      if (!rules) return false;
      policies.push({ depth, rules });
      let stat;
      try { stat = fs.lstatSync(path.join(root, ...parts.slice(0, depth + 1))); }
      catch (error) { if (error.code !== 'ENOENT') return false; }
      if (stat?.isSymbolicLink()) return false;
      const directory = depth < parts.length - 1 || stat?.isDirectory() || normalized.endsWith('/');
      let ignored = false;
      for (const policy of policies) {
        const candidate = parts.slice(policy.depth, depth + 1).join('/') + (directory ? '/' : '');
        const result = policy.rules.test(candidate);
        if (result.ignored) ignored = true;
        else if (result.unignored) ignored = false;
      }
      if (ignored) return false;
    }
    return true;
  };
}

export function canonicalizeWorkspace(workspace, fieldDir, globalSensitive = []) {
  const requested = path.resolve(fieldDir, workspace.path);
  let canonicalPath = null;
  let mounted = false;
  try {
    const stat = fs.statSync(requested);
    if (stat.isDirectory()) {
      canonicalPath = realpath(requested);
      mounted = true;
    }
  } catch { /* surfaced as an unmounted workspace */ }
  const root = canonicalPath ?? requested;
  return {
    ...workspace,
    path: root,
    canonicalPath,
    mounted,
    isVisible: mounted
      ? createWorkspaceVisibility(root, [...globalSensitive, ...(workspace.sensitive?.deny ?? [])])
      : () => false,
  };
}

function inspectChain(root, absolute, { allowMissingLeaf }) {
  const relative = path.relative(root, absolute);
  const parts = relative ? relative.split(path.sep) : [];
  let cursor = root;
  for (let index = 0; index < parts.length; index += 1) {
    cursor = path.join(cursor, parts[index]);
    const leaf = index === parts.length - 1;
    let stat;
    try { stat = fs.lstatSync(cursor); } catch (error) {
      if (allowMissingLeaf && leaf && error.code === 'ENOENT') return { missingLeaf: true };
      throw error;
    }
    if (stat.isSymbolicLink()) throw new Error('symlink or junction traversal is not allowed');
    if (!leaf && !stat.isDirectory()) throw new Error('workspace path has a non-directory parent');
  }
  return { missingLeaf: false };
}

export function resolveWorkspacePath(cfg, workspaceId, relativePath = '', {
  operation = 'read',
  type = 'any',
  allowSensitive = false,
} = {}) {
  const workspace = cfg.workspaces.find((item) => item.id === workspaceId);
  if (!workspace) throw new Error(`unknown workspace: ${workspaceId}`);
  if (!workspace.mounted || !workspace.canonicalPath) throw new Error(`workspace ${workspaceId} is not mounted`);
  const relative = normalizeRelative(relativePath);
  const display = relative.replace(/\\/g, '/');
  if (!allowSensitive && display && !workspace.isVisible(display)) {
    throw new Error('path is hidden by the workspace security policy');
  }

  const root = workspace.canonicalPath;
  const absolute = path.resolve(root, relative);
  if (!isWithin(root, absolute)) throw new Error('path escapes the workspace root');
  const chain = inspectChain(root, absolute, { allowMissingLeaf: operation === 'write' });

  if (chain.missingLeaf) {
    if (type === 'directory') throw new Error('workspace directory does not exist');
    return { ws: workspace, root, abs: absolute, relative: display, exists: false };
  }

  const canonical = realpath(absolute);
  if (!isWithin(root, canonical)) throw new Error('canonical path escapes the workspace root');
  const stat = fs.statSync(canonical);
  if (type === 'file' && !stat.isFile()) throw new Error('workspace target is not a regular file');
  if (type === 'directory' && !stat.isDirectory()) throw new Error('workspace target is not a directory');
  if (type === 'any' && !stat.isFile() && !stat.isDirectory()) {
    throw new Error('unsupported workspace file type');
  }
  return { ws: workspace, root, abs: canonical, relative: display, exists: true, stat };
}

export function readWorkspaceFile(resolved, encoding = 'utf8') {
  const flags = fs.constants.O_RDONLY | (fs.constants.O_NOFOLLOW ?? 0);
  const fd = fs.openSync(resolved.abs, flags);
  try {
    if (!fs.fstatSync(fd).isFile()) throw new Error('workspace target is not a regular file');
    return fs.readFileSync(fd, encoding);
  } finally {
    fs.closeSync(fd);
  }
}

export function writeWorkspaceFile(resolved, content, encoding = 'utf8') {
  const flags = fs.constants.O_WRONLY
    | fs.constants.O_CREAT
    | fs.constants.O_TRUNC
    | (fs.constants.O_NOFOLLOW ?? 0);
  const fd = fs.openSync(resolved.abs, flags, 0o600);
  try {
    const stat = fs.fstatSync(fd);
    if (!stat.isFile()) throw new Error('workspace target is not a regular file');
    fs.writeFileSync(fd, content, encoding);
  } finally {
    fs.closeSync(fd);
  }
}
