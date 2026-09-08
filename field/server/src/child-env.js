const BASE_KEYS = [
  'PATH', 'PATHEXT', 'SYSTEMROOT', 'WINDIR', 'COMSPEC',
  'TEMP', 'TMP', 'TMPDIR', 'USERPROFILE', 'HOME', 'APPDATA', 'LOCALAPPDATA', 'PROGRAMDATA',
  'LANG', 'LC_ALL', 'LC_CTYPE', 'TERM', 'COLORTERM', 'NO_COLOR', 'FORCE_COLOR',
  'SSL_CERT_FILE', 'SSL_CERT_DIR', 'NODE_EXTRA_CA_CERTS',
  'CLAUDE_CONFIG_DIR',
];

const PROVIDER_KEYS = {
  anthropic: ['ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_BASE_URL'],
};

/** Operator credentials that must never reach a harness child, even if listed. */
const PRIVILEGED_NEVER = new Set([
  'CAMEO_CONSOLE_KEY',
  'CAMEO_UPDATE_PRIVATE_PEM',
  'CAMEO_UPDATE_PRIVATE',
  'SSH_AUTH_SOCK',
  'SSH_PRIVATE_KEY',
]);

function isPrivilegedEnvKey(key) {
  const name = String(key ?? '').toUpperCase();
  if (PRIVILEGED_NEVER.has(name)) return true;
  return name.startsWith('CAMEO_') && (name.includes('KEY') || name.includes('PRIVATE') || name.includes('SECRET') || name.includes('TOKEN'));
}

function sourceValue(source, wanted) {
  if (process.platform !== 'win32') return source[wanted];
  const found = Object.keys(source).find((key) => key.toLowerCase() === wanted.toLowerCase());
  return found ? source[found] : undefined;
}

/** Build a minimal environment for a harness and every subprocess it launches. */
export function buildChildEnvironment({
  source = process.env,
  provider,
  explicitKeys = [],
  overrides = {},
} = {}) {
  const result = {};
  const keys = new Set([
    ...BASE_KEYS,
    ...(PROVIDER_KEYS[provider] ?? []),
    ...explicitKeys.map(String),
  ]);
  for (const key of keys) {
    if (isPrivilegedEnvKey(key)) continue;
    const value = sourceValue(source, key);
    if (value != null && value !== '') result[key] = value;
  }
  for (const [key, value] of Object.entries(overrides)) {
    if (value == null || isPrivilegedEnvKey(key)) continue;
    result[key] = String(value);
  }
  return result;
}
