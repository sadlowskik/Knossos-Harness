import assert from 'node:assert/strict';
import { buildChildEnvironment } from '../src/child-env.js';

const source = {
  PATH: '/bin',
  USERPROFILE: 'C:/fixture',
  ANTHROPIC_API_KEY: 'anthropic-canary',
  OPENAI_API_KEY: 'openai-canary',
  AWS_SECRET_ACCESS_KEY: 'aws-canary',
  GITHUB_TOKEN: 'github-canary',
  NPM_TOKEN: 'npm-canary',
  SSH_AUTH_SOCK: 'ssh-canary',
  CI_JOB_TOKEN: 'ci-canary',
  CAMEO_CONSOLE_KEY: 'cameo-operator-canary',
};

const anthropic = buildChildEnvironment({
  source,
  provider: 'anthropic',
  overrides: { FIELD_SESSION_ID: 'session-1' },
});
assert.equal(anthropic.PATH, '/bin');
assert.equal(anthropic.ANTHROPIC_API_KEY, 'anthropic-canary');
assert.equal(anthropic.FIELD_SESSION_ID, 'session-1');
for (const secret of ['OPENAI_API_KEY', 'AWS_SECRET_ACCESS_KEY', 'GITHUB_TOKEN', 'NPM_TOKEN', 'SSH_AUTH_SOCK', 'CI_JOB_TOKEN', 'CAMEO_CONSOLE_KEY']) {
  assert.equal(secret in anthropic, false, `${secret} leaked into Anthropic child`);
}

const cameo = buildChildEnvironment({ source, provider: 'openai-compatible' });
assert.equal('ANTHROPIC_API_KEY' in cameo, false);
assert.equal('OPENAI_API_KEY' in cameo, false);

const explicit = buildChildEnvironment({
  source,
  provider: 'openai-compatible',
  explicitKeys: ['OPENAI_API_KEY', 'CAMEO_CONSOLE_KEY'],
  overrides: { CAMEO_CONSOLE_KEY: 'override-operator-canary', FIELD_SESSION_ID: 'session-2' },
});
assert.equal(explicit.OPENAI_API_KEY, 'openai-canary');
assert.equal('ANTHROPIC_API_KEY' in explicit, false);
assert.equal('CAMEO_CONSOLE_KEY' in explicit, false, 'Cameo operator key must not reach a child even if requested');
assert.equal(explicit.FIELD_SESSION_ID, 'session-2');

console.log('child env: base allowlist, provider isolation, explicit credentials, and canary scrubbing passed');
