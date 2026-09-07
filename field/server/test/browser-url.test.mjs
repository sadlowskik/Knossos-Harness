import assert from 'node:assert/strict';
import { validateBrowserUrl } from '../../web/src/workspace/browser-url.js';

const allowed = ['docs.example.test', 'localhost'];
assert.deepEqual(
  validateBrowserUrl('https://docs.example.test/guide?q=1', allowed),
  { ok: true, url: 'https://docs.example.test/guide?q=1' },
);
assert.equal(validateBrowserUrl('javascript:alert(1)', allowed).ok, false);
assert.equal(validateBrowserUrl('data:text/html,<script>alert(1)</script>', allowed).ok, false);
assert.equal(validateBrowserUrl('https://user:pass@docs.example.test/', allowed).ok, false);
assert.equal(validateBrowserUrl('https://evil.example.test/', allowed).ok, false);
assert.equal(validateBrowserUrl('//docs.example.test/path', allowed).ok, false);
assert.equal(validateBrowserUrl('http://localhost:8080/status', allowed).ok, true);

console.log('browser URL: protocols, credentials, complete URLs, and configured domains enforced');
