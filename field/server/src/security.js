import { createHash, randomBytes, timingSafeEqual } from 'node:crypto';

const TOKEN_BYTES = 32;
const COOKIE_NAME = 'field_session';
const UNSAFE_METHODS = new Set(['POST', 'PUT', 'PATCH', 'DELETE']);

export function generateToken() {
  return randomBytes(TOKEN_BYTES).toString('base64url');
}

function equalSecret(left, right) {
  if (typeof left !== 'string' || typeof right !== 'string') return false;
  const a = Buffer.from(left);
  const b = Buffer.from(right);
  return a.length === b.length && timingSafeEqual(a, b);
}

function cookieValue(header, name) {
  if (typeof header !== 'string') return null;
  for (const part of header.split(';')) {
    const index = part.indexOf('=');
    if (index < 1) continue;
    if (part.slice(0, index).trim() === name) return part.slice(index + 1).trim();
  }
  return null;
}

export function isLoopbackAddress(address) {
  if (typeof address !== 'string') return false;
  const normalized = address.toLowerCase();
  return normalized === '127.0.0.1'
    || normalized === '::1'
    || normalized === '::ffff:127.0.0.1';
}

function acceptedAuthority(hostHeader, port) {
  if (typeof hostHeader !== 'string' || /[\s/@\\]/.test(hostHeader)) return null;
  try {
    const parsed = new URL(`http://${hostHeader}`);
    const hostname = parsed.hostname.toLowerCase();
    const expectedPort = String(port);
    if (!['127.0.0.1', 'localhost', '[::1]'].includes(hostname)) return null;
    if (parsed.port !== expectedPort) return null;
    return parsed.host.toLowerCase();
  } catch {
    return null;
  }
}

function normalizedLoopbackOrigin(value) {
  try {
    const parsed = new URL(value);
    if (parsed.protocol !== 'http:' || parsed.username || parsed.password) return null;
    if (!['127.0.0.1', 'localhost', '[::1]'].includes(parsed.hostname.toLowerCase())) return null;
    if (!parsed.port || parsed.pathname !== '/' || parsed.search || parsed.hash) return null;
    return parsed.origin.toLowerCase();
  } catch {
    return null;
  }
}

function sameOrigin(req, port, trustedOrigins) {
  const authority = acceptedAuthority(req.headers.host, port);
  if (!authority || typeof req.headers.origin !== 'string') return false;
  try {
    const origin = new URL(req.headers.origin);
    const structurallyValid = origin.protocol === 'http:'
      && origin.username === ''
      && origin.password === ''
      && origin.pathname === '/'
      && origin.search === ''
      && origin.hash === '';
    if (!structurallyValid) return false;
    return origin.host.toLowerCase() === authority
      || trustedOrigins.has(origin.origin.toLowerCase());
  } catch {
    return false;
  }
}

function bearerToken(header) {
  if (typeof header !== 'string') return null;
  const match = /^Bearer ([A-Za-z0-9_-]+)$/.exec(header);
  return match?.[1] ?? null;
}

function denied(status, code, error) {
  return { ok: false, status, code, error };
}

/**
 * Ephemeral authority for one Field server process. Browser and harness credentials are
 * deliberately separate: neither grants the other's capabilities.
 */
export function createControlSecurity({
  port,
  bootstrapToken = generateToken(),
  browserToken = generateToken(),
  trustedOrigins = [],
  bootstrapRedirect = '/',
  frameSources = [],
  now = Date.now,
  bootstrapTtlMs = 10 * 60 * 1000,
} = {}) {
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    throw new Error('Field security requires a valid listening port');
  }
  const normalizedOrigins = new Set(trustedOrigins.map(normalizedLoopbackOrigin));
  if (normalizedOrigins.has(null)) throw new Error('Field trusted origins must be exact loopback HTTP origins');
  const normalizedRedirect = bootstrapRedirect === '/'
    ? '/'
    : normalizedLoopbackOrigin(bootstrapRedirect);
  if (!normalizedRedirect) throw new Error('Field bootstrap redirect must be / or an exact loopback HTTP origin');
  const normalizedFrameSources = frameSources.map((value) => {
    try {
      const parsed = new URL(value);
      if (!['http:', 'https:'].includes(parsed.protocol) || parsed.username || parsed.password
        || parsed.pathname !== '/' || parsed.search || parsed.hash) throw new Error();
      return parsed.origin;
    } catch {
      throw new Error('Field frame sources must be exact HTTP(S) origins');
    }
  });

  let bootstrapConsumed = false;
  let browserEnabled = false;
  const bootstrapExpiresAt = now() + bootstrapTtlMs;
  const harnessCapabilities = new Map();

  const capabilityKey = (token) => createHash('sha256').update(String(token)).digest('base64url');

  function mintHarnessToken(sessionId) {
    revokeHarnessToken(sessionId);
    const token = generateToken();
    harnessCapabilities.set(capabilityKey(token), String(sessionId));
    return token;
  }

  function revokeHarnessToken(sessionId) {
    const wanted = String(sessionId);
    for (const [key, owner] of harnessCapabilities) {
      if (owner === wanted) harnessCapabilities.delete(key);
    }
  }

  function revokeAllHarnessTokens() {
    harnessCapabilities.clear();
  }

  function networkCheck(req) {
    if (!isLoopbackAddress(req.socket?.remoteAddress)) {
      return denied(403, 'loopback_required', 'Field only accepts loopback clients.');
    }
    if (!acceptedAuthority(req.headers.host, port)) {
      return denied(421, 'invalid_host', 'Field rejected the Host header.');
    }
    return { ok: true };
  }

  function hasBrowserSession(req) {
    return browserEnabled
      && equalSecret(cookieValue(req.headers.cookie, COOKIE_NAME), browserToken);
  }

  function revokeBrowserSession() {
    browserEnabled = false;
    bootstrapConsumed = true;
    return `${COOKIE_NAME}=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0`;
  }

  function consumeBootstrap(req, url) {
    if (url.pathname !== '/' || !url.searchParams.has('bootstrap')) return { handled: false };
    const network = networkCheck(req);
    if (!network.ok) return { handled: true, ...network };
    if (req.method !== 'GET') {
      return { handled: true, ...denied(405, 'method_not_allowed', 'Bootstrap requires GET.') };
    }
    if (bootstrapConsumed) {
      if (hasBrowserSession(req)) return { handled: true, ok: true, redirect: normalizedRedirect };
      return {
        handled: true,
        ...denied(410, 'bootstrap_consumed', 'This bootstrap link has already been used. Restart Field to mint a new link.'),
      };
    }
    if (now() >= bootstrapExpiresAt || !equalSecret(url.searchParams.get('bootstrap'), bootstrapToken)) {
      return { handled: true, ...denied(401, 'invalid_bootstrap', 'The bootstrap token is invalid.') };
    }

    bootstrapConsumed = true;
    browserEnabled = true;
    return {
      handled: true,
      ok: true,
      redirect: normalizedRedirect,
      cookie: `${COOKIE_NAME}=${browserToken}; HttpOnly; SameSite=Strict; Path=/`,
    };
  }

  function authorizeRequest(req, url) {
    const network = networkCheck(req);
    if (!network.ok) return network;

    if (url.pathname === '/api/internal/permission') {
      if (req.method !== 'POST') {
        return denied(405, 'method_not_allowed', 'The internal permission endpoint requires POST.');
      }
      const token = bearerToken(req.headers.authorization);
      const capabilitySessionId = token ? harnessCapabilities.get(capabilityKey(token)) : null;
      if (!capabilitySessionId) {
        return denied(401, 'internal_auth_required', 'Valid harness authorization is required.');
      }
      if (!String(req.headers['content-type'] ?? '').toLowerCase().startsWith('application/json')) {
        return denied(415, 'json_required', 'The internal permission endpoint requires JSON.');
      }
      return { ok: true, authority: 'harness', sessionId: capabilitySessionId };
    }

    if (!hasBrowserSession(req)) {
      return denied(401, 'browser_auth_required', 'Open the fresh bootstrap URL printed by the Field server.');
    }
    if (UNSAFE_METHODS.has(req.method) && !sameOrigin(req, port, normalizedOrigins)) {
      return denied(403, 'origin_required', 'State-changing Field requests require the exact local origin.');
    }
    return { ok: true, authority: 'operator' };
  }

  function authorizeUpgrade(req, url) {
    const auth = authorizeRequest(req, url);
    if (!auth.ok) return auth;
    if (url.pathname !== '/ws') return denied(404, 'not_found', 'Unknown WebSocket endpoint.');
    if (req.method !== 'GET' || !sameOrigin(req, port, normalizedOrigins)) {
      return denied(403, 'origin_required', 'Field WebSockets require the exact local origin.');
    }
    return auth;
  }

  function applyHeaders(res) {
    res.setHeader('cache-control', 'no-store');
    res.setHeader('content-security-policy', [
      "default-src 'self'",
      "script-src 'self'",
      "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
      "font-src 'self' https://fonts.gstatic.com",
      "img-src 'self' data:",
      `connect-src 'self' ws://127.0.0.1:${port} ws://localhost:${port}`,
      `frame-src 'self' ${normalizedFrameSources.join(' ')}`.trim(),
      "frame-ancestors 'none'",
      "base-uri 'self'",
      "object-src 'none'",
      "form-action 'none'",
    ].join('; '));
    res.setHeader('cross-origin-opener-policy', 'same-origin');
    res.setHeader('cross-origin-resource-policy', 'same-origin');
    res.setHeader('permissions-policy', 'camera=(), microphone=(), geolocation=()');
    res.setHeader('referrer-policy', 'no-referrer');
    res.setHeader('x-content-type-options', 'nosniff');
    res.setHeader('x-frame-options', 'DENY');
  }

  return {
    bootstrapToken,
    browserToken,
    bootstrapUrl: `http://127.0.0.1:${port}/?bootstrap=${bootstrapToken}`,
    mintHarnessToken,
    revokeHarnessToken,
    revokeAllHarnessTokens,
    revokeBrowserSession,
    consumeBootstrap,
    authorizeNetwork: networkCheck,
    authorizeRequest,
    authorizeUpgrade,
    applyHeaders,
  };
}
