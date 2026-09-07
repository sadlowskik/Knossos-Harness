import assert from 'node:assert/strict';
import http from 'node:http';
import { WebSocket } from 'ws';
import { createControlSecurity, generateToken, isLoopbackAddress } from '../src/security.js';
import { createHub } from '../src/ws.js';

const port = 7749;
const security = createControlSecurity({
  port,
  bootstrapToken: 'bootstrap-fixture-token',
  browserToken: 'browser-fixture-token',
});
const internalToken = security.mintHarnessToken('session-fixture');

function request({
  method = 'GET',
  host = `127.0.0.1:${port}`,
  origin,
  cookie,
  authorization,
  contentType,
  remoteAddress = '127.0.0.1',
} = {}) {
  return {
    method,
    headers: {
      host,
      ...(origin ? { origin } : {}),
      ...(cookie ? { cookie } : {}),
      ...(authorization ? { authorization } : {}),
      ...(contentType ? { 'content-type': contentType } : {}),
    },
    socket: { remoteAddress },
  };
}

assert.equal(generateToken().length >= 43, true);
assert.notEqual(generateToken(), generateToken());
assert.equal(isLoopbackAddress('::ffff:127.0.0.1'), true);
assert.equal(isLoopbackAddress('192.168.1.20'), false);

const root = new URL('http://127.0.0.1:7749/?bootstrap=bootstrap-fixture-token');
const boot = security.consumeBootstrap(request(), root);
assert.equal(boot.ok, true);
assert.match(boot.cookie, /^field_session=browser-fixture-token; HttpOnly; SameSite=Strict; Path=\/$/);
assert.equal(boot.redirect, '/');

const cookie = 'unrelated=1; field_session=browser-fixture-token; another=2';
assert.equal(
  security.authorizeRequest(request({ cookie }), new URL('http://local/api/state')).ok,
  true,
);
assert.equal(
  security.authorizeRequest(request({ cookie: 'field_session=wrong' }), new URL('http://local/api/state')).status,
  401,
);
assert.equal(
  security.authorizeRequest(request({ cookie, host: 'evil.example' }), new URL('http://local/api/state')).status,
  421,
);
assert.equal(
  security.authorizeRequest(request({ cookie, remoteAddress: '10.0.0.8' }), new URL('http://local/api/state')).status,
  403,
);

const mutatingUrl = new URL('http://local/api/command');
assert.equal(security.authorizeRequest(request({ method: 'POST', cookie }), mutatingUrl).status, 403);
assert.equal(security.authorizeRequest(request({
  method: 'POST', cookie, origin: `http://127.0.0.1:${port}`,
}), mutatingUrl).ok, true);
assert.equal(security.authorizeRequest(request({
  method: 'POST', cookie, origin: `http://localhost:${port}`,
}), mutatingUrl).status, 403);

const internalUrl = new URL('http://local/api/internal/permission');
assert.equal(security.authorizeRequest(request({
  method: 'POST',
  authorization: `Bearer ${internalToken}`,
  contentType: 'application/json',
}), internalUrl).sessionId, 'session-fixture');
assert.equal(security.authorizeRequest(request({
  method: 'POST', cookie, contentType: 'application/json',
}), internalUrl).status, 401);
assert.equal(security.authorizeRequest(request({
  method: 'POST',
  authorization: `Bearer ${internalToken}`,
  contentType: 'text/plain',
}), internalUrl).status, 415);

const wsUrl = new URL('http://local/ws');
assert.equal(security.authorizeUpgrade(request({
  cookie, origin: `http://127.0.0.1:${port}`,
}), wsUrl).ok, true);
assert.equal(security.authorizeUpgrade(request({ cookie }), wsUrl).status, 403);

const replay = security.consumeBootstrap(request(), root);
assert.equal(replay.status, 410);
const alreadyAuthenticated = security.consumeBootstrap(request({ cookie }), root);
assert.equal(alreadyAuthenticated.ok, true);

// Exercise the real HTTP upgrade boundary, not only the pure policy functions.
const socketServer = http.createServer((_req, res) => { res.writeHead(404); res.end(); });
await new Promise((resolve) => socketServer.listen(0, '127.0.0.1', resolve));
const socketPort = socketServer.address().port;
const socketSecurity = createControlSecurity({
  port: socketPort,
  bootstrapToken: 'socket-bootstrap-token',
  browserToken: 'socket-browser-token',
});
socketSecurity.consumeBootstrap(
  request({ host: `127.0.0.1:${socketPort}` }),
  new URL(`http://127.0.0.1:${socketPort}/?bootstrap=socket-bootstrap-token`),
);
const socketHub = createHub(socketServer, {
  projection: { snapshot: () => ({ secured: true }) },
  authorizeUpgrade: socketSecurity.authorizeUpgrade,
});

const rejectedStatus = await new Promise((resolve, reject) => {
  const socket = new WebSocket(`ws://127.0.0.1:${socketPort}/ws`, {
    headers: { Origin: `http://127.0.0.1:${socketPort}` },
  });
  socket.once('unexpected-response', (_request, response) => resolve(response.statusCode));
  socket.once('open', () => reject(new Error('unauthorized WebSocket opened')));
  socket.once('error', () => {});
});
assert.equal(rejectedStatus, 401);

const firstMessage = await new Promise((resolve, reject) => {
  const socket = new WebSocket(`ws://127.0.0.1:${socketPort}/ws`, {
    headers: {
      Origin: `http://127.0.0.1:${socketPort}`,
      Cookie: 'field_session=socket-browser-token',
    },
  });
  socket.once('message', (data) => {
    socket.close();
    resolve(JSON.parse(String(data)));
  });
  socket.once('error', reject);
});
assert.deepEqual(firstMessage, { type: 'snapshot', state: { secured: true } });
socketHub.close();
await new Promise((resolve) => socketServer.close(resolve));

const devSecurity = createControlSecurity({
  port,
  trustedOrigins: ['http://127.0.0.1:7748'],
  bootstrapRedirect: 'http://127.0.0.1:7748',
  bootstrapToken: 'dev-bootstrap-token',
  browserToken: 'dev-browser-token',
});
const devBoot = devSecurity.consumeBootstrap(
  request(),
  new URL(`http://127.0.0.1:${port}/?bootstrap=dev-bootstrap-token`),
);
assert.equal(devBoot.redirect, 'http://127.0.0.1:7748');
assert.equal(devSecurity.authorizeRequest(request({
  method: 'POST',
  cookie: 'field_session=dev-browser-token',
  origin: 'http://127.0.0.1:7748',
}), mutatingUrl).ok, true);
assert.equal(devSecurity.authorizeRequest(request({
  method: 'POST',
  cookie: 'field_session=dev-browser-token',
  origin: 'http://127.0.0.1:7750',
}), mutatingUrl).status, 403);
assert.throws(
  () => createControlSecurity({ port, trustedOrigins: ['https://evil.example'] }),
  /exact loopback HTTP origins/,
);

security.revokeHarnessToken('session-fixture');
assert.equal(security.authorizeRequest(request({
  method: 'POST', authorization: `Bearer ${internalToken}`, contentType: 'application/json',
}), internalUrl).status, 401);

const preservedHarness = security.mintHarnessToken('still-running');
assert.match(security.revokeBrowserSession(), /Max-Age=0/);
assert.equal(security.authorizeRequest(request({ cookie }), mutatingUrl).status, 401);
assert.equal(security.authorizeUpgrade(request({ cookie, origin: `http://127.0.0.1:${port}` }), wsUrl).status, 401);
assert.equal(security.consumeBootstrap(request(), root).status, 410);
assert.equal(security.authorizeRequest(request({ method: 'POST', authorization: `Bearer ${preservedHarness}`, contentType: 'application/json' }), internalUrl).sessionId, 'still-running');
let clock = 0;
const expiring = createControlSecurity({ port, now: () => clock, bootstrapTtlMs: 100, bootstrapToken: 'expires' });
clock = 100;
assert.equal(expiring.consumeBootstrap(request(), new URL('http://local/?bootstrap=expires')).status, 401);
const restarted = createControlSecurity({ port });
assert.equal(restarted.authorizeRequest(request({ cookie }), mutatingUrl).status, 401);

console.log('security: bootstrap, split authority, loopback, host, origin, cookie, and websocket gates passed');
