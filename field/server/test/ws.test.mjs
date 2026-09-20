import assert from 'node:assert/strict';
import http from 'node:http';
import { WebSocket } from 'ws';
import { createHub } from '../src/ws.js';

let seq = 0;
let snapshotCalls = 0;
const projection = {
  snapshot() {
    snapshotCalls += 1;
    return {
      seq,
      permissions: [{ id: 'pending-1', sessionId: 'agent-1', status: 'pending' }],
    };
  },
};
const server = http.createServer((_req, res) => { res.writeHead(404); res.end(); });
const hub = createHub(server, { projection, authorizeUpgrade: () => ({ ok: true }) });
await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
const port = server.address().port;

async function connect() {
  const socket = new WebSocket(`ws://127.0.0.1:${port}/ws`);
  const messages = [];
  socket.on('message', (data) => messages.push(JSON.parse(String(data))));
  await new Promise((resolve, reject) => {
    socket.once('open', resolve);
    socket.once('error', reject);
  });
  await waitFor(() => messages.some((message) => message.type === 'snapshot'));
  return { socket, messages };
}

const first = await connect();
for (let i = 0; i < 500; i += 1) {
  seq += 1;
  hub.pushEvent({ seq, ts: i, kind: 'fixture.burst', data: { i } });
}
await waitFor(() => first.messages.filter((message) => message.type === 'event').length === 500);
await new Promise((resolve) => setTimeout(resolve, 180));
assert.equal(first.messages.filter((message) => message.type === 'snapshot').length, 2);
assert.equal(first.messages.at(-1).state.seq, 500);
first.socket.close();
await new Promise((resolve) => first.socket.once('close', resolve));

const second = await connect();
const recovered = second.messages.find((message) => message.type === 'snapshot').state;
assert.equal(recovered.seq, 500);
assert.equal(recovered.permissions[0].id, 'pending-1');
const revoked = new Promise((resolve) => second.socket.once('close', resolve));
hub.revokeClients();
await revoked;
assert.equal(second.socket.readyState, WebSocket.CLOSED, 'logout terminates an authenticated socket');

hub.close();
await new Promise((resolve) => server.close(resolve));
assert.ok(snapshotCalls <= 3, `burst produced ${snapshotCalls} projection snapshots`);
console.log('websocket: 500-event burst coalescing, reconnect, and pending approval recovery passed');

async function waitFor(predicate) {
  const deadline = Date.now() + 2000;
  while (!predicate()) {
    if (Date.now() > deadline) throw new Error('timed out waiting for websocket state');
    await new Promise((resolve) => setTimeout(resolve, 5));
  }
}
