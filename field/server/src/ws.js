// One socket per operator window. Events go out immediately (they drive the trace tail
// and the route pulses); the folded snapshot is coalesced so a burst of tool calls
// cannot flood the client.
import { WebSocketServer } from 'ws';

const SNAPSHOT_INTERVAL_MS = 120;
const MAX_BUFFERED_BYTES = 1024 * 1024;

export function createHub(server, { projection, authorizeUpgrade }) {
  if (typeof authorizeUpgrade !== 'function') {
    throw new Error('createHub requires an explicit WebSocket authorization policy');
  }
  const wss = new WebSocketServer({ noServer: true });
  let dirty = false;
  let timer = null;

  const onUpgrade = (req, socket, head) => {
    const url = new URL(req.url, 'http://127.0.0.1');
    const auth = authorizeUpgrade(req, url);
    if (!auth.ok) {
      const body = JSON.stringify({ error: auth.error, code: auth.code });
      socket.write(
        `HTTP/1.1 ${auth.status} Unauthorized\r\n`
        + 'Connection: close\r\n'
        + 'Content-Type: application/json; charset=utf-8\r\n'
        + `Content-Length: ${Buffer.byteLength(body)}\r\n\r\n${body}`,
      );
      socket.destroy();
      return;
    }
    wss.handleUpgrade(req, socket, head, (client) => wss.emit('connection', client, req));
  };
  server.on('upgrade', onUpgrade);

  wss.on('connection', (socket) => {
    socket.send(JSON.stringify({ type: 'snapshot', state: projection.snapshot() }));
    socket.on('error', () => { /* client vanished; nothing to do */ });
  });

  function sendAll(payload) {
    const text = JSON.stringify(payload);
    for (const client of wss.clients) {
      if (client.readyState === 1) {
        if (client.bufferedAmount > MAX_BUFFERED_BYTES) {
          client.terminate();
          continue;
        }
        try { client.send(text); } catch { /* dropped client */ }
      }
    }
  }

  function flush() {
    timer = null;
    if (!dirty) return;
    dirty = false;
    sendAll({ type: 'snapshot', state: projection.snapshot() });
  }

  return {
    pushEvent(evt) {
      sendAll({ type: 'event', event: evt });
      dirty = true;
      if (!timer) timer = setTimeout(flush, SNAPSHOT_INTERVAL_MS);
    },
    broadcast: sendAll,
    get clients() { return wss.clients.size; },
    revokeClients() {
      for (const client of wss.clients) client.terminate();
    },
    close() {
      if (timer) clearTimeout(timer);
      server.off('upgrade', onUpgrade);
      wss.close();
    },
  };
}
