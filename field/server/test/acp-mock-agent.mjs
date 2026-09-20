// A tiny mock ACP agent used by acp-session.test.mjs. It speaks the *agent* side of the Agent
// Client Protocol (JSON-RPC 2.0 over newline-delimited stdio) well enough to exercise the client
// adapter: initialize + session/new handshake, a prompt turn that narrates via session/update, an
// optional agent->client session/request_permission round trip, and a stop reason.
//
// Nothing here touches the network or a real agent — it is a fixture, driven only by the adapter
// it is spawned by. Human-readable diagnostics go to stderr; stdout is the protocol channel.
import { createInterface } from 'node:readline';

function send(message) {
  process.stdout.write(JSON.stringify(message) + '\n');
}
function reply(id, result) { send({ jsonrpc: '2.0', id, result }); }
function update(sessionId, u) { send({ jsonrpc: '2.0', method: 'session/update', params: { sessionId, update: u } }); }

let nextOutboundId = 1000;
const pendingPermission = new Map(); // outbound request id -> resolve(optionId)

async function requestPermission(sessionId) {
  const id = ++nextOutboundId;
  return new Promise((resolve) => {
    pendingPermission.set(id, resolve);
    send({
      jsonrpc: '2.0',
      id,
      method: 'session/request_permission',
      params: {
        sessionId,
        toolCall: { toolCallId: 'call-danger', title: 'write_file config.json', rawInput: { path: 'config.json' } },
        options: [
          { optionId: 'allow', name: 'Allow', kind: 'allow_once' },
          { optionId: 'reject', name: 'Reject', kind: 'reject_once' },
        ],
      },
    });
  });
}

async function runPrompt(sessionId, text) {
  update(sessionId, {
    sessionUpdate: 'plan',
    entries: [
      { content: 'inspect the workspace', priority: 'medium', status: 'completed' },
      { content: 'make the change', priority: 'medium', status: 'pending' },
    ],
  });
  update(sessionId, { sessionUpdate: 'agent_thought_chunk', content: { type: 'text', text: 'thinking about it' } });
  update(sessionId, { sessionUpdate: 'agent_message_chunk', content: { type: 'text', text: 'here is what I found' } });
  update(sessionId, {
    sessionUpdate: 'tool_call',
    toolCallId: 'call-read-1',
    title: 'read_file src/lib.rs',
    kind: 'read',
    status: 'completed',
    rawInput: { path: 'src/lib.rs' },
    content: [{ type: 'content', content: { type: 'text', text: 'pub fn one() -> u32 { 1 }' } }],
    locations: [{ path: 'src/lib.rs' }],
  });

  if (/permission/i.test(text)) {
    const optionId = await requestPermission(sessionId);
    update(sessionId, {
      sessionUpdate: 'tool_call_update',
      toolCallId: 'call-danger',
      status: optionId === 'allow' ? 'completed' : 'failed',
      content: [{ type: 'content', content: { type: 'text', text: optionId === 'allow' ? 'wrote config.json' : 'write rejected' } }],
    });
  }

  return { stopReason: 'end_turn', missionId: 'mock-m1' };
}

const rl = createInterface({ input: process.stdin });
rl.on('line', async (line) => {
  const trimmed = line.trim();
  if (!trimmed) return;
  let msg;
  try { msg = JSON.parse(trimmed); } catch { return; }

  // A reply to one of OUR outbound requests (the permission handshake).
  if (msg.id !== undefined && msg.method === undefined) {
    const resolve = pendingPermission.get(msg.id);
    if (resolve) {
      pendingPermission.delete(msg.id);
      resolve(msg.result?.outcome?.optionId ?? 'reject');
    }
    return;
  }

  switch (msg.method) {
    case 'initialize':
      reply(msg.id, {
        protocolVersion: 1,
        agentInfo: { name: 'mock-acp', version: '0.0.0' },
        agentCapabilities: { loadSession: true, promptCapabilities: { embeddedContext: true } },
        authMethods: [],
      });
      return;
    case 'session/new':
      reply(msg.id, {
        sessionId: 'mock-s1',
        modes: { currentModeId: 'write', availableModes: [{ id: 'ask' }, { id: 'preview' }, { id: 'write' }] },
      });
      return;
    case 'session/set_mode':
      reply(msg.id, { mode: msg.params?.modeId ?? 'write' });
      return;
    case 'session/prompt': {
      const sessionId = msg.params?.sessionId ?? 'mock-s1';
      const text = (msg.params?.prompt ?? []).map((b) => b.text ?? '').join('\n');
      const result = await runPrompt(sessionId, text);
      reply(msg.id, result);
      return;
    }
    case 'session/cancel':
      // Notification; nothing to reply.
      return;
    default:
      if (msg.id !== undefined) {
        send({ jsonrpc: '2.0', id: msg.id, error: { code: -32601, message: `unknown method ${msg.method}` } });
      }
      return;
  }
});
