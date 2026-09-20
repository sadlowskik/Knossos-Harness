#!/usr/bin/env node
// A minimal stdio MCP server whose only tool is the Field permission gate.
// The harness calls it via --permission-prompt-tool before any consequential action.
// It parks the request with the Field server and blocks until a human decides.
// stdout carries JSON-RPC only; anything else would corrupt the protocol stream.
import { createInterface } from 'node:readline';

const API = process.env.FIELD_API || 'http://127.0.0.1:7749';
const SESSION = process.env.FIELD_SESSION || 'unknown';
const INTERNAL_TOKEN = process.env.FIELD_INTERNAL_TOKEN;
const TOOL_NAME = 'approve';

function send(obj) { process.stdout.write(JSON.stringify(obj) + '\n'); }
function reply(id, result) { send({ jsonrpc: '2.0', id, result }); }
function fail(id, message) { send({ jsonrpc: '2.0', id, error: { code: -32603, message } }); }

async function askOperator(toolName, input, toolUseId) {
  if (!INTERNAL_TOKEN) throw new Error('internal Field authorization is not configured');
  const res = await fetch(`${API}/api/internal/permission`, {
    method: 'POST',
    headers: {
      authorization: `Bearer ${INTERNAL_TOKEN}`,
      'content-type': 'application/json',
    },
    body: JSON.stringify({ sessionId: SESSION, toolName, input, toolUseId }),
  });
  if (!res.ok) throw new Error(`field server refused the request (${res.status})`);
  return res.json(); // { decision: 'allow' | 'deny', message?, updatedInput? }
}

const rl = createInterface({ input: process.stdin });

rl.on('line', async (line) => {
  const text = line.trim();
  if (!text) return;
  let msg;
  try { msg = JSON.parse(text); } catch { return; }
  const { id, method, params } = msg;

  if (method === 'initialize') {
    reply(id, {
      protocolVersion: params?.protocolVersion ?? '2024-11-05',
      capabilities: { tools: { listChanged: false } },
      serverInfo: { name: 'field', version: '0.2.0-beta.2' },
    });
    return;
  }

  if (method === 'notifications/initialized' || id === undefined) return;

  if (method === 'tools/list') {
    reply(id, {
      tools: [{
        name: TOOL_NAME,
        description:
          'Ask the Field operator to approve a consequential tool call. ' +
          'Blocks until a human decides.',
        inputSchema: {
          type: 'object',
          properties: {
            tool_name: { type: 'string' },
            input: { type: 'object' },
            tool_use_id: { type: 'string' },
          },
          required: ['tool_name', 'input'],
        },
      }],
    });
    return;
  }

  if (method === 'tools/call') {
    const args = params?.arguments ?? {};
    try {
      const decision = await askOperator(args.tool_name, args.input, args.tool_use_id);
      const payload = decision.decision === 'allow'
        ? { behavior: 'allow', updatedInput: decision.updatedInput ?? args.input }
        : { behavior: 'deny', message: decision.message || 'Denied by the Field operator.' };
      reply(id, { content: [{ type: 'text', text: JSON.stringify(payload) }] });
    } catch (err) {
      // A permission gate that fails open is not a permission gate.
      reply(id, {
        content: [{
          type: 'text',
          text: JSON.stringify({
            behavior: 'deny',
            message: `Field permission gate unreachable: ${err.message}`,
          }),
        }],
      });
    }
    return;
  }

  fail(id, `unsupported method: ${method}`);
});
