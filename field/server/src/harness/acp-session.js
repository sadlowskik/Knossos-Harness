// Generic ACP (Agent Client Protocol) adapter.
//
// ACP is JSON-RPC 2.0 over newline-delimited stdio. Knossos implements the *agent* side of
// this protocol (see knossos-rs/src/acp.rs); THIS adapter is the *client* side, so Field can
// drive ANY ACP-compliant coding agent as a harness. It owns one external agent process,
// performs the ACP handshake, sends prompts, and translates the agent's `session/update`
// notifications into Field's canonical event vocabulary (contracts/field-event-v1.schema.json).
//
// Nothing here is Knossos-specific: the launch spec (command/args/env) comes from the manifest
// or the orders, and the protocol is spoken exactly as the ACP schema defines it.
import { spawn } from 'node:child_process';
import { createInterface } from 'node:readline';
import { HarnessAdapter } from './adapter.js';
import { buildChildEnvironment } from '../child-env.js';

// The ACP protocol version this client speaks. Matches PROTOCOL_VERSION in knossos-rs/src/acp.rs.
const ACP_PROTOCOL_VERSION = 1;

// How long the initialize -> session/new handshake may take before we give up. A prompt turn
// itself is deliberately un-timed (a turn can legitimately run for minutes); only the handshake,
// which is a couple of cheap round trips, is bounded so a broken agent cannot hang a session.
const HANDSHAKE_TIMEOUT_MS = 15_000;

export class AcpSession extends HarnessAdapter {
  constructor(opts) {
    super();
    this.id = opts.id;                       // Field session id (the sessionId every event carries)
    this.agentId = opts.agentId;
    this.name = opts.name;
    this.role = opts.role;
    this.model = opts.model;
    this.endpointId = opts.endpointId;
    this.effort = opts.effort ?? 'medium';
    this.cwd = opts.cwd;
    this.workspaceId = opts.workspaceId;
    this.systemPrompt = opts.systemPrompt ?? '';
    this.env = opts.env ?? {};
    this.providerKind = opts.providerKind;
    this.readOnly = opts.readOnly === true;
    this.environmentScope = opts.environmentScope ?? null;
    this.credentialEnvKeys = opts.credentialEnvKeys ?? [];

    // Launch spec for the external ACP agent. Command/args/env come from the manifest or orders;
    // FIELD_ACP_BIN / FIELD_ACP_ARGS provide operator-level overrides. There is no single default
    // binary because ACP is generic — the operator names the agent they want to drive.
    this.command = opts.command ?? opts.acpCommand ?? process.env.FIELD_ACP_BIN ?? 'acp-agent';
    this.args = normalizeArgs(opts.args ?? opts.acpArgs ?? process.env.FIELD_ACP_ARGS);
    this.launchEnv = opts.acpEnv ?? {};

    this.proc = null;
    this.ownedProcesses = new Set();
    this.state = 'created';
    this.started = false;
    this.ready = false;            // handshake complete (initialize + session/new)
    this.hasTurn = false;          // at least one prompt sent (drives systemPrompt prepend)
    this.acpSessionId = null;      // the ACP-side session id from session/new (distinct from this.id)
    this.pendingOrders = null;     // orders queued until the handshake completes
    this.stderr = '';

    // JSON-RPC client bookkeeping.
    this.nextRpcId = 0;
    this.pendingRpc = new Map();          // jsonrpc id -> { resolve, reject, timer }
    this.pendingPermissions = new Map();  // Field requestId -> inbound jsonrpc id to answer
    this.toolNames = new Map();           // toolCallId -> tool name (for tool_result labelling)
  }

  // emitEvent is inherited from HarnessAdapter (canonical Field-event emission point).

  capabilities() {
    return {
      kind: 'acp',
      duplex: true,               // session/prompt can be sent repeatedly within a session
      resumable: true,            // pause() stops the process; resume() re-handshakes
      permissions: 'inline-handshake', // session/request_permission over the same JSON-RPC channel
      delegation: false,
      browser: false,
      verification: false,
      dryRun: true,               // session/set_mode ask/preview gives a read-only launch mode
    };
  }

  // --------------------------------------------------------------------- lifecycle

  start(orders) {
    if (this.proc) throw new Error(`session ${this.id} already running`);
    this.beforeStart?.();
    this.pendingOrders = orders ?? 'Await orders.';
    this.stderr = '';
    this.ready = false;

    this.proc = spawn(this.command, this.args, {
      cwd: this.cwd,
      env: buildChildEnvironment({
        provider: this.providerKind,
        explicitKeys: this.credentialEnvKeys,
        overrides: {
          ...this.env,
          ...this.launchEnv,
          FIELD_SESSION_ID: this.id,
        },
      }),
      stdio: ['pipe', 'pipe', 'pipe'],
      windowsHide: true,
    });
    const child = this.proc;
    this.ownedProcesses.add(child);
    this.started = true;
    this.state = 'spawning';

    child.on('error', (error) => {
      this.ownedProcesses.delete(child);
      if (this.proc !== child) return;
      this.state = 'error';
      this.rejectAllRpc(new Error(`ACP spawn failed: ${error.message}`));
      this.emitEvent('session.ended', { reason: 'error', error: `ACP spawn failed: ${error.message}` });
      this.proc = null;
    });

    createInterface({ input: child.stdout }).on('line', (line) => {
      if (this.proc === child) this.handleLine(line);
    });
    child.stderr.on('data', (chunk) => {
      this.stderr += chunk.toString();
      if (this.stderr.length > 8000) this.stderr = this.stderr.slice(-4000);
    });

    child.on('close', (code) => {
      this.ownedProcesses.delete(child);
      if (this.proc !== child) return;
      this.proc = null;
      this.rejectAllRpc(new Error('ACP agent exited'));
      if (this.state === 'paused' || this.state === 'cancelled') return;
      if (code !== 0) {
        this.state = 'error';
        this.emitEvent('session.ended', {
          reason: 'error',
          error: this.stderr.trim().split(/\r?\n/).slice(-4).join('\n') || `ACP agent exited ${code}`,
        });
      } else if (this.state !== 'done') {
        this.state = 'done';
        this.emitEvent('session.ended', { reason: 'exit' });
      }
    });

    // The handshake is async (request/response). Kick it off but return synchronously, like the
    // other adapters, so the registry's start() call site is unchanged.
    this.handshake().catch((error) => {
      if (this.state === 'cancelled' || this.state === 'paused') return;
      this.state = 'error';
      this.emitEvent('session.state', { state: 'error', detail: String(error?.message ?? error) });
    });
    return this;
  }

  async handshake() {
    await this.rpc('initialize', {
      protocolVersion: ACP_PROTOCOL_VERSION,
      clientInfo: { name: 'field', title: 'Knossos Field' },
      clientCapabilities: {},
    }, HANDSHAKE_TIMEOUT_MS);
    if (!this.proc) return;

    const created = await this.rpc('session/new', {
      cwd: this.cwd,
      mcpServers: [],
    }, HANDSHAKE_TIMEOUT_MS);
    if (!this.proc) return;
    this.acpSessionId = created?.sessionId ?? null;

    // Read-only / snapshot roles map onto an ACP mode that does not touch the workspace. Best
    // effort: an agent that does not support session/set_mode simply errors and we proceed.
    if (this.readOnly || ['snapshot', 'production-readonly'].includes(this.environmentScope)) {
      try { await this.rpc('session/set_mode', { sessionId: this.acpSessionId, modeId: 'ask' }, HANDSHAKE_TIMEOUT_MS); }
      catch { /* mode is advisory; a turn still runs without it */ }
    }

    this.ready = true;
    this.state = 'ready';
    this.emitEvent('session.state', { state: 'ready' });
    this.emitEvent('endpoint.routed', {
      endpointId: this.endpointId,
      model: this.model,
      reason: 'ACP session/new',
    });

    if (this.pendingOrders != null) {
      const orders = this.pendingOrders;
      this.pendingOrders = null;
      this.send(orders);
    }
  }

  /** Push a real user turn into the running ACP session via session/prompt. */
  send(text) {
    if (!this.proc || this.state === 'cancelled' || this.state === 'paused') return false;
    // Before the handshake finishes there is no ACP session to prompt; queue the orders and let
    // handshake() flush them, mirroring KnossosSession's pending-orders handling.
    if (!this.ready || !this.acpSessionId) {
      this.pendingOrders = text;
      return true;
    }
    const full = this.hasTurn || !this.systemPrompt
      ? String(text)
      : `${this.systemPrompt}\n\n---\n\n# Active orders\n\n${text}`;

    this.emitEvent('session.message', { role: 'user', text: String(text) });
    this.state = 'thinking';
    this.emitEvent('session.state', { state: 'thinking' });
    this.hasTurn = true;

    this.rpc('session/prompt', {
      sessionId: this.acpSessionId,
      prompt: [{ type: 'text', text: full }],
    }).then((result) => {
      this.onTurnComplete(result, false);
    }).catch((error) => {
      if (this.state === 'cancelled' || this.state === 'paused' || !this.proc) return;
      this.onTurnComplete({ error: String(error?.message ?? error) }, true);
    });
    return true;
  }

  onTurnComplete(result, isError) {
    this.state = isError ? 'error' : 'idle';
    this.emitEvent('session.state', { state: this.state, detail: result?.stopReason ?? null });
    this.emitEvent('session.turn_complete', {
      result: typeof result?.stopReason === 'string' ? result.stopReason
        : (typeof result?.error === 'string' ? result.error.slice(0, 4000) : null),
      stopReason: result?.stopReason ?? null,
      missionId: result?.missionId ?? null,
      isError: !!isError,
    });
  }

  /** Stop the process but keep the Field session id so resume() can re-handshake. */
  pause() {
    if (!this.proc) return false;
    this.state = 'paused';
    if (this.acpSessionId) this.notify('session/cancel', { sessionId: this.acpSessionId });
    this.proc.stdin?.end?.();
    this.proc.kill();
    this.proc = null;
    this.ready = false;
    this.acpSessionId = null;
    this.emitEvent('session.state', { state: 'paused', detail: 'ACP process stopped; workspace retained' });
    return true;
  }

  resume(orders) {
    if (this.proc) return false;
    this.hasTurn = false;
    this.start(orders ?? 'Resume from the workspace, restate current progress, and continue.');
    this.emitEvent('session.state', { state: 'thinking', detail: 'ACP restarted from workspace state' });
    return true;
  }

  cancel() {
    this.state = 'cancelled';
    if (this.proc) {
      if (this.acpSessionId) this.notify('session/cancel', { sessionId: this.acpSessionId });
      this.proc.stdin?.end?.();
      this.proc.kill();
      this.proc = null;
    }
    this.rejectAllRpc(new Error('session cancelled'));
    this.emitEvent('session.ended', { reason: 'cancelled' });
    return true;
  }

  /** Answer a `harness.permission_requested` by replying to the agent's ACP request. */
  decidePermission(requestId, decision) {
    const rpcId = this.pendingPermissions.get(requestId);
    if (rpcId === undefined) return false;
    this.pendingPermissions.delete(requestId);
    this.writeMessage({
      jsonrpc: '2.0',
      id: rpcId,
      result: {
        outcome: {
          outcome: 'selected',
          optionId: decision === 'allow' ? 'allow' : 'reject',
        },
      },
    });
    return true;
  }

  // --------------------------------------------------------------------- JSON-RPC transport

  writeMessage(message) {
    if (!this.proc?.stdin?.writable) return false;
    this.proc.stdin.write(JSON.stringify(message) + '\n');
    return true;
  }

  /** Send a JSON-RPC request and resolve with its result (or reject with its error). */
  rpc(method, params, timeoutMs) {
    return new Promise((resolve, reject) => {
      const id = ++this.nextRpcId;
      if (!this.writeMessage({ jsonrpc: '2.0', id, method, params })) {
        reject(new Error(`cannot send ${method}: ACP agent stdin is not writable`));
        return;
      }
      let timer = null;
      if (timeoutMs) {
        timer = setTimeout(() => {
          if (this.pendingRpc.delete(id)) reject(new Error(`ACP ${method} timed out after ${timeoutMs}ms`));
        }, timeoutMs);
        timer.unref?.();
      }
      this.pendingRpc.set(id, { resolve, reject, timer });
    });
  }

  /** Send a JSON-RPC notification (no id, no reply expected). */
  notify(method, params) {
    return this.writeMessage({ jsonrpc: '2.0', method, params });
  }

  rejectAllRpc(error) {
    for (const { reject, timer } of this.pendingRpc.values()) {
      if (timer) clearTimeout(timer);
      reject(error);
    }
    this.pendingRpc.clear();
  }

  handleLine(line) {
    const trimmed = String(line).trim();
    if (!trimmed) return;
    let msg;
    try { msg = JSON.parse(trimmed); } catch { return; }

    const hasId = msg.id !== undefined && msg.id !== null;
    const hasMethod = typeof msg.method === 'string';

    if (hasMethod && hasId) { this.handleInboundRequest(msg); return; }   // agent -> client request
    if (hasMethod) { this.handleNotification(msg); return; }              // agent -> client notification
    if (hasId) { this.handleResponse(msg); return; }                     // reply to one of our requests
  }

  handleResponse(msg) {
    const pending = this.pendingRpc.get(msg.id);
    if (!pending) return;
    this.pendingRpc.delete(msg.id);
    if (pending.timer) clearTimeout(pending.timer);
    if (msg.error) pending.reject(new Error(msg.error.message ?? `ACP error ${msg.error.code ?? ''}`.trim()));
    else pending.resolve(msg.result ?? {});
  }

  handleInboundRequest(msg) {
    if (msg.method === 'session/request_permission') {
      const params = msg.params ?? {};
      const toolCall = params.toolCall ?? {};
      const requestId = String(msg.id);
      this.pendingPermissions.set(requestId, msg.id);
      this.emitEvent('harness.permission_requested', {
        requestId,
        toolName: toolCall.title || toolCall.toolCallId || 'tool',
        toolCallId: toolCall.toolCallId ?? null,
        options: Array.isArray(params.options) ? params.options : [],
        input: toolCall.rawInput ?? null,
      });
      return;
    }
    // We advertise no fs/terminal client capabilities, so no other agent->client request is
    // expected. Answer rather than ignore: an unanswered request blocks the agent's turn.
    this.writeMessage({
      jsonrpc: '2.0',
      id: msg.id,
      error: { code: -32601, message: `${msg.method} is not supported by this client` },
    });
  }

  handleNotification(msg) {
    if (msg.method !== 'session/update') return;
    const update = msg.params?.update;
    if (!update || typeof update !== 'object') return;
    this.translateUpdate(update);
  }

  /**
   * Translate one ACP `session/update` into Field canonical events. Field-name corrections vs.
   * the plain-English mapping in the orders are documented per case below.
   */
  translateUpdate(update) {
    switch (update.sessionUpdate) {
      case 'agent_message_chunk':
        // ACP text lives at update.content.text (a ContentBlock), not update.text.
        this.emitEvent('session.message', { role: 'assistant', text: contentText(update.content) });
        return;

      case 'agent_thought_chunk':
        this.emitEvent('session.thinking', { text: contentText(update.content).slice(0, 2000) });
        return;

      case 'plan':
        // ACP plans are `entries[]` with {content, priority, status}, not a `steps[]` array.
        this.emitEvent('session.progress', {
          done: (update.entries ?? []).filter((e) => e.status === 'completed').length,
          total: (update.entries ?? []).length,
          steps: (update.entries ?? []).map((e) => e.content),
        });
        return;

      case 'tool_call': {
        // The initial call. ACP identifies it by `toolCallId`, describes it by `title`, and
        // classifies it by `kind`. There is no machine tool-name field, so name falls back
        // through title -> kind -> 'tool' to satisfy the canonical schema's non-empty `name`.
        const name = firstNonEmpty(update.title, update.kind, 'tool');
        this.toolNames.set(update.toolCallId, name);
        this.emitEvent('session.tool_use', {
          toolId: String(update.toolCallId),
          name,
          summary: update.title ?? name,
          kind: update.kind ?? null,
          input: update.rawInput ?? null,
          locations: update.locations ?? null,
        });
        // Some agents (Knossos among them) emit tool_call already in a terminal state instead
        // of a follow-up tool_call_update. Surface the result too when that happens.
        if (update.status === 'completed' || update.status === 'failed') {
          this.emitToolResult(update);
        }
        return;
      }

      case 'tool_call_update':
        // The status/result of a previously announced call.
        this.emitToolResult(update);
        return;

      default:
        // available_commands_update, current_mode_update, user_message_chunk, etc. — real ACP
        // narration, but nothing the Field event vocabulary models. Ignore.
        return;
    }
  }

  emitToolResult(update) {
    const name = this.toolNames.get(update.toolCallId) ?? null;
    this.emitEvent('session.tool_result', {
      toolId: String(update.toolCallId),
      name,
      ok: update.status !== 'failed',
      preview: extractToolOutput(update).slice(0, 400),
    });
  }
}

function normalizeArgs(args) {
  if (Array.isArray(args)) return args.map(String);
  if (typeof args === 'string' && args.trim()) return args.trim().split(/\s+/);
  return [];
}

function firstNonEmpty(...values) {
  for (const v of values) {
    if (typeof v === 'string' && v.length) return v;
  }
  return 'tool';
}

/** Pull the text out of an ACP ContentBlock ({type:'text', text}) or a bare string. */
function contentText(content) {
  if (typeof content === 'string') return content;
  if (content && typeof content === 'object') {
    if (typeof content.text === 'string') return content.text;
  }
  return '';
}

/** Flatten an ACP tool call's output content / rawOutput into preview text. */
function extractToolOutput(update) {
  if (typeof update.rawOutput === 'string') return update.rawOutput;
  const blocks = Array.isArray(update.content) ? update.content : [];
  return blocks
    .map((b) => contentText(b?.content ?? b))
    .filter(Boolean)
    .join(' ');
}
