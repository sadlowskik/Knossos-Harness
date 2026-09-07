// One HarnessSession is one real `claude` process running in stream-json duplex mode.
// Every Field event about an agent originates in this file, parsed from that process's
// actual stdout. Nothing is synthesized.
import { spawn } from 'node:child_process';
import { EventEmitter } from 'node:events';
import { createInterface } from 'node:readline';
import { describeTool } from './tools.js';
import { buildChildEnvironment } from '../child-env.js';

const CLAUDE_BIN = process.env.FIELD_CLAUDE_BIN || 'claude';

// Field exposes low / medium / high / adaptive. The harness takes an effort level.
// `adaptive` is resolved by the registry before it reaches here, and the resolved
// level is recorded on the spawn event so the operator can see what was chosen.
const EFFORT = new Set(['low', 'medium', 'high', 'xhigh', 'max']);

export class HarnessSession extends EventEmitter {
  constructor(opts) {
    super();
    this.id = opts.id;                       // uuid, also the harness --session-id
    this.agentId = opts.agentId;
    this.name = opts.name;
    this.role = opts.role;
    this.model = opts.model;
    this.endpointId = opts.endpointId;
    this.effort = EFFORT.has(opts.effort) ? opts.effort : 'medium';
    this.cwd = opts.cwd;
    this.workspaceId = opts.workspaceId;
    this.systemPrompt = opts.systemPrompt;
    this.addDirs = opts.addDirs ?? [];
    this.allowedTools = opts.allowedTools ?? null;
    this.disallowedTools = opts.disallowedTools ?? null;
    this.mcpConfig = opts.mcpConfig ?? null;   // JSON string for --mcp-config
    this.permissionTool = opts.permissionTool ?? null;
    this.permissionMode = opts.permissionMode ?? null;
    this.workspaces = opts.workspaces ?? [];
    this.env = opts.env ?? {};
    this.providerKind = opts.providerKind;
    this.credentialEnvKeys = opts.credentialEnvKeys ?? [];

    this.proc = null;
    this.ownedProcesses = new Set();
    this.state = 'created';
    this.started = false;
    this.buffer = '';
    this.pendingTools = new Map();
    this.lifetimeUsage = { input: 0, output: 0, cacheRead: 0 };
    this.turnSawUsage = false;
  }

  emitEvent(kind, data) {
    this.emit('event', kind, { sessionId: this.id, ...data });
  }

  buildArgs({ resume = false } = {}) {
    const a = [
      '-p',
      '--output-format', 'stream-json',
      '--input-format', 'stream-json',
      '--verbose',
    ];
    if (resume) a.push('--resume', this.id);
    else a.push('--session-id', this.id);

    if (this.model) a.push('--model', this.model);
    a.push('--effort', this.effort);
    if (this.systemPrompt) a.push('--append-system-prompt', this.systemPrompt);
    for (const d of this.addDirs) a.push('--add-dir', d);
    // --tools is the actual role capability boundary. Do not also pass --allowedTools:
    // that flag auto-approves consequential tools and would bypass Field's permission
    // bridge for writes, commands, and network requests.
    if (this.allowedTools?.length) {
      a.push('--tools', this.allowedTools.join(','));
    } else {
      a.push('--tools', '');
    }
    if (this.disallowedTools?.length) a.push('--disallowedTools', this.disallowedTools.join(','));

    if (this.permissionTool && this.mcpConfig) {
      a.push('--mcp-config', this.mcpConfig);
      a.push('--permission-prompt-tool', this.permissionTool);
    } else if (this.permissionMode) {
      a.push('--permission-mode', this.permissionMode);
    }
    return a;
  }

  start(orders, { resume = false } = {}) {
    if (this.proc) throw new Error(`session ${this.id} already running`);
    this.beforeStart?.();
    const args = this.buildArgs({ resume });

    this.proc = spawn(CLAUDE_BIN, args, {
      cwd: this.cwd,
      env: buildChildEnvironment({
        provider: this.providerKind,
        explicitKeys: this.credentialEnvKeys,
        overrides: { ...this.env, FIELD_SESSION_ID: this.id },
      }),
      stdio: ['pipe', 'pipe', 'pipe'],
      windowsHide: true,
    });

    const child = this.proc;
    this.ownedProcesses.add(child);
    this.started = true;
    this.state = 'spawning';

    child.on('error', (err) => {
      this.ownedProcesses.delete(child);
      if (this.proc !== child) return;
      this.state = 'error';
      this.emitEvent('session.ended', { reason: 'error', error: `spawn failed: ${err.message}` });
      this.proc = null;
    });

    const rl = createInterface({ input: this.proc.stdout });
    rl.on('line', (line) => { if (this.proc === child) this.handleLine(line); });

    let stderr = '';
    this.proc.stderr.on('data', (b) => {
      stderr += b.toString();
      if (stderr.length > 8000) stderr = stderr.slice(-4000);
    });

    child.on('close', (code) => {
      this.ownedProcesses.delete(child);
      if (this.proc !== child) return;
      this.proc = null;
      if (this.state === 'paused' || this.state === 'cancelled') return;
      if (code !== 0) {
        this.state = 'error';
        this.emitEvent('session.ended', {
          reason: 'error',
          error: (stderr.trim().split('\n').slice(-4).join('\n')) || `harness exited ${code}`,
        });
      } else if (this.state !== 'done') {
        this.state = 'done';
        this.emitEvent('session.ended', { reason: 'exit' });
      }
    });

    if (orders) this.send(orders);
    return this;
  }

  /** Push a real user turn into the running harness session. */
  send(text) {
    if (!this.proc?.stdin?.writable) return false;
    const msg = {
      type: 'user',
      message: { role: 'user', content: [{ type: 'text', text }] },
    };
    this.proc.stdin.write(JSON.stringify(msg) + '\n');
    this.turnSawUsage = false;
    this.emitEvent('session.message', { role: 'user', text });
    this.state = 'thinking';
    this.emitEvent('session.state', { state: 'thinking' });
    return true;
  }

  /** Stop the process but keep the session id so `--resume` can pick it back up. */
  pause() {
    if (!this.proc) return false;
    this.state = 'paused';
    this.proc.stdin.end();
    this.proc.kill();
    this.proc = null;
    this.emitEvent('session.state', { state: 'paused' });
    return true;
  }

  resume(orders) {
    if (this.proc) return false;
    this.state = 'spawning';
    this.start(orders ?? 'Continue.', { resume: true });
    this.emitEvent('session.state', { state: 'thinking', detail: 'resumed' });
    return true;
  }

  cancel() {
    this.state = 'cancelled';
    if (this.proc) {
      this.proc.stdin.end();
      this.proc.kill();
      this.proc = null;
    }
    this.emitEvent('session.ended', { reason: 'cancelled' });
    return true;
  }

  handleLine(line) {
    const trimmed = line.trim();
    if (!trimmed) return;
    let msg;
    try { msg = JSON.parse(trimmed); } catch { return; }

    switch (msg.type) {
      case 'system':
        if (msg.subtype === 'init') {
          this.state = 'ready';
          this.emitEvent('session.state', { state: 'ready' });
          this.emitEvent('endpoint.routed', {
            endpointId: this.endpointId,
            model: msg.model ?? this.model,
            reason: 'session start',
          });
        }
        return;

      case 'assistant':
        this.handleAssistant(msg.message ?? {});
        return;

      case 'user':
        this.handleToolResults(msg.message ?? {});
        return;

      case 'result':
        this.handleResult(msg);
        return;

      default:
        return;
    }
  }

  handleAssistant(message) {
    const content = Array.isArray(message.content) ? message.content : [];
    for (const part of content) {
      if (part.type === 'text' && part.text?.trim()) {
        this.emitEvent('session.message', { role: 'assistant', text: part.text });
      } else if (part.type === 'thinking' && part.thinking) {
        this.emitEvent('session.thinking', { text: String(part.thinking).slice(0, 2000) });
      } else if (part.type === 'tool_use') {
        this.handleToolUse(part);
      }
    }
    if (message.usage) this.emitUsage(message.usage);
  }

  handleToolUse(part) {
    const info = describeTool(part.name, part.input ?? {}, {
      workspaces: this.workspaces,
      cwd: this.cwd,
    });
    this.pendingTools.set(part.id, { name: part.name, info });

    this.emitEvent('session.tool_use', {
      toolId: part.id,
      name: part.name,
      summary: info.summary,
      workspaceId: info.workspaceId ?? null,
      dir: info.dir ?? null,
      path: info.path ?? null,
      command: info.command ?? null,
      input: truncateInput(part.input),
    });

    if (info.browser) {
      this.emitEvent('browser.navigated', {
        url: info.browser.url,
        domain: info.browser.domain,
      });
    }
    if (info.delegation) {
      // A real child harness session, created by the agent itself, inside the
      // delegation limits the Field configured for it.
      this.emitEvent('session.delegated', {
        parentSessionId: this.id,
        childSessionId: `${this.id}:${part.id}`,
        description: info.delegation.description,
        subagentType: info.delegation.type,
      });
    }
    if (info.progress) {
      this.emitEvent('session.progress', { ...info.progress });
    }
  }

  handleToolResults(message) {
    const content = Array.isArray(message.content) ? message.content : [];
    for (const part of content) {
      if (part.type !== 'tool_result') continue;
      const pending = this.pendingTools.get(part.tool_use_id);
      this.pendingTools.delete(part.tool_use_id);
      const text = typeof part.content === 'string'
        ? part.content
        : Array.isArray(part.content)
          ? part.content.map((c) => c.text ?? '').join(' ')
          : '';
      this.emitEvent('session.tool_result', {
        toolId: part.tool_use_id,
        name: pending?.name ?? null,
        ok: !part.is_error,
        preview: text.slice(0, 400),
      });
    }
  }

  emitUsage(usage, costUsd) {
    const deltaInput = Math.max(0, Number(usage.input_tokens) || 0);
    const deltaOutput = Math.max(0, Number(usage.output_tokens) || 0);
    const deltaCacheRead = Math.max(0, Number(usage.cache_read_input_tokens) || 0);
    this.lifetimeUsage.input += deltaInput;
    this.lifetimeUsage.output += deltaOutput;
    this.lifetimeUsage.cacheRead += deltaCacheRead;
    this.turnSawUsage = true;
    this.emitEvent('session.usage', {
      inputTokens: this.lifetimeUsage.input,
      outputTokens: this.lifetimeUsage.output,
      cacheRead: this.lifetimeUsage.cacheRead,
      contextTokens:
        (usage.input_tokens ?? 0) +
        (usage.cache_read_input_tokens ?? 0) +
        (usage.cache_creation_input_tokens ?? 0) +
        (usage.output_tokens ?? 0),
      deltaInput,
      deltaOutput,
      ...(typeof costUsd === 'number' ? { costUsd } : {}),
    });
  }

  handleResult(msg) {
    if (msg.usage && !this.turnSawUsage) this.emitUsage(msg.usage, msg.total_cost_usd);
    else if (typeof msg.total_cost_usd === 'number') {
      this.emitEvent('session.usage', { costUsd: msg.total_cost_usd });
    }
    this.state = msg.is_error ? 'error' : 'idle';
    this.emitEvent('session.state', {
      state: this.state,
      detail: msg.subtype ?? null,
    });
    this.emitEvent('session.turn_complete', {
      result: typeof msg.result === 'string' ? msg.result.slice(0, 4000) : null,
      turns: msg.num_turns ?? null,
      durationMs: msg.duration_ms ?? null,
      isError: !!msg.is_error,
    });
  }
}

function truncateInput(input) {
  if (!input || typeof input !== 'object') return input ?? null;
  const out = {};
  for (const [k, v] of Object.entries(input)) {
    out[k] = typeof v === 'string' && v.length > 600 ? v.slice(0, 600) + '…' : v;
  }
  return out;
}
