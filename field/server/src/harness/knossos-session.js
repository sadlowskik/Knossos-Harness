// Knossos adapter for OpenAI-compatible and Cameo-backed endpoints.
//
// Knossos `serve` is a long-lived NDJSON protocol: commands on stdin, events on stdout.
// This adapter translates those real events into the same Field vocabulary as the direct
// Claude adapter, so campaigns do not care which harness/provider powers a unit.
import { spawn } from 'node:child_process';
import { EventEmitter } from 'node:events';
import { createInterface } from 'node:readline';
import fs from 'node:fs';
import path from 'node:path';
import { buildChildEnvironment } from '../child-env.js';

const KNOSSOS_BIN = resolveKnossosBinary();

export function resolveKnossosBinary() {
  if (process.env.FIELD_KNOSSOS_BIN) return process.env.FIELD_KNOSSOS_BIN;
  const exe = process.platform === 'win32' ? 'knossos.exe' : 'knossos';
  const legacyExe = process.platform === 'win32' ? 'daedalus.exe' : 'daedalus';
  const roots = [
    path.resolve(process.cwd(), '..', 'knossos-rs'),
    path.resolve(process.cwd(), '..', 'Knossos-Harness', 'knossos-rs'),
    path.resolve(process.cwd(), '..', 'knossos-harness', 'knossos-rs'),
    path.resolve(process.cwd(), '..', 'daedalus', 'knossos-rs'),
    path.resolve(process.cwd(), '..', '..', 'daedalus', 'knossos-rs'),
  ];
  const candidates = roots.flatMap((root) => [
    path.join(root, 'target', 'release', exe),
    path.join(root, 'target', 'debug', exe),
    path.join(root, 'target', 'release', legacyExe),
    path.join(root, 'target', 'debug', legacyExe),
  ]);
  return candidates.find((candidate) => fs.existsSync(candidate)) ?? 'knossos';
}

export class KnossosSession extends EventEmitter {
  constructor(opts) {
    super();
    this.id = opts.id;
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
    this.engine = opts.engine ?? 'cameo';
    this.providerKind = opts.providerKind;
    this.readOnly = opts.readOnly === true;
    this.environmentScope = opts.environmentScope ?? null;
    this.credentialEnvKeys = opts.credentialEnvKeys ?? [];

    this.proc = null;
    this.ownedProcesses = new Set();
    this.state = 'created';
    this.started = false;
    this.hasTask = false;
    this.pendingOrders = null;
    this.stderr = '';
  }

  emitEvent(kind, data) { this.emit('event', kind, { sessionId: this.id, ...data }); }

  buildArgs() {
    const args = ['serve', '--workspace', this.cwd, '--engine', this.engine];
    if (this.model) args.push('--model', this.model);
    if (this.readOnly || ['snapshot', 'production-readonly'].includes(this.environmentScope)) {
      args.push('--dry-run');
    }
    return args;
  }

  start(orders) {
    if (this.proc) throw new Error(`session ${this.id} already running`);
    this.beforeStart?.();
    this.pendingOrders = orders ?? 'Await orders.';
    this.stderr = '';
    this.proc = spawn(KNOSSOS_BIN, this.buildArgs(), {
      cwd: this.cwd,
      env: buildChildEnvironment({
        provider: this.providerKind,
        explicitKeys: this.credentialEnvKeys,
        overrides: {
        ...this.env,
        KNOSSOS_SESSION_ID: this.id,
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
      this.emitEvent('session.ended', { reason: 'error', error: `Knossos spawn failed: ${error.message}` });
      this.proc = null;
    });
    createInterface({ input: this.proc.stdout }).on('line', (line) => { if (this.proc === child) this.handleLine(line); });
    this.proc.stderr.on('data', (chunk) => {
      this.stderr += chunk.toString();
      if (this.stderr.length > 8000) this.stderr = this.stderr.slice(-4000);
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
          error: this.stderr.trim().split(/\r?\n/).slice(-4).join('\n') || `Knossos exited ${code}`,
        });
      } else if (this.state !== 'done') {
        this.state = 'done';
        this.emitEvent('session.ended', { reason: 'exit' });
      }
    });
    return this;
  }

  write(command) {
    if (!this.proc?.stdin?.writable) return false;
    this.proc.stdin.write(JSON.stringify(command) + '\n');
    return true;
  }

  send(text) {
    if (!this.proc || this.state === 'paused') return false;
    const full = this.hasTask || !this.systemPrompt
      ? String(text)
      : `${this.systemPrompt}\n\n---\n\n# Active orders\n\n${text}`;
    const cmd = this.hasTask ? 'resume' : 'task';
    if (!this.write({ cmd, text: full })) return false;
    this.hasTask = true;
    this.emitEvent('session.message', { role: 'user', text: String(text) });
    this.state = 'thinking';
    this.emitEvent('session.state', { state: 'thinking' });
    return true;
  }

  pause() {
    if (!this.proc) return false;
    // Knossos serve cannot cancel an in-flight turn through its queued command channel.
    // Terminating the process is the only honest immediate pause. Workspace and trace are
    // durable; resume starts a fresh harness turn and says so explicitly.
    this.state = 'paused';
    this.proc.kill();
    this.proc = null;
    this.emitEvent('session.state', {
      state: 'paused', detail: 'Knossos process stopped; workspace and trace retained',
    });
    return true;
  }

  resume(orders) {
    if (this.proc) return false;
    this.hasTask = false;
    this.start(orders ?? 'Recover from the durable workspace and trace, restate current progress, and continue.');
    this.emitEvent('session.state', { state: 'thinking', detail: 'Knossos restarted from durable workspace state' });
    return true;
  }

  cancel() {
    this.state = 'cancelled';
    if (this.proc) {
      this.proc.kill();
      this.proc = null;
    }
    this.emitEvent('session.ended', { reason: 'cancelled' });
    return true;
  }

  decidePermission(requestId, decision) {
    return this.write({ cmd: 'permission', id: requestId, allow: decision === 'allow' });
  }

  handleLine(line) {
    let msg;
    try { msg = JSON.parse(String(line).trim()); } catch { return; }
    switch (msg.event) {
      case 'ready':
        this.state = 'ready';
        this.emitEvent('session.state', { state: 'ready' });
        this.emitEvent('endpoint.routed', {
          endpointId: this.endpointId, model: this.model,
          reason: `Knossos ready via ${msg.engine ?? this.engine}`,
        });
        this.write({ cmd: 'capabilities', permissions: true });
        if (this.pendingOrders) {
          const orders = this.pendingOrders;
          this.pendingOrders = null;
          this.send(orders);
        }
        return;
      case 'plan':
        this.emitEvent('session.progress', { done: 0, total: msg.steps?.length ?? 0, steps: msg.steps ?? [] });
        return;
      case 'outcome':
        this.state = msg.succeeded ? 'idle' : 'blocked';
        if (msg.summary) this.emitEvent('session.message', { role: 'assistant', text: msg.summary });
        this.emitEvent('session.state', { state: this.state, detail: msg.halt ?? null });
        this.emitEvent('session.turn_complete', {
          result: String(msg.summary ?? '').slice(0, 4000),
          turns: msg.steps_used ?? null, isError: !msg.succeeded,
          changedFiles: msg.changed ?? [], dryRun: !!msg.dry_run,
        });
        return;
      case 'verdict':
        this.emitEvent('session.verification', {
          passed: !!msg.passed, summary: msg.summary ?? '', tiers: msg.tiers ?? [], dryRun: !!msg.dry_run,
        });
        return;
      case 'permission_request':
        this.emitEvent('harness.permission_requested', {
          requestId: msg.id, toolName: msg.tool, input: msg.input,
        });
        return;
      case 'error':
        this.state = 'error';
        this.emitEvent('session.state', { state: 'error', detail: msg.message ?? 'Knossos error' });
        return;
      case 'idle':
        if (this.state !== 'blocked' && this.state !== 'error') this.state = 'idle';
        this.emitEvent('session.state', { state: this.state });
        return;
      default:
        return;
    }
  }
}
