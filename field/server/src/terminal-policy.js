import { spawn } from 'node:child_process';

export const TERMINAL_LIMITS = Object.freeze({
  concurrent: 4,
  timeoutMs: 10 * 60 * 1000,
  outputBytes: 2 * 1024 * 1024,
  outputLines: 20_000,
  commandChars: 16_000,
});

export function validateTerminalCommand(value) {
  if (typeof value !== 'string' || !value.trim()) throw new Error('terminal command is required');
  if (value.length > TERMINAL_LIMITS.commandChars) throw new Error('terminal command is too long');
  if (value.includes('\0')) throw new Error('terminal command contains a null byte');
  return value;
}

export function redactCommand(value) {
  return String(value)
    .replace(/\bBearer\s+[A-Za-z0-9._~+/=-]{8,}\b/gi, 'Bearer [REDACTED]')
    .replace(/\b(sk-[A-Za-z0-9_-]{8,})\b/g, '[REDACTED]')
    .replace(/\b(password|passwd|secret|token|api[_-]?key|private[_-]?key)=([^\s;&|]+)/gi, '$1=[REDACTED]');
}

export class OutputBudget {
  constructor({ bytes = TERMINAL_LIMITS.outputBytes, lines = TERMINAL_LIMITS.outputLines } = {}) {
    this.maxBytes = bytes;
    this.maxLines = lines;
    this.bytes = 0;
    this.lines = 0;
    this.exceeded = false;
  }

  accept(chunk) {
    if (this.exceeded) return { text: '', exceeded: true };
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk);
    const remaining = Math.max(0, this.maxBytes - this.bytes);
    const accepted = buffer.subarray(0, remaining);
    let text = accepted.toString();
    const remainingLines = Math.max(0, this.maxLines - this.lines);
    const breaks = [...text.matchAll(/\n/g)];
    if (breaks.length > remainingLines) {
      text = remainingLines === 0 ? '' : text.slice(0, breaks[remainingLines - 1].index + 1);
    }
    const emittedBytes = Buffer.byteLength(text);
    this.bytes += emittedBytes;
    this.lines += Math.min(breaks.length, remainingLines);
    this.exceeded = buffer.length > remaining || breaks.length > remainingLines;
    return { text, exceeded: this.exceeded };
  }
}

export function stopProcessTree(proc) {
  if (!proc?.pid) return;
  if (process.platform === 'win32') {
    const killer = spawn('taskkill.exe', ['/pid', String(proc.pid), '/t', '/f'], {
      windowsHide: true,
      stdio: 'ignore',
    });
    killer.on('error', () => { try { proc.kill(); } catch { /* already gone */ } });
    return;
  }
  try { process.kill(-proc.pid, 'SIGTERM'); } catch { try { proc.kill('SIGTERM'); } catch { /* already gone */ } }
}
