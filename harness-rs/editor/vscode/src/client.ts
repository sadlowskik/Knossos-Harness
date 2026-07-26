// NDJSON client for `daedalus serve`.
//
// Keeps one long-lived process alive so the panel can hold a conversation.
// stdout is the protocol channel — one JSON object per line — and stderr is
// human-readable diagnostics, kept strictly separate.

import { ChildProcess, spawn } from "child_process";

export interface DaedalusEvent {
  event: string;
  [key: string]: unknown;
}

export type EventHandler = (event: DaedalusEvent) => void;
export type ExitHandler = (code: number | null) => void;

export class DaedalusClient {
  private child: ChildProcess | undefined;
  /** Partial line carried between stdout chunks. */
  private buffer = "";
  private eventHandlers: EventHandler[] = [];
  private exitHandlers: ExitHandler[] = [];
  private stderrHandlers: ((text: string) => void)[] = [];

  get running(): boolean {
    return this.child !== undefined && !this.child.killed;
  }

  start(
    binary: string,
    args: string[],
    cwd: string,
    env?: NodeJS.ProcessEnv
  ): void {
    this.stop();
    this.buffer = "";

    const child = spawn(binary, args, { cwd, env });
    this.child = child;

    child.stdout?.on("data", (chunk: Buffer) => this.consume(chunk.toString()));

    child.stderr?.on("data", (chunk: Buffer) => {
      const text = chunk.toString();
      for (const handler of this.stderrHandlers) {
        handler(text);
      }
    });

    child.on("error", (err: NodeJS.ErrnoException) => {
      // `missing_binary` is distinct from a generic error so the caller can
      // offer a file picker rather than repeating an unhelpful message.
      this.emit(
        err.code === "ENOENT"
          ? { event: "missing_binary", binary }
          : { event: "error", message: err.message }
      );
      this.emit({ event: "idle" });
    });

    child.on("close", (code) => {
      this.child = undefined;
      for (const handler of this.exitHandlers) {
        handler(code);
      }
    });
  }

  /**
   * Split stdout into whole lines. Chunks arrive at arbitrary boundaries, so a
   * partial line is held back rather than parsed and discarded.
   */
  private consume(text: string): void {
    this.buffer += text;
    let newline = this.buffer.indexOf("\n");

    while (newline !== -1) {
      const line = this.buffer.slice(0, newline).trim();
      this.buffer = this.buffer.slice(newline + 1);
      newline = this.buffer.indexOf("\n");

      if (!line) {
        continue;
      }
      try {
        this.emit(JSON.parse(line) as DaedalusEvent);
      } catch {
        // A non-JSON line means the protocol channel was polluted. Surface it
        // rather than silently dropping it — it is always a bug worth seeing.
        this.emit({ event: "error", message: `unparseable output: ${line}` });
      }
    }
  }

  private emit(event: DaedalusEvent): void {
    for (const handler of this.eventHandlers) {
      handler(event);
    }
  }

  send(command: Record<string, unknown>): boolean {
    if (!this.child?.stdin?.writable) {
      return false;
    }
    this.child.stdin.write(`${JSON.stringify(command)}\n`);
    return true;
  }

  onEvent(handler: EventHandler): void {
    this.eventHandlers.push(handler);
  }

  onExit(handler: ExitHandler): void {
    this.exitHandlers.push(handler);
  }

  onStderr(handler: (text: string) => void): void {
    this.stderrHandlers.push(handler);
  }

  stop(): void {
    if (!this.child) {
      return;
    }
    // Ask politely first so the server can finish flushing.
    this.send({ cmd: "shutdown" });
    this.child.stdin?.end();
    this.child.kill();
    this.child = undefined;
  }
}
