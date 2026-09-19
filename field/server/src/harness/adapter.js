// The formal Harness Adapter contract.
//
// A harness adapter owns one real external agent process and translates that process's
// native output into Field's canonical event vocabulary (see contracts/field-event-v1.schema.json).
// Both existing adapters — HarnessSession (the direct `claude` CLI, stream-json duplex) and
// KnossosSession (`knossos serve`, NDJSON) — extend this base so the registry, store projection,
// and campaign director can treat any harness uniformly.
//
// Two different "kind" axes exist in this codebase and MUST NOT be conflated:
//   * endpoint/engine kind  — anthropic | openai-compatible | ollama | cameo (endpoints-store.js).
//     Describes the inference backend and its credentials.
//   * harness kind          — the values in HARNESS_KINDS below. Describes HOW Field launches and
//     speaks to the agent process. This is the axis a manifest keys on.
import { EventEmitter, on } from 'node:events';

// The canonical Field event kinds every adapter emits. This is the single source of truth
// the schema-conformance and adapter-conformance tests check against, and it mirrors the
// `kind` enum in contracts/field-event-v1.schema.json.
export const CANONICAL_EVENTS = Object.freeze([
  'session.state',
  'session.message',
  'session.thinking',
  'session.tool_use',
  'session.tool_result',
  'session.usage',
  'session.progress',
  'session.turn_complete',
  'session.verification',
  'session.delegated',
  'session.ended',
  'endpoint.routed',
  'browser.navigated',
  'harness.permission_requested',
]);

// The harness-kind axis. `cli-stream-json` and `ndjson-serve` are the two shipping transports;
// the rest are declared points on the axis reserved for later adapters (do not implement here).
export const HARNESS_KINDS = Object.freeze([
  'cli-stream-json',
  'ndjson-serve',
  'acp',
  'http-api',
  'mcp',
]);

/**
 * The formal adapter interface. Concrete adapters extend this and implement the lifecycle.
 *
 * Contract methods (subclasses MUST implement start/send/cancel; capabilities() is expected):
 *   start(orders, opts?)   -> starts the real process and begins streaming canonical events.
 *   send(input)            -> pushes a user turn / order into the running process.
 *   cancel()               -> terminates the process, emitting `session.ended`.
 *   capabilities()         -> a declarative snapshot (see below) describing what this adapter supports.
 *   stream(opts?)          -> async iterator of { kind, data } canonical events (default provided here).
 *
 * Lifecycle methods the registry also relies on (already implemented by both adapters):
 *   pause() / resume(orders?)  -> stop/restart the process while keeping the session id.
 *   decidePermission(id, decision) -> answer a `harness.permission_requested` (adapters that gate).
 *   beforeStart                -> optional hook the registry assigns for admission/budget checks.
 *
 * Canonical capabilities() shape:
 *   {
 *     kind: <one of HARNESS_KINDS>,
 *     duplex: boolean,        // can accept further turns mid-session
 *     resumable: boolean,     // supports pause()/resume()
 *     permissions: 'mcp-bridge' | 'inline-handshake' | 'none',
 *     delegation: boolean,    // can emit session.delegated
 *     browser: boolean,       // can emit browser.navigated
 *     verification: boolean,  // can emit session.verification
 *     dryRun: boolean,        // supports a read-only / dry-run launch mode
 *   }
 */
export class HarnessAdapter extends EventEmitter {
  /** Canonical emission point: every Field event carries the owning sessionId. */
  emitEvent(kind, data) {
    this.emit('event', kind, { sessionId: this.id, ...data });
  }

  start() {
    throw new Error('HarnessAdapter.start() must be implemented by the adapter');
  }

  send() {
    throw new Error('HarnessAdapter.send() must be implemented by the adapter');
  }

  cancel() {
    throw new Error('HarnessAdapter.cancel() must be implemented by the adapter');
  }

  capabilities() {
    return { kind: null };
  }

  /**
   * Consume this adapter's canonical events as an async iterator. Additive over the
   * existing `'event'` emitter interface — it adds a listener and never changes emission.
   * Pass an AbortSignal to stop iterating.
   */
  async *stream({ signal } = {}) {
    for await (const [kind, data] of on(this, 'event', signal ? { signal } : undefined)) {
      yield { kind, data };
    }
  }
}

/**
 * Validate and normalize a declarative adapter manifest. The manifest is the seam the
 * registry keys on to instantiate an adapter for a given endpoint kind.
 *
 * Manifest shape:
 *   {
 *     id: string,                 // stable adapter id, e.g. 'claude-code'
 *     name: string,               // human label
 *     kind: <one of HARNESS_KINDS>,
 *     endpointKinds: string[],    // endpoint/engine kinds that route to this adapter
 *     launch?: { bin, argv },     // descriptive launch spec (cli-stream-json / ndjson-serve)
 *     endpoint?: object,          // descriptive endpoint spec (http-api / acp / mcp)
 *     auth?: { via, keys },       // how credentials reach the child
 *     capabilities: object,       // matches capabilities() above
 *     Adapter: Function,          // the adapter class/constructor
 *     default?: boolean,          // chosen when no endpointKinds match
 *   }
 */
export function defineAdapterManifest(manifest) {
  const m = manifest ?? {};
  if (!m.id || typeof m.id !== 'string') throw new Error('adapter manifest requires a string id');
  if (!HARNESS_KINDS.includes(m.kind)) {
    throw new Error(`adapter manifest ${m.id} has unknown harness kind: ${m.kind}`);
  }
  if (!Array.isArray(m.endpointKinds)) throw new Error(`adapter manifest ${m.id} requires an endpointKinds array`);
  if (typeof m.Adapter !== 'function') throw new Error(`adapter manifest ${m.id} requires an Adapter constructor`);
  return {
    id: m.id,
    name: m.name ?? m.id,
    kind: m.kind,
    endpointKinds: [...m.endpointKinds],
    launch: m.launch ?? null,
    endpoint: m.endpoint ?? null,
    auth: m.auth ?? null,
    capabilities: m.capabilities ?? {},
    Adapter: m.Adapter,
    default: m.default === true,
  };
}
