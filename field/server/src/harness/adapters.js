// The declarative adapter manifest registry. The Registry keys on this to select and
// instantiate a harness adapter for a given endpoint kind — replacing the previous inline
// `ep.kind === 'openai-compatible' ? KnossosSession : HarnessSession` ternary with the same
// mapping expressed as data.
//
// This module is the only place that imports both concrete adapters AND the base contract,
// so the base (adapter.js) stays free of concrete imports and there is no import cycle.
import { defineAdapterManifest } from './adapter.js';
import { HarnessSession } from './session.js';
import { KnossosSession } from './knossos-session.js';
import { AcpSession } from './acp-session.js';

export const ADAPTER_MANIFESTS = [
  defineAdapterManifest({
    id: 'claude-code',
    name: 'Claude Code (stream-json duplex)',
    kind: 'cli-stream-json',
    // Any endpoint kind that is not openai-compatible falls through to this default,
    // preserving the historical selection behavior exactly.
    endpointKinds: ['anthropic'],
    default: true,
    launch: { bin: 'FIELD_CLAUDE_BIN|claude', argv: '-p --output-format stream-json --input-format stream-json --verbose' },
    auth: { via: 'child-env', keys: ['ANTHROPIC_API_KEY', 'ANTHROPIC_BASE_URL'] },
    capabilities: {
      kind: 'cli-stream-json',
      duplex: true,
      resumable: true,
      permissions: 'mcp-bridge',
      delegation: true,
      browser: true,
      verification: false,
      dryRun: false,
    },
    Adapter: HarnessSession,
  }),
  defineAdapterManifest({
    id: 'knossos',
    name: 'Knossos serve (NDJSON)',
    kind: 'ndjson-serve',
    endpointKinds: ['openai-compatible'],
    launch: { bin: 'FIELD_KNOSSOS_BIN|knossos', argv: 'serve --workspace <cwd> --engine <engine>' },
    auth: { via: 'child-env', keys: ['CAMEO_SERVE_KEY', 'CAMEO_BASE_URL', 'CAMEO_MODEL'] },
    capabilities: {
      kind: 'ndjson-serve',
      duplex: true,
      resumable: true,
      permissions: 'inline-handshake',
      delegation: false,
      browser: false,
      verification: true,
      dryRun: true,
    },
    Adapter: KnossosSession,
  }),
  defineAdapterManifest({
    id: 'acp',
    name: 'Agent Client Protocol (JSON-RPC over stdio)',
    kind: 'acp',
    // Intentionally empty: ACP is a HARNESS-kind axis, not an endpoint/engine kind. No endpoint
    // kind auto-routes to it, so endpoint-kind resolution stays byte-identical to before. The
    // adapter is reached only by an explicit harness selector (see resolveAdapterManifest below).
    endpointKinds: [],
    launch: { bin: 'FIELD_ACP_BIN', argv: 'FIELD_ACP_ARGS' },
    auth: { via: 'child-env', keys: [] },
    capabilities: {
      kind: 'acp',
      duplex: true,
      resumable: true,
      permissions: 'inline-handshake',
      delegation: false,
      browser: false,
      verification: false,
      dryRun: true,
    },
    Adapter: AcpSession,
  }),
];

/**
 * Resolve the adapter manifest for an endpoint/engine kind. A kind that no manifest claims
 * falls back to the default manifest — identical to the old ternary, where every non
 * `openai-compatible` endpoint used the direct Claude adapter.
 *
 * `explicitHarness` is an ADDITIVE second axis: when an endpoint/orders names a harness by id or
 * kind (e.g. 'acp'), that manifest is selected regardless of endpoint kind. This is the seam that
 * lets an operator say "drive this endpoint via the ACP harness." It is opt-in only — when
 * absent (every existing call site and config), resolution is byte-identical to before, so the
 * default selection behaviour is unchanged.
 */
export function resolveAdapterManifest(endpointKind, explicitHarness) {
  if (explicitHarness) {
    const chosen = ADAPTER_MANIFESTS.find((m) => m.id === explicitHarness || m.kind === explicitHarness);
    if (chosen) return chosen;
  }
  return ADAPTER_MANIFESTS.find((m) => m.endpointKinds.includes(endpointKind))
    ?? ADAPTER_MANIFESTS.find((m) => m.default);
}
