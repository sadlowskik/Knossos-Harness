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
];

/**
 * Resolve the adapter manifest for an endpoint/engine kind. A kind that no manifest claims
 * falls back to the default manifest — identical to the old ternary, where every non
 * `openai-compatible` endpoint used the direct Claude adapter.
 */
export function resolveAdapterManifest(endpointKind) {
  return ADAPTER_MANIFESTS.find((m) => m.endpointKinds.includes(endpointKind))
    ?? ADAPTER_MANIFESTS.find((m) => m.default);
}
