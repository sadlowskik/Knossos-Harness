// Real health for real inference connections.
//
// What each probe actually measures, stated plainly so the Field never overclaims:
//   probe: http  — a GET against the endpoint's own /models route. Proves the server
//                  is listening and answering. Reported latency is that round trip.
//   probe: cli   — TCP+TLS reachability of the provider API host. It proves the route
//                  out is open; it does not prove the credentials are valid. Credential
//                  failures surface as real session errors and mark the endpoint down.
const TIMEOUT_MS = 2500;

async function probeHttp(url) {
  const started = Date.now();
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(url, { signal: ctrl.signal });
    return {
      status: res.ok || res.status === 401 || res.status === 403 ? 'up' : 'degraded',
      latencyMs: Date.now() - started,
      detail: `HTTP ${res.status}`,
    };
  } catch (e) {
    return {
      status: 'down',
      latencyMs: null,
      detail: e.name === 'AbortError' ? `no response in ${TIMEOUT_MS}ms` : e.message,
    };
  } finally {
    clearTimeout(t);
  }
}

async function probeEndpoint(ep) {
  if (ep.kind === 'openai-compatible' && ep.base_url) {
    return probeHttp(ep.base_url.replace(/\/$/, '') + '/models');
  }
  if (ep.kind === 'anthropic') {
    const r = await probeHttp('https://api.anthropic.com/v1/models');
    return { ...r, detail: r.status === 'up' ? `reachable · ${r.detail}` : r.detail };
  }
  return { status: 'unknown', latencyMs: null, detail: `no probe defined for kind "${ep.kind}"` };
}

// The log is append-only and permanent, so an unchanged probe result is not worth an
// event. Health is recorded when it changes, or on a slow heartbeat so a long-running
// Field can still show that the probe is alive.
const HEARTBEAT_MS = 5 * 60 * 1000;

export function startEndpointProbes(cfg, emit, registry, intervalMs = 20000) {
  const last = new Map();
  let stopped = false;

  async function tick() {
    await Promise.all(cfg.endpoints.map(async (ep) => {
      const r = await probeEndpoint(ep);
      const prev = last.get(ep.id);
      const changed = prev?.status !== r.status;
      const stale = !prev || Date.now() - prev.at > HEARTBEAT_MS;
      last.set(ep.id, { status: r.status, at: changed || stale ? Date.now() : prev.at });

      if (changed || stale) {
        emit('endpoint.health', {
          endpointId: ep.id,
          status: r.status,
          latencyMs: r.latencyMs,
          detail: r.detail,
        }, { subject: ep.id });
      }
      if (changed) registry?.onEndpointHealth(ep.id, r.status);
    }));
    if (!stopped) setTimeout(tick, intervalMs);
  }

  tick();
  return () => { stopped = true; };
}
