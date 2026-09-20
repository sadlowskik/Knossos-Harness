const WORLD_TYPES = new Set(['folder', 'website', 'service', 'endpoint', 'workspace']);
const ACTIVE_STATES = new Set(['active_worksite', 'established', 'promoted']);

export class GraphProjection {
  constructor() { this.reset(); }

  reset() {
    this.nodes = new Map();
    this.edges = new Map();
  }

  apply(evt) {
    const d = evt.data ?? {};
    const fn = MAPPERS[evt.kind];
    if (fn) fn.call(this, d, evt);
  }

  observeNode(type, id, attrs, evt, weight = 1) {
    if (id == null || id === '') return null;
    const key = `${type}:${id}`;
    let node = this.nodes.get(key);
    if (!node) {
      node = {
        key, type, id: String(id), label: attrs?.label ?? String(id),
        firstSeen: evt.ts, lastSeen: evt.ts, observations: 0,
        firstSeq: evt.seq, lastSeq: evt.seq, state: WORLD_TYPES.has(type) ? 'contact' : 'active',
        attrs: {},
      };
      this.nodes.set(key, node);
    }
    node.lastSeen = Math.max(node.lastSeen, evt.ts);
    node.lastSeq = Math.max(node.lastSeq, evt.seq);
    node.observations += weight;
    node.attrs = { ...node.attrs, ...withoutUndefined(attrs) };
    if (attrs?.label) node.label = attrs.label;
    if (WORLD_TYPES.has(type)) node.state = lifecycle(node, evt.ts);
    return node;
  }

  observeEdge(type, from, to, attrs, evt, weight = 1) {
    if (!from || !to) return null;
    const key = `${type}:${from}->${to}`;
    let edge = this.edges.get(key);
    if (!edge) {
      edge = {
        key, type, from, to, firstSeen: evt.ts, lastSeen: evt.ts,
        observations: 0, firstSeq: evt.seq, lastSeq: evt.seq, active: true, attrs: {},
      };
      this.edges.set(key, edge);
    }
    edge.lastSeen = Math.max(edge.lastSeen, evt.ts);
    edge.lastSeq = Math.max(edge.lastSeq, evt.seq);
    edge.observations += weight;
    edge.attrs = { ...edge.attrs, ...withoutUndefined(attrs) };
    return edge;
  }

  promote(type, id, evt, attrs = {}) {
    const node = this.observeNode(type, id, attrs, evt);
    if (node) node.state = 'promoted';
    return node;
  }

  snapshot(now = Date.now(), limits = {}) {
    const allNodes = [...this.nodes.values()].map((node) => {
      const state = WORLD_TYPES.has(node.type) && node.state !== 'promoted'
        ? lifecycle(node, now)
        : node.state;
      return { ...node, state };
    });
    const maxNodes = limits.maxNodes ?? Infinity;
    const nodes = allNodes.length <= maxNodes ? allNodes : allNodes
      .sort((a, b) => nodePriority(b, now) - nodePriority(a, now) || b.lastSeen - a.lastSeen)
      .slice(0, maxNodes);
    const visible = new Set(nodes.filter((n) => !WORLD_TYPES.has(n.type) || ACTIVE_STATES.has(n.state)).map((n) => n.key));
    let edges = [...this.edges.values()].filter((edge) => visible.has(edge.from) && visible.has(edge.to)).map((edge) => ({
      ...edge,
      active: visible.has(edge.from) && visible.has(edge.to) && now - edge.lastSeen < 30 * 60_000,
    }));
    const maxEdges = limits.maxEdges ?? Infinity;
    if (edges.length > maxEdges) edges = edges
      .sort((a, b) => Number(b.active) - Number(a.active) || b.lastSeen - a.lastSeen || b.observations - a.observations)
      .slice(0, maxEdges);
    return {
      nodes, edges, visibleNodeKeys: [...visible],
      totals: { nodes: this.nodes.size, edges: this.edges.size },
      truncated: nodes.length < this.nodes.size || edges.length < this.edges.size,
    };
  }
}

function nodePriority(node, now) {
  if (node.state === 'promoted') return 1_000_000;
  const active = !WORLD_TYPES.has(node.type) || ACTIVE_STATES.has(node.state);
  const recent = Math.max(0, 100_000 - Math.floor((now - node.lastSeen) / 1000));
  return (active ? 500_000 : 0) + recent + Math.min(10_000, node.observations);
}

function lifecycle(node, now) {
  if (node.state === 'promoted') return 'promoted';
  const age = Math.max(0, now - node.lastSeen);
  const span = Math.max(0, node.lastSeen - node.firstSeen);
  if (age > 30 * 60_000) return 'dormant';
  if (node.observations >= 8 || (node.observations >= 4 && span >= 5 * 60_000)) return 'established';
  if (node.observations >= 3 || span >= 60_000) return 'active_worksite';
  return 'contact';
}

function nodeKey(type, id) { return `${type}:${id}`; }

const MAPPERS = {
  'campaign.created'(d, evt) {
    this.observeNode('campaign', d.campaignId, { label: d.name, phase: 'draft' }, evt);
    if (d.target?.workspaceId) {
      this.observeNode('workspace', d.target.workspaceId, { label: d.target.workspaceId }, evt);
      this.observeEdge('working_on', nodeKey('campaign', d.campaignId), nodeKey('workspace', d.target.workspaceId), {}, evt);
    }
  },
  'campaign.phase_changed'(d, evt) {
    this.observeNode('campaign', d.campaignId, { phase: d.to }, evt);
  },
  'objective.created'(d, evt) {
    this.observeNode('objective', d.objectiveId, { label: d.statement, campaignId: d.campaignId, status: 'queued' }, evt);
    this.observeEdge('depends_on', nodeKey('objective', d.objectiveId), nodeKey('campaign', d.campaignId), {}, evt);
    if (d.target?.workspaceId) {
      this.observeNode('workspace', d.target.workspaceId, { label: d.target.workspaceId }, evt);
      this.observeEdge('working_on', nodeKey('objective', d.objectiveId), nodeKey('workspace', d.target.workspaceId), {}, evt);
    }
  },
  'objective.assigned'(d, evt) {
    this.observeNode('objective', d.objectiveId, { status: 'active', team: d.team }, evt);
    for (const id of d.sessionIds ?? []) {
      this.observeEdge('assigned_to', nodeKey('agent', id), nodeKey('objective', d.objectiveId), { team: d.team }, evt);
    }
  },
  'objective.satisfied'(d, evt) { this.observeNode('objective', d.objectiveId, { status: 'satisfied' }, evt); },
  'objective.blocked'(d, evt) { this.observeNode('objective', d.objectiveId, { status: 'blocked' }, evt); },
  'team.member_assigned'(d, evt) {
    const teamId = `${d.campaignId}:${d.team}`;
    this.observeNode('team', teamId, { label: d.team, campaignId: d.campaignId, kind: d.team }, evt);
    this.observeNode('agent', d.sessionId, { label: d.agentId ?? d.sessionId, role: d.role, team: d.team }, evt);
    this.observeEdge('member_of', nodeKey('agent', d.sessionId), nodeKey('team', teamId), { role: d.role }, evt);
    this.observeEdge('member_of', nodeKey('team', teamId), nodeKey('campaign', d.campaignId), {}, evt);
  },
  'session.spawned'(d, evt) {
    this.observeNode('agent', d.sessionId, {
      label: d.name ?? d.agentId ?? d.sessionId, role: d.role, state: 'spawning',
      campaignId: d.campaignId, team: d.team,
    }, evt);
    if (d.workspaceId) {
      this.observeNode('workspace', d.workspaceId, { label: d.workspaceId }, evt);
      this.observeEdge('located_at', nodeKey('agent', d.sessionId), nodeKey('workspace', d.workspaceId), {}, evt);
    }
    if (d.endpointId) {
      this.observeNode('endpoint', d.endpointId, { label: d.endpointId, model: d.model }, evt);
      this.observeEdge('routed_through', nodeKey('agent', d.sessionId), nodeKey('endpoint', d.endpointId), {}, evt);
    }
  },
  'session.state'(d, evt) { this.observeNode('agent', d.sessionId, { state: d.state }, evt); },
  'session.tool_use'(d, evt) {
    this.observeNode('agent', d.sessionId, {}, evt);
    this.observeNode('tool', d.name, { label: d.name }, evt);
    this.observeEdge('used', nodeKey('agent', d.sessionId), nodeKey('tool', d.name), {}, evt);
    if (d.workspaceId && d.dir != null) {
      const id = `${d.workspaceId}:${d.dir}`;
      this.observeNode('folder', id, { label: d.dir || '/', workspaceId: d.workspaceId, path: d.dir }, evt);
      this.observeEdge('working_on', nodeKey('agent', d.sessionId), nodeKey('folder', id), {}, evt);
      this.observeEdge('located_at', nodeKey('folder', id), nodeKey('workspace', d.workspaceId), {}, evt);
    }
    if (d.path) {
      const id = `${d.workspaceId ?? 'external'}:${d.path}`;
      this.observeNode('file', id, { label: d.path, workspaceId: d.workspaceId, path: d.path }, evt);
      this.observeEdge('working_on', nodeKey('agent', d.sessionId), nodeKey('file', id), {}, evt);
    }
  },
  'session.delegated'(d, evt) {
    this.observeNode('agent', d.parentSessionId, {}, evt);
    this.observeNode('agent', d.childSessionId, { label: d.description ?? d.childSessionId, delegated: true }, evt);
    this.observeEdge('communicates_with', nodeKey('agent', d.parentSessionId), nodeKey('agent', d.childSessionId), { delegation: true }, evt);
  },
  'agent.communication'(d, evt) {
    this.observeNode('agent', d.fromSessionId, {}, evt);
    this.observeNode('agent', d.toSessionId, {}, evt);
    this.observeEdge('communicates_with', nodeKey('agent', d.fromSessionId), nodeKey('agent', d.toSessionId), { channel: d.channel }, evt);
  },
  'browser.navigated'(d, evt) {
    this.observeNode('website', d.domain, { label: d.domain, url: d.url }, evt);
    this.observeEdge('visited', nodeKey('agent', d.sessionId), nodeKey('website', d.domain), { url: d.url }, evt);
  },
  'endpoint.health'(d, evt) {
    this.observeNode('endpoint', d.endpointId, { label: d.endpointId, status: d.status, latencyMs: d.latencyMs }, evt);
  },
  'endpoint.routed'(d, evt) {
    this.observeNode('endpoint', d.endpointId, { label: d.endpointId, model: d.model }, evt);
    this.observeEdge('routed_through', nodeKey('agent', d.sessionId), nodeKey('endpoint', d.endpointId), { reason: d.reason }, evt);
  },
  'finding.reported'(d, evt) {
    this.observeNode('agent', d.authorSessionId, { team: 'red' }, evt);
    this.observeNode('finding', d.findingId, { label: d.claim, severity: d.severity, status: 'open' }, evt);
    this.observeEdge('found', nodeKey('agent', d.authorSessionId), nodeKey('finding', d.findingId), {}, evt);
    if (d.objectiveId) this.observeEdge('challenged_by', nodeKey('objective', d.objectiveId), nodeKey('finding', d.findingId), {}, evt);
  },
  'mitigation.proposed'(d, evt) {
    this.observeNode('mitigation', d.mitigationId, { label: d.claim, status: 'proposed' }, evt);
    for (const id of d.findingIds ?? []) {
      this.observeEdge('mitigated_by', nodeKey('finding', id), nodeKey('mitigation', d.mitigationId), {}, evt);
    }
  },
  'retest.completed'(d, evt) {
    this.observeNode('agent', d.sessionId, { team: 'red' }, evt);
    this.observeEdge('retested_by', nodeKey('finding', d.findingId), nodeKey('agent', d.sessionId), { result: d.result }, evt);
  },
  'referee.verdict'(d, evt) {
    this.observeNode('agent', d.sessionId, { team: 'referee' }, evt);
    this.observeNode('verdict', d.verdictId, { label: d.verdict, verdict: d.verdict }, evt);
    this.observeEdge('verified_by', nodeKey('campaign', d.campaignId), nodeKey('verdict', d.verdictId), {}, evt);
    this.observeEdge('produced', nodeKey('agent', d.sessionId), nodeKey('verdict', d.verdictId), {}, evt);
  },
  'campaign.checkpoint_created'(d, evt) {
    this.observeNode('checkpoint', d.checkpointId, { label: d.name, campaignId: d.campaignId }, evt);
    this.observeEdge('produced', nodeKey('campaign', d.campaignId), nodeKey('checkpoint', d.checkpointId), {}, evt);
  },
  'campaign.promoted'(d, evt) {
    for (const capability of d.capabilities ?? []) {
      this.promote('capability', capability.id, evt, { label: capability.name ?? capability.id });
      this.observeEdge('promoted_into', nodeKey('campaign', d.campaignId), nodeKey('capability', capability.id), { checkpointId: d.checkpointId }, evt);
    }
  },
};

function withoutUndefined(obj = {}) {
  return Object.fromEntries(Object.entries(obj).filter(([, value]) => value !== undefined));
}
