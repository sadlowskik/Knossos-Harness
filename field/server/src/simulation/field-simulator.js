import { randomUUID } from 'node:crypto';

const TIMELINE_SCALE = 3;
const OPERATIONS_DURATION_MS = 24_400 * TIMELINE_SCALE;

const UNITS = [
  ['praetor', 'Praetor', 'architect', 'high'],
  ['forge-a', 'Forge I', 'builder', 'medium'],
  ['forge-b', 'Forge II', 'builder', 'medium'],
  ['forge-c', 'Forge III', 'builder', 'low'],
  ['pathfinder', 'Pathfinder', 'scout', 'low'],
  ['courier', 'Courier', 'scout', 'low'],
  ['sentinel-a', 'Sentinel I', 'verifier', 'high'],
  ['sentinel-b', 'Sentinel II', 'verifier', 'medium'],
  ['challenger', 'Adversary', 'challenger', 'high'],
  ['archivist', 'Archivist', 'archivist', 'low'],
  ['relay-a', 'Relay I', 'builder', 'low'],
  ['relay-b', 'Relay II', 'builder', 'low'],
].map(([key, name, role, thinking]) => ({ key, name, role, thinking }));

const WORK = {
  praetor: ['docs', 'docs/architecture.md', 'Read'],
  'forge-a': ['src/core', 'src/core/service.ts', 'Read'],
  'forge-b': ['src/ui', 'src/ui/App.tsx', 'Edit'],
  'forge-c': ['src/agents', 'src/agents/orchestrator.ts', 'Edit'],
  pathfinder: ['docs', 'docs/architecture.md', 'Grep'],
  courier: ['site', 'site/index.html', 'Read'],
  'sentinel-a': ['tests', 'tests/integration.test.ts', 'Bash'],
  'sentinel-b': ['tests/e2e', 'tests/e2e/release.test.ts', 'Bash'],
  challenger: ['src/core', 'src/core/api.ts', 'Read'],
  archivist: ['docs', 'docs/operations.md', 'Write'],
  'relay-a': ['infra', 'infra/compose.yaml', 'Read'],
  'relay-b': ['deploy', 'deploy/service.yaml', 'Read'],
};

const item = (at, kind, data, subject) => ({
  at, kind, data: { ...data, simulated: true }, subject,
});

/** Pure timeline so safety can be tested without clocks or provider access. */
export function operationsTimeline(runId, { workspaceId = 'cameo', workspaceName = 'the project' } = {}) {
  const ids = Object.fromEntries(UNITS.map((u) => [u.key, `sim-${runId}-${u.key}`]));
  const events = [];

  UNITS.forEach((u, i) => {
    const sessionId = ids[u.key];
    events.push(item(i * 110, 'session.spawned', {
      sessionId, agentId: `simulation-${u.key}`, name: u.name, role: u.role,
      model: `rehearsal-${u.role}`, endpointId: 'rehearsal', thinking: u.thinking,
      workspaceId, simulationRunId: runId,
      initialOrders: `Serve as ${u.name}, the ${u.role} assigned to ${workspaceName}. Survey your lane, coordinate with the cohort, verify evidence, and report a durable result.`,
      systemPrompt: `You are a synthetic ${u.role} in Cameo's safe operations rehearsal. Make no provider calls or filesystem writes. Report simulated observations clearly to the other agents.`,
    }, sessionId));
    events.push(item(900 + i * 45, 'session.state', {
      sessionId, state: 'ready', detail: 'Awaiting formation orders', simulationRunId: runId,
    }, sessionId));
  });

  events.push(item(1_700, 'session.message', {
    sessionId: ids.praetor, role: 'assistant',
    text: 'Formation online. Splitting reconnaissance, build, and verification lanes.',
    simulationRunId: runId,
  }, ids.praetor));

  UNITS.forEach((u, i) => {
    const [dir, path, tool] = WORK[u.key];
    const sessionId = ids[u.key];
    const assignmentId = `sim-${runId}-assignment-${u.key}`;
    events.push(item(2_100 + i * 95, 'assignment.created', {
      assignmentId, sessionIds: [sessionId], targetType: 'folder', targetId: dir,
      targetLabel: dir, workspaceId, orders: `Advance ${workspaceName} through ${dir}`, simulationRunId: runId,
    }, assignmentId));
    events.push(item(3_500 + i * 150, 'session.state', {
      sessionId, state: 'moving', detail: `Moving to ${dir}`, simulationRunId: runId,
    }, sessionId));
    events.push(item(5_500 + i * 170, 'session.tool_use', {
      sessionId, name: tool, workspaceId, dir, path, summary: `Surveying ${path}`,
      simulationRunId: runId,
    }, sessionId));
    events.push(item(6_300 + i * 145, 'session.progress', {
      sessionId, done: 1, total: 5,
      steps: ['survey', 'plan', 'execute', 'verify', 'report'], simulationRunId: runId,
    }, sessionId));
  });

  const visits = [
    [5_200, 'pathfinder', 'https://docs.github.com/actions', 'docs.github.com'],
    [5_750, 'courier', 'https://developer.mozilla.org/', 'developer.mozilla.org'],
    [6_250, 'challenger', 'https://docs.docker.com/', 'docs.docker.com'],
  ];
  visits.forEach(([at, key, url, domain]) => events.push(item(at, 'browser.navigated', {
    sessionId: ids[key], url, domain, simulationRunId: runId,
  }, ids[key])));

  const links = [
    ['pathfinder', 'praetor'], ['courier', 'praetor'], ['praetor', 'forge-a'],
    ['praetor', 'forge-b'], ['praetor', 'forge-c'], ['forge-a', 'sentinel-a'],
    ['forge-c', 'sentinel-b'], ['challenger', 'sentinel-a'], ['challenger', 'sentinel-b'],
    ['relay-a', 'forge-a'], ['relay-b', 'forge-a'], ['archivist', 'praetor'],
  ];
  links.forEach(([from, to], i) => events.push(item(6_500 + i * 230, 'agent.communication', {
    fromSessionId: ids[from], toSessionId: ids[to], channel: 'operations',
    summary: 'Shared status and evidence', simulationRunId: runId,
  }, ids[from])));

  UNITS.forEach((u, i) => events.push(item(8_500 + i * 185, 'session.progress', {
    sessionId: ids[u.key], done: 2, total: 5,
    steps: ['survey', 'plan', 'execute', 'verify', 'report'], simulationRunId: runId,
  }, ids[u.key])));

  // A visible incident keeps the scenario from being a perfect canned march.
  events.push(item(10_900, 'session.state', {
    sessionId: ids['relay-b'], state: 'blocked', detail: 'Deployment manifest drift detected', simulationRunId: runId,
  }, ids['relay-b']));
  events.push(item(11_250, 'session.state', {
    sessionId: ids['forge-a'], state: 'blocked', detail: 'Waiting on deployment topology', simulationRunId: runId,
  }, ids['forge-a']));
  events.push(item(11_700, 'agent.communication', {
    fromSessionId: ids['relay-b'], toSessionId: ids.praetor, channel: 'incident',
    summary: 'Escalated deployment drift', simulationRunId: runId,
  }, ids['relay-b']));
  events.push(item(12_150, 'agent.communication', {
    fromSessionId: ids.praetor, toSessionId: ids['sentinel-b'], channel: 'incident',
    summary: 'Requested independent manifest comparison', simulationRunId: runId,
  }, ids.praetor));
  events.push(item(12_900, 'session.tool_use', {
    sessionId: ids['sentinel-b'], name: 'Diff', workspaceId, dir: 'deploy',
    path: 'deploy/service.yaml', summary: 'Comparing desired and observed topology', simulationRunId: runId,
  }, ids['sentinel-b']));
  events.push(item(13_800, 'session.message', {
    sessionId: ids['sentinel-b'], role: 'assistant',
    text: 'Drift isolated to one stale service selector. Safe correction prepared.', simulationRunId: runId,
  }, ids['sentinel-b']));
  events.push(item(14_350, 'session.state', {
    sessionId: ids['relay-b'], state: 'working', detail: 'Applying verified correction', simulationRunId: runId,
  }, ids['relay-b']));
  events.push(item(14_650, 'session.state', {
    sessionId: ids['forge-a'], state: 'working', detail: 'Dependency restored', simulationRunId: runId,
  }, ids['forge-a']));

  const changed = [
    ['forge-a', 'src/core', 'src/core/service.ts'],
    ['forge-b', 'src/ui', 'src/ui/App.tsx'],
    ['forge-c', 'src/agents', 'src/agents/orchestrator.ts'],
    ['relay-b', 'deploy', 'deploy/service.yaml'],
    ['archivist', 'docs', 'docs/operations.md'],
  ];
  changed.forEach(([key, dir, path], i) => events.push(item(15_000 + i * 430, 'fs.changed', {
    sessionId: ids[key], workspaceId, dir, path, change: i === 4 ? 'add' : 'change', simulationRunId: runId,
  }, ids[key])));

  UNITS.forEach((u, i) => events.push(item(17_400 + i * 165, 'session.progress', {
    sessionId: ids[u.key], done: u.role === 'verifier' ? 3 : 4, total: 5,
    steps: ['survey', 'plan', 'execute', 'verify', 'report'], simulationRunId: runId,
  }, ids[u.key])));
  events.push(item(19_650, 'agent.communication', {
    fromSessionId: ids.challenger, toSessionId: ids['sentinel-a'], channel: 'red-blue',
    summary: 'Challenge case delivered for independent replay', simulationRunId: runId,
  }, ids.challenger));
  events.push(item(20_300, 'agent.communication', {
    fromSessionId: ids['sentinel-a'], toSessionId: ids.praetor, channel: 'verification',
    summary: 'Red-team challenge reproduced and closed', simulationRunId: runId,
  }, ids['sentinel-a']));

  UNITS.forEach((u, i) => {
    const sessionId = ids[u.key];
    events.push(item(21_000 + i * 120, 'session.progress', {
      sessionId, done: 5, total: 5,
      steps: ['survey', 'plan', 'execute', 'verify', 'report'], simulationRunId: runId,
    }, sessionId));
    events.push(item(22_800 + i * 120, 'session.ended', {
      sessionId, reason: 'completed', result: 'Synthetic objective completed', simulationRunId: runId,
    }, sessionId));
  });
  events.push(item(24_400, 'simulation.completed', {
    simulationRunId: runId, scenario: 'operations-cycle', sessionIds: Object.values(ids),
  }, runId));

  return events.sort((a, b) => a.at - b.at).map((event) => ({ ...event, at: event.at * TIMELINE_SCALE }));
}

export class FieldSimulator {
  constructor({ emit, workspaces = [] }) {
    this.emit = emit;
    this.workspaces = workspaces.filter((workspace) => workspace.mounted !== false);
    this.timers = new Set();
    this.active = null;
  }

  scenarios() {
    return [{
      id: 'operations-cycle', name: 'Long-horizon operations cycle', durationMs: OPERATIONS_DURATION_MS,
      description: 'Twelve agents muster, fan out, coordinate, recover from drift, verify, and report.',
    }];
  }

  status() {
    return this.active ? { ...this.active, sessionIds: [...this.active.sessionIds] } : null;
  }

  run(scenario = 'operations-cycle', { speed = 1, workspaceId = null } = {}) {
    if (scenario !== 'operations-cycle') throw new Error(`unknown simulation scenario: ${scenario}`);
    this.stop('replaced');
    const safeSpeed = Math.max(0.25, Math.min(4, Number(speed) || 1));
    const workspace = this.workspaces.find((item) => item.id === workspaceId) ?? this.workspaces[0] ?? { id: workspaceId || 'cameo', name: 'the project' };
    const runId = randomUUID().slice(0, 8);
    const timeline = operationsTimeline(runId, { workspaceId: workspace.id, workspaceName: workspace.name });
    const sessionIds = [...new Set(timeline.map((x) => x.data.sessionId).filter(Boolean))];
    this.active = {
      runId, scenario, speed: safeSpeed, startedAt: Date.now(), workspaceId: workspace.id,
      durationMs: Math.ceil(timeline.at(-1).at / safeSpeed), sessionIds,
    };
    this.emit('simulation.started', {
      simulationRunId: runId, scenario, speed: safeSpeed, sessionIds, simulated: true,
    }, { subject: runId, simulated: true });

    for (const event of timeline) {
      const timer = setTimeout(() => {
        this.timers.delete(timer);
        this.emit(event.kind, event.data, { subject: event.subject, simulated: true });
        if (event.kind === 'simulation.completed' && this.active?.runId === runId) this.active = null;
      }, Math.ceil(event.at / safeSpeed));
      this.timers.add(timer);
    }
    return this.status();
  }

  stop(reason = 'operator') {
    for (const timer of this.timers) clearTimeout(timer);
    this.timers.clear();
    if (!this.active) return { stopped: false };
    const previous = this.active;
    this.active = null;
    for (const sessionId of previous.sessionIds) {
      this.emit('session.ended', {
        sessionId, reason: 'cancelled', simulated: true, simulationRunId: previous.runId,
      }, { subject: sessionId, simulated: true });
    }
    this.emit('simulation.stopped', {
      simulationRunId: previous.runId, scenario: previous.scenario, reason, simulated: true,
    }, { subject: previous.runId, simulated: true });
    return { stopped: true, runId: previous.runId };
  }
}

export function assertSafeTimeline(timeline) {
  for (const event of timeline) {
    if (!event.data.simulated) throw new Error(`unmarked simulation event: ${event.kind}`);
    if (event.kind === 'session.usage' && Number(event.data.costUsd) > 0) throw new Error('simulation may not emit cost');
    if (event.kind === 'terminal.run' || event.kind === 'permission.requested') {
      throw new Error(`simulation may not emit executable event ${event.kind}`);
    }
  }
  return true;
}
