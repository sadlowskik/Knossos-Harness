import { useEffect, useMemo, useState } from 'react';
import { api } from '../net/client.js';
import { clearActiveAgent, openSenate, selectCampaign, useField } from '../state/store.js';
import { useModalFocus } from '../ui/useModalFocus.js';
import AgentControls from '../hud/AgentControls.jsx';

// The server's director still keys rosters by lane. The client shows one flat list of
// agents per plan and always files them under this lane until the server is simplified.
const LANE = 'blue';

// Server phases → plain words. A plan is objectives plus the agents working on them.
const PHASE_WORD = {
  draft: 'not started', mobilizing: 'starting', blue_building: 'in progress', red_challenging: 'in review',
  contested: 'in review', blue_mitigating: 'fixing', red_retesting: 'in review', referee_review: 'verifying',
  verified: 'verified', promoted: 'promoted', cancelled: 'cancelled',
};
const OPEN_PHASES = new Set(['mobilizing', 'blue_building', 'red_challenging', 'contested', 'blue_mitigating', 'red_retesting', 'referee_review']);

export default function CampaignMode() {
  const st = useField();
  const agent = st.snap.sessions.find((session) => session.id === st.activeSessionId);
  return agent ? <AgentChamber session={agent} /> : <PlansCommand />;
}

function AgentChamber({ session }) {
  const st = useField();
  const [trace, setTrace] = useState([]);
  const role = st.config?.roles?.find((item) => item.id === session.role);
  const campaign = st.snap.campaigns?.find((item) => item.id === session.campaignId);
  const workspace = st.snap.workspaces.find((item) => item.id === session.workspaceId);
  const pct = session.progress?.total ? Math.round((session.progress.done / session.progress.total) * 100) : 0;
  const collaborators = useMemo(() => {
    const ids = new Set();
    for (const edge of st.snap.graph?.edges ?? []) {
      if (edge.type !== 'communicates_with') continue;
      const from = String(edge.from ?? '').replace(/^agent:/, '');
      const to = String(edge.to ?? '').replace(/^agent:/, '');
      if (from === session.id) ids.add(to);
      if (to === session.id) ids.add(from);
    }
    return [...ids].map((id) => st.snap.sessions.find((item) => item.id === id)).filter(Boolean);
  }, [session.id, st.snap.graph?.edges, st.snap.sessions]);

  useEffect(() => {
    let alive = true;
    api.trace(session.id, 0, 2000).then((result) => { if (alive) setTrace(result.events ?? []); }).catch(() => { if (alive) setTrace([]); });
    return () => { alive = false; };
  }, [session.id, session.messageCount, session.toolCount, session.state, session.progress?.done]);

  const spawn = trace.find((event) => event.kind === 'session.spawned')?.data ?? {};
  const messages = trace.filter((event) => event.kind === 'session.message').slice(-10);
  const activity = trace.filter((event) => ['session.tool_use', 'session.tool_result', 'session.state'].includes(event.kind)).slice(-8).reverse();
  const endpoint = st.snap.endpoints.find((item) => item.id === session.endpointId);
  return <main className="senate-session-shell">
    <header className="senate-session-header"><div><span>AGENT</span><h1>{session.name}</h1><p>{session.role} · {session.state?.replaceAll('_', ' ')}</p></div><button type="button" onClick={clearActiveAgent}>Back to plans</button></header>
    <section className="senate-dais">
      <div className="senate-agent-seal">{String(session.name || 'A').slice(0, 2).toUpperCase()}</div>
      <div><span>CURRENT OBJECTIVE</span><h2>{session.target?.label ?? session.stateDetail ?? 'Awaiting orders'}</h2><p>{workspace?.name ?? session.workspaceId}{campaign ? ` · ${campaign.name}` : ''}</p></div>
      <div className="senate-progress"><b>{pct}%</b><span role="progressbar" aria-label={`${session.name} progress`} aria-valuemin="0" aria-valuemax="100" aria-valuenow={pct}><i style={{ width: `${pct}%` }} /></span><small>{session.progress?.done ?? 0} / {session.progress?.total ?? 0} stages</small></div>
    </section>
    <div className="senate-chamber-grid">
      {trace.length >= 2000 && <div className="label">Showing the first 2,000 events. Open History to load the rest.</div>}
      <section className="senate-brief"><span>PROMPT</span><p>{spawn.initialOrders ?? spawn.systemPrompt ?? 'No prompt was recorded for this session.'}</p>{spawn.systemPrompt && spawn.systemPrompt !== spawn.initialOrders && <details><summary>System instructions</summary><p>{spawn.systemPrompt}</p></details>}<dl><div><dt>Model</dt><dd>{session.model ?? 'unreported'}</dd></div><div><dt>Endpoint</dt><dd>{endpoint?.name ?? session.endpointId ?? 'local'}</dd></div><div className={session.budgetExhausted ? 'budget-exhausted' : ''}><dt>Cost</dt><dd>${(session.costUsd ?? 0).toFixed(4)}{session.budgetUsd ? ` of ${Number(session.budgetUsd).toFixed(2)}` : ''}{session.budgetRemainingUsd != null && !session.budgetExhausted ? ` · ${Number(session.budgetRemainingUsd).toFixed(2)} left` : ''}{session.budgetExhausted ? ' · exhausted' : ''}</dd></div><div><dt>Tools</dt><dd>{(role?.tools_allow ?? []).join(', ') || 'none declared'}</dd></div></dl><AgentControls session={session} /></section>
      <section className="senate-conversation"><span>CONVERSATION</span>{messages.length ? messages.map((event) => <article key={event.id ?? event.seq}><b>{event.data?.role ?? 'agent'}</b><p>{event.data?.text ?? event.data?.content ?? event.data?.summary}</p></article>) : <em>No messages yet.</em>}</section>
      <section className="senate-cohort"><span>WORKING WITH</span>{collaborators.length ? collaborators.map((item) => <button type="button" key={item.id} onClick={() => openSenate(item.id)}><i className={`state-${item.state}`} /><b>{item.name}</b><small>{item.role} · {item.state}</small></button>) : <em>No active links.</em>}<span className="senate-activity-title">RECENT ACTIVITY</span>{activity.length ? activity.map((event) => <article key={event.id ?? event.seq}><b>{event.kind.replace('session.', '')}</b><p>{event.data?.summary ?? event.data?.detail ?? event.data?.name ?? 'Status updated'}</p></article>) : <em>Nothing yet.</em>}</section>
    </div>
  </main>;
}

function PlansCommand() {
  const st = useField();
  const campaigns = st.snap.campaigns ?? [];
  const [selected, setSelected] = useState(st.activeCampaignId ?? campaigns[0]?.id ?? null);
  const [creating, setCreating] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [historyOpen, setHistoryOpen] = useState(false);
  const [replay, setReplay] = useState(null);

  useEffect(() => {
    if (!selected || !campaigns.some((c) => c.id === selected)) setSelected(campaigns[0]?.id ?? null);
  }, [campaigns, selected]);

  useEffect(() => {
    if (st.activeCampaignId && campaigns.some((campaign) => campaign.id === st.activeCampaignId)) setSelected(st.activeCampaignId);
  }, [campaigns, st.activeCampaignId]);

  useEffect(() => { setHistoryOpen(false); setReplay(null); }, [selected]);

  const liveCampaign = campaigns.find((c) => c.id === selected) ?? null;
  const campaign = replay?.campaign ?? liveCampaign;
  const sessions = useMemo(() => new Map(st.snap.sessions.map((s) => [s.id, s])), [st.snap.sessions]);

  const act = async (kind, payload = {}) => {
    if (!liveCampaign || busy || historyOpen) return;
    setBusy(true); setError(null);
    try {
      await api.campaignAction({
        commandId: crypto.randomUUID(), campaignId: liveCampaign.id, kind, ...payload,
      });
    } catch (e) { setError(e.message); }
    finally { setBusy(false); }
  };

  const start = () => {
    const objectiveId = campaign.objectives[0]?.id;
    const builders = st.config?.agents?.filter((a) => a.role === 'builder').slice(0, 2) ?? [];
    const roster = (builders.length ? builders : (st.config?.agents ?? []).slice(0, 1))
      .map((a) => ({ team: LANE, agentId: a.id, role: a.role, objectiveId }));
    if (!roster.length) { setError('No agents are configured in field.yaml, so nothing can start.'); return; }
    act('mobilize', { roster });
  };

  const members = campaign ? planMembers(campaign) : [];
  const spent = members.reduce((sum, m) => sum + (sessions.get(m.sessionId)?.costUsd ?? 0), 0);

  return (
    <div className="campaign-shell">
      <aside className="campaign-index">
        <div className="campaign-index-head">
          <span className="label">plans</span>
          <button className="campaign-add" type="button" onClick={() => setCreating(true)} aria-label="Create plan" title="Create plan">+</button>
        </div>
        {campaigns.map((c) => {
          const live = planMembers(c).filter((m) => m.status === 'active').length;
          return (
            <button
              type="button" key={c.id}
              className={`campaign-item${c.id === selected ? ' on' : ''}`}
              onClick={() => { setSelected(c.id); selectCampaign(c.id); }}
            >
              <span className={`campaign-sigil phase-${c.phase}`} />
              <span><b>{c.name}</b><small>{phaseLabel(c.phase)}{live ? ` · ${live} working` : ''}</small></span>
            </button>
          );
        })}
        {!campaigns.length && <div className="campaign-none">No plans yet.</div>}
      </aside>

      {campaign ? (
        <main className="campaign-table">
          <header className="campaign-titlebar">
            <div>
              <span className="label">plan · {phaseLabel(campaign.phase)}{campaign.paused ? ' · on hold' : ''}</span>
              <h1>{campaign.name}</h1>
              {campaign.intent && <p className="campaign-intent">{campaign.intent}</p>}
            </div>
            <div className="campaign-meters mono">
              {historyOpen && <span className="history-status">replay · event {replay?.actualSeq ?? '…'}</span>}
              <span>{campaign.scope}</span>
              <span>{members.filter((m) => m.status === 'active').length} working</span>
              <span className={campaign.budgetExhausted || spent >= campaign.budgetUsd ? 'budget-exhausted' : ''} title="spent of budget · remaining">
                ${spent.toFixed(2)} of ${campaign.budgetUsd.toFixed(0)} · ${Math.max(0, campaign.budgetUsd - spent).toFixed(2)} left
              </span>
              <button type="button" onClick={() => { setHistoryOpen((value) => !value); setReplay(null); }}>
                {historyOpen ? 'return live' : 'history'}
              </button>
            </div>
          </header>

          {historyOpen && liveCampaign && (
            <CampaignReplayRail campaign={liveCampaign} onReplay={setReplay} />
          )}

          <section className="campaign-fronts" aria-label="Objectives">
            {campaign.objectives.map((objective) => (
              <ObjectiveCard key={objective.id} objective={objective} campaign={campaign} sessions={sessions} />
            ))}
          </section>

          <PlanAgents
            campaign={campaign}
            members={members}
            sessions={sessions}
            config={st.config}
            busy={busy || historyOpen}
            act={act}
            onOpen={(id) => openSenate(id)}
          />

          <div className="campaign-actions">
            {historyOpen && <span className="history-readonly mono">historical state · controls locked</span>}
            {campaign.phase === 'draft' && <Action primary disabled={busy || historyOpen} onClick={start}>Start plan</Action>}
            {!campaign.paused && OPEN_PHASES.has(campaign.phase) && (
              <Action disabled={busy || historyOpen} onClick={() => act('pause')}>Hold</Action>
            )}
            {campaign.paused && <Action disabled={busy || historyOpen} onClick={() => act('resume')}>Resume</Action>}
            {!['cancelled', 'promoted'].includes(campaign.phase) && (
              <Action danger disabled={busy || historyOpen} onClick={() => {
                if (window.confirm('Cancel this plan? Its agents are stopped.')) act('cancel', { reason: 'operator cancelled from Plans' });
              }}>Cancel plan</Action>
            )}
          </div>
          {error && <div className="campaign-error mono" role="alert">{error}</div>}
        </main>
      ) : (
        <main className="campaign-empty">
          <div className="campaign-empty-mark" />
          <b>No plan yet.</b>
          <span>A plan is an objective with a definition of done and the agents assigned to reach it. Field starts them, tracks their spend, and keeps the history replayable.</span>
          <Action primary onClick={() => setCreating(true)}>Create a plan</Action>
        </main>
      )}

      {creating && <CreateCampaign config={st.config} onClose={() => setCreating(false)} onCreated={(id) => { setSelected(id); selectCampaign(id); setCreating(false); }} />}
    </div>
  );
}

function planMembers(campaign) {
  const teams = campaign.teams ?? {};
  return Object.keys(teams).flatMap((lane) => (teams[lane]?.members ?? []).map((member) => ({ ...member, lane })));
}

function CampaignReplayRail({ campaign, onReplay }) {
  const [trace, setTrace] = useState(null);
  const [cursor, setCursor] = useState(campaign.lastSeq);
  const [error, setError] = useState(null);

  useEffect(() => {
    let alive = true;
    api.campaignTrace(campaign.id, 0, 2000).then((result) => {
      if (!alive) return;
      setTrace(result);
      const last = result.events.at(-1)?.seq ?? campaign.lastSeq;
      setCursor(last);
    }).catch((cause) => alive && setError(cause.message));
    return () => { alive = false; };
  }, [campaign.id, campaign.lastSeq]);

  useEffect(() => {
    if (!trace) return undefined;
    let alive = true;
    const timer = setTimeout(() => {
      api.campaignReplay(campaign.id, cursor)
        .then((result) => alive && onReplay(result))
        .catch((cause) => alive && setError(cause.message));
    }, 80);
    return () => { alive = false; clearTimeout(timer); };
  }, [campaign.id, cursor, onReplay, trace]);

  if (error) return <div className="campaign-replay error mono">{error}</div>;
  if (!trace) return <div className="campaign-replay mono">loading plan history…</div>;
  const first = trace.events[0]?.seq ?? 0;
  const last = trace.events.at(-1)?.seq ?? first;
  const event = [...trace.events].reverse().find((item) => item.seq <= cursor);
  return (
    <div className="campaign-replay">
      <span className="label">replay</span>
      <input type="range" min={first} max={last} value={Math.min(cursor, last)} onChange={(e) => setCursor(Number(e.target.value))} aria-label="Replay position" />
      <span className="mono">{cursor} / {last}{trace.nextFrom != null && ' · first page'}</span>
      <b>{event ? eventLabel(event) : 'before start'}</b>
    </div>
  );
}

function ObjectiveCard({ objective, campaign, sessions }) {
  const working = planMembers(campaign)
    .filter((m) => m.status === 'active' && (!m.objectiveId || m.objectiveId === objective.id))
    .map((m) => sessions.get(m.sessionId)?.name ?? m.agentId ?? m.sessionId.slice(0, 6));
  return (
    <article className={`objective-front ${objective.status}`}>
      <div className="front-rank mono">{String(objective.priority).padStart(2, '0')}</div>
      <div className="front-copy">
        <span className="label">objective · {String(objective.status).replaceAll('_', ' ')}</span>
        <h2>{objective.statement}</h2>
        <div className="front-done">
          {objective.definitionOfDone.map((x, i) => <span key={i}>{x}</span>)}
        </div>
      </div>
      <div className="front-state">
        {working.length ? <span className="front-working">{working.join(', ')}</span> : <span className="front-working idle">no agent yet</span>}
        {objective.evidence?.length > 0 && <span className="front-evidence mono">{objective.evidence.length} evidence</span>}
      </div>
    </article>
  );
}

function PlanAgents({ campaign, members, sessions, config, busy, act, onOpen }) {
  const [panel, setPanel] = useState(null);
  const [orders, setOrders] = useState('');
  const [agentId, setAgentId] = useState(config?.agents?.[0]?.id ?? '');
  const active = members.filter((member) => member.status === 'active');
  const objectiveId = campaign.objectives[0]?.id;
  const submit = () => {
    if (panel === 'orders' && orders.trim()) {
      act('issue_orders', { team: active[0]?.lane ?? LANE, objectiveId, orders: orders.trim() });
      setOrders(''); setPanel(null);
    }
    if (panel === 'add' && agentId) {
      const agent = config?.agents?.find((item) => item.id === agentId);
      act('reinforce', {
        team: LANE, objectiveId, agentId, role: agent?.role,
        workspaceId: campaign.target?.workspaceId,
      });
      setPanel(null);
    }
  };
  return (
    <section className="plan-agents" aria-label="Agents on this plan">
      <div className="formation-head">
        <div><b>Agents</b><small>{active.length} working{members.length > active.length ? ` · ${members.length - active.length} finished` : ''}</small></div>
        <div className="formation-tools">
          <button type="button" disabled={busy || !active.length} onClick={() => setPanel(panel === 'orders' ? null : 'orders')}>Orders to all</button>
          <button type="button" disabled={busy || campaign.phase === 'draft'} onClick={() => setPanel(panel === 'add' ? null : 'add')}>+ Add agent</button>
        </div>
      </div>
      <div className="formation-units">
        {members.map((member) => {
          const session = sessions.get(member.sessionId);
          return (
            <div className="formation-unit" key={member.sessionId}>
              <button type="button" onClick={() => onOpen(member.sessionId)} title={session?.state ?? member.status}>
                <i className={`dot ${session?.state ?? 'idle'}`} />
                <span>{session?.name ?? member.agentId ?? member.sessionId.slice(0, 6)}</span>
                <small>{member.role ?? session?.role ?? 'agent'} · {String(session?.state ?? member.status).replaceAll('_', ' ')}{session?.costUsd ? ` · $${session.costUsd.toFixed(3)}` : ''}</small>
              </button>
              {member.status === 'active' && (
                <button className="formation-retreat" type="button" disabled={busy} title="Remove from plan" aria-label={`Remove ${session?.name ?? member.agentId ?? 'agent'} from the plan`} onClick={() => act('retreat', { team: member.lane, sessionId: member.sessionId })}>×</button>
              )}
            </div>
          );
        })}
        {!members.length && <span className="formation-vacant">{campaign.phase === 'draft' ? 'Start the plan to assign agents.' : 'No agents assigned.'}</span>}
      </div>
      {panel === 'orders' && (
        <div className="formation-order">
          <textarea value={orders} onChange={(event) => setOrders(event.target.value)} placeholder="Orders for every agent on this plan" aria-label="Orders" />
          <button type="button" disabled={busy || !orders.trim()} onClick={submit}>Send</button>
        </div>
      )}
      {panel === 'add' && (
        <div className="formation-order">
          <select value={agentId} onChange={(event) => setAgentId(event.target.value)} aria-label="Agent to add">
            {(config?.agents ?? []).map((agent) => <option key={agent.id} value={agent.id}>{agent.name ?? agent.id} · {agent.role}</option>)}
          </select>
          <button type="button" disabled={busy || !agentId} onClick={submit}>Add</button>
        </div>
      )}
    </section>
  );
}

function CreateCampaign({ config, onClose, onCreated }) {
  const dialogRef = useModalFocus(onClose);
  const [form, setForm] = useState({ name: '', intent: '', objective: '', done: '', workspaceId: config?.workspaces?.[0]?.id ?? '' });
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const submit = async (e) => {
    e.preventDefault(); setBusy(true); setError(null);
    try {
      const result = await api.createCampaign({
        commandId: crypto.randomUUID(), name: form.name, intent: form.intent,
        scope: 'sandbox', target: { type: 'workspace', id: form.workspaceId, workspaceId: form.workspaceId },
        objectives: [{
          statement: form.objective, definitionOfDone: [form.done],
          target: { type: 'workspace', id: form.workspaceId, workspaceId: form.workspaceId },
        }],
      });
      onCreated(result.campaignId);
    } catch (e2) { setError(e2.message); }
    finally { setBusy(false); }
  };
  return (
    <div className="campaign-modal-shade" onPointerDown={(e) => e.target === e.currentTarget && onClose()}>
      <form ref={dialogRef} className="campaign-modal" role="dialog" aria-modal="true" aria-label="Create a plan" onSubmit={submit}>
        <div><span className="label">new plan</span><button type="button" onClick={onClose} aria-label="Close new plan dialog">×</button></div>
        <label><span>name</span><input required value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} placeholder="Ship the settings page" /></label>
        <label><span>why</span><textarea required value={form.intent} onChange={(e) => setForm({ ...form, intent: e.target.value })} placeholder="What this plan is for, in a sentence or two." /></label>
        <label><span>project</span><select value={form.workspaceId} onChange={(e) => setForm({ ...form, workspaceId: e.target.value })}>{config?.workspaces?.map((w) => <option key={w.id} value={w.id}>{w.name}</option>)}</select></label>
        <label><span>objective</span><input required value={form.objective} onChange={(e) => setForm({ ...form, objective: e.target.value })} placeholder="What must be true when it is done" /></label>
        <label><span>definition of done</span><input required value={form.done} onChange={(e) => setForm({ ...form, done: e.target.value })} placeholder="e.g. npm test passes and the page renders" /></label>
        {error && <div className="campaign-error mono" role="alert">{error}</div>}
        <button className="btn primary" disabled={busy} type="submit">{busy ? 'Creating…' : 'Create plan'}</button>
      </form>
    </div>
  );
}

function Action({ children, onClick, disabled, danger, primary }) {
  return <button className={`campaign-action${danger ? ' danger' : ''}${primary ? ' primary' : ''}`} type="button" onClick={onClick} disabled={disabled}>{children}</button>;
}

function phaseLabel(phase) { return PHASE_WORD[phase] ?? phase?.replaceAll('_', ' ') ?? 'unknown'; }
function eventLabel(event) {
  const d = event.data ?? {};
  if (event.kind === 'campaign.phase_changed') return `${phaseLabel(d.from)} → ${phaseLabel(d.to)}`;
  if (event.kind === 'campaign.checkpoint_created') return `checkpoint · ${d.name}`;
  if (event.kind === 'team.member_assigned') return 'agent assigned';
  if (event.kind === 'objective.satisfied') return 'objective satisfied';
  return event.kind.replaceAll('.', ' · ').replaceAll('_', ' ');
}
