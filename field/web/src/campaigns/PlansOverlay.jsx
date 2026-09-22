/* Plans, drawn on the territory.

   A plan is an objective, a definition of done and the agents working towards it. That is
   never a separate place: it is a claim over part of the map. So Plans is not a
   destination any more — it is an overlay on Rome that draws each plan across the city it
   targets and the districts its agents are standing in, with the plan's objectives, its
   roster and its controls in a panel beside them. A plan with no folder to stand on is
   listed in the panel rather than given invented geography.

   The per-agent "chamber" that used to live here is gone: it showed the prompt, the
   model, the cost and the messages, which is the conversation panel, and the conversation
   panel is one click away in the folder the agent is standing in. */

import { useEffect, useMemo, useState } from 'react';
import { X } from 'lucide-react';
import { api } from '../net/client.js';
import { selectCampaign, useField } from '../state/store.js';
import { useModalFocus } from '../ui/useModalFocus.js';
import ReplayRail from '../ui/ReplayRail.jsx';
import { plainState } from '../ui/WorkCard.jsx';

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

function phaseLabel(phase) { return PHASE_WORD[phase] ?? phase?.replaceAll('_', ' ') ?? 'unknown'; }

export function planMembers(campaign) {
  const teams = campaign.teams ?? {};
  return Object.keys(teams).flatMap((lane) => (teams[lane]?.members ?? []).map((member) => ({ ...member, lane })));
}

const normalizeDir = (value) => (typeof value === 'string' ? value.replaceAll('\\', '/').replace(/^\/+|\/+$/g, '') : '');

/**
 * Where a plan stands on the map: the city it targets, plus every district one of its
 * agents is currently in. `map` is Rome's own geometry, so a plan is drawn over the same
 * shapes you clicked to get here.
 */
export function planGround(campaign, map, sessions) {
  const workspaceId = campaign.target?.workspaceId ?? campaign.target?.id ?? null;
  const city = map.find((item) => item.workspace.id === workspaceId) ?? null;
  const memberIds = new Set(planMembers(campaign).map((member) => member.sessionId));
  const points = [];
  if (city) points.push({ key: `city:${city.workspace.id}`, x: city.cityCentre.x, y: city.cityCentre.y, label: city.workspace.name });
  for (const entry of map) {
    for (const district of entry.districts) {
      const occupied = sessions.some((session) => (
        memberIds.has(session.id)
        && session.workspaceId === entry.workspace.id
        && normalizeDir(session.focusDir) === district.dir
      ));
      if (occupied) points.push({ key: `d:${entry.workspace.id}:${district.dir}`, x: district.x, y: district.y, label: district.name });
    }
  }
  return { workspaceId, city, points };
}

export default function PlansOverlay({ map = [], onClose, onOpenAgent = null }) {
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

  const grounds = useMemo(
    () => new Map(campaigns.map((item) => [item.id, planGround(item, map, st.snap.sessions)])),
    [campaigns, map, st.snap.sessions],
  );
  const unplaced = campaigns.filter((item) => !grounds.get(item.id)?.points.length);

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
    <>
      {/* The territory each plan covers, over the map itself. */}
      <div className="plans-ground" aria-hidden="true">
        {campaigns.map((item) => {
          const ground = grounds.get(item.id);
          if (!ground?.points.length) return null;
          const on = item.id === selected;
          return ground.points.map((point) => (
            <span
              key={`${item.id}:${point.key}`}
              className={`plan-claim phase-${item.phase}${on ? ' on' : ''}`}
              style={{ left: `${point.x}%`, top: `${point.y}%` }}
            ><i />{on ? item.name : ''}</span>
          ));
        })}
      </div>

      <aside className="plans-panel" onClick={(event) => event.stopPropagation()} role="dialog" aria-label="Plans on this territory">
        <header className="plans-panel-head">
          <div>
            <span className="label">plans on this territory</span>
            <h2>{campaigns.length === 1 ? 'One plan' : `${campaigns.length} plans`}</h2>
          </div>
          <button type="button" className="btn sm ghost icon" aria-label="Close the plans overlay" onClick={onClose}>
            <X aria-hidden="true" />
          </button>
        </header>

        <div className="plans-index" role="list">
          {campaigns.map((c) => {
            const live = planMembers(c).filter((m) => m.status === 'active').length;
            const placed = grounds.get(c.id)?.points.length ?? 0;
            return (
              <button
                type="button" key={c.id} role="listitem"
                className={`campaign-item${c.id === selected ? ' on' : ''}`}
                onClick={() => { setSelected(c.id); selectCampaign(c.id); }}
              >
                <span className={`campaign-sigil phase-${c.phase}`} />
                <span>
                  <b>{c.name}</b>
                  <small>
                    {phaseLabel(c.phase)}{live ? ` · ${live} working` : ''}
                    {placed ? ` · ${placed} on the map` : ' · nowhere on the map yet'}
                  </small>
                </span>
              </button>
            );
          })}
          {!campaigns.length && <div className="campaign-none">No plans yet.</div>}
        </div>

        {unplaced.length > 0 && campaigns.length > unplaced.length && (
          <p className="plans-unplaced">
            {unplaced.length === 1 ? 'One plan has' : `${unplaced.length} plans have`} no folder to stand on yet,
            so {unplaced.length === 1 ? 'it is' : 'they are'} only in this list.
          </p>
        )}

        {campaign ? (
          <div className="plans-detail">
            <header className="plans-detail-head">
              <span className="label">plan · {phaseLabel(campaign.phase)}{campaign.paused ? ' · on hold' : ''}</span>
              <h3>{campaign.name}</h3>
              {campaign.intent && <p className="campaign-intent">{campaign.intent}</p>}
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
              onOpen={(id) => onOpenAgent?.(id)}
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
                  if (window.confirm('Cancel this plan? Its agents are stopped.')) act('cancel', { reason: 'operator cancelled from the plans overlay' });
                }}>Cancel plan</Action>
              )}
            </div>
            {error && <div className="campaign-error mono" role="alert">{error}</div>}
          </div>
        ) : (
          <div className="campaign-empty">
            <div className="campaign-empty-mark" />
            <b>No plan yet.</b>
            <span>A plan is an objective with a definition of done and the agents assigned to reach it. Field starts them, tracks their spend, and keeps the history replayable.</span>
          </div>
        )}

        <footer className="plans-panel-foot">
          <button type="button" className="btn primary" onClick={() => setCreating(true)}>Create a plan</button>
        </footer>
      </aside>

      {creating && <CreateCampaign config={st.config} onClose={() => setCreating(false)} onCreated={(id) => { setSelected(id); selectCampaign(id); setCreating(false); }} />}
    </>
  );
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

  if (error) return <div className="replay-rail error mono">{error}</div>;
  if (!trace) return <div className="replay-rail mono">loading plan history…</div>;
  const first = trace.events[0]?.seq ?? 0;
  const last = trace.events.at(-1)?.seq ?? first;
  const event = [...trace.events].reverse().find((item) => item.seq <= cursor);
  // The same scrubber the map's time control uses; a plan's own history is per-plan, so
  // it stays here rather than moving the whole map back with it.
  return (
    <ReplayRail
      className="campaign-replay"
      min={first}
      max={last}
      value={Math.min(cursor, last)}
      onChange={setCursor}
      detail={event ? eventLabel(event) : 'before start'}
    />
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
              <button type="button" onClick={() => onOpen(member.sessionId)} title="Open this agent where it is standing">
                <i className={`dot tone-${plainState(session).tone}`} />
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

function eventLabel(event) {
  const d = event.data ?? {};
  if (event.kind === 'campaign.phase_changed') return `${phaseLabel(d.from)} → ${phaseLabel(d.to)}`;
  if (event.kind === 'campaign.checkpoint_created') return `checkpoint · ${d.name}`;
  if (event.kind === 'team.member_assigned') return 'agent assigned';
  if (event.kind === 'objective.satisfied') return 'objective satisfied';
  return event.kind.replaceAll('.', ' · ').replaceAll('_', ' ');
}
