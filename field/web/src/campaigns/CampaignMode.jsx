import { useEffect, useMemo, useState } from 'react';
import { api } from '../net/client.js';
import { clearActiveAgent, openSenate, selectCampaign, useField } from '../state/store.js';
import { useModalFocus } from '../ui/useModalFocus.js';

const PHASES = [
  'draft', 'mobilizing', 'blue_building', 'red_challenging', 'contested',
  'blue_mitigating', 'red_retesting', 'referee_review', 'verified', 'promoted',
];

const TEAM_ORDER = ['blue', 'red', 'referee', 'purple'];

export default function CampaignMode() {
  const st = useField();
  const senator = st.snap.sessions.find((session) => session.id === st.activeSessionId);
  return senator ? <SenateSessionChamber session={senator} /> : <CampaignCommand />;
}

function SenateSessionChamber({ session }) {
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
  return <main className="senate-session-shell">
    <header className="senate-session-header"><div><span>SENATE CHAMBER</span><h1>{session.name}</h1><p>{session.role} · {session.state?.replaceAll('_', ' ')}</p></div><button type="button" onClick={clearActiveAgent}>Campaign command</button></header>
    <section className="senate-dais">
      <div className="senate-agent-seal">{String(session.name || 'A').slice(0, 2).toUpperCase()}</div>
      <div><span>CURRENT OBJECTIVE</span><h2>{session.target?.label ?? session.stateDetail ?? 'Awaiting orders'}</h2><p>{workspace?.name ?? session.workspaceId}{campaign ? ` · ${campaign.name}` : ''}</p></div>
      <div className="senate-progress"><b>{pct}%</b><span role="progressbar" aria-label={`${session.name} progress`} aria-valuemin="0" aria-valuemax="100" aria-valuenow={pct}><i style={{ width: `${pct}%` }} /></span><small>{session.progress?.done ?? 0} / {session.progress?.total ?? 0} stages</small></div>
    </section>
    <div className="senate-chamber-grid">
      {trace.length >= 2000 && <div className="label">Showing the first 2,000 events. Open Traces to load the rest.</div>}
      <section className="senate-brief"><span>PROMPT</span><p>{spawn.initialOrders ?? spawn.systemPrompt ?? 'No prompt was recorded for this session.'}</p>{spawn.systemPrompt && spawn.systemPrompt !== spawn.initialOrders && <details><summary>System instructions</summary><p>{spawn.systemPrompt}</p></details>}<dl><div><dt>Model</dt><dd>{session.model ?? 'unreported'}</dd></div><div><dt>Endpoint</dt><dd>{session.endpointId ?? 'local'}</dd></div><div><dt>Authority</dt><dd>{(role?.tools_allow ?? []).join(', ') || 'none declared'}</dd></div></dl></section>
      <section className="senate-conversation"><span>CONVERSATION</span>{messages.length ? messages.map((event) => <article key={event.id ?? event.seq}><b>{event.data?.role ?? 'agent'}</b><p>{event.data?.text ?? event.data?.content ?? event.data?.summary}</p></article>) : <em>No messages yet.</em>}</section>
      <section className="senate-cohort"><span>WORKING WITH</span>{collaborators.length ? collaborators.map((item) => <button type="button" key={item.id} onClick={() => openSenate(item.id)}><i className={`state-${item.state}`} /><b>{item.name}</b><small>{item.role} · {item.state}</small></button>) : <em>No active links.</em>}<span className="senate-activity-title">RECENT ACTIVITY</span>{activity.map((event) => <article key={event.id ?? event.seq}><b>{event.kind.replace('session.', '')}</b><p>{event.data?.summary ?? event.data?.detail ?? event.data?.name ?? 'Status updated'}</p></article>)}</section>
    </div>
  </main>;
}

function CampaignCommand() {
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

  const mobilize = () => {
    const objectiveId = campaign.objectives[0]?.id;
    const byRole = (role) => st.config?.agents?.find((a) => a.role === role)?.id;
    const builders = st.config?.agents?.filter((a) => a.role === 'builder').slice(0, 2) ?? [];
    const roster = [
      ...builders.map((a) => ({ team: 'blue', agentId: a.id, role: a.role, objectiveId })),
      { team: 'red', agentId: byRole('challenger') ?? byRole('scout'), role: 'challenger', objectiveId },
      { team: 'referee', agentId: byRole('verifier'), role: 'verifier', objectiveId },
    ].filter((x) => x.agentId);
    act('mobilize', { roster });
  };

  return (
    <div className="campaign-shell">
      <aside className="campaign-index">
        <div className="campaign-index-head">
          <span className="label">operations</span>
          <button className="campaign-add" type="button" onClick={() => setCreating(true)} aria-label="Create operation">+</button>
        </div>
        {campaigns.map((c) => (
          <button
            type="button" key={c.id}
            className={`campaign-item${c.id === selected ? ' on' : ''}`}
            onClick={() => { setSelected(c.id); selectCampaign(c.id); }}
          >
            <span className={`campaign-sigil phase-${c.phase}`} />
            <span><b>{c.name}</b><small>{phaseLabel(c.phase)}</small></span>
            {c.findings.some((f) => ['open', 'acknowledged', 'mitigating', 'ready_for_retest'].includes(f.status)) && (
              <i>{c.findings.filter((f) => ['open', 'acknowledged', 'mitigating', 'ready_for_retest'].includes(f.status)).length}</i>
            )}
          </button>
        ))}
        {!campaigns.length && <div className="campaign-none">No active operations.</div>}
      </aside>

      {campaign ? (
        <main className="campaign-table">
          <header className="campaign-titlebar">
            <div>
              <span className="label">campaign</span>
              <h1>{campaign.name}</h1>
            </div>
            <div className="campaign-meters mono">
              {historyOpen && <span className="history-status">replay · event {replay?.actualSeq ?? '…'}</span>}
              {historyOpen && replay?.campaign?.checkpoints?.length > 0 && (
                <span>baseline · {replay.campaign.checkpoints.at(-1).name}</span>
              )}
              <span>{campaign.scope}</span>
              <span>{activeMembers(campaign)} deployed</span>
              <span>${campaign.budgetUsd.toFixed(0)} ceiling</span>
              <button type="button" onClick={() => { setHistoryOpen((value) => !value); setReplay(null); }}>
                {historyOpen ? 'return live' : 'history'}
              </button>
            </div>
          </header>

          {historyOpen && liveCampaign && (
            <CampaignReplayRail campaign={liveCampaign} onReplay={setReplay} />
          )}

          <PhaseRail campaign={campaign} onAdvance={(to) => act('advance', { to })} busy={busy || historyOpen} />

          <section className="campaign-fronts">
            {campaign.objectives.map((objective) => (
              <ObjectiveFront
                key={objective.id}
                objective={objective}
                campaign={campaign}
                busy={busy || historyOpen}
                onReady={(evidence) => act('satisfy_objective', { objectiveId: objective.id, evidence: [evidence] })}
              />
            ))}
          </section>

          <section className="formations">
            {TEAM_ORDER.map((team) => (
              <Formation
                 key={team} kind={team} data={campaign.teams[team]}
                 sessions={sessions} config={st.config} campaign={campaign}
                 busy={busy || historyOpen} act={act}
                 onOpen={(id) => openSenate(id)}
              />
            ))}
          </section>

          <div className="campaign-actions">
            {historyOpen && <span className="history-readonly mono">historical state · controls locked</span>}
            {campaign.phase === 'draft' && <Action disabled={busy || historyOpen} onClick={mobilize}>Mobilize core</Action>}
            {!campaign.paused && !['draft', 'promoted', 'cancelled'].includes(campaign.phase) && (
              <Action disabled={busy || historyOpen} onClick={() => act('pause')}>Hold</Action>
            )}
            {campaign.paused && <Action disabled={busy || historyOpen} onClick={() => act('resume')}>Resume</Action>}
            {campaign.phase === 'verified' && !campaign.checkpoints.length && (
              <Action disabled={busy || historyOpen} onClick={() => {
                if (window.confirm('Record the campaign against the target workspace’s current clean Git revision? Uncommitted changes make the checkpoint fail.')) {
                  act('checkpoint', { name: `${campaign.name} verified`, confirmRisk: true });
                }
              }}>Record Git checkpoint</Action>
            )}
            {campaign.phase === 'verified' && campaign.checkpoints.length > 0 && (
              <PromotionControl campaign={campaign} busy={busy || historyOpen} act={act} />
            )}
            {campaign.phase === 'promoted' && campaign.checkpoints.length > 0 && (
              <RollbackControl campaign={campaign} busy={busy || historyOpen} act={act} />
            )}
            {!['cancelled', 'promoted'].includes(campaign.phase) && (
              <Action danger disabled={busy || historyOpen} onClick={() => act('cancel', { reason: 'operator cancelled from Campaign view' })}>Stand down</Action>
            )}
          </div>
          {error && <div className="campaign-error mono">{error}</div>}
        </main>
      ) : (
        <main className="campaign-empty">
          <div className="campaign-empty-mark" />
          <b>Campaign command is quiet.</b>
          <span>Create an operation to mobilize blue, red, and referee formations.</span>
          <Action onClick={() => setCreating(true)}>Create operation</Action>
        </main>
      )}

      {campaign && <ContestPanel campaign={campaign} act={act} busy={busy || historyOpen} sessions={sessions} />}
      {creating && <CreateCampaign config={st.config} onClose={() => setCreating(false)} onCreated={(id) => { setSelected(id); selectCampaign(id); setCreating(false); }} />}
    </div>
  );
}

function PhaseRail({ campaign, onAdvance, busy }) {
  const current = PHASES.indexOf(campaign.phase);
  const legal = new Set(campaign.legalActions ?? []);
  const options = new Map((campaign.transitionOptions ?? []).map((option) => [option.to, option]));
  return (
    <div className="phase-rail">
      {PHASES.map((phase, i) => {
        const can = legal.has(`advance:${phase}`);
        return (
          <button
            key={phase} type="button" disabled={!can || busy}
            className={`${i < current ? 'past ' : ''}${i === current ? 'now ' : ''}${can ? 'legal' : ''}`}
            onClick={() => can && onAdvance(phase)}
            title={can ? `Advance to ${phaseLabel(phase)}` : (options.get(phase)?.reason ?? phaseLabel(phase))}
          >
            <i />
            <span>{phaseLabel(phase)}</span>
          </button>
        );
      })}
    </div>
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

  if (error) return <div className="campaign-replay error mono">{error}</div>;
  if (!trace) return <div className="campaign-replay mono">loading campaign history…</div>;
  const first = trace.events[0]?.seq ?? 0;
  const last = trace.events.at(-1)?.seq ?? first;
  const event = [...trace.events].reverse().find((item) => item.seq <= cursor);
  return (
    <div className="campaign-replay">
      <span className="label">operational replay</span>
      <input type="range" min={first} max={last} value={Math.min(cursor, last)} onChange={(e) => setCursor(Number(e.target.value))} />
      <span className="mono">{cursor} / {last}{trace.nextFrom != null && ' · first page'}</span>
      <b>{event ? eventLabel(event) : 'before mobilization'}</b>
    </div>
  );
}

function ObjectiveFront({ objective, campaign, onReady, busy }) {
  const [evidence, setEvidence] = useState('');
  const linked = campaign.findings.filter((f) => f.objectiveId === objective.id);
  return (
    <article className={`objective-front ${objective.status}`}>
      <div className="front-rank mono">{String(objective.priority).padStart(2, '0')}</div>
      <div className="front-copy">
        <span className="label">objective · {objective.status}</span>
        <h2>{objective.statement}</h2>
        <div className="front-done">
          {objective.definitionOfDone.map((x, i) => <span key={i}>{x}</span>)}
        </div>
      </div>
      <div className="front-state">
        {linked.length > 0 && <span className="front-contested">{linked.length} challenge{linked.length === 1 ? '' : 's'}</span>}
        {campaign.phase === 'blue_building' && objective.status !== 'satisfied' && (
          <div className="inline-order">
            <input value={evidence} onChange={(e) => setEvidence(e.target.value)} placeholder="readiness evidence" />
            <button type="button" disabled={busy || !evidence.trim()} onClick={() => { onReady(evidence.trim()); setEvidence(''); }}>attach</button>
          </div>
        )}
      </div>
    </article>
  );
}

function Formation({ kind, data, sessions, config, campaign, busy, act, onOpen }) {
  const [panel, setPanel] = useState(null);
  const [orders, setOrders] = useState('');
  const [agentId, setAgentId] = useState(config?.agents?.[0]?.id ?? '');
  const active = data?.members?.filter((member) => member.status === 'active') ?? [];
  const external = data?.members?.filter((member) => member.status === 'external') ?? [];
  const objectiveId = campaign.objectives[0]?.id;
  const submit = () => {
    if (panel === 'orders' && orders.trim()) {
      act('issue_orders', { team: kind, objectiveId, orders: orders.trim() });
      setOrders(''); setPanel(null);
    }
    if (panel === 'reinforce' && agentId) {
      const agent = config?.agents?.find((item) => item.id === agentId);
      act('reinforce', {
        team: kind, objectiveId, agentId, role: agent?.role,
        workspaceId: campaign.target?.workspaceId,
      });
      setPanel(null);
    }
  };
  return (
    <div className={`formation team-${kind}`}>
      <div className="formation-head">
        <span className="team-crest">{crest(kind)}</span>
        <div><b>{kind}</b><small>{active.length} live{external.length ? ` · ${external.length} external` : ''}</small></div>
        <div className="formation-tools">
          <button type="button" disabled={busy || !active.length} title={`Issue orders to ${kind}`} onClick={() => setPanel(panel === 'orders' ? null : 'orders')}>orders</button>
          <button type="button" disabled={busy} title={`Reinforce ${kind}`} aria-label={`Reinforce ${kind}`} onClick={() => setPanel(panel === 'reinforce' ? null : 'reinforce')}>+</button>
        </div>
      </div>
      <div className="formation-units">
        {(data?.members ?? []).map((member) => {
          const session = sessions.get(member.sessionId);
          return (
            <div className="formation-unit" key={member.sessionId}>
              <button type="button" onClick={() => onOpen(member.sessionId)} title={session?.state ?? member.status}>
                <i className={`unit-glyph role-${member.role ?? session?.role ?? 'builder'}`} />
                <span>{session?.name ?? member.agentId ?? member.sessionId.slice(0, 6)}</span>
                <small>{session?.state ?? member.status}</small>
              </button>
              {member.status === 'active' && (
                <button className="formation-retreat" type="button" disabled={busy} title="Retreat unit" aria-label={`Retreat ${session?.name ?? member.agentId ?? 'unit'}`} onClick={() => act('retreat', { team: kind, sessionId: member.sessionId })}>×</button>
              )}
            </div>
          );
        })}
        {!data?.members?.length && <span className="formation-vacant">vacant</span>}
      </div>
      {panel === 'orders' && (
        <div className="formation-order">
          <textarea value={orders} onChange={(event) => setOrders(event.target.value)} placeholder={`orders for ${kind}`} />
          <button type="button" disabled={busy || !orders.trim()} onClick={submit}>dispatch</button>
        </div>
      )}
      {panel === 'reinforce' && (
        <div className="formation-order">
          <select value={agentId} onChange={(event) => setAgentId(event.target.value)}>
            {(config?.agents ?? []).map((agent) => <option key={agent.id} value={agent.id}>{agent.name ?? agent.id} · {agent.role}</option>)}
          </select>
          <button type="button" disabled={busy || !agentId} onClick={submit}>deploy</button>
        </div>
      )}
    </div>
  );
}

function ContestPanel({ campaign, act, busy }) {
  const [form, setForm] = useState({ severity: 'high', category: 'correctness', claim: '', evidence: '', reproduction: '' });
  const red = campaign.teams.red.members.find((m) => ['active', 'external'].includes(m.status));
  const referee = campaign.teams.referee.members.find((m) => ['active', 'external'].includes(m.status));
  const objective = campaign.objectives[0];
  const submitFinding = () => {
    if (!form.claim || !form.evidence || !form.reproduction || !red) return;
    act('report_finding', {
      objectiveId: objective.id, authorSessionId: red.sessionId,
      severity: form.severity, category: form.category, claim: form.claim,
      scope: objective.target?.id ?? campaign.target?.id ?? '',
      evidence: [form.evidence], reproduction: [form.reproduction], confidence: 0.85,
    });
    setForm({ ...form, claim: '', evidence: '', reproduction: '' });
  };
  return (
    <aside className="contest-panel">
      <div className="contest-head"><span className="label">contest ledger</span><b>{campaign.findings.length}</b></div>
      <div className="contest-list">
        {campaign.findings.map((finding) => (
          <FindingCard key={finding.id} finding={finding} campaign={campaign} act={act} busy={busy} red={red} />
        ))}
        {!campaign.findings.length && <div className="contest-clear">No findings recorded.</div>}
      </div>
      {campaign.phase === 'red_challenging' && (
        <div className="finding-form">
          <div className="finding-form-row">
            <select value={form.severity} onChange={(e) => setForm({ ...form, severity: e.target.value })}>
              {['low', 'medium', 'high', 'critical'].map((x) => <option key={x}>{x}</option>)}
            </select>
            <input value={form.category} onChange={(e) => setForm({ ...form, category: e.target.value })} placeholder="category" />
          </div>
          <textarea value={form.claim} onChange={(e) => setForm({ ...form, claim: e.target.value })} placeholder="claim" />
          <input value={form.evidence} onChange={(e) => setForm({ ...form, evidence: e.target.value })} placeholder="evidence reference" />
          <input value={form.reproduction} onChange={(e) => setForm({ ...form, reproduction: e.target.value })} placeholder="reproduction" />
          <Action disabled={busy || !red} onClick={submitFinding}>Record challenge</Action>
        </div>
      )}
      {campaign.phase === 'referee_review' && referee && (
        <VerdictControl campaign={campaign} referee={referee} act={act} busy={busy} />
      )}
    </aside>
  );
}

function FindingCard({ finding, campaign, act, busy, red }) {
  const [claim, setClaim] = useState('');
  const [evidence, setEvidence] = useState('');
  const [retestResult, setRetestResult] = useState('fixed');
  const blueOwner = campaign.teams.blue.members.find((m) => ['active', 'external'].includes(m.status));
  const linked = finding.mitigationIds.map((id) => campaign.mitigations.find((m) => m.id === id)).filter(Boolean);
  const liveMitigation = linked.some((m) => ['proposed', 'active', 'ready'].includes(m.status));
  return (
    <article className={`finding severity-${finding.severity}`}>
      <div><span>{finding.severity}</span><i>{finding.status.replaceAll('_', ' ')}</i></div>
      <b>{finding.claim}</b>
      <small>{finding.category} · {Math.round(finding.confidence * 100)}% confidence</small>
      {campaign.phase === 'contested' && finding.status === 'open' && (
        <button type="button" disabled={busy} onClick={() => act('acknowledge_finding', { findingId: finding.id })}>acknowledge</button>
      )}
      {campaign.phase === 'blue_mitigating' && ['acknowledged', 'disputed'].includes(finding.status) && !liveMitigation && (
        <div className="finding-inline">
          <input value={claim} onChange={(e) => setClaim(e.target.value)} placeholder="mitigation claim" />
          <button type="button" disabled={busy || !claim.trim() || !blueOwner} onClick={() => {
            act('propose_mitigation', { findingIds: [finding.id], claim: claim.trim(), ownerSessionId: blueOwner?.sessionId });
            setClaim('');
          }}>link</button>
        </div>
      )}
      {campaign.phase === 'blue_mitigating' && linked.map((mitigation) => {
        if (mitigation.status === 'proposed') return (
          <button key={mitigation.id} type="button" disabled={busy} onClick={() => act('start_mitigation', { mitigationId: mitigation.id, sessionId: mitigation.ownerSessionId })}>begin mitigation</button>
        );
        if (mitigation.status === 'active') return (
          <div className="finding-inline" key={mitigation.id}>
            <input value={evidence} onChange={(e) => setEvidence(e.target.value)} placeholder="mitigation evidence" />
            <button type="button" disabled={busy || !evidence.trim()} onClick={() => {
              act('mark_mitigation_ready', { mitigationId: mitigation.id, sessionId: mitigation.ownerSessionId, evidence: [evidence.trim()] });
              setEvidence('');
            }}>ready</button>
          </div>
        );
        return null;
      })}
      {campaign.phase === 'red_retesting' && finding.status === 'ready_for_retest' && red && (
        <div className="finding-inline retest">
          <select value={retestResult} onChange={(e) => setRetestResult(e.target.value)}>
            {['fixed', 'persists', 'false_positive', 'inconclusive'].map((x) => <option key={x} value={x}>{x.replaceAll('_', ' ')}</option>)}
          </select>
          <input value={evidence} onChange={(e) => setEvidence(e.target.value)} placeholder="retest evidence" />
          <button type="button" disabled={busy || !evidence.trim()} onClick={() => {
            act('record_retest', { findingId: finding.id, sessionId: red.sessionId, result: retestResult, evidence: [evidence.trim()] });
            setEvidence('');
          }}>record</button>
        </div>
      )}
    </article>
  );
}

function VerdictControl({ campaign, referee, act, busy }) {
  const [evidence, setEvidence] = useState('');
  const [rationale, setRationale] = useState('');
  return (
    <div className="verdict-strip">
      <span className="label">independent verdict</span>
      {campaign.review?.status !== 'active' && campaign.review?.status !== 'complete' && (
        <button type="button" disabled={busy} onClick={() => act('begin_referee_review', { sessionId: referee.sessionId })}>begin review</button>
      )}
      {campaign.review?.status === 'active' && (
        <>
          <input value={evidence} onChange={(e) => setEvidence(e.target.value)} placeholder="independent evidence" />
          <textarea value={rationale} onChange={(e) => setRationale(e.target.value)} placeholder="rationale" />
          <div>
            {['verified', 'rejected', 'inconclusive'].map((verdict) => (
              <button key={verdict} type="button" disabled={busy || !evidence.trim() || !rationale.trim()} onClick={() => {
                act('record_verdict', { sessionId: referee.sessionId, verdict, evidence: [evidence.trim()], rationale: rationale.trim() });
              }}>{verdict}</button>
            ))}
          </div>
        </>
      )}
    </div>
  );
}

function PromotionControl({ campaign, busy, act }) {
  const [name, setName] = useState(campaign.name);
  return (
    <div className="promotion-control">
      <input value={name} onChange={(e) => setName(e.target.value)} placeholder="capability name" />
      <Action disabled={busy || !name.trim()} onClick={() => act('promote', {
        checkpointId: campaign.checkpoints.at(-1).id,
        capabilities: [{ id: crypto.randomUUID(), name: name.trim(), evidence: ['campaign referee verdict'] }],
      })}>Promote</Action>
    </div>
  );
}

function RollbackControl({ campaign, busy, act }) {
  const [reason, setReason] = useState('');
  return (
    <div className="promotion-control rollback-control">
      <input value={reason} onChange={(event) => setReason(event.target.value)} placeholder="rollback reason" />
      <Action danger disabled={busy || !reason.trim()} onClick={() => {
        if (window.confirm('Record this campaign as rolled back? Field is currently record-only: this will not change workspace files.')) {
          act('rollback', {
            checkpointId: campaign.checkpoints.at(-1).id,
            reason: reason.trim(),
            confirmRisk: true,
          });
        }
      }}>Record rollback</Action>
    </div>
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
      <form ref={dialogRef} className="campaign-modal" role="dialog" aria-modal="true" aria-label="Create a new operation" onSubmit={submit}>
        <div><span className="label">new operation</span><button type="button" onClick={onClose} aria-label="Close new operation dialog">×</button></div>
        <label><span>name</span><input required value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} /></label>
        <label><span>intent</span><textarea required value={form.intent} onChange={(e) => setForm({ ...form, intent: e.target.value })} /></label>
        <label><span>workspace</span><select value={form.workspaceId} onChange={(e) => setForm({ ...form, workspaceId: e.target.value })}>{config?.workspaces?.map((w) => <option key={w.id} value={w.id}>{w.name}</option>)}</select></label>
        <label><span>objective</span><input required value={form.objective} onChange={(e) => setForm({ ...form, objective: e.target.value })} /></label>
        <label><span>definition of done</span><input required value={form.done} onChange={(e) => setForm({ ...form, done: e.target.value })} /></label>
        {error && <div className="campaign-error mono" role="alert">{error}</div>}
        <button className="btn primary" disabled={busy} type="submit">Establish campaign</button>
      </form>
    </div>
  );
}

function Action({ children, onClick, disabled, danger }) {
  return <button className={`campaign-action${danger ? ' danger' : ''}`} type="button" onClick={onClick} disabled={disabled}>{children}</button>;
}

function phaseLabel(phase) { return phase?.replaceAll('_', ' ') ?? 'unknown'; }
function eventLabel(event) {
  const d = event.data ?? {};
  if (event.kind === 'campaign.phase_changed') return `${phaseLabel(d.from)} → ${phaseLabel(d.to)}`;
  if (event.kind === 'finding.reported') return `red finding · ${d.severity}`;
  if (event.kind === 'retest.completed') return `retest · ${phaseLabel(d.result)}`;
  if (event.kind === 'referee.verdict') return `referee · ${d.verdict}`;
  if (event.kind === 'campaign.checkpoint_created') return `baseline · ${d.name}`;
  if (event.kind === 'campaign.promoted') return 'capability promoted';
  return phaseLabel(event.kind.replaceAll('.', ' · '));
}
function activeMembers(c) { return TEAM_ORDER.flatMap((x) => c.teams[x]?.members ?? []).filter((m) => m.status === 'active').length; }
function crest(kind) { return ({ blue: 'B', red: 'R', referee: 'V', purple: 'P' })[kind]; }
