import { useEffect, useRef, useState } from 'react';
import { Landmark, Send, Plus, X } from 'lucide-react';
import { api } from '../net/client.js';

const TIER_LABEL = { outpost: 'Outpost', town: 'Town', city: 'City', capital: 'Capital' };

// City command hub. A city IS a workspace: pick one, see its agents and their chat, send
// orders to the whole city, or deploy a new agent — no right-click required.
export default function CityPanel({ onClose }) {
  const [cities, setCities] = useState([]);
  const [selected, setSelected] = useState(null);
  const [detail, setDetail] = useState(null);
  const [order, setOrder] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [note, setNote] = useState('');
  const feedRef = useRef(null);

  async function loadCities() {
    try {
      const res = await api.cities();
      setCities(res.cities ?? []);
      setSelected((cur) => cur ?? res.cities?.[0]?.id ?? null);
    } catch (e) { setError(e.message); }
  }
  async function loadDetail(id) {
    if (!id) return;
    try { setDetail(await api.city(id)); } catch (e) { setError(e.message); }
  }

  useEffect(() => { loadCities(); }, []);
  useEffect(() => { loadDetail(selected); }, [selected]);
  useEffect(() => {
    const t = setInterval(() => { loadCities(); if (selected) loadDetail(selected); }, 4000);
    return () => clearInterval(t);
  }, [selected]);
  useEffect(() => {
    if (feedRef.current) feedRef.current.scrollTop = feedRef.current.scrollHeight;
  }, [detail?.feed?.length]);

  async function sendOrder() {
    const text = order.trim();
    if (!text || !selected) return;
    setBusy(true); setError(''); setNote('');
    try {
      const res = await api.cityOrder(selected, text);
      setOrder('');
      setNote(res.delivered != null
        ? `Order delivered to ${res.delivered} agent(s).`
        : `No agents were live — deployed one (${String(res.spawned).slice(0, 6)}).`);
      loadDetail(selected);
    } catch (e) { setError(e.message); } finally { setBusy(false); }
  }
  async function deploy() {
    if (!selected) return;
    setBusy(true); setError(''); setNote('');
    try {
      const res = await api.deployToCity(selected, {});
      setNote(`Deployed agent ${String(res.sessionId).slice(0, 6)}.`);
      loadDetail(selected);
    } catch (e) { setError(e.message); } finally { setBusy(false); }
  }

  return (
    <div className="cityhub-veil" onClick={onClose}>
      <div className="cityhub" role="dialog" aria-modal="true" onClick={(e) => e.stopPropagation()}>
        <header className="cityhub-head">
          <span><Landmark /> Cities</span>
          <button type="button" onClick={onClose} aria-label="Close cities"><X /></button>
        </header>
        <div className="cityhub-body">
          <aside className="cityhub-list">
            {cities.map((c) => (
              <button key={c.id} type="button" className={c.id === selected ? 'on' : ''} onClick={() => setSelected(c.id)}>
                <b>{c.name}</b>
                <small>{TIER_LABEL[c.tier] ?? c.tier} · lvl {c.level} · {c.agentCount} live</small>
              </button>
            ))}
            {!cities.length && <p className="cityhub-empty">No cities mounted.</p>}
          </aside>

          <section className="cityhub-detail">
            {!detail ? <p className="cityhub-empty">Select a city.</p> : (
              <>
                <div className="cityhub-rank">
                  <h2>{detail.name}</h2>
                  <span className={`cityhub-tier tier-${detail.tier}`}>{TIER_LABEL[detail.tier] ?? detail.tier} · level {detail.level}</span>
                  <div className="cityhub-xp"><i style={{ width: `${Math.min(100, detail.score)}%` }} /></div>
                  <small>{detail.agentCount} live agent(s) · maturity {detail.score}/100</small>
                </div>

                <div className="cityhub-roster">
                  {detail.agents?.length ? detail.agents.map((a) => (
                    <div key={a.sessionId} className={`cityhub-agent ${a.active ? 'live' : 'idle'}`}>
                      <b>{a.name}</b><small>{a.role} · {a.state}</small>
                      {a.lastSay && <p>{a.lastSay}</p>}
                    </div>
                  )) : <p className="cityhub-empty">No agents here yet. Deploy one below.</p>}
                </div>

                <div className="cityhub-feed" ref={feedRef}>
                  {detail.feed?.length ? detail.feed.map((f) => (
                    <div key={f.seq} className="cityhub-feed-row">
                      <span className="cityhub-feed-kind">{f.role ?? (f.kind.split('.')[1] ?? f.kind)}</span>
                      <span className="cityhub-feed-text">{f.text}</span>
                    </div>
                  )) : <p className="cityhub-empty">No activity yet.</p>}
                </div>

                {error && <p className="cityhub-error">{error}</p>}
                {note && <p className="cityhub-note">{note}</p>}

                <div className="cityhub-compose">
                  <textarea
                    value={order}
                    onChange={(e) => setOrder(e.target.value)}
                    placeholder={detail.agentCount ? 'Tell the city’s agents what to do next… (Ctrl+Enter)' : 'No agents yet — an order will deploy one.'}
                    rows={2}
                    onKeyDown={(e) => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) sendOrder(); }}
                  />
                  <div className="cityhub-actions">
                    <button type="button" className="ghost" disabled={busy} onClick={deploy}><Plus /> Deploy agent</button>
                    <button type="button" className="primary" disabled={busy || !order.trim()} onClick={sendOrder}>
                      <Send /> {detail.agentCount ? 'Send order' : 'Deploy with order'}
                    </button>
                  </div>
                </div>
              </>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}
