import { lazy, Suspense, useEffect, useState } from 'react';
import { api } from './net/client.js';
import { setMode, useField } from './state/store.js';
import RoutinesMode from './routines/RoutinesMode.jsx';
import TracesMode from './traces/TracesMode.jsx';
import CampaignMode from './campaigns/CampaignMode.jsx';
import TheaterMode from './theater/TheaterMode.jsx';

const WorkspaceMode = lazy(() => import('./workspace/WorkspaceMode.jsx'));

const MODES = [
  { id: 'theater', name: 'Field', key: 'F' },
  { id: 'routines', name: 'Routines', key: 'R' },
  { id: 'traces', name: 'Traces', key: 'T' },
];
const FIELD_LENSES = [
  { id: 'theater', name: 'Board', key: 'V' },
  { id: 'workspace', name: 'City', key: 'C' },
  { id: 'campaigns', name: 'Senate', key: 'S' },
];
const FIELD_MODES = new Set(['theater', 'field', 'campaigns', 'workspace']);

export default function App() {
  const st = useField();
  const [loggedOut, setLoggedOut] = useState(false);
  const [logoutError, setLogoutError] = useState(null);

  useEffect(() => {
    const onKey = (e) => {
      const tag = document.activeElement?.tagName;
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(tag) || document.activeElement?.isContentEditable) return;
      if (e.defaultPrevented || e.repeat || !e.altKey || !e.ctrlKey || e.metaKey) return;
      const hit = [...MODES, ...FIELD_LENSES].find((m) => m.key.toLowerCase() === e.key.toLowerCase());
      if (hit) {
        e.preventDefault();
        setMode(hit.id);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [st.mode]);

  const live = st.snap.sessions.filter(
    (s) => !['done', 'cancelled', 'interrupted', 'error'].includes(s.state),
  );
  const waiting = st.snap.permissions?.length ?? 0;
  const simulated = live.filter((s) => s.simulated).length;
  const realSessionCount = st.snap.sessions.filter((s) => !s.simulated).length;
  const budgetReservations = st.snap.budgetReservations ?? [];
  const reservedUsd = budgetReservations.reduce((sum, row) => sum + row.reservedUsd, 0);
  const unknownCosts = budgetReservations.filter(row => row.costStatus === 'missing' || (row.terminal && !row.finalCostKnown)).length;

  if (loggedOut) return <main className="stage"><p role="status">Signed out. Restart Field and open its new bootstrap URL to sign in again.</p></main>;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <span className="brand-name">FIELD</span>
          <span className="brand-sub label">{st.config?.field?.name ?? 'loading…'}</span>
        </div>

        <nav className="modes">
          {MODES.map((m) => (
            <button
              key={m.id}
              className={`mode${m.id === 'theater' ? (FIELD_MODES.has(st.mode) ? ' on' : '') : (st.mode === m.id ? ' on' : '')}`}
              onClick={() => setMode(m.id)}
              type="button"
              aria-current={m.id === 'theater' ? (FIELD_MODES.has(st.mode) ? 'page' : undefined) : (st.mode === m.id ? 'page' : undefined)}
              aria-keyshortcuts={`Control+Alt+${m.key}`}
            >
              {m.name}
              <kbd>Ctrl+Alt+{m.key}</kbd>
            </button>
          ))}
        </nav>

        {FIELD_MODES.has(st.mode) && (
          <nav className="field-lenses" aria-label="Field view">
            {FIELD_LENSES.map((lens) => (
              <button key={lens.id} className={st.mode === lens.id || (lens.id === 'theater' && st.mode === 'field') ? 'on' : ''} onClick={() => setMode(lens.id)} type="button" aria-current={st.mode === lens.id || (lens.id === 'theater' && st.mode === 'field') ? 'page' : undefined} aria-keyshortcuts={`Control+Alt+${lens.key}`}>
                {lens.name}<kbd>Ctrl+Alt+{lens.key}</kbd>
              </button>
            ))}
          </nav>
        )}

        <div className="topstats mono" aria-live="polite" aria-atomic="true">
          <button type="button" onClick={async () => {
            try { await api.logout(); setLoggedOut(true); }
            catch { setLogoutError('Sign out failed. Check the Field server and try again.'); }
          }}>Sign out</button>
          {simulated > 0 && <span className="stat warn">SIMULATION · {simulated} units</span>}
          {waiting > 0 && <span className="stat warn">{waiting} awaiting approval</span>}
          <span className="stat">{live.length} active</span>
          <span className="stat">{realSessionCount} sessions</span>
          <span className="stat">${(st.snap.totals?.costUsd ?? 0).toFixed(3)}</span>
          {reservedUsd > 0 && <span className="stat">${reservedUsd.toFixed(2)} reserved</span>}
          {unknownCosts > 0 && <span className="stat warn">{unknownCosts} costs unconfirmed</span>}
        </div>
      </header>

      <main className="stage">
        {logoutError && <p role="alert">{logoutError}</p>}
        {st.error && <div className="fatal">Field server unreachable — {st.error}</div>}
        {(st.mode === 'theater' || st.mode === 'field') && <TheaterMode />}
        {st.mode === 'campaigns' && <CampaignMode />}
        {st.mode === 'workspace' && (
          <Suspense fallback={<p role="status">Loading City…</p>}>
            <WorkspaceMode />
          </Suspense>
        )}
        {st.mode === 'routines' && <RoutinesMode />}
        {st.mode === 'traces' && <TracesMode />}
      </main>
    </div>
  );
}
