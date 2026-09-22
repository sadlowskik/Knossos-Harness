import { lazy, Suspense, useEffect, useState } from 'react';
import { api } from './net/client.js';
import { setMode, useField } from './state/store.js';
import RoutinesMode from './routines/RoutinesMode.jsx';
import TracesMode from './traces/TracesMode.jsx';
import CampaignMode from './campaigns/CampaignMode.jsx';
import TheaterMode from './theater/TheaterMode.jsx';

const WorkspaceMode = lazy(() => import('./workspace/WorkspaceMode.jsx'));
// The canvas RTS renderer is an opt-in lens; lazy so it never weighs on the default path.
const FieldMode = lazy(() => import('./field/FieldMode.jsx'));

const MODES = [
  { id: 'theater', name: 'Field', key: 'F' },
  { id: 'routines', name: 'Routines', key: 'R' },
  { id: 'traces', name: 'History', key: 'T' },
];
const FIELD_LENSES = [
  { id: 'theater', name: 'Board', key: 'V' },
  { id: 'rts', name: 'Map', key: 'G' },
  { id: 'workspace', name: 'Project', key: 'C' },
  { id: 'campaigns', name: 'Plans', key: 'S' },
];
const FIELD_MODES = new Set(['theater', 'field', 'rts', 'campaigns', 'workspace']);

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
        return;
      }
      // N — start an agent. Whichever Field screen is mounted opens its starter on the
      // first mounted project; from Routines or History, come back to the Board first.
      if (e.key.toLowerCase() === 'n') {
        e.preventDefault();
        if (!FIELD_MODES.has(st.mode)) setMode('theater');
        requestAnimationFrame(() => window.dispatchEvent(new CustomEvent('field:start-agent')));
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [st.mode]);

  const live = st.snap.sessions.filter(
    (s) => !['done', 'cancelled', 'interrupted', 'error'].includes(s.state),
  );
  const waiting = st.snap.permissions?.length ?? 0;
  const realSessionCount = st.snap.sessions.length;
  const budgetReservations = st.snap.budgetReservations ?? [];
  const reservedUsd = budgetReservations.reduce((sum, row) => sum + row.reservedUsd, 0);
  const unknownCosts = budgetReservations.filter(row => row.costStatus === 'missing' || (row.terminal && !row.finalCostKnown)).length;

  if (loggedOut) return <main className="stage"><p role="status">Signed out. Restart Field and open its new bootstrap URL to sign in again.</p></main>;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <span className="brand-name">Field</span>
          <span className="brand-sub">{st.config?.field?.name ?? 'connecting…'}</span>
        </div>

        <nav className="modes segctl" aria-label="Mode">
          {MODES.map((m) => (
            <button
              key={m.id}
              className={`mode segctl-item${m.id === 'theater' ? (FIELD_MODES.has(st.mode) ? ' on' : '') : (st.mode === m.id ? ' on' : '')}`}
              onClick={() => setMode(m.id)}
              type="button"
              aria-current={m.id === 'theater' ? (FIELD_MODES.has(st.mode) ? 'page' : undefined) : (st.mode === m.id ? 'page' : undefined)}
              aria-keyshortcuts={`Control+Alt+${m.key}`}
            >
              {m.name}
              <kbd className="segctl-hint">Ctrl+Alt+{m.key}</kbd>
            </button>
          ))}
        </nav>

        {FIELD_MODES.has(st.mode) && (
          <nav className="field-lenses segctl" aria-label="Field view">
            {FIELD_LENSES.map((lens) => (
              <button key={lens.id} className={`segctl-item${st.mode === lens.id || (lens.id === 'theater' && st.mode === 'field') ? ' on' : ''}`} onClick={() => setMode(lens.id)} type="button" aria-current={st.mode === lens.id || (lens.id === 'theater' && st.mode === 'field') ? 'page' : undefined} aria-keyshortcuts={`Control+Alt+${lens.key}`}>
                {lens.name}<kbd className="segctl-hint">Ctrl+Alt+{lens.key}</kbd>
              </button>
            ))}
          </nav>
        )}

        <div className="topstats" aria-live="polite" aria-atomic="true">
          {waiting > 0 && <span className="stat warn"><i aria-hidden="true" />{waiting} awaiting approval</span>}
          <span className="stat"><i aria-hidden="true" />{live.length} active</span>
          <span className="stat quiet">{realSessionCount} {realSessionCount === 1 ? 'session' : 'sessions'}</span>
          <span className="stat quiet mono">${(st.snap.totals?.costUsd ?? 0).toFixed(3)}</span>
          {reservedUsd > 0 && <span className="stat quiet mono">${reservedUsd.toFixed(2)} reserved</span>}
          {unknownCosts > 0 && <span className="stat warn"><i aria-hidden="true" />{unknownCosts} costs unconfirmed</span>}
        </div>

        <button type="button" className="topbar-signout" onClick={async () => {
          try { await api.logout(); setLoggedOut(true); }
          catch { setLogoutError('Sign out failed. Check the Field server and try again.'); }
        }}>Sign out</button>
      </header>

      <main className="stage">
        {logoutError && <p role="alert" className="stage-alert">{logoutError}</p>}
        {st.error && (
          <div className="fatal" role="alert">
            <div className="fatal-card">
              <b>Can't reach the Field server</b>
              <p>Field keeps trying in the background. Check that the server is running, then this page will recover on its own.</p>
              <code>{st.error}</code>
            </div>
          </div>
        )}
        {(st.mode === 'theater' || st.mode === 'field') && <TheaterMode />}
        {st.mode === 'rts' && (
          <Suspense fallback={<p role="status" className="stage-status">Loading map…</p>}>
            <FieldMode />
          </Suspense>
        )}
        {st.mode === 'campaigns' && <CampaignMode />}
        {st.mode === 'workspace' && (
          <Suspense fallback={<p role="status" className="stage-status">Loading project…</p>}>
            <WorkspaceMode />
          </Suspense>
        )}
        {st.mode === 'routines' && <RoutinesMode />}
        {st.mode === 'traces' && <TracesMode />}
      </main>
    </div>
  );
}
