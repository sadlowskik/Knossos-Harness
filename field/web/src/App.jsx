import { lazy, Suspense, useEffect, useState } from 'react';
import { api } from './net/client.js';
import { setMode, useField } from './state/store.js';
import RoutinesMode from './routines/RoutinesMode.jsx';
import TracesMode from './traces/TracesMode.jsx';
import CampaignMode from './campaigns/CampaignMode.jsx';
import AtlasMode from './atlas/AtlasMode.jsx';
import EmptyState from './ui/EmptyState.jsx';

const WorkspaceMode = lazy(() => import('./workspace/WorkspaceMode.jsx'));
// The canvas RTS renderer is an opt-in lens; lazy so it never weighs on the default path.
const FieldMode = lazy(() => import('./field/FieldMode.jsx'));
// Rome is a heavy illustrated map. It used to be reachable only through a buried
// `theme` setting; it is a destination now, and still lazy.
const TheaterMode = lazy(() => import('./theater/TheaterMode.jsx'));

/* One row of destinations. There used to be two: a MODES row where "Field" was an alias
   for Board, and a FIELD_LENSES row underneath it — seven buttons to say seven things.
   `gap` marks the hairline after Rome, which separates the three live views of the work
   from the four surfaces you work in. */
const NAV = [
  { id: 'theater', name: 'Board', key: 'V', alias: 'F' },
  { id: 'rts', name: 'Map', key: 'G' },
  { id: 'rome', name: 'Rome', key: 'O' },
  { id: 'workspace', name: 'Project', key: 'C', gap: true },
  { id: 'campaigns', name: 'Plans', key: 'S' },
  { id: 'routines', name: 'Routines', key: 'R' },
  { id: 'traces', name: 'History', key: 'T' },
];
const FIELD_MODES = new Set(['theater', 'field', 'rome', 'rts', 'campaigns', 'workspace']);
const isOn = (id, mode) => mode === id || (id === 'theater' && mode === 'field');

export default function App() {
  const st = useField();
  const [loggedOut, setLoggedOut] = useState(false);
  const [logoutError, setLogoutError] = useState(null);

  useEffect(() => {
    const onKey = (e) => {
      const tag = document.activeElement?.tagName;
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(tag) || document.activeElement?.isContentEditable) return;
      if (e.defaultPrevented || e.repeat || !e.altKey || !e.ctrlKey || e.metaKey) return;
      const pressed = e.key.toLowerCase();
      const hit = NAV.find((m) => m.key.toLowerCase() === pressed || m.alias?.toLowerCase() === pressed);
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

        <nav className="modes segctl" aria-label="Field view">
          {NAV.map((m) => (
            <button
              key={m.id}
              className={`mode segctl-item${m.gap ? ' segctl-gap' : ''}${isOn(m.id, st.mode) ? ' on' : ''}`}
              onClick={() => setMode(m.id)}
              type="button"
              aria-current={isOn(m.id, st.mode) ? 'page' : undefined}
              aria-keyshortcuts={`Control+Alt+${m.key}`}
            >
              {m.name}
              <kbd className="segctl-hint">Ctrl+Alt+{m.key}</kbd>
            </button>
          ))}
        </nav>

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
        {(st.mode === 'theater' || st.mode === 'field') && <AtlasMode />}
        {st.mode === 'rts' && (
          <Suspense fallback={<EmptyState status title="Drawing the map…">The canvas renderer is loading. Agents appear on it as soon as it is ready.</EmptyState>}>
            <FieldMode />
          </Suspense>
        )}
        {st.mode === 'rome' && (
          <Suspense fallback={<EmptyState status title="Raising Rome…">The illustrated operations map is loading.</EmptyState>}>
            <TheaterMode />
          </Suspense>
        )}
        {st.mode === 'campaigns' && <CampaignMode />}
        {st.mode === 'workspace' && (
          <Suspense fallback={<EmptyState status title="Opening the project…">Files, changes and the terminal are loading.</EmptyState>}>
            <WorkspaceMode />
          </Suspense>
        )}
        {st.mode === 'routines' && <RoutinesMode />}
        {st.mode === 'traces' && <TracesMode />}
      </main>
    </div>
  );
}
