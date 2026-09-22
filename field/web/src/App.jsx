import { lazy, Suspense, useEffect, useState } from 'react';
import { api } from './net/client.js';
import { setMode, useField } from './state/store.js';
import AtlasMode from './atlas/AtlasMode.jsx';
import EmptyState from './ui/EmptyState.jsx';

// Rome is a heavy illustrated map; it stays lazy even though it is the default.
const TheaterMode = lazy(() => import('./theater/TheaterMode.jsx'));

/* Two destinations and a gear, which is the whole navigation.

   There were seven: Board, Map, Rome, Project, Plans, Routines and History. Map was Rome
   with a different skin, Project was a folder's detail behind a route, Plans was an
   overlay on the territory, Routines was a settings panel and History was a time control
   on the map. Each of those is now where it belongs, and the row says the two things
   Field actually shows: a territory and every conversation on it. */
const NAV = [
  { id: 'rome', name: 'Rome', key: 'O' },
  { id: 'atlas', name: 'Atlas', key: 'A' },
];

export default function App() {
  const st = useField();
  const [loggedOut, setLoggedOut] = useState(false);
  const [logoutError, setLogoutError] = useState(null);
  const [tallyOpen, setTallyOpen] = useState(false);

  useEffect(() => {
    const onKey = (e) => {
      const tag = document.activeElement?.tagName;
      if (['INPUT', 'TEXTAREA', 'SELECT'].includes(tag) || document.activeElement?.isContentEditable) return;
      if (e.defaultPrevented || e.repeat || !e.altKey || !e.ctrlKey || e.metaKey) return;
      const pressed = e.key.toLowerCase();
      const hit = NAV.find((m) => m.key.toLowerCase() === pressed);
      if (hit) {
        e.preventDefault();
        setMode(hit.id);
        return;
      }
      // N — start an agent. Both screens answer it where they stand.
      if (pressed === 'n') {
        e.preventDefault();
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
              className={`mode segctl-item${st.mode === m.id ? ' on' : ''}`}
              onClick={() => setMode(m.id)}
              type="button"
              aria-current={st.mode === m.id ? 'page' : undefined}
              aria-keyshortcuts={`Control+Alt+${m.key}`}
            >
              {m.name}
              <kbd className="segctl-hint">Ctrl+Alt+{m.key}</kbd>
            </button>
          ))}
        </nav>

        {/* One signal, not six chips.

            The topbar used to print: awaiting approval, active, sessions, spent,
            reserved, and costs unconfirmed — six counters competing for the same corner
            of the eye, five of which were ledger. What is left is the one thing that
            decides whether you look: a badge with a number when something needs you,
            otherwise how many agents are working. The ledger is one tap below. */}
        <div className="topstats" aria-live="polite" aria-atomic="true">
          <button
            type="button"
            className={`tally${waiting > 0 ? ' warn' : ''}${tallyOpen ? ' on' : ''}`}
            aria-expanded={tallyOpen}
            onClick={() => setTallyOpen((open) => !open)}
          >
            {waiting > 0
              ? <><span className="tally-badge mono">{waiting}</span>{waiting === 1 ? 'needs you' : 'need you'}</>
              : <><i aria-hidden="true" />{live.length ? `${live.length} working` : 'all quiet'}</>}
          </button>
          {tallyOpen && (
            <div className="tally-sheet" role="group" aria-label="Field totals">
              <p><span className="label">working</span><b className="mono">{live.length}</b></p>
              <p><span className="label">sessions</span><b className="mono">{realSessionCount}</b></p>
              <p><span className="label">spent</span><b className="mono">${(st.snap.totals?.costUsd ?? 0).toFixed(3)}</b></p>
              {reservedUsd > 0 && <p><span className="label">reserved</span><b className="mono">${reservedUsd.toFixed(2)}</b></p>}
              {unknownCosts > 0 && (
                <p className="bad"><span className="label">unconfirmed</span><b className="mono">{unknownCosts}</b></p>
              )}
            </div>
          )}
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
        {st.mode === 'atlas' && <AtlasMode />}
        {st.mode === 'rome' && (
          <Suspense fallback={<EmptyState status title="Raising Rome…">The illustrated operations map is loading.</EmptyState>}>
            <TheaterMode />
          </Suspense>
        )}
      </main>
    </div>
  );
}
