import { lazy, Suspense, useEffect, useState } from 'react';
import { ChevronRight, Plus, Settings } from 'lucide-react';
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

const fire = (name, detail = null) => window.dispatchEvent(new CustomEvent(name, { detail }));

export default function App() {
  const st = useField();
  const [loggedOut, setLoggedOut] = useState(false);
  const [logoutError, setLogoutError] = useState(null);
  const [tallyOpen, setTallyOpen] = useState(false);

  // Signing out lives in the settings sheet now, one tap behind the gear, because the
  // bar has room for the five things an operator looks at and not for a sixth.
  useEffect(() => {
    const done = () => setLoggedOut(true);
    const failed = (event) => setLogoutError(event.detail ?? 'Sign out failed. Check the Field server and try again.');
    window.addEventListener('field:signed-out', done);
    window.addEventListener('field:signout-failed', failed);
    return () => {
      window.removeEventListener('field:signed-out', done);
      window.removeEventListener('field:signout-failed', failed);
    };
  }, []);

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

  /* The one bar. It was three: this one, Rome's 68px page header and Atlas's 44px action
     row, which between them printed the project's name three times and changed the
     chrome height by 68px when you switched tabs. What is left is the five things the
     first second is allowed to contain plus the two destinations: where you are (the
     breadcrumb), what needs you (the badge), and the one thing you do (start an agent).
     The page title is gone outright — the island's own plaque names the project. */
  const chrome = st.chrome ?? { crumbs: [], picker: null };
  const crumbs = chrome.crumbs ?? [];

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark" aria-hidden="true" />
          <span className="brand-name">Field</span>
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

        {/* Where you are. Rome fills it with the open folder's path; Atlas fills it with
            the project filter, which is the same question asked the other way round. */}
        <nav className="crumbs" aria-label="Where you are">
          {chrome.picker && (
            <label className="crumb-pick">
              <span className="label">{chrome.picker.label ?? 'project'}</span>
              <select
                value={chrome.picker.value}
                onChange={(event) => fire('field:pick-project', { value: event.target.value })}
              >
                {chrome.picker.options.map((option) => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
              </select>
            </label>
          )}
          {crumbs.map((crumb, index) => (
            <span key={crumb.key ?? crumb.label} className="crumb-step">
              {index > 0 && <ChevronRight aria-hidden="true" />}
              {index === crumbs.length - 1
                ? <b>{crumb.label}</b>
                : <button type="button" onClick={() => fire('field:navigate', { dir: crumb.dir })}>{crumb.label}</button>}
            </span>
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

        {/* The one primary per screen, and the gear. Both reach whichever screen is
            mounted through the same events the keyboard already used. */}
        <button
          type="button"
          className={`btn topbar-primary${chrome.primaryQuiet ? '' : ' primary'}`}
          disabled={chrome.primaryDisabled}
          title={chrome.primaryHint || undefined}
          aria-keyshortcuts="Control+Alt+N"
          onClick={() => fire('field:start-agent')}
        ><Plus aria-hidden="true" /><span>Start an agent</span></button>

        <button
          type="button"
          className="btn ghost icon topbar-gear"
          aria-label="Field settings"
          onClick={() => fire('field:open-settings')}
        ><Settings aria-hidden="true" /></button>
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
