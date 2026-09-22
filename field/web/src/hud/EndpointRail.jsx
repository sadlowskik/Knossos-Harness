import { useField } from '../state/store.js';

/* The rail keeps status and spend; the modal owns the detail. An endpoint used to be
   described in three places — this rail, a Models tab in settings, and the PowerSources
   modal — so the rows are buttons now and all three lead to the same one surface. */
export default function EndpointRail({ onOpen = null }) {
  const st = useField();
  const sessions = st.snap.sessions.filter(
    (s) => !['done', 'cancelled', 'interrupted'].includes(s.state),
  );

  return (
    <div className="rail">
      <span className="label rail-title">endpoints</span>
      {st.snap.endpoints.map((e) => {
        const users = sessions.filter((s) => s.endpointId === e.id);
        return (
          <button
            key={e.id}
            type="button"
            className={`ep ${e.status}`}
            title={e.detail ? `${e.detail} — open in Models` : 'Open in Models'}
            onClick={() => onOpen?.(e.id)}
            disabled={!onOpen}
          >
            <span className={`dot ${e.status}`} />
            <span className="ep-name">{e.name}</span>
            <span className="ep-meta mono">{e.status}</span>
            {users.length > 0 && (
              <span className="ep-load mono" title={`${users.length} session(s) on this endpoint`}>
                ×{users.length}
              </span>
            )}
          </button>
        );
      })}

      <div className="rail-spacer" />

      <div className="rail-totals mono">
        <span title="sessions currently live">{sessions.length} live</span>
        <span title="total spend recorded in the event log">
          ${(st.snap.totals?.costUsd ?? 0).toFixed(4)}
        </span>
        <span title="events in the log">{st.snap.seq} evt</span>
        <span className={st.connected ? 'ok' : 'bad'}>
          {st.connected ? 'connected' : 'reconnecting…'}
        </span>
      </div>
    </div>
  );
}
