import { useField } from '../state/store.js';

export default function EndpointRail() {
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
          <div key={e.id} className={`ep ${e.status}`} title={e.detail ?? ''}>
            <span className={`dot ${e.status}`} />
            <span className="ep-name">{e.name}</span>
            <span className="ep-meta mono">
              {e.latencyMs != null ? `${e.latencyMs}ms` : e.status}
            </span>
            {users.length > 0 && (
              <span className="ep-load mono" title={`${users.length} session(s) on this endpoint`}>
                ×{users.length}
              </span>
            )}
          </div>
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
