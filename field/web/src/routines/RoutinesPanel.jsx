/* Routines, behind the gear.

   Routines were a destination, which said they were a place you go. They are not: they
   are a setting — which standing work is armed, when it last fired and what came of it.
   So this is the same panel, whole, inside Field settings: the enable switch with its
   risk confirmation, the trigger, the next scheduled run, the last outcome, the run
   history and Run now. Nothing about it changed except where it lives. */

import { useEffect, useState } from 'react';
import { api } from '../net/client.js';
import { useField } from '../state/store.js';

export default function RoutinesPanel() {
  const st = useField();
  const [states, setStates] = useState({});
  const [details, setDetails] = useState({});
  const [busy, setBusy] = useState({});
  const [err, setErr] = useState(null);
  const routines = st.config?.routines ?? [];

  useEffect(() => {
    let mounted = true;
    const refresh = () => api.routineStates().then((r) => {
      if (!mounted) return;
      setStates(r.states); setDetails(r.details ?? {});
    }).catch(() => {});
    refresh();
    const timer = setInterval(refresh, 15_000);
    return () => { mounted = false; clearInterval(timer); };
  }, []);

  const toggle = async (id) => {
    const next = !states[id];
    const confirmRisk = next
      ? window.confirm('Enable this routine? It can start model sessions and use the configured tools, workspace, endpoint, and budget whenever its trigger fires.')
      : false;
    if (next && !confirmRisk) return;
    setBusy((b) => ({ ...b, [id]: true }));
    setErr(null);
    try {
      const r = await api.toggleRoutine(id, next, confirmRisk);
      setStates(r.states);
      setDetails(r.details ?? {});
    } catch (e) { setErr(e.message); }
    finally { setBusy((b) => ({ ...b, [id]: false })); }
  };

  const runNow = async (id) => {
    setBusy((b) => ({ ...b, [id]: true }));
    setErr(null);
    try { await api.runRoutine(id); }
    catch (e) { setErr(e.message); }
    finally { setBusy((b) => ({ ...b, [id]: false })); }
  };

  return (
    <section className="routines-panel">
      <label>Routines</label>
      <p className="settings-help">
        Persistent and scheduled work. Each routine is a file in <code>field/routines/</code> and
        lives in Git. Enablement and run outcomes are retained in the Field event log. A routine
        that fires spawns a real session with the role, endpoint, budget, and orders written here.
      </p>

      {err && <div className="ctx-err" role="alert">{err}</div>}

      {routines.length === 0 && (
        <div className="empty"><b>No routines defined.</b>Add a YAML file under <code>field/routines/</code> with a trigger (cron or file change), a role, orders and a budget, then restart Field. It appears here with an enable switch and a Run now button.</div>
      )}

      {routines.map((r) => {
        const spawned = st.snap.sessions.filter((s) => s.routineId === r.id);
        const on = !!states[r.id];
        const projected = st.snap.routines?.find((item) => item.id === r.id);
        const runtime = { ...details[r.id], ...projected };
        return (
          <div className="rt" key={r.id}>
            <div className="rt-head">
              <button
                className={`toggle${on ? ' on' : ''}`}
                onClick={() => toggle(r.id)}
                disabled={busy[r.id]}
                type="button"
                aria-label={`${on ? 'Disable' : 'Enable'} ${r.name ?? r.id} routine`}
                title={on ? 'enabled — will fire on its trigger' : 'disabled — inert'}
              ><i /></button>

              <span className="rt-name">{r.name ?? r.id}</span>
              <span style={{ flex: '1 1 auto' }} />
              {spawned.length > 0 && (
                <span className="label">{spawned.length} run{spawned.length > 1 ? 's' : ''}</span>
              )}
              <button
                className="btn sm"
                onClick={() => runNow(r.id)}
                disabled={busy[r.id]}
                type="button"
              >Run now</button>
            </div>

            <div className="rt-body">
              <span className="rt-trigger">
                {r.trigger?.kind === 'schedule'
                  ? `cron ${r.trigger.cron}`
                  : r.trigger?.kind === 'fs_change'
                    ? `on change · ${(r.trigger.paths ?? []).join(', ')} · ${r.trigger.debounce_seconds ?? 60}s debounce`
                    : r.trigger?.kind ?? 'manual'}
              </span>
              <div className="rt-orders">{r.orders?.trim() ?? '(no orders)'}</div>
              <div className="rt-meta">
                <Meta k="role" v={r.role} />
                <Meta k="endpoint" v={r.endpoint} />
                <Meta k="thinking" v={r.thinking} />
                <Meta k="workspace" v={r.workspace} />
                <Meta k="budget" v={r.budget_usd != null ? `$${r.budget_usd.toFixed(2)}` : '—'} />
                <Meta k="completion" v={r.completion?.kind ?? '—'} />
                <Meta k="timezone" v={runtime.timezone ?? 'UTC'} />
                <Meta k="last run" v={formatTime(runtime.lastRunAt)} />
                <Meta k="next scheduled run" v={runtime.nextRunAt ? formatTime(runtime.nextRunAt) : 'none'} />
                <Meta k="owner" v={runtime.currentOwner ?? 'none'} />
                <Meta k="outcome" v={runtime.activeRunId ? 'running' : runtime.lastOutcome ?? 'never run'} />
              </div>
              {(runtime.history ?? projected?.history ?? []).length > 0 && (
                <details><summary>Recent run history</summary><ol>
                  {(runtime.history ?? projected?.history ?? []).slice(-20).reverse().map((run, i) => (
                    <li key={`${run.runId}-${i}`}>{formatTime(run.at)} — {run.status}{run.reason ? `: ${run.reason}` : ''}</li>
                  ))}
                </ol></details>
              )}
            </div>
          </div>
        );
      })}
    </section>
  );
}

function formatTime(value) {
  if (!value) return 'never';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? 'unknown' : date.toLocaleString();
}

function Meta({ k, v }) {
  return (
    <div>
      <div className="label">{k}</div>
      <div className="mono rt-meta-value">{v ?? '—'}</div>
    </div>
  );
}
