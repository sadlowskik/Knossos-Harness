/* The shared vocabulary for a piece of work.

   This was the one card, plus the parts a card is made of. The card shell itself belonged
   to the canvas Map's strip, which is gone, so what is left is what Rome and Atlas both
   still use: the plain-language state, the pill that says it, an agent's mark, and the
   flush-set fact row. Four tones, one word list. */

import { initials, identityHue } from '../theater/fieldPreferences.js';

// Raw session states → plain language + a semantic tone the stylesheet knows about.
// tone: working | attention | done | failed | idle
export function plainState(session) {
  const state = session?.state ?? 'idle';
  if (!session) return { label: 'idle', tone: 'idle' };
  if (state === 'waiting_permission' || session.pendingPermission) return { label: 'needs approval', tone: 'attention' };
  if (state === 'blocked') return { label: 'blocked', tone: 'attention' };
  if (state === 'error') return { label: 'failed', tone: 'failed' };
  if (state === 'done') return { label: 'done', tone: 'done' };
  if (state === 'cancelled' || state === 'interrupted') return { label: 'stopped', tone: 'idle' };
  if (state === 'thinking') return { label: 'thinking', tone: 'working' };
  if (['running', 'working', 'active', 'spawning', 'starting'].includes(state)) return { label: 'working', tone: 'working' };
  if (state === 'paused') return { label: 'paused', tone: 'idle' };
  if (state === 'ready') return { label: 'ready', tone: 'idle' };
  return { label: 'idle', tone: 'idle' };
}

/* What the agent is doing right now, as one plain sentence.

   This replaced the fact row the footer used to lead with — tool, progress, elapsed,
   context, spent — which asked the operator to read five labelled numbers and assemble
   the sentence themselves. The numbers are all still there, one tap down under
   "Details"; this is the thing you actually wanted to know. */

const TOOL_VERB = [
  [/(edit|write|str_?replace|notebook|patch|create)/i, 'Editing'],
  [/(bash|shell|powershell|terminal|exec|command|run)/i, 'Running a command'],
  [/(grep|search|glob|find|list|ls)/i, 'Searching'],
  [/(read|view|open|cat)/i, 'Reading'],
  [/(fetch|web|http|browser|navigate|url)/i, 'Looking something up'],
  [/(task|agent|delegate|spawn|subagent)/i, 'Delegating to another agent'],
];

const leaf = (path) => String(path ?? '').replaceAll('\\', '/').split('/').filter(Boolean).at(-1) ?? '';

export function plainActivity(session, { connected = true } = {}) {
  if (!connected) return 'Not connected to the Field server.';
  if (!session) return 'Nothing is running.';
  const state = session.state ?? 'idle';
  if (state === 'waiting_permission' || session.pendingPermission) return 'Waiting for you to approve something.';
  if (state === 'blocked') return session.stateDetail ? String(session.stateDetail).slice(0, 160) : 'Blocked, waiting on you.';
  if (state === 'error') return String(session.error ?? session.lastError ?? 'It ran into an error.').slice(0, 160);
  if (state === 'done') return session.result ? String(session.result).slice(0, 160) : 'Finished.';
  if (state === 'cancelled' || state === 'interrupted') return 'Stopped.';
  if (state === 'paused') return session.budgetExhausted ? 'Paused — it spent its budget.' : 'Paused.';
  if (state === 'thinking') return 'Thinking it through.';

  const tool = typeof session.lastTool === 'string' ? session.lastTool : session.lastTool?.name;
  const file = leaf(session.focusPath);
  if (tool) {
    const verb = TOOL_VERB.find(([pattern]) => pattern.test(tool))?.[1];
    if (verb === 'Running a command') return 'Running a command.';
    if (verb) return file ? `${verb} ${file}.` : `${verb}.`;
    return file ? `Working on ${file}.` : `Using ${tool}.`;
  }
  if (file) return `Working on ${file}.`;
  if (['spawning', 'starting'].includes(state)) return 'Starting up.';
  if (['running', 'working', 'active'].includes(state)) return 'Working.';
  if (state === 'ready') return 'Ready for an assignment.';
  return 'Idle.';
}

/** The state word in its tone. The only place a session state is spelled out. */
export function StatusPill({ session, state = null, compact = false }) {
  const shown = state ?? plainState(session);
  return (
    <span className={`work-pill tone-${shown.tone}${compact ? ' compact' : ''}`}>
      <i aria-hidden="true" />{shown.label}
    </span>
  );
}

/** An agent's emblem, or a project's initials when no agent has claimed the card. */
export function AgentMark({ identity, name = null, project = false, size = 'md' }) {
  const label = identity?.displayName ?? name ?? '?';
  if (!project && identity?.iconUrl) {
    return <img className={`work-mark size-${size}`} src={identity.iconUrl} alt="" />;
  }
  if (project) {
    return <span className={`work-mark size-${size} is-project`} aria-hidden="true">{initials(label)}</span>;
  }
  const hue = identityHue(label);
  return (
    <span
      className={`work-mark size-${size}`}
      style={{ background: `hsl(${hue} 30% 26%)`, color: `hsl(${hue} 60% 86%)` }}
      aria-hidden="true"
    >{initials(label)}</span>
  );
}

/* Flush-set mono facts. No chips, no bordered pills: a label in the label style and a
   value in the data style, separated by hairlines rather than boxes. */
export function MetaRow({ items = [], className = '' }) {
  const rows = items.filter((item) => item && item.value != null && item.value !== '');
  if (!rows.length) return null;
  return (
    <div className={`work-meta${className ? ` ${className}` : ''}`}>
      {rows.map((item) => (
        <span
          key={item.key ?? item.label}
          className={`work-meta-item${item.tone ? ` tone-${item.tone}` : ''}`}
          title={item.title ?? undefined}
        >
          {item.label && <span className="work-meta-label">{item.label}</span>}
          {item.onClick
            ? <button type="button" className="work-meta-value work-meta-action" onClick={item.onClick}>{item.value}</button>
            : <span className="work-meta-value">{item.value}</span>}
        </span>
      ))}
    </div>
  );
}

