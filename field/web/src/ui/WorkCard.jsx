/* The one card. The Board and the Map strip used to keep two card components and two
   state vocabularies for the same project-and-agent facts; both now render .work-card,
   and the Map strip renders .work-card.compact — the same shell at 24px rows, without
   the transcript. Four tones, one word list, one padding rhythm. */

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

/* The card shell. Slots, not variants: whichever screen fills them decides what the
   card is about. `compact` is the Map strip — narrower, 24px rows, no transcript. */
export default function WorkCard({
  tone = 'idle',
  compact = false,
  selected = false,
  needsYou = false,
  hollow = false,
  ariaLabel,
  role = 'listitem',
  mark = null,
  title,
  subtitle = null,
  pill = null,
  action = null,
  meta = null,
  notices = null,
  children = null,
  footer = null,
  onTitleClick = null,
  titleTitle = undefined,
}) {
  const heading = onTitleClick
    ? <button type="button" className="work-who" onClick={onTitleClick} title={titleTitle}>
        {mark}
        <span className="work-who-text"><b>{title}</b>{subtitle && <small className="work-model">{subtitle}</small>}</span>
      </button>
    : <div className="work-who idle">
        {mark}
        <span className="work-who-text"><b>{title}</b>{subtitle && <small className="work-model">{subtitle}</small>}</span>
      </div>;

  return (
    <article
      className={`work-card tone-${tone}${compact ? ' compact' : ''}${selected ? ' selected' : ''}${needsYou ? ' needs-you' : ''}${hollow ? ' hollow' : ''}`}
      role={role}
      aria-label={ariaLabel}
    >
      <header className="work-card-head">
        {heading}
        {pill}
        {action}
      </header>
      {meta}
      {notices}
      {children != null && <div className="work-card-body">{children}</div>}
      {footer && <footer className="work-card-foot">{footer}</footer>}
    </article>
  );
}
