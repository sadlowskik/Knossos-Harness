/* One empty state, one loading state: a mark, one line, one action. Bare strings like
   "Loading map…" and centred sentences in a list are the same thing said worse. */

export default function EmptyState({ title, children = null, action = null, status = false, className = '' }) {
  return (
    <div className={`atlas-empty${className ? ` ${className}` : ''}`} role={status ? 'status' : undefined}>
      <span className={`atlas-empty-mark${status ? ' pending' : ''}`} aria-hidden="true" />
      <b>{title}</b>
      {children && <p>{children}</p>}
      {action && <div className="atlas-empty-actions">{action}</div>}
    </div>
  );
}
