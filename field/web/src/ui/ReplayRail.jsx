/* The one replay control. History and Plans each grew their own scrubber over the same
   event log; this is that control, once: a label, the range, `pos / last`, and End. */

export default function ReplayRail({
  label = 'replay',
  min = 1,
  max = 1,
  value,
  onChange,
  onEnd = null,
  detail = null,
  ariaLabel = 'Replay position',
  className = '',
  disabled = false,
}) {
  const last = Math.max(min, max);
  const position = Math.min(Math.max(value ?? min, min), last);
  return (
    <div className={`replay-rail${className ? ` ${className}` : ''}`}>
      <span className="label replay-rail-label">{label}</span>
      <input
        type="range"
        min={min}
        max={last}
        value={position}
        disabled={disabled}
        aria-label={ariaLabel}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      <span className="mono replay-rail-pos">{position} / {last}</span>
      {detail && <b className="replay-rail-detail">{detail}</b>}
      <button
        type="button"
        className="btn sm"
        disabled={disabled || position >= last}
        onClick={() => (onEnd ? onEnd() : onChange(last))}
      >End</button>
    </div>
  );
}
