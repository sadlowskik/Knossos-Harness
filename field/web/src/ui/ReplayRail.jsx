/* The one replay control.

   It used to be a raw `<input type="range">` with no class on it — OS blue, OS track,
   the platform's thumb — followed by `3811 / 3811`, then the moment in words, then a
   disabled `End`. Three of those four are the same fact said differently, and the
   sequence numbers are a debug value: an operator does not scrub to event 3,811.

   What is left is a drawn control and one label. The track carries the position as a
   filled portion, which is the same information `pos / last` was printing, and the
   caller supplies the single word or timestamp that says where you are. */

export default function ReplayRail({
  label = null,
  min = 1,
  max = 1,
  value,
  onChange,
  ariaLabel = 'Replay position',
  ariaValueText = null,
  className = '',
  disabled = false,
}) {
  const last = Math.max(min, max);
  const position = Math.min(Math.max(value ?? min, min), last);
  const fill = last > min ? ((position - min) / (last - min)) * 100 : 100;
  return (
    <div className={`replay-rail${className ? ` ${className}` : ''}`}>
      <input
        className="rail-slider"
        type="range"
        min={min}
        max={last}
        value={position}
        disabled={disabled}
        aria-label={ariaLabel}
        aria-valuetext={ariaValueText ?? (label || undefined)}
        style={{ '--rail-fill': `${fill.toFixed(2)}%` }}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      {label && <b className="rail-label">{label}</b>}
    </div>
  );
}
