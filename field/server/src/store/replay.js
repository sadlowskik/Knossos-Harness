// Page through the append-only log without assuming an upper event count. Keeping this
// in one helper prevents boot, trace export, and historical UI replay from drifting apart.
export function readEventRange(log, { from = 0, to = Infinity, pageSize = 10_000 } = {}) {
  const events = [];
  let cursor = Math.max(0, Number(from) || 0);
  const ceiling = Number.isFinite(Number(to)) ? Math.max(cursor, Number(to)) : Infinity;

  while (cursor < ceiling) {
    const batch = log.read(cursor, pageSize);
    if (!batch.length) break;
    for (const event of batch) {
      if (event.seq > ceiling) return events;
      events.push(event);
    }
    const next = batch.at(-1).seq;
    if (next <= cursor) break;
    cursor = next;
    if (batch.length < pageSize) break;
  }
  return events;
}

export function replayInto(log, projection, options = {}) {
  const events = readEventRange(log, options);
  for (const event of events) projection.apply(event);
  return { count: events.length, lastEvent: events.at(-1) ?? null };
}
