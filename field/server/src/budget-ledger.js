// Replayable dollar reservations. Missing cost is never treated as zero cost.
// Unsettled interrupted work retains its reservation because its final bill is unknown.
export class BudgetLedger {
  constructor() { this.reservations = new Map(); }

  apply(event) {
    const d = event.data ?? {};
    if (event.source === 'synthetic' || d.simulated) return;
    if (event.kind === 'budget.reserved') {
      if (!this.reservations.has(d.sessionId) && Number.isFinite(d.limitUsd) && d.limitUsd > 0) {
        this.reservations.set(d.sessionId, { sessionId: d.sessionId, campaignId: d.campaignId ?? null,
          limitUsd: d.limitUsd, spentUsd: null, terminal: false, settled: false });
      }
      return;
    }
    const entry = this.reservations.get(d.sessionId);
    if (!entry) return;
    if (event.kind === 'budget.reactivated') { entry.terminal = false; entry.settled = false; }
    if (event.kind === 'session.usage' && Number.isFinite(d.costUsd) && d.costUsd >= 0) {
      entry.spentUsd = Math.max(entry.spentUsd ?? 0, d.costUsd);
    }
    if (event.kind === 'session.turn_complete') entry.settled = true;
    if (event.kind === 'session.state' && ['spawning', 'thinking', 'working'].includes(d.state)) {
      entry.settled = false; entry.terminal = false;
    }
    if (event.kind === 'session.ended' || (event.kind === 'session.state' && d.state === 'interrupted')) entry.terminal = true;
  }

  reserved(entry) {
    if (entry.terminal && entry.settled && entry.spentUsd !== null) return 0;
    return Math.max(0, entry.limitUsd - (entry.spentUsd ?? 0));
  }

  campaign(campaignId, spentUsd = 0, limitUsd = 0) {
    const entries = [...this.reservations.values()].filter(e => e.campaignId === campaignId);
    const reservedUsd = entries.reduce((sum, entry) => sum + this.reserved(entry), 0);
    return { spentUsd, reservedUsd, remainingUsd: Math.max(0, limitUsd - spentUsd - reservedUsd),
      unknownCostSessions: entries.filter(e => e.spentUsd === null || (e.terminal && !e.settled)).length };
  }

  snapshot() {
    return [...this.reservations.values()].map(entry => ({ ...entry, reservedUsd: this.reserved(entry),
      costStatus: entry.spentUsd === null ? 'missing' : entry.spentUsd === 0 ? 'reported_zero' : 'reported',
      finalCostKnown: entry.terminal && entry.settled && entry.spentUsd !== null }));
  }
}
