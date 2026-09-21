// The Oracle verdict a harness reported for a session: one row per tier with its
// pass / skip / forgiven state. Forgiven tiers were already failing before the change,
// so they say nothing about it either way and are labelled as such.

export function tierState(tier) {
  if (tier?.skipped) return 'skipped';
  if (tier?.forgiven) return 'forgiven';
  return tier?.passed ? 'passed' : 'failed';
}

const STATE_WORD = {
  passed: 'pass', failed: 'FAIL', skipped: 'skipped', forgiven: 'forgiven',
};

export default function VerdictLadder({ verdict, compact = false }) {
  if (!verdict || typeof verdict !== 'object') return null;
  const tiers = Array.isArray(verdict.tiers) ? verdict.tiers : [];
  const forgiven = Number(verdict.forgivenCount ?? tiers.filter((t) => t?.forgiven).length) || 0;
  const tone = verdict.passed ? 'passed' : 'failed';
  return (
    <div className={`verdict ${tone}${compact ? ' compact' : ''}`} role="group" aria-label="Verification verdict">
      <div className="verdict-head">
        <span className={`verdict-pill ${tone}`}>{verdict.passed ? 'verified' : 'not verified'}</span>
        {verdict.dryRun && <span className="verdict-note">dry run · nothing was written</span>}
        {verdict.reachedTier != null && <span className="verdict-note">reached tier {verdict.reachedTier}</span>}
        {verdict.ts && <span className="verdict-note">{new Date(verdict.ts).toLocaleTimeString()}</span>}
      </div>
      {verdict.summary && !compact && <p className="verdict-summary">{String(verdict.summary).slice(0, 400)}</p>}
      {tiers.length > 0 && (
        <ol className="verdict-tiers mono">
          {tiers.map((tier, i) => {
            const state = tierState(tier);
            return (
              <li key={`${tier?.tier ?? i}-${i}`} className={`verdict-tier ${state}`}>
                <span className="verdict-tier-mark" aria-hidden="true" />
                <span className="verdict-tier-label">{tier?.label ?? `tier ${tier?.tier ?? i}`}</span>
                <span className="verdict-tier-state">{STATE_WORD[state]}</span>
                {!compact && tier?.detail && state !== 'passed' && (
                  <span className="verdict-tier-detail" title={tier.detail}>{String(tier.detail).slice(0, 160)}</span>
                )}
              </li>
            );
          })}
        </ol>
      )}
      {forgiven > 0 && (
        <p className="verdict-forgiven">
          {forgiven} forgiven tier{forgiven === 1 ? '' : 's'}: already failing before this change, so not evidence about it.
        </p>
      )}
      {!compact && Array.isArray(verdict.residualRisk) && verdict.residualRisk.length > 0 && (
        <div className="verdict-list"><span className="label">residual risk</span><ul>{verdict.residualRisk.map((r, i) => <li key={i}>{String(r)}</li>)}</ul></div>
      )}
      {!compact && Array.isArray(verdict.recovery) && verdict.recovery.length > 0 && (
        <div className="verdict-list"><span className="label">recovery</span><ul>{verdict.recovery.map((r, i) => <li key={i}>{String(r)}</li>)}</ul></div>
      )}
    </div>
  );
}
