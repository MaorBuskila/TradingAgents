/** Verdict badge — renders BUY / HOLD / SELL with semantic color. */
export type Verdict = 'BUY' | 'HOLD' | 'SELL'

function verdictClass(v: string): string {
  const upper = v.toUpperCase()
  if (upper === 'BUY')  return 'buy'
  if (upper === 'SELL') return 'sell'
  return 'hold'
}

interface StatusBadgeProps {
  verdict: string
  size?: 'sm' | 'lg'
}

export default function StatusBadge({ verdict, size = 'sm' }: StatusBadgeProps) {
  const cls = verdictClass(verdict)
  if (size === 'lg') {
    return <span className={`verdict-badge ${cls}`}>{verdict.toUpperCase()}</span>
  }
  return <span className={`badge badge-${cls === 'buy' ? 'pos' : cls === 'sell' ? 'neg' : 'warn'}`}>{verdict.toUpperCase()}</span>
}

/** Full verdict banner shown at completion of an analysis run. */
interface VerdictBannerProps {
  verdict: string
  ticker: string
  reportId?: string | null
  onViewReport?: () => void
}

export function VerdictBanner({ verdict, ticker, reportId, onViewReport }: VerdictBannerProps) {
  const cls = verdictClass(verdict)
  return (
    <div className={`verdict-banner ${cls} fade-in`} role="status" aria-live="polite">
      <div>
        <div style={{ fontSize: 'var(--text-xs)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', opacity: 0.7, marginBottom: '0.25rem' }}>
          Final decision — {ticker}
        </div>
        <div className="verdict-label">{verdict.toUpperCase()}</div>
      </div>
      <div className="verdict-meta" style={{ flex: 1 }}>
        Analysis complete. The portfolio manager has issued a final verdict.
        {reportId && (
          <> Saved as report <code>{reportId}</code>.</>
        )}
      </div>
      {onViewReport && (
        <button type="button" className="btn btn-secondary btn-sm" onClick={onViewReport}>
          View full report
        </button>
      )}
    </div>
  )
}
