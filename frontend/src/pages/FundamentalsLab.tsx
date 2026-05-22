import { useState, useEffect, useRef, useCallback } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ComposedChart, Line, ReferenceLine,
} from 'recharts'
import { ExternalLink, RefreshCw, Square, TrendingUp } from 'lucide-react'
import { useTabLogger } from '../hooks/useTabLogger'

const API_BASE = (import.meta as any).env?.VITE_API_BASE ?? 'http://127.0.0.1:8000/api'
const SESSION_KEY = 'tradingagents_fundamentals_v1'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface CalendarRow {
  ticker: string
  short_name: string | null
  next_earnings_date: string | null
  days_until: number | null
  earnings_time: string | null
  eps_estimate: number | null
  revenue_estimate: number | null
  last_eps_actual: number | null
  last_surprise_pct: number | null
}

interface CalendarResponse {
  rows: CalendarRow[]
  upcoming_count: number
}

interface Ratios {
  ticker: string
  short_name: string | null
  sector: string | null
  industry: string | null
  current_price: number | null
  market_cap: number | null
  pe_ratio: number | null
  forward_pe: number | null
  peg_ratio: number | null
  price_to_book: number | null
  roe: number | null
  roa: number | null
  profit_margin: number | null
  operating_margin: number | null
  beta: number | null
  week52_high: number | null
  week52_low: number | null
  dividend_yield: number | null
}

interface IncomePeriod {
  period: string
  revenue: number | null
  gross_profit: number | null
  net_income: number | null
  eps: number | null
  operating_income: number | null
}

interface BalanceSheet {
  period: string | null
  total_assets: number | null
  current_assets: number | null
  total_liabilities: number | null
  stockholders_equity: number | null
  total_debt: number | null
  current_liabilities: number | null
  debt_to_equity: number | null
  current_ratio: number | null
}

interface CashFlow {
  period: string | null
  operating_cf: number | null
  free_cf: number | null
  capex: number | null
}

interface EarningsHistoryRow {
  date: string
  eps_estimate: number | null
  eps_actual: number | null
  surprise_pct: number | null
}

interface AnalystConsensus {
  strong_buy: number
  buy: number
  hold: number
  sell: number
  strong_sell: number
  period: string | null
  target_high: number | null
  target_low: number | null
  target_mean: number | null
  target_median: number | null
  num_analysts: number | null
}

interface TickerFundamentals {
  ticker: string
  ratios: Ratios
  income: IncomePeriod[]
  balance: BalanceSheet
  cashflow: CashFlow
  earnings_history: EarningsHistoryRow[]
  analyst: AnalystConsensus | null
}

interface SecFiling {
  accession_number: string
  form_type: string
  filing_date: string | null
  filing_url: string
  llm_summary: string | null
  summarized_at: string | null
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const fmtB = (v: number | null | undefined): string => {
  if (v == null) return '—'
  const abs = Math.abs(v)
  if (abs >= 1e12) return `$${(v / 1e12).toFixed(2)}T`
  if (abs >= 1e9) return `$${(v / 1e9).toFixed(2)}B`
  if (abs >= 1e6) return `$${(v / 1e6).toFixed(1)}M`
  return `$${v.toFixed(0)}`
}

const fmtPct = (v: number | null | undefined): string =>
  v == null ? '—' : `${(v * 100).toFixed(1)}%`

const fmtRatio = (v: number | null | undefined): string =>
  v == null ? '—' : v.toFixed(2)

const surpClr = (v: number | null): string => {
  if (v == null) return 'inherit'
  if (v > 0) return 'var(--success, #22c55e)'
  if (v < 0) return 'var(--danger, #ef4444)'
  return 'inherit'
}

const daysBadge = (days: number | null): string => {
  if (days == null) return '—'
  if (days < 0) return `${Math.abs(days)}d ago`
  if (days === 0) return 'Today'
  if (days === 1) return 'Tomorrow'
  return `${days}d`
}

const daysColor = (days: number | null): string => {
  if (days == null) return 'var(--text-muted)'
  if (days <= 3) return 'var(--danger, #ef4444)'
  if (days <= 7) return 'var(--warning, #f59e0b)'
  return 'var(--success, #22c55e)'
}

const incomeChartData = (income: IncomePeriod[]) =>
  [...income].reverse().map(p => ({
    period: p.period.slice(0, 7),
    Revenue: p.revenue ? +(p.revenue / 1e9).toFixed(2) : null,
    'Gross Profit': p.gross_profit ? +(p.gross_profit / 1e9).toFixed(2) : null,
    'Net Income': p.net_income ? +(p.net_income / 1e9).toFixed(2) : null,
  }))

const earningsChartData = (history: EarningsHistoryRow[]) =>
  [...history].reverse().map(h => ({
    date: h.date.slice(0, 7),
    'EPS Estimate': h.eps_estimate,
    'EPS Actual': h.eps_actual,
    'Surprise %': h.surprise_pct,
  }))

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function MetricCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div style={{
      background: 'var(--card-bg)',
      border: '1px solid var(--border)',
      borderRadius: 8,
      padding: '1rem 1.25rem',
      minWidth: 140,
      flex: '1 1 140px',
    }}>
      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: '1.25rem', fontWeight: 700 }}>{value}</div>
      {sub && <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <h3 style={{ fontSize: '0.9rem', fontWeight: 700, marginBottom: '0.75rem', color: 'var(--text)' }}>
      {children}
    </h3>
  )
}

function AnalystBar({ consensus }: { consensus: AnalystConsensus }) {
  const total = consensus.strong_buy + consensus.buy + consensus.hold + consensus.sell + consensus.strong_sell
  if (total === 0) return <p style={{ color: 'var(--text-muted)' }}>No analyst ratings available.</p>

  const segments = [
    { label: 'Str. Buy', count: consensus.strong_buy, color: '#16a34a' },
    { label: 'Buy', count: consensus.buy, color: '#22c55e' },
    { label: 'Hold', count: consensus.hold, color: '#f59e0b' },
    { label: 'Sell', count: consensus.sell, color: '#ef4444' },
    { label: 'Str. Sell', count: consensus.strong_sell, color: '#991b1b' },
  ]

  return (
    <div>
      <div style={{ display: 'flex', height: 24, borderRadius: 4, overflow: 'hidden', marginBottom: 8 }}>
        {segments.map(s => s.count > 0 && (
          <div
            key={s.label}
            title={`${s.label}: ${s.count}`}
            style={{ flex: s.count / total, background: s.color }}
          />
        ))}
      </div>
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', fontSize: '0.78rem' }}>
        {segments.map(s => (
          <span key={s.label} style={{ color: s.color, fontWeight: 600 }}>
            {s.label} {s.count}
          </span>
        ))}
        <span style={{ color: 'var(--text-muted)' }}>({total} analysts)</span>
        {consensus.target_mean && (
          <span style={{ color: 'var(--text)' }}>
            Avg target: <strong>${consensus.target_mean.toFixed(2)}</strong>
          </span>
        )}
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Earnings Calendar Tab
// ---------------------------------------------------------------------------

function EarningsCalendarTab({
  onTickerSelect,
}: {
  onTickerSelect: (ticker: string) => void
}) {
  const [data, setData] = useState<CalendarRow[]>([])
  const [upcomingCount, setUpcomingCount] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const log = useTabLogger('FundamentalsLab.Calendar')

  const fetchCalendar = useCallback(async (force = false) => {
    setLoading(true)
    setError(null)
    log('api:start', { endpoint: 'fundamentals/calendar', force })
    try {
      const res = await axios.get<CalendarResponse>(`${API_BASE}/fundamentals/calendar`, {
        params: force ? { force: true } : {},
      })
      setData(res.data.rows)
      setUpcomingCount(res.data.upcoming_count)
      log('api:success', { count: res.data.rows.length })
    } catch (err: any) {
      const msg = err?.response?.data?.detail || err?.message || 'Failed to load calendar'
      setError(msg)
      log('api:error', err)
    } finally {
      setLoading(false)
    }
  }, [log])

  useEffect(() => { fetchCalendar() }, [fetchCalendar])

  const thStyle: React.CSSProperties = {
    padding: '8px 12px', textAlign: 'left', fontWeight: 600,
    fontSize: '0.78rem', color: 'var(--text-muted)', whiteSpace: 'nowrap',
    borderBottom: '1px solid var(--border)',
  }
  const tdStyle: React.CSSProperties = {
    padding: '8px 12px', fontSize: '0.82rem',
    borderBottom: '1px solid var(--border)', whiteSpace: 'nowrap',
  }

  return (
    <div>
      {/* Alert banner */}
      {upcomingCount > 0 && (
        <div style={{
          padding: '0.75rem 1rem',
          background: 'var(--danger-soft, rgba(239,68,68,0.1))',
          border: '1px solid var(--danger, #ef4444)',
          borderRadius: 6,
          marginBottom: '1rem',
          fontSize: '0.85rem',
          fontWeight: 600,
          color: 'var(--danger, #ef4444)',
        }}>
          {upcomingCount} watchlist stock{upcomingCount > 1 ? 's' : ''} reporting within 3 days
        </div>
      )}

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
        <SectionTitle>Earnings Calendar — Watchlist</SectionTitle>
        <button
          onClick={() => fetchCalendar(true)}
          disabled={loading}
          className="btn"
          style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: '0.8rem' }}
        >
          <RefreshCw size={13} />
          Refresh
        </button>
      </div>

      {loading && (
        <div className="card" style={{ padding: '2rem', textAlign: 'center' }}>
          <p style={{ fontWeight: 600 }}>Loading earnings calendar…</p>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>Fetching from yfinance for all watchlist tickers</p>
        </div>
      )}

      {error && (
        <div className="card" style={{ padding: '1.25rem', borderLeft: '4px solid var(--danger)' }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {!loading && !error && data.length === 0 && (
        <div className="card" style={{ padding: '2rem', textAlign: 'center' }}>
          <p style={{ color: 'var(--text-muted)' }}>No watchlist tickers found. Add tickers in the Watchlist tab.</p>
        </div>
      )}

      {!loading && data.length > 0 && (
        <div className="card" style={{ padding: 0, overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse' }}>
            <thead>
              <tr>
                <th style={thStyle}>Ticker</th>
                <th style={thStyle}>Company</th>
                <th style={thStyle}>Earnings Date</th>
                <th style={thStyle}>Days Until</th>
                <th style={thStyle}>Time</th>
                <th style={{ ...thStyle, textAlign: 'right' }}>EPS Estimate</th>
                <th style={{ ...thStyle, textAlign: 'right' }}>Last EPS</th>
                <th style={{ ...thStyle, textAlign: 'right' }}>Surprise</th>
                <th style={{ ...thStyle, textAlign: 'right' }}>Rev Estimate</th>
                <th style={thStyle}>Action</th>
              </tr>
            </thead>
            <tbody>
              {data.map(row => (
                <tr key={row.ticker} style={{ cursor: 'pointer' }} onClick={() => onTickerSelect(row.ticker)}>
                  <td style={{ ...tdStyle, fontWeight: 700 }}>{row.ticker}</td>
                  <td style={{ ...tdStyle, maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {row.short_name || '—'}
                  </td>
                  <td style={tdStyle}>{row.next_earnings_date || '—'}</td>
                  <td style={{ ...tdStyle, fontWeight: 700, color: daysColor(row.days_until) }}>
                    {daysBadge(row.days_until)}
                  </td>
                  <td style={{ ...tdStyle, color: 'var(--text-muted)' }}>{row.earnings_time || '—'}</td>
                  <td style={{ ...tdStyle, textAlign: 'right' }}>
                    {row.eps_estimate != null ? `$${row.eps_estimate.toFixed(2)}` : '—'}
                  </td>
                  <td style={{ ...tdStyle, textAlign: 'right' }}>
                    {row.last_eps_actual != null ? `$${row.last_eps_actual.toFixed(2)}` : '—'}
                  </td>
                  <td style={{ ...tdStyle, textAlign: 'right', color: surpClr(row.last_surprise_pct), fontWeight: 600 }}>
                    {row.last_surprise_pct != null ? `${row.last_surprise_pct > 0 ? '+' : ''}${row.last_surprise_pct.toFixed(1)}%` : '—'}
                  </td>
                  <td style={{ ...tdStyle, textAlign: 'right' }}>
                    {row.revenue_estimate != null ? fmtB(row.revenue_estimate) : '—'}
                  </td>
                  <td style={tdStyle}>
                    <button
                      className="btn"
                      style={{ fontSize: '0.72rem', padding: '2px 8px' }}
                      onClick={e => { e.stopPropagation(); onTickerSelect(row.ticker) }}
                    >
                      Deep Dive
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Deep Dive Tab
// ---------------------------------------------------------------------------

function DeepDiveTab({ initialTicker }: { initialTicker: string }) {
  const log = useTabLogger('FundamentalsLab.DeepDive')
  const [ticker, setTicker] = useState(initialTicker)
  const [inputTicker, setInputTicker] = useState(initialTicker)

  const [fundamentals, setFundamentals] = useState<TickerFundamentals | null>(null)
  const [filings, setFilings] = useState<SecFiling[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [llmAnalysis, setLlmAnalysis] = useState<string | null>(null)
  const [llmLoading, setLlmLoading] = useState(false)
  const abortAnalyzeRef = useRef<AbortController | null>(null)

  const [filingSummaries, setFilingSummaries] = useState<Record<string, string>>({})
  const [summarizingAcc, setSummarizingAcc] = useState<string | null>(null)

  // Sync if parent passes a new ticker
  useEffect(() => {
    if (initialTicker && initialTicker !== ticker) {
      setInputTicker(initialTicker)
      setTicker(initialTicker)
    }
  }, [initialTicker])

  const fetchFundamentals = useCallback(async (t: string, force = false) => {
    if (!t.trim()) return
    setLoading(true)
    setError(null)
    setFundamentals(null)
    setFilings([])
    log('api:start', { endpoint: 'fundamentals/ticker', ticker: t })
    try {
      const [fundRes, filingsRes] = await Promise.all([
        axios.get<TickerFundamentals>(`${API_BASE}/fundamentals/ticker/${t}`, {
          params: force ? { force: true } : {},
        }),
        axios.get(`${API_BASE}/fundamentals/filings/${t}`),
      ])
      setFundamentals(fundRes.data)
      setFilings(filingsRes.data.filings || [])
      // Pre-populate any cached summaries
      const cached: Record<string, string> = {}
      for (const f of filingsRes.data.filings || []) {
        if (f.llm_summary) cached[f.accession_number] = f.llm_summary
      }
      setFilingSummaries(cached)
      log('api:success')
    } catch (err: any) {
      const msg = err?.response?.data?.detail || err?.message || 'Failed to fetch fundamentals'
      setError(msg)
      log('api:error', err)
    } finally {
      setLoading(false)
    }
  }, [log])

  useEffect(() => {
    if (ticker) fetchFundamentals(ticker)
  }, [ticker])

  const handleFetch = () => {
    const t = inputTicker.trim().toUpperCase()
    if (!t) return
    log('action:fetch', { ticker: t })
    setTicker(t)
  }

  const handleAnalyze = async () => {
    if (!fundamentals) return
    log('action:analyze', { ticker: fundamentals.ticker })
    setLlmAnalysis(null)
    setLlmLoading(true)
    abortAnalyzeRef.current = new AbortController()
    try {
      const res = await axios.post(
        `${API_BASE}/fundamentals/analyze/${fundamentals.ticker}`,
        {},
        { signal: abortAnalyzeRef.current.signal },
      )
      // SSE data comes back as a single event in axios mode
      const raw = typeof res.data === 'string' ? res.data : JSON.stringify(res.data)
      const match = raw.match(/data: ({.*})/s)
      if (match) {
        const parsed = JSON.parse(match[1])
        if (parsed.chunk) setLlmAnalysis(parsed.chunk)
        if (parsed.error) setError(parsed.error)
      } else {
        setLlmAnalysis(raw)
      }
      log('api:success', { endpoint: 'fundamentals/analyze' })
    } catch (err: any) {
      if (axios.isCancel(err)) return
      log('api:error', err)
    } finally {
      setLlmLoading(false)
    }
  }

  const stopAnalyze = () => {
    abortAnalyzeRef.current?.abort()
    setLlmLoading(false)
  }

  const summarizeFiling = async (filing: SecFiling) => {
    if (summarizingAcc) return
    log('action:summarize-filing', { accession: filing.accession_number })
    setSummarizingAcc(filing.accession_number)
    try {
      const res = await axios.post(`${API_BASE}/fundamentals/filings/${ticker}/summarize`, {
        accession_number: filing.accession_number,
        filing_url: filing.filing_url,
        form_type: filing.form_type,
        filing_date: filing.filing_date,
      })
      const raw = typeof res.data === 'string' ? res.data : JSON.stringify(res.data)
      const match = raw.match(/data: ({.*})/s)
      if (match) {
        const parsed = JSON.parse(match[1])
        if (parsed.chunk) {
          setFilingSummaries(prev => ({ ...prev, [filing.accession_number]: parsed.chunk }))
        }
      }
      log('api:success', { endpoint: 'fundamentals/filings/summarize' })
    } catch (err: any) {
      log('api:error', err)
    } finally {
      setSummarizingAcc(null)
    }
  }

  const thStyle: React.CSSProperties = {
    padding: '7px 10px', textAlign: 'left', fontWeight: 600,
    fontSize: '0.77rem', color: 'var(--text-muted)',
    borderBottom: '1px solid var(--border)',
  }
  const tdStyle: React.CSSProperties = {
    padding: '7px 10px', fontSize: '0.82rem',
    borderBottom: '1px solid var(--border)',
  }

  const f = fundamentals

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Ticker input row */}
      <div className="card" style={{ padding: '1rem 1.25rem' }}>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
          <input
            className="input"
            style={{ width: 120, textTransform: 'uppercase', fontWeight: 700, fontSize: '0.95rem' }}
            value={inputTicker}
            onChange={e => setInputTicker(e.target.value.toUpperCase())}
            onKeyDown={e => e.key === 'Enter' && handleFetch()}
            placeholder="NVDA"
          />
          <button className="btn btn-primary" onClick={handleFetch} disabled={loading}>
            Fetch
          </button>
          {f && !llmLoading && (
            <button className="btn" onClick={handleAnalyze} disabled={loading}>
              <TrendingUp size={14} style={{ marginRight: 4 }} />
              Analyze
            </button>
          )}
          {llmLoading && (
            <button className="btn btn-stop" onClick={stopAnalyze}>
              <Square size={13} style={{ marginRight: 4 }} />
              Stop
            </button>
          )}
          {f && (
            <button
              className="btn"
              style={{ marginLeft: 'auto', fontSize: '0.78rem' }}
              onClick={() => fetchFundamentals(ticker, true)}
            >
              <RefreshCw size={12} style={{ marginRight: 4 }} />
              Refresh cache
            </button>
          )}
        </div>
      </div>

      {loading && (
        <div className="card" style={{ padding: '2.5rem', textAlign: 'center' }}>
          <p style={{ fontWeight: 600 }}>Fetching fundamentals for {ticker}…</p>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: 4 }}>
            Pulling from yfinance, SEC EDGAR, and Finnhub in parallel
          </p>
        </div>
      )}

      {error && (
        <div className="card" style={{ padding: '1.25rem', borderLeft: '4px solid var(--danger)' }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {f && (
        <>
          {/* Company header */}
          <div className="card" style={{ padding: '1rem 1.25rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8 }}>
              <div>
                <div style={{ fontSize: '1.2rem', fontWeight: 800 }}>{f.ticker}</div>
                {f.ratios.short_name && (
                  <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                    {f.ratios.short_name}
                    {f.ratios.sector && ` · ${f.ratios.sector}`}
                    {f.ratios.industry && ` · ${f.ratios.industry}`}
                  </div>
                )}
              </div>
              {f.ratios.current_price && (
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: '1.4rem', fontWeight: 800 }}>
                    ${f.ratios.current_price.toFixed(2)}
                  </div>
                  {f.ratios.week52_high && f.ratios.week52_low && (
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      52w: ${f.ratios.week52_low.toFixed(2)} – ${f.ratios.week52_high.toFixed(2)}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Key Ratios */}
          <div className="card" style={{ padding: '1rem 1.25rem' }}>
            <SectionTitle>Key Ratios</SectionTitle>
            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
              <MetricCard label="Market Cap" value={fmtB(f.ratios.market_cap)} />
              <MetricCard label="P/E (TTM)" value={fmtRatio(f.ratios.pe_ratio)} sub={`Forward: ${fmtRatio(f.ratios.forward_pe)}`} />
              <MetricCard label="PEG" value={fmtRatio(f.ratios.peg_ratio)} />
              <MetricCard label="P/B" value={fmtRatio(f.ratios.price_to_book)} />
              <MetricCard label="ROE" value={fmtPct(f.ratios.roe)} sub={`ROA: ${fmtPct(f.ratios.roa)}`} />
              <MetricCard label="Profit Margin" value={fmtPct(f.ratios.profit_margin)} sub={`Op: ${fmtPct(f.ratios.operating_margin)}`} />
              <MetricCard label="Beta" value={fmtRatio(f.ratios.beta)} />
              {f.ratios.dividend_yield && (
                <MetricCard label="Div Yield" value={fmtPct(f.ratios.dividend_yield)} />
              )}
            </div>
          </div>

          {/* Analyst Consensus */}
          {f.analyst && (
            <div className="card" style={{ padding: '1rem 1.25rem' }}>
              <SectionTitle>Analyst Consensus</SectionTitle>
              <AnalystBar consensus={f.analyst} />
              {f.analyst.target_high && f.ratios.current_price && (
                <div style={{ marginTop: 8, fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  Target range: ${f.analyst.target_low?.toFixed(2)} – ${f.analyst.target_high?.toFixed(2)}
                  {f.analyst.target_mean && (
                    <span style={{ marginLeft: 8, color: 'var(--text)', fontWeight: 600 }}>
                      ({((f.analyst.target_mean / f.ratios.current_price - 1) * 100).toFixed(1)}% upside)
                    </span>
                  )}
                </div>
              )}
            </div>
          )}

          {/* EPS Surprise Chart */}
          {f.earnings_history.length > 0 && (
            <div className="card" style={{ padding: '1rem 1.25rem' }}>
              <SectionTitle>EPS Surprise History (last {f.earnings_history.length} quarters)</SectionTitle>
              <ResponsiveContainer width="100%" height={220}>
                <ComposedChart data={earningsChartData(f.earnings_history)} margin={{ top: 4, right: 20, bottom: 0, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                  <YAxis yAxisId="eps" tick={{ fontSize: 10 }} />
                  <YAxis yAxisId="surp" orientation="right" tick={{ fontSize: 10 }} unit="%" />
                  <Tooltip formatter={(v: any) => (v != null ? +v.toFixed(3) : '—')} />
                  <Legend iconSize={10} wrapperStyle={{ fontSize: 11 }} />
                  <Bar yAxisId="eps" dataKey="EPS Estimate" fill="var(--accent-soft, #6366f1)" opacity={0.7} barSize={18} />
                  <Bar yAxisId="eps" dataKey="EPS Actual" fill="var(--success, #22c55e)" barSize={18} />
                  <Line yAxisId="surp" type="monotone" dataKey="Surprise %" stroke="var(--warning, #f59e0b)" strokeWidth={2} dot={{ r: 3 }} />
                  <ReferenceLine yAxisId="surp" y={0} stroke="var(--border)" strokeDasharray="4 4" />
                </ComposedChart>
              </ResponsiveContainer>

              {/* Surprise table */}
              <div style={{ overflowX: 'auto', marginTop: '0.75rem' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.8rem' }}>
                  <thead>
                    <tr>
                      {['Quarter', 'EPS Est', 'EPS Actual', 'Surprise'].map(h => (
                        <th key={h} style={thStyle}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {f.earnings_history.map(h => (
                      <tr key={h.date}>
                        <td style={tdStyle}>{h.date}</td>
                        <td style={tdStyle}>{h.eps_estimate != null ? `$${h.eps_estimate.toFixed(3)}` : '—'}</td>
                        <td style={tdStyle}>{h.eps_actual != null ? `$${h.eps_actual.toFixed(3)}` : '—'}</td>
                        <td style={{ ...tdStyle, color: surpClr(h.surprise_pct), fontWeight: 700 }}>
                          {h.surprise_pct != null ? `${h.surprise_pct > 0 ? '+' : ''}${h.surprise_pct.toFixed(1)}%` : '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Income Statement Trend */}
          {f.income.length > 0 && (
            <div className="card" style={{ padding: '1rem 1.25rem' }}>
              <SectionTitle>Income Statement Trend — {f.income.length} Quarters (in $B)</SectionTitle>
              <ResponsiveContainer width="100%" height={220}>
                <BarChart data={incomeChartData(f.income)} margin={{ top: 4, right: 20, bottom: 0, left: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="period" tick={{ fontSize: 10 }} />
                  <YAxis tick={{ fontSize: 10 }} unit="B" />
                  <Tooltip formatter={(v: any) => v != null ? `$${v}B` : '—'} />
                  <Legend iconSize={10} wrapperStyle={{ fontSize: 11 }} />
                  <Bar dataKey="Revenue" fill="var(--accent-soft, #6366f1)" opacity={0.85} />
                  <Bar dataKey="Gross Profit" fill="var(--success, #22c55e)" opacity={0.85} />
                  <Bar dataKey="Net Income" fill="var(--warning, #f59e0b)" opacity={0.85} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Balance Sheet */}
          <div className="card" style={{ padding: '1rem 1.25rem' }}>
            <SectionTitle>Balance Sheet Snapshot {f.balance.period ? `(${f.balance.period})` : ''}</SectionTitle>
            <div style={{ overflowX: 'auto' }}>
              <table style={{ borderCollapse: 'collapse', fontSize: '0.82rem', minWidth: 360 }}>
                <tbody>
                  {[
                    ['Total Assets', fmtB(f.balance.total_assets)],
                    ['Current Assets', fmtB(f.balance.current_assets)],
                    ['Current Liabilities', fmtB(f.balance.current_liabilities)],
                    ['Total Liabilities', fmtB(f.balance.total_liabilities)],
                    ['Stockholders Equity', fmtB(f.balance.stockholders_equity)],
                    ['Total Debt', fmtB(f.balance.total_debt)],
                    ['Debt / Equity', fmtRatio(f.balance.debt_to_equity)],
                    ['Current Ratio', fmtRatio(f.balance.current_ratio)],
                  ].map(([label, value]) => (
                    <tr key={label}>
                      <td style={{ ...tdStyle, color: 'var(--text-muted)', width: '55%' }}>{label}</td>
                      <td style={{ ...tdStyle, fontWeight: 600, textAlign: 'right' }}>{value}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Cash Flow */}
          <div className="card" style={{ padding: '1rem 1.25rem' }}>
            <SectionTitle>Cash Flow Health {f.cashflow.period ? `(${f.cashflow.period})` : ''}</SectionTitle>
            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
              <MetricCard label="Operating CF" value={fmtB(f.cashflow.operating_cf)} />
              <MetricCard label="Free Cash Flow" value={fmtB(f.cashflow.free_cf)} />
              <MetricCard label="CapEx" value={fmtB(f.cashflow.capex)} />
            </div>
          </div>

          {/* SEC Filings */}
          {filings.length > 0 && (
            <div className="card" style={{ padding: '1rem 1.25rem' }}>
              <SectionTitle>Recent SEC Filings</SectionTitle>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                  <thead>
                    <tr>
                      {['Date', 'Type', 'Link', 'Summary'].map(h => (
                        <th key={h} style={thStyle}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {filings.map(f => (
                      <>
                        <tr key={f.accession_number}>
                          <td style={tdStyle}>{f.filing_date || '—'}</td>
                          <td style={{ ...tdStyle, fontWeight: 700 }}>{f.form_type}</td>
                          <td style={tdStyle}>
                            <a
                              href={f.filing_url}
                              target="_blank"
                              rel="noopener noreferrer"
                              style={{ color: 'var(--accent)', display: 'flex', alignItems: 'center', gap: 4 }}
                            >
                              SEC.gov <ExternalLink size={11} />
                            </a>
                          </td>
                          <td style={tdStyle}>
                            {filingSummaries[f.accession_number] ? (
                              <span style={{ color: 'var(--success)', fontSize: '0.75rem' }}>Summarized</span>
                            ) : (
                              <button
                                className="btn"
                                style={{ fontSize: '0.72rem', padding: '2px 8px' }}
                                disabled={summarizingAcc === f.accession_number}
                                onClick={() => summarizeFiling(f)}
                              >
                                {summarizingAcc === f.accession_number ? 'Summarizing…' : 'Summarize'}
                              </button>
                            )}
                          </td>
                        </tr>
                        {filingSummaries[f.accession_number] && (
                          <tr key={`${f.accession_number}-summary`}>
                            <td colSpan={4} style={{ padding: '0.75rem 1rem', background: 'var(--surface)', fontSize: '0.8rem' }}>
                              <div style={{ whiteSpace: 'pre-wrap', lineHeight: 1.6 }}>
                                {filingSummaries[f.accession_number]}
                              </div>
                            </td>
                          </tr>
                        )}
                      </>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* LLM Analysis Panel */}
          {(llmLoading || llmAnalysis) && (
            <div className="card" style={{ padding: '1rem 1.25rem' }}>
              <SectionTitle>LLM Fundamental Analysis</SectionTitle>
              {llmLoading && (
                <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                  Analyzing {ticker} with LLM…
                </p>
              )}
              {llmAnalysis && (
                <div style={{ whiteSpace: 'pre-wrap', lineHeight: 1.7, fontSize: '0.85rem' }}>
                  {llmAnalysis}
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

type Tab = 'calendar' | 'deep-dive'

export default function FundamentalsLab() {
  const log = useTabLogger('FundamentalsLab')

  const [activeTab, setActiveTab] = useState<Tab>(() => {
    try {
      const s = sessionStorage.getItem(SESSION_KEY)
      return (s ? JSON.parse(s).activeTab : null) ?? 'calendar'
    } catch {
      return 'calendar'
    }
  })

  const [deepDiveTicker, setDeepDiveTicker] = useState<string>(() => {
    try {
      const s = sessionStorage.getItem(SESSION_KEY)
      return (s ? JSON.parse(s).deepDiveTicker : null) ?? ''
    } catch {
      return ''
    }
  })

  const persist = (tab: Tab, ticker: string) => {
    try {
      sessionStorage.setItem(SESSION_KEY, JSON.stringify({ activeTab: tab, deepDiveTicker: ticker }))
    } catch {}
  }

  const setTab = (t: Tab) => {
    setActiveTab(t)
    persist(t, deepDiveTicker)
    log('action:tab', { tab: t })
  }

  const handleTickerSelect = (ticker: string) => {
    setDeepDiveTicker(ticker)
    setActiveTab('deep-dive')
    persist('deep-dive', ticker)
    log('action:ticker-select', { ticker })
  }

  const TABS: { key: Tab; label: string; desc: string }[] = [
    { key: 'calendar', label: 'Earnings Calendar', desc: 'Upcoming reports across watchlist' },
    { key: 'deep-dive', label: 'Deep Dive', desc: 'Per-ticker financials + LLM analysis' },
  ]

  return (
    <div style={{ padding: '1.5rem', maxWidth: 1100, margin: '0 auto' }}>
      <h1 style={{ fontSize: '1.35rem', fontWeight: 800, marginBottom: 4 }}>Fundamentals Lab</h1>
      <p style={{ fontSize: '0.83rem', color: 'var(--text-muted)', marginBottom: '1.5rem' }}>
        Earnings calendar · SEC filings · Financial statements · Analyst consensus · LLM analysis
      </p>

      {/* Tab bar */}
      <div style={{
        display: 'flex',
        gap: 0,
        borderBottom: '2px solid var(--border)',
        marginBottom: '1.5rem',
      }}>
        {TABS.map(tab => (
          <button
            key={tab.key}
            onClick={() => setTab(tab.key)}
            style={{
              background: 'none',
              border: 'none',
              borderBottom: activeTab === tab.key ? '3px solid var(--accent)' : '3px solid transparent',
              marginBottom: -2,
              padding: '0.5rem 1.25rem',
              cursor: 'pointer',
              fontWeight: activeTab === tab.key ? 700 : 400,
              color: activeTab === tab.key ? 'var(--text)' : 'var(--text-muted)',
              fontSize: '0.88rem',
              transition: 'all 0.15s',
            }}
          >
            <div>{tab.label}</div>
            <div style={{ fontSize: '0.72rem', opacity: 0.75 }}>{tab.desc}</div>
          </button>
        ))}
      </div>

      {activeTab === 'calendar' && (
        <EarningsCalendarTab onTickerSelect={handleTickerSelect} />
      )}
      {activeTab === 'deep-dive' && (
        <DeepDiveTab initialTicker={deepDiveTicker} />
      )}
    </div>
  )
}
