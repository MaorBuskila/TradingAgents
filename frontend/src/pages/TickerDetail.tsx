import { useState, useEffect, useCallback } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import axios from 'axios'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  ComposedChart,
  Bar,
} from 'recharts'
import { ExternalLink, PlayCircle, FlaskConical, Activity, Crosshair, ChevronDown, ChevronUp, Newspaper } from 'lucide-react'

const API_BASE = import.meta.env.VITE_API_BASE ?? '/api'

// ── Types ─────────────────────────────────────────────────────────────────────

type CatalogItem = {
  id: number
  ticker: string
  name: string | null
  category: string
  category_label: string | null
  asset_type: string
  is_favorite: boolean
}

type TestResult = {
  test_id: string
  job_type: string
  ticker: string | null
  created_at: string
  status: string
  results: Record<string, unknown> | null
  error_log: string | null
}

type RsiSignal = {
  action: string | null
  current_rsi: number | null
  rsi_zone: string | null
  period: number | null
  overbought: number | null
  oversold: number | null
  confidence: string | null
  latest_close: number | null
  rsi_history: Array<{ date: string; rsi: number; close: number }>
  error: string | null
}

type RsiCache = {
  optimal_period: number
  optimal_upper: number
  optimal_lower: number
  is_sharpe: number
  oos_sharpe: number
  confidence: string
  regime: string | null
  optimized_at: string | null
}

type MacdSignal = {
  action: string | null
  macd_line: number | null
  signal_line: number | null
  histogram: number | null
  fast_period: number | null
  slow_period: number | null
  signal_period: number | null
  confidence: string | null
  latest_close: number | null
  macd_history: Array<{ date: string; macd: number; signal: number; histogram: number; close: number }>
  error: string | null
}

type MacdCache = {
  optimal_fast: number
  optimal_slow: number
  optimal_signal: number
  is_sharpe: number
  oos_sharpe: number
  confidence: string
  regime: string | null
  optimized_at: string | null
}

type SniperSignal = {
  action: string | null
  bull_score: number | null
  bear_score: number | null
  confidence: string | null
  stop_loss: number | null
  tp1: number | null
  latest_close: number | null
  error: string | null
}

type NewsItem = {
  title: string | null
  publisher: string | null
  link: string | null
  published_at: string | null
}

type LatestDecision = {
  ticker: string
  report_id: string
  source: string
  content: string
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function actionColor(action: string | null) {
  if (!action) return 'var(--text-muted)'
  if (action === 'BUY') return '#22c55e'
  if (action === 'SELL') return '#ef4444'
  return '#f59e0b'
}

function confidenceBadge(conf: string | null) {
  if (!conf) return null
  const color = conf === 'HIGH' ? '#22c55e' : conf === 'MEDIUM' ? '#f59e0b' : '#ef4444'
  return (
    <span style={{ fontSize: '0.65rem', fontWeight: 700, color, border: `1px solid ${color}`, borderRadius: 4, padding: '1px 5px', letterSpacing: '0.05em' }}>
      {conf}
    </span>
  )
}

function formatDate(raw: string | null) {
  if (!raw) return '—'
  try {
    // epoch seconds
    const n = Number(raw)
    if (!isNaN(n) && n > 1_000_000_000) return new Date(n * 1000).toLocaleDateString()
    return new Date(raw).toLocaleDateString()
  } catch {
    return raw
  }
}

function formatDatetime(raw: string | null) {
  if (!raw) return '—'
  try {
    return new Date(raw).toLocaleString()
  } catch {
    return raw
  }
}

function Collapsible({ title, icon, children, defaultOpen = true }: {
  title: string
  icon: React.ReactNode
  children: React.ReactNode
  defaultOpen?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div style={{ border: '1px solid var(--border)', borderRadius: 8, overflow: 'hidden', marginBottom: '1.25rem' }}>
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          width: '100%', display: 'flex', alignItems: 'center', gap: 8,
          padding: '0.75rem 1rem', background: 'var(--bg-subtle)',
          border: 'none', cursor: 'pointer', fontWeight: 600, fontSize: '0.92rem',
          color: 'var(--text-primary)', textAlign: 'left',
        }}
      >
        {icon}
        <span style={{ flex: 1 }}>{title}</span>
        {open ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
      </button>
      {open && <div style={{ padding: '1rem' }}>{children}</div>}
    </div>
  )
}

// ── Signal card ───────────────────────────────────────────────────────────────

function SignalCard({ label, action, detail, confidence }: {
  label: string
  action: string | null
  detail?: string
  confidence?: string | null
}) {
  return (
    <div style={{
      flex: 1, minWidth: 140, border: '1px solid var(--border)', borderRadius: 8,
      padding: '0.75rem 1rem', background: 'var(--bg-subtle)',
    }}>
      <div style={{ fontSize: '0.7rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 4 }}>
        {label}
      </div>
      <div style={{ fontSize: '1.4rem', fontWeight: 800, color: actionColor(action) }}>
        {action ?? '—'}
      </div>
      {detail && <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: 2 }}>{detail}</div>}
      {confidence && <div style={{ marginTop: 4 }}>{confidenceBadge(confidence)}</div>}
    </div>
  )
}

// ── Job type label ────────────────────────────────────────────────────────────

function jobTypeLabel(t: string) {
  const map: Record<string, string> = {
    analysis: 'Full Analysis',
    'rsi-optimize': 'RSI Lab',
    'macd-optimize': 'MACD Lab',
    'sniper-optimize': 'Sniper Lab',
    'rsi-oos-test': 'RSI OOS Test',
    'macd-oos-test': 'MACD OOS Test',
  }
  return map[t] ?? t
}

function statusColor(s: string) {
  if (s === 'complete') return '#22c55e'
  if (s === 'error') return '#ef4444'
  if (s === 'running') return '#3b82f6'
  return 'var(--text-muted)'
}

function extractDecision(row: TestResult): string {
  if (!row.results) return '—'
  const r = row.results
  if (r.decision) return String(r.decision)
  if (r.oos_winner) return `OOS: ${r.oos_winner}`
  if (r.oos_sharpe !== undefined) return `Sharpe ${Number(r.oos_sharpe).toFixed(2)}`
  return '—'
}

// ── Main component ────────────────────────────────────────────────────────────

export default function TickerDetail() {
  const { ticker } = useParams<{ ticker: string }>()
  const sym = (ticker ?? '').toUpperCase()
  const navigate = useNavigate()

  const [catalog, setCatalog] = useState<CatalogItem | null>(null)
  const [decision, setDecision] = useState<LatestDecision | null>(null)
  const [rsiSignal, setRsiSignal] = useState<RsiSignal | null>(null)
  const [rsiCache, setRsiCache] = useState<RsiCache | null>(null)
  const [macdSignal, setMacdSignal] = useState<MacdSignal | null>(null)
  const [macdCache, setMacdCache] = useState<MacdCache | null>(null)
  const [sniperSignal, setSniperSignal] = useState<SniperSignal | null>(null)
  const [history, setHistory] = useState<TestResult[]>([])
  const [news, setNews] = useState<NewsItem[]>([])
  const [newsLoading, setNewsLoading] = useState(false)
  const [loading, setLoading] = useState(true)

  const load = useCallback(async () => {
    if (!sym) return
    setLoading(true)
    const get = <T,>(url: string) => axios.get<T>(url).then(r => r.data).catch(() => null)

    const [cat, dec, rsiSig, rsiC, macdSig, macdC, sniperSig, hist] = await Promise.all([
      get<CatalogItem[]>(`${API_BASE}/catalog/items?q=${encodeURIComponent(sym)}`),
      get<LatestDecision>(`${API_BASE}/reports/by-ticker/${encodeURIComponent(sym)}/latest-decision`),
      get<RsiSignal>(`${API_BASE}/rsi-signal/${encodeURIComponent(sym)}`),
      get<RsiCache>(`${API_BASE}/rsi-cache/${encodeURIComponent(sym)}`),
      get<MacdSignal>(`${API_BASE}/macd-signal/${encodeURIComponent(sym)}`),
      get<MacdCache>(`${API_BASE}/macd-cache/${encodeURIComponent(sym)}`),
      get<SniperSignal>(`${API_BASE}/sniper-signal/${encodeURIComponent(sym)}`),
      get<TestResult[]>(`${API_BASE}/test-results?ticker=${encodeURIComponent(sym)}&limit=50`),
    ])

    setCatalog(Array.isArray(cat) ? (cat.find(c => c.ticker.toUpperCase() === sym) ?? cat[0] ?? null) : null)
    setDecision(dec)
    setRsiSignal(rsiSig)
    setRsiCache(rsiC)
    setMacdSignal(macdSig)
    setMacdCache(macdC)
    setSniperSignal(sniperSig)
    setHistory(Array.isArray(hist) ? hist : [])
    setLoading(false)

    // Load news separately (can be slower)
    setNewsLoading(true)
    const n = await get<NewsItem[]>(`${API_BASE}/ticker/${encodeURIComponent(sym)}/news`)
    setNews(Array.isArray(n) ? n : [])
    setNewsLoading(false)
  }, [sym])

  useEffect(() => { load() }, [load])

  if (!sym) return <div style={{ padding: '2rem' }}>No ticker specified.</div>

  const decisionText = decision?.content?.match(/\b(BUY|SELL|HOLD)\b/)?.[0] ?? null

  return (
    <div style={{ padding: '1.5rem', maxWidth: 900, margin: '0 auto' }}>
      {/* ── Header ── */}
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: '1rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
        <div style={{ flex: 1, minWidth: 200 }}>
          <h1 style={{ margin: 0, fontSize: '2rem', fontWeight: 800, letterSpacing: '-0.02em' }}>{sym}</h1>
          {catalog && (
            <div style={{ marginTop: 4, display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap' }}>
              {catalog.name && <span style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>{catalog.name}</span>}
              {catalog.category_label && (
                <span style={{ fontSize: '0.7rem', padding: '2px 7px', borderRadius: 12, background: 'var(--bg-subtle)', border: '1px solid var(--border)', color: 'var(--text-muted)' }}>
                  {catalog.category_label}
                </span>
              )}
              <span style={{ fontSize: '0.65rem', fontWeight: 700, textTransform: 'uppercase', padding: '2px 7px', borderRadius: 12, background: 'var(--bg-subtle)', border: '1px solid var(--border)', color: 'var(--text-muted)', letterSpacing: '0.06em' }}>
                {catalog.asset_type}
              </span>
            </div>
          )}
          {loading && !catalog && <div style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginTop: 4 }}>Loading…</div>}
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <button
            className="btn btn-sm"
            onClick={() => navigate(`/?ticker=${encodeURIComponent(sym)}`)}
            style={{ display: 'flex', alignItems: 'center', gap: 5 }}
          >
            <PlayCircle size={14} />
            Run Analysis
          </button>
          <a
            href={`https://www.tradingview.com/chart/?symbol=${encodeURIComponent(sym)}`}
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-sm"
            style={{ display: 'flex', alignItems: 'center', gap: 5, textDecoration: 'none' }}
          >
            <ExternalLink size={14} />
            TradingView
          </a>
        </div>
      </div>

      {/* ── Signal summary cards ── */}
      <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
        <SignalCard
          label="Analysis"
          action={decisionText}
          detail={decision ? `Report: ${decision.report_id}` : undefined}
        />
        <SignalCard
          label="RSI Signal"
          action={rsiSignal?.action ?? null}
          detail={rsiSignal?.current_rsi != null ? `RSI ${rsiSignal.current_rsi.toFixed(1)} · ${rsiSignal.rsi_zone ?? ''}` : undefined}
          confidence={rsiSignal?.confidence}
        />
        <SignalCard
          label="MACD Signal"
          action={macdSignal?.action ?? null}
          detail={macdSignal?.histogram != null ? `Hist ${macdSignal.histogram.toFixed(4)}` : undefined}
          confidence={macdSignal?.confidence}
        />
        <SignalCard
          label="Sniper Signal"
          action={sniperSignal?.action ?? null}
          detail={sniperSignal?.bull_score != null ? `Bull ${(sniperSignal.bull_score * 100).toFixed(0)}%` : undefined}
          confidence={sniperSignal?.confidence}
        />
      </div>

      {/* ── Analysis History ── */}
      <Collapsible title="Analysis History" icon={<PlayCircle size={16} />}>
        {history.length === 0 ? (
          <p style={{ color: 'var(--text-muted)', margin: 0 }}>No runs found for {sym} yet.</p>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border)' }}>
                  {['Date', 'Type', 'Status', 'Decision / Result'].map(h => (
                    <th key={h} style={{ padding: '6px 10px', textAlign: 'left', fontWeight: 600, color: 'var(--text-muted)', fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.06em' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {history.map(row => (
                  <tr key={row.test_id} style={{ borderBottom: '1px solid var(--border-subtle, var(--border))' }}>
                    <td style={{ padding: '6px 10px', color: 'var(--text-secondary)' }}>{formatDatetime(row.created_at)}</td>
                    <td style={{ padding: '6px 10px' }}>{jobTypeLabel(row.job_type)}</td>
                    <td style={{ padding: '6px 10px', fontWeight: 600, color: statusColor(row.status) }}>{row.status}</td>
                    <td style={{ padding: '6px 10px', fontWeight: 600, color: actionColor(extractDecision(row) as string | null) }}>
                      {extractDecision(row)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Collapsible>

      {/* ── RSI Lab ── */}
      <Collapsible title="RSI Lab" icon={<FlaskConical size={16} />} defaultOpen={!!rsiCache}>
        {!rsiCache && !rsiSignal ? (
          <p style={{ color: 'var(--text-muted)', margin: 0 }}>No RSI optimization run yet for {sym}. <button className="btn btn-sm" onClick={() => navigate('/rsi-lab')} style={{ marginLeft: 8 }}>Open RSI Lab</button></p>
        ) : (
          <div style={{ display: 'flex', gap: '1.5rem', flexWrap: 'wrap' }}>
            {rsiCache && (
              <div style={{ minWidth: 220 }}>
                <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 8 }}>Optimal Params</div>
                <table style={{ borderCollapse: 'collapse', fontSize: '0.85rem', width: '100%' }}>
                  <tbody>
                    {[
                      ['Period', rsiCache.optimal_period],
                      ['Overbought', rsiCache.optimal_upper],
                      ['Oversold', rsiCache.optimal_lower],
                      ['IS Sharpe', rsiCache.is_sharpe?.toFixed(2)],
                      ['OOS Sharpe', rsiCache.oos_sharpe?.toFixed(2)],
                      ['Regime', rsiCache.regime ?? '—'],
                      ['Optimized', formatDate(rsiCache.optimized_at)],
                    ].map(([k, v]) => (
                      <tr key={String(k)}>
                        <td style={{ padding: '3px 0', color: 'var(--text-muted)', paddingRight: 12 }}>{k}</td>
                        <td style={{ padding: '3px 0', fontWeight: 600 }}>{String(v)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div style={{ marginTop: 8 }}>{confidenceBadge(rsiCache.confidence)}</div>
                <button className="btn btn-sm" onClick={() => navigate('/rsi-lab')} style={{ marginTop: 10, display: 'flex', alignItems: 'center', gap: 5 }}>
                  <FlaskConical size={12} /> Open RSI Lab
                </button>
              </div>
            )}
            {rsiSignal && rsiSignal.rsi_history.length > 0 && (
              <div style={{ flex: 1, minWidth: 280, minHeight: 160 }}>
                <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 8 }}>
                  RSI History (30 bars)
                </div>
                <ResponsiveContainer width="100%" height={160}>
                  <LineChart data={rsiSignal.rsi_history} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
                    <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={d => d.slice(5)} minTickGap={30} />
                    <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} width={28} />
                    <Tooltip formatter={(v: number) => v.toFixed(1)} labelFormatter={l => `Date: ${l}`} />
                    {rsiCache?.optimal_upper && <ReferenceLine y={rsiCache.optimal_upper} stroke="#ef4444" strokeDasharray="3 3" label={{ value: 'OB', position: 'insideTopRight', fontSize: 9, fill: '#ef4444' }} />}
                    {rsiCache?.optimal_lower && <ReferenceLine y={rsiCache.optimal_lower} stroke="#22c55e" strokeDasharray="3 3" label={{ value: 'OS', position: 'insideBottomRight', fontSize: 9, fill: '#22c55e' }} />}
                    <Line type="monotone" dataKey="rsi" stroke="#6366f1" strokeWidth={1.5} dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        )}
      </Collapsible>

      {/* ── MACD Lab ── */}
      <Collapsible title="MACD Lab" icon={<Activity size={16} />} defaultOpen={!!macdCache}>
        {!macdCache && !macdSignal ? (
          <p style={{ color: 'var(--text-muted)', margin: 0 }}>No MACD optimization run yet for {sym}. <button className="btn btn-sm" onClick={() => navigate('/macd-lab')} style={{ marginLeft: 8 }}>Open MACD Lab</button></p>
        ) : (
          <div style={{ display: 'flex', gap: '1.5rem', flexWrap: 'wrap' }}>
            {macdCache && (
              <div style={{ minWidth: 220 }}>
                <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 8 }}>Optimal Params</div>
                <table style={{ borderCollapse: 'collapse', fontSize: '0.85rem', width: '100%' }}>
                  <tbody>
                    {[
                      ['Fast', macdCache.optimal_fast],
                      ['Slow', macdCache.optimal_slow],
                      ['Signal', macdCache.optimal_signal],
                      ['IS Sharpe', macdCache.is_sharpe?.toFixed(2)],
                      ['OOS Sharpe', macdCache.oos_sharpe?.toFixed(2)],
                      ['Regime', macdCache.regime ?? '—'],
                      ['Optimized', formatDate(macdCache.optimized_at)],
                    ].map(([k, v]) => (
                      <tr key={String(k)}>
                        <td style={{ padding: '3px 0', color: 'var(--text-muted)', paddingRight: 12 }}>{k}</td>
                        <td style={{ padding: '3px 0', fontWeight: 600 }}>{String(v)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div style={{ marginTop: 8 }}>{confidenceBadge(macdCache.confidence)}</div>
                <button className="btn btn-sm" onClick={() => navigate('/macd-lab')} style={{ marginTop: 10, display: 'flex', alignItems: 'center', gap: 5 }}>
                  <Activity size={12} /> Open MACD Lab
                </button>
              </div>
            )}
            {macdSignal && macdSignal.macd_history.length > 0 && (
              <div style={{ flex: 1, minWidth: 280, minHeight: 200 }}>
                <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.07em', marginBottom: 8 }}>
                  MACD History (30 bars)
                </div>
                <ResponsiveContainer width="100%" height={200}>
                  <ComposedChart data={macdSignal.macd_history} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
                    <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={d => d.slice(5)} minTickGap={30} />
                    <YAxis tick={{ fontSize: 10 }} width={40} />
                    <Tooltip labelFormatter={l => `Date: ${l}`} />
                    <ReferenceLine y={0} stroke="var(--border)" />
                    <Bar dataKey="histogram" fill="#6366f1" opacity={0.5} name="Histogram" />
                    <Line type="monotone" dataKey="macd" stroke="#22c55e" strokeWidth={1.5} dot={false} name="MACD" />
                    <Line type="monotone" dataKey="signal" stroke="#f59e0b" strokeWidth={1.5} dot={false} name="Signal" />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        )}
      </Collapsible>

      {/* ── Sniper Lab ── */}
      <Collapsible title="Sniper Lab" icon={<Crosshair size={16} />} defaultOpen={!!sniperSignal?.action}>
        {!sniperSignal || sniperSignal.error ? (
          <p style={{ color: 'var(--text-muted)', margin: 0 }}>No Sniper optimization run yet for {sym}. <button className="btn btn-sm" onClick={() => navigate('/sniper-lab')} style={{ marginLeft: 8 }}>Open Sniper Lab</button></p>
        ) : (
          <div style={{ display: 'flex', gap: '1.5rem', flexWrap: 'wrap', alignItems: 'flex-start' }}>
            <div style={{ minWidth: 220 }}>
              <table style={{ borderCollapse: 'collapse', fontSize: '0.85rem', width: '100%' }}>
                <tbody>
                  {[
                    ['Action', <span style={{ fontWeight: 800, color: actionColor(sniperSignal.action) }}>{sniperSignal.action ?? '—'}</span>],
                    ['Bull Score', sniperSignal.bull_score != null ? `${(sniperSignal.bull_score * 100).toFixed(0)}%` : '—'],
                    ['Bear Score', sniperSignal.bear_score != null ? `${(sniperSignal.bear_score * 100).toFixed(0)}%` : '—'],
                    ['Stop Loss', sniperSignal.stop_loss != null ? `$${sniperSignal.stop_loss.toFixed(2)}` : '—'],
                    ['TP1', sniperSignal.tp1 != null ? `$${sniperSignal.tp1.toFixed(2)}` : '—'],
                    ['Close', sniperSignal.latest_close != null ? `$${sniperSignal.latest_close.toFixed(2)}` : '—'],
                  ].map(([k, v]) => (
                    <tr key={String(k)}>
                      <td style={{ padding: '3px 0', color: 'var(--text-muted)', paddingRight: 12 }}>{k}</td>
                      <td style={{ padding: '3px 0', fontWeight: 600 }}>{v as React.ReactNode}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {sniperSignal.confidence && <div style={{ marginTop: 8 }}>{confidenceBadge(sniperSignal.confidence)}</div>}
              <button className="btn btn-sm" onClick={() => navigate('/sniper-lab')} style={{ marginTop: 10, display: 'flex', alignItems: 'center', gap: 5 }}>
                <Crosshair size={12} /> Open Sniper Lab
              </button>
            </div>
          </div>
        )}
      </Collapsible>

      {/* ── News & Social ── */}
      <Collapsible title="News & Social" icon={<Newspaper size={16} />}>
        {newsLoading ? (
          <p style={{ color: 'var(--text-muted)', margin: 0 }}>Loading news…</p>
        ) : news.length === 0 ? (
          <p style={{ color: 'var(--text-muted)', margin: 0 }}>No recent news found for {sym}.</p>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
            {news.map((item, i) => (
              <div key={i} style={{ borderBottom: i < news.length - 1 ? '1px solid var(--border-subtle, var(--border))' : 'none', paddingBottom: i < news.length - 1 ? '0.75rem' : 0 }}>
                {item.link ? (
                  <a
                    href={item.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{ fontWeight: 600, color: 'var(--text-primary)', textDecoration: 'none', fontSize: '0.9rem', lineHeight: 1.4 }}
                    onMouseEnter={e => (e.currentTarget.style.textDecoration = 'underline')}
                    onMouseLeave={e => (e.currentTarget.style.textDecoration = 'none')}
                  >
                    {item.title ?? 'Untitled'}
                  </a>
                ) : (
                  <span style={{ fontWeight: 600, fontSize: '0.9rem' }}>{item.title ?? 'Untitled'}</span>
                )}
                <div style={{ display: 'flex', gap: 8, marginTop: 3, fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                  {item.publisher && <span>{item.publisher}</span>}
                  {item.published_at && <span>· {formatDate(item.published_at)}</span>}
                </div>
              </div>
            ))}
          </div>
        )}
      </Collapsible>
    </div>
  )
}
