/**
 * TPTracker — WillyAlgoTrader-style TP/SL progress monitor.
 *
 * Left panel: list of active trackers with TP dot progress.
 * Right panel: detail card with price chart + sidebar metrics.
 */
import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'
import { useTabLogger } from '../hooks/useTabLogger'
import {
  ComposedChart, Line, ReferenceLine, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, Label,
} from 'recharts'
import {
  Target, TrendingUp, TrendingDown, Minus,
  Check, Circle, Loader2, RefreshCw, Plus, X,
} from 'lucide-react'

const API_BASE = import.meta.env.VITE_API_BASE ?? '/api'

// ── types ──────────────────────────────────────────────────────────────────

interface TPTracker {
  id: number
  ticker: string
  side: 'long' | 'short'
  entry: number
  sl: number
  trail_sl: number
  tp1: number
  tp2: number
  tp3: number
  tp1_hit: boolean
  tp2_hit: boolean
  tp3_hit: boolean
  trail_active: boolean
  closed: boolean
  exit_reason: string | null
  rsi_at_entry: number | null
  htf_bias: number | null
  vol_regime: string | null
  bull_score: number | null
  as_of_date: string | null
  created_at: string
  updated_at: string
  current_price: number | null
  trail_status: string | null
}

interface PricePoint {
  date: string
  close: number
  rsi: number | null
}

// ── helpers ────────────────────────────────────────────────────────────────

function fmt(v: number | null | undefined, decimals = 2): string {
  if (v == null) return '—'
  return v.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals })
}

function actionBadge(side: string) {
  const a = side === 'long' ? 'BUY' : 'SELL'
  const cfg = {
    BUY:  { bg: 'rgba(5,150,105,0.12)',  color: '#059669', Icon: TrendingUp },
    SELL: { bg: 'rgba(220,38,38,0.12)',  color: '#dc2626', Icon: TrendingDown },
  } as const
  const c = cfg[a as keyof typeof cfg] ?? { bg: 'rgba(99,102,241,0.10)', color: '#6366f1', Icon: Minus }
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 5,
      padding: '3px 10px', borderRadius: 999,
      background: c.bg, color: c.color,
      fontSize: '0.78rem', fontWeight: 800, letterSpacing: '0.05em',
    }}>
      <c.Icon size={12} strokeWidth={2.5} />
      {a}
    </span>
  )
}

function TpDots({ tracker }: { tracker: TPTracker }) {
  const dots = [tracker.tp1_hit, tracker.tp2_hit, tracker.tp3_hit]
  return (
    <span style={{ display: 'inline-flex', gap: 4, alignItems: 'center' }}>
      {dots.map((hit, i) => (
        <span
          key={i}
          title={`TP${i + 1}`}
          style={{
            width: 8, height: 8, borderRadius: '50%',
            background: hit ? '#059669' : 'var(--border)',
            border: `1.5px solid ${hit ? '#059669' : 'var(--text-muted)'}`,
            display: 'inline-block',
          }}
        />
      ))}
    </span>
  )
}

function MetricRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div style={{
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      padding: '8px 0', borderBottom: '1px solid var(--border)',
    }}>
      <span style={{ fontSize: '0.83rem', color: 'var(--text-secondary)', fontWeight: 500 }}>
        {label}
      </span>
      <span style={{ fontSize: '0.88rem', fontWeight: 600, color: 'var(--text)' }}>
        {value}
      </span>
    </div>
  )
}

function TpRow({ label, price, hit }: { label: string; price: number; hit: boolean }) {
  return (
    <div style={{
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      padding: '8px 0', borderBottom: '1px solid var(--border)',
    }}>
      <span style={{
        display: 'flex', alignItems: 'center', gap: 6,
        fontSize: '0.83rem', color: hit ? '#059669' : 'var(--text-secondary)', fontWeight: 600,
      }}>
        {hit
          ? <Check size={14} strokeWidth={2.5} color="#059669" />
          : <Circle size={14} strokeWidth={1.5} color="var(--text-muted)" />
        }
        {label}
      </span>
      <span style={{ fontSize: '0.88rem', fontWeight: 600, color: hit ? '#059669' : 'var(--text)' }}>
        ${fmt(price)}
      </span>
    </div>
  )
}

// ── component ──────────────────────────────────────────────────────────────

export default function TPTracker() {
  const log = useTabLogger('TPTracker')
  const [trackers, setTrackers] = useState<TPTracker[]>([])
  const [selectedId, setSelectedId] = useState<number | null>(null)
  const [priceHistory, setPriceHistory] = useState<PricePoint[]>([])
  const [loading, setLoading] = useState(false)
  const [refreshingId, setRefreshingId] = useState<number | null>(null)
  const [trackTicker, setTrackTicker] = useState('')
  const [trackDate, setTrackDate] = useState('')
  const [tracking, setTracking] = useState(false)
  const [trackError, setTrackError] = useState<string | null>(null)
  const [showTrackForm, setShowTrackForm] = useState(false)
  const [refreshAllLoading, setRefreshAllLoading] = useState(false)

  const selected = trackers.find(t => t.id === selectedId) ?? null

  // ── data loaders ──────────────────────────────────────────────────────────

  const loadTrackers = useCallback(async () => {
    setLoading(true)
    try {
      const { data } = await axios.get<TPTracker[]>(`${API_BASE}/tp-tracker`)
      setTrackers(data)
      if (data.length > 0 && selectedId == null) {
        setSelectedId(data[0].id)
      }
    } catch {
      // ignore
    } finally {
      setLoading(false)
    }
  }, [selectedId])

  const loadPriceHistory = useCallback(async (ticker: string) => {
    setPriceHistory([])
    try {
      const { data } = await axios.get<{ rsi_history?: PricePoint[] }>(
        `${API_BASE}/rsi-signal/${ticker}`
      )
      if (data.rsi_history) {
        setPriceHistory(data.rsi_history.slice(-90))
      }
    } catch {
      // chart stays empty
    }
  }, [])

  useEffect(() => { loadTrackers() }, [])

  useEffect(() => {
    if (selected) loadPriceHistory(selected.ticker)
  }, [selected?.ticker])

  // ── actions ───────────────────────────────────────────────────────────────

  async function handleRefresh(tracker: TPTracker) {
    setRefreshingId(tracker.id)
    try {
      const { data } = await axios.post<TPTracker>(
        `${API_BASE}/tp-tracker/${tracker.ticker}/refresh`
      )
      setTrackers(prev => prev.map(t => t.id === data.id ? data : t))
    } catch {
      // ignore
    } finally {
      setRefreshingId(null)
    }
  }

  async function handleRefreshAll() {
    setRefreshAllLoading(true)
    for (const t of trackers.filter(t => !t.closed)) {
      try {
        const { data } = await axios.post<TPTracker>(`${API_BASE}/tp-tracker/${t.ticker}/refresh`)
        setTrackers(prev => prev.map(x => x.id === data.id ? data : x))
      } catch {
        // continue
      }
    }
    setRefreshAllLoading(false)
  }

  async function handleTrackSignal() {
    if (!trackTicker.trim()) return
    setTracking(true)
    setTrackError(null)
    log('action:add-trade', { ticker: trackTicker.trim().toUpperCase() })
    try {
      log('api:start', { endpoint: 'tp-tracker/from-signal' })
      const { data } = await axios.post<TPTracker>(`${API_BASE}/tp-tracker/from-signal`, {
        ticker: trackTicker.trim().toUpperCase(),
        date: trackDate || undefined,
      })
      setTrackers(prev => [data, ...prev])
      setSelectedId(data.id)
      setTrackTicker('')
      setTrackDate('')
      setShowTrackForm(false)
      log('api:success')
    } catch (err: unknown) {
      log('api:error', err)
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail
        ?? 'Failed to create tracker'
      setTrackError(msg)
    } finally {
      setTracking(false)
    }
  }

  async function handleClose(tracker: TPTracker) {
    try {
      await axios.delete(`${API_BASE}/tp-tracker/${tracker.ticker}`)
      setTrackers(prev => prev.filter(t => t.id !== tracker.id))
      if (selectedId === tracker.id) setSelectedId(null)
    } catch {
      // ignore
    }
  }

  // ── chart data ────────────────────────────────────────────────────────────

  const chartData = priceHistory.map(p => ({
    date: p.date.slice(5),  // MM-DD
    close: p.close,
  }))

  const yDomain = selected && chartData.length > 0
    ? (() => {
        const prices = chartData.map(p => p.close)
        const levels = [selected.sl, selected.trail_sl, selected.tp1, selected.tp2, selected.tp3]
        const all = [...prices, ...levels]
        const lo = Math.min(...all) * 0.98
        const hi = Math.max(...all) * 1.02
        return [lo, hi] as [number, number]
      })()
    : (['auto', 'auto'] as [string, string])

  // ── render ────────────────────────────────────────────────────────────────

  return (
    <div className="page" style={{ display: 'grid', gridTemplateColumns: '280px 1fr', gap: 0, height: '100%', minHeight: 0 }}>

      {/* ── Left panel ── */}
      <aside style={{
        borderRight: '1px solid var(--border)',
        display: 'flex', flexDirection: 'column',
        background: 'var(--bg)',
        overflow: 'hidden',
      }}>
        {/* Header */}
        <div style={{
          padding: '16px 16px 12px',
          borderBottom: '1px solid var(--border)',
          display: 'flex', flexDirection: 'column', gap: 10,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: 8, fontWeight: 700, fontSize: '0.95rem' }}>
              <Target size={16} color="#6366f1" />
              TP Tracker
            </span>
            <button
              className="btn btn-sm"
              onClick={handleRefreshAll}
              disabled={refreshAllLoading || trackers.filter(t => !t.closed).length === 0}
              title="Refresh all active trackers"
              style={{ display: 'flex', alignItems: 'center', gap: 5, padding: '4px 10px' }}
            >
              {refreshAllLoading
                ? <Loader2 size={12} className="icon-spin" />
                : <RefreshCw size={12} />
              }
              Refresh
            </button>
          </div>

          {/* Track new signal */}
          {!showTrackForm ? (
            <button
              className="btn"
              onClick={() => setShowTrackForm(true)}
              style={{ display: 'flex', alignItems: 'center', gap: 6, width: '100%', justifyContent: 'center' }}
            >
              <Plus size={13} /> Track Signal
            </button>
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              <input
                className="input"
                placeholder="Ticker (e.g. IREN)"
                value={trackTicker}
                onChange={e => setTrackTicker(e.target.value.toUpperCase())}
                onKeyDown={e => e.key === 'Enter' && handleTrackSignal()}
                style={{ fontSize: '0.85rem', padding: '5px 8px' }}
              />
              <input
                className="input"
                type="date"
                value={trackDate}
                onChange={e => setTrackDate(e.target.value)}
                style={{ fontSize: '0.85rem', padding: '5px 8px' }}
              />
              {trackError && (
                <span style={{ fontSize: '0.78rem', color: '#dc2626' }}>{trackError}</span>
              )}
              <div style={{ display: 'flex', gap: 6 }}>
                <button
                  className="btn"
                  onClick={handleTrackSignal}
                  disabled={tracking || !trackTicker.trim()}
                  style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 5 }}
                >
                  {tracking ? <Loader2 size={12} className="icon-spin" /> : <Target size={12} />}
                  Track
                </button>
                <button
                  className="btn btn-ghost"
                  onClick={() => { setShowTrackForm(false); setTrackError(null) }}
                  style={{ padding: '4px 8px' }}
                >
                  <X size={12} />
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Tracker list */}
        <ul style={{ listStyle: 'none', margin: 0, padding: 0, overflowY: 'auto', flex: 1 }}>
          {loading && trackers.length === 0 && (
            <li style={{ padding: '20px 16px', color: 'var(--text-muted)', fontSize: '0.85rem', textAlign: 'center' }}>
              <Loader2 size={16} className="icon-spin" style={{ display: 'inline-block' }} />
            </li>
          )}
          {!loading && trackers.length === 0 && (
            <li style={{ padding: '24px 16px', color: 'var(--text-muted)', fontSize: '0.85rem', textAlign: 'center' }}>
              No active trackers.<br />Track a signal above.
            </li>
          )}
          {trackers.map(t => (
            <li key={t.id}>
              <button
                onClick={() => setSelectedId(t.id)}
                style={{
                  width: '100%', textAlign: 'left', padding: '12px 16px',
                  background: selectedId === t.id ? 'var(--bg-elevated)' : 'transparent',
                  borderBottom: '1px solid var(--border)',
                  border: 'none', cursor: 'pointer',
                  borderLeft: selectedId === t.id ? '3px solid #6366f1' : '3px solid transparent',
                  transition: 'background 0.15s',
                  opacity: t.closed ? 0.5 : 1,
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
                  <span style={{ fontWeight: 700, fontSize: '0.95rem', letterSpacing: '-0.01em' }}>
                    {t.ticker}
                  </span>
                  {actionBadge(t.side)}
                </div>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <TpDots tracker={t} />
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    {t.trail_status ?? 'Open'}
                  </span>
                </div>
                {t.trail_active && !t.closed && (
                  <span style={{
                    display: 'inline-block', marginTop: 4,
                    padding: '1px 6px', borderRadius: 4,
                    background: 'rgba(217,119,6,0.12)', color: '#d97706',
                    fontSize: '0.72rem', fontWeight: 700,
                  }}>
                    TRAIL
                  </span>
                )}
              </button>
            </li>
          ))}
        </ul>
      </aside>

      {/* ── Right panel ── */}
      <div style={{ overflowY: 'auto', padding: 24, display: 'flex', flexDirection: 'column', gap: 20 }}>
        {!selected && (
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-muted)', fontSize: '0.9rem' }}>
            Select a tracker or create a new one to get started.
          </div>
        )}

        {selected && (
          <>
            {/* Header row */}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <span style={{ fontSize: '1.5rem', fontWeight: 800, letterSpacing: '-0.03em' }}>
                  {selected.ticker}
                </span>
                {actionBadge(selected.side)}
                {selected.as_of_date && (
                  <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>
                    {selected.as_of_date}
                  </span>
                )}
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <button
                  className="btn btn-sm"
                  onClick={() => handleRefresh(selected)}
                  disabled={refreshingId === selected.id || selected.closed}
                  style={{ display: 'flex', alignItems: 'center', gap: 5 }}
                >
                  {refreshingId === selected.id
                    ? <Loader2 size={13} className="icon-spin" />
                    : <RefreshCw size={13} />
                  }
                  Refresh
                </button>
                <button
                  className="btn btn-sm btn-ghost"
                  onClick={() => handleClose(selected)}
                  title="Close tracker"
                  style={{ display: 'flex', alignItems: 'center', gap: 5 }}
                >
                  <X size={13} /> Close
                </button>
              </div>
            </div>

            {/* Main card: chart + sidebar */}
            <div style={{
              display: 'grid', gridTemplateColumns: '1fr 260px', gap: 20,
              background: 'var(--bg-elevated)',
              borderRadius: 'var(--radius-lg)',
              border: '1px solid var(--border)',
              boxShadow: 'var(--shadow-md)',
              overflow: 'hidden',
            }}>
              {/* Chart */}
              <div style={{ padding: 20 }}>
                <div style={{ fontWeight: 600, fontSize: '0.88rem', color: 'var(--text-secondary)', marginBottom: 12 }}>
                  Price + Levels
                </div>
                <ResponsiveContainer width="100%" height={340}>
                  <ComposedChart data={chartData} margin={{ top: 10, right: 80, bottom: 0, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                    <XAxis
                      dataKey="date"
                      tick={{ fontSize: 11, fill: 'var(--text-muted)' }}
                      tickLine={false}
                      interval="preserveStartEnd"
                    />
                    <YAxis
                      domain={yDomain}
                      tick={{ fontSize: 11, fill: 'var(--text-muted)' }}
                      tickLine={false}
                      axisLine={false}
                      tickFormatter={v => `$${v.toFixed(0)}`}
                      width={55}
                    />
                    <Tooltip
                      contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', borderRadius: 8, fontSize: '0.82rem' }}
                      formatter={(v: number) => [`$${v.toFixed(2)}`, 'Close']}
                    />
                    <Line
                      type="monotone"
                      dataKey="close"
                      stroke="#6366f1"
                      strokeWidth={2}
                      dot={false}
                    />

                    {/* Entry */}
                    <ReferenceLine y={selected.entry} stroke="#6366f1" strokeDasharray="6 3" strokeWidth={1.5}>
                      <Label value={`Entry $${fmt(selected.entry)}`} position="right" fontSize={11} fill="#6366f1" />
                    </ReferenceLine>

                    {/* Trail SL */}
                    <ReferenceLine y={selected.trail_sl} stroke="#dc2626" strokeWidth={1.5}>
                      <Label
                        value={`${selected.trail_active ? 'Trail' : 'SL'} $${fmt(selected.trail_sl)}`}
                        position="right" fontSize={11} fill="#dc2626"
                      />
                    </ReferenceLine>

                    {/* TP1 */}
                    <ReferenceLine
                      y={selected.tp1}
                      stroke="#059669"
                      strokeWidth={selected.tp1_hit ? 2 : 1.5}
                      strokeDasharray={selected.tp1_hit ? undefined : '4 2'}
                    >
                      <Label
                        value={`TP1${selected.tp1_hit ? ' ✓' : ''} $${fmt(selected.tp1)}`}
                        position="right" fontSize={11} fill="#059669"
                      />
                    </ReferenceLine>

                    {/* TP2 */}
                    <ReferenceLine
                      y={selected.tp2}
                      stroke="#059669"
                      strokeWidth={selected.tp2_hit ? 2 : 1.5}
                      strokeDasharray={selected.tp2_hit ? undefined : '4 2'}
                    >
                      <Label
                        value={`TP2${selected.tp2_hit ? ' ✓' : ''} $${fmt(selected.tp2)}`}
                        position="right" fontSize={11} fill="#059669"
                      />
                    </ReferenceLine>

                    {/* TP3 */}
                    <ReferenceLine
                      y={selected.tp3}
                      stroke="#059669"
                      strokeWidth={selected.tp3_hit ? 2 : 1.5}
                      strokeDasharray={selected.tp3_hit ? undefined : '4 2'}
                    >
                      <Label
                        value={`TP3${selected.tp3_hit ? ' ✓' : ''} $${fmt(selected.tp3)}`}
                        position="right" fontSize={11} fill="#059669"
                      />
                    </ReferenceLine>

                    {/* Current price if available */}
                    {selected.current_price && (
                      <ReferenceLine y={selected.current_price} stroke="#d97706" strokeWidth={1.5} strokeDasharray="2 2">
                        <Label value={`Now $${fmt(selected.current_price)}`} position="right" fontSize={11} fill="#d97706" />
                      </ReferenceLine>
                    )}
                  </ComposedChart>
                </ResponsiveContainer>
              </div>

              {/* Sidebar metrics */}
              <div style={{
                borderLeft: '1px solid var(--border)',
                padding: '20px 16px',
                display: 'flex', flexDirection: 'column', gap: 0,
              }}>
                {/* Status banner */}
                <div style={{
                  background: selected.closed
                    ? 'rgba(220,38,38,0.08)'
                    : selected.trail_active
                    ? 'rgba(217,119,6,0.10)'
                    : 'rgba(99,102,241,0.08)',
                  borderRadius: 8, padding: '8px 12px', marginBottom: 16,
                  fontWeight: 700, fontSize: '0.82rem',
                  color: selected.closed
                    ? '#dc2626'
                    : selected.trail_active
                    ? '#d97706'
                    : '#6366f1',
                }}>
                  {selected.trail_status ?? 'Open'}
                </div>

                {/* Levels */}
                <MetricRow label="Entry" value={`$${fmt(selected.entry)}`} />
                <MetricRow label="SL (original)" value={<span style={{ color: '#dc2626' }}>${fmt(selected.sl)}</span>} />
                <MetricRow
                  label={selected.trail_active ? 'Trail SL' : 'SL'}
                  value={<span style={{ color: '#dc2626', fontWeight: 700 }}>${fmt(selected.trail_sl)}</span>}
                />

                <div style={{ margin: '12px 0 4px', fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                  Take Profits
                </div>
                <TpRow label="TP1" price={selected.tp1} hit={selected.tp1_hit} />
                <TpRow label="TP2" price={selected.tp2} hit={selected.tp2_hit} />
                <TpRow label="TP3" price={selected.tp3} hit={selected.tp3_hit} />

                {/* Indicators */}
                <div style={{ margin: '12px 0 4px', fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                  Indicators
                </div>
                <MetricRow label="RSI" value={selected.rsi_at_entry != null ? selected.rsi_at_entry.toFixed(1) : '—'} />
                <MetricRow label="HTF Bias" value={selected.htf_bias != null ? (selected.htf_bias > 0 ? '+' : '') + selected.htf_bias.toFixed(4) : '—'} />
                <MetricRow label="Vol Regime" value={selected.vol_regime ?? '—'} />
                <MetricRow label="Bull Score" value={selected.bull_score != null ? selected.bull_score.toFixed(2) : '—'} />

                {selected.current_price && (
                  <>
                    <div style={{ margin: '12px 0 4px', fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                      Current
                    </div>
                    <MetricRow label="Price" value={<span style={{ color: '#d97706', fontWeight: 700 }}>${fmt(selected.current_price)}</span>} />
                  </>
                )}

                {selected.exit_reason && (
                  <div style={{ marginTop: 12, padding: '8px 12px', borderRadius: 8, background: 'rgba(220,38,38,0.08)', color: '#dc2626', fontSize: '0.82rem', fontWeight: 700 }}>
                    Closed: {selected.exit_reason}
                  </div>
                )}
              </div>
            </div>
          </>
        )}
      </div>

      <style>{`
        .icon-spin { animation: spin 1s linear infinite; }
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        .btn-ghost { background: transparent; border-color: var(--border); }
        .btn-ghost:hover { background: var(--bg-elevated); }
      `}</style>
    </div>
  )
}
