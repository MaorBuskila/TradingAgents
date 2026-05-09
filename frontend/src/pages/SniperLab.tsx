/**
 * SniperLab — Precision Sniper classical WFO + DT ensemble (API-backed).
 */
import { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import { Crosshair, Loader2, RefreshCw, TrendingUp, TrendingDown, Minus, ShieldCheck, ShieldAlert, Activity, Target, BarChart2, Zap, Square } from 'lucide-react'

const API_BASE = import.meta.env.VITE_API_BASE ?? '/api'

interface SniperAlgo {
  optimal_ema_fast: number
  optimal_ema_slow: number
  optimal_ema_trend: number
  optimal_min_score: number
  optimal_sl_mult: number
  optimal_vol_mult: number
  is_sharpe: number
  oos_sharpe: number
  confidence: string
  combos_tested: number
  default_is_sharpe: number
  default_oos_sharpe: number
}

interface SniperDt {
  last_signal: number
  last_prob_a: number
  last_prob_b: number
  threshold_a: number
  threshold_b: number
  oos_sharpe: number
  oos_hit_rate: number
  bull_score_today: number
  grade_veto_ok: boolean
  confidence: string
}

interface RsiWfoResult {
  optimal_period: number
  optimal_upper: number
  optimal_lower: number
  is_sharpe: number
  oos_sharpe: number
  confidence: string
  combos_tested: number
  is_days: number
  oos_days: number
  default_is_sharpe: number
  default_oos_sharpe: number
}

interface MacdWfoResult {
  optimal_fast: number
  optimal_slow: number
  optimal_signal: number
  is_sharpe: number
  oos_sharpe: number
  confidence: string
  combos_tested: number
  is_days: number
  oos_days: number
  default_is_sharpe: number
  default_oos_sharpe: number
}

interface SniperOptimizeResponse {
  symbol: string
  date: string
  classical: SniperAlgo
  dt: SniperDt
  summary: string
  rsi: RsiWfoResult | null
  macd: MacdWfoResult | null
}

interface SniperSignal {
  ticker: string
  as_of_date: string
  action: string | null
  bull_grade: string | null
  bear_grade: string | null
  bull_score: number | null
  latest_close: number | null
  stop_loss: number | null
  tp1: number | null
  tp2: number | null
  tp3: number | null
  vol_regime: string | null
  error?: string | null
}

// ── helpers ────────────────────────────────────────────────────────────────

function confidenceBadge(level: string) {
  const map: Record<string, { bg: string; color: string; dot: string }> = {
    HIGH:   { bg: 'rgba(5,150,105,0.10)',  color: '#059669', dot: '#059669' },
    MEDIUM: { bg: 'rgba(217,119,6,0.10)',  color: '#d97706', dot: '#d97706' },
    LOW:    { bg: 'rgba(220,38,38,0.10)',   color: '#dc2626', dot: '#dc2626' },
  }
  const s = map[level?.toUpperCase()] ?? map.LOW
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 5,
      padding: '3px 10px', borderRadius: 999,
      background: s.bg, color: s.color,
      fontSize: '0.78rem', fontWeight: 700, letterSpacing: '0.04em',
    }}>
      <span style={{ width: 6, height: 6, borderRadius: '50%', background: s.dot, display: 'inline-block' }} />
      {level}
    </span>
  )
}

function sharpeChip(value: number) {
  const good = value > 0
  return (
    <span style={{
      fontWeight: 700, fontSize: '0.95rem',
      color: good ? '#059669' : '#dc2626',
    }}>
      {value > 0 ? '+' : ''}{value.toFixed ? value.toFixed(3) : value}
    </span>
  )
}

function actionBadge(action: string | null) {
  if (!action) return <span style={{ color: 'var(--text-muted)' }}>—</span>
  const a = action.toUpperCase()
  const cfg: Record<string, { bg: string; color: string; Icon: typeof TrendingUp }> = {
    BUY:  { bg: 'rgba(5,150,105,0.12)',  color: '#059669', Icon: TrendingUp },
    SELL: { bg: 'rgba(220,38,38,0.12)',  color: '#dc2626', Icon: TrendingDown },
    HOLD: { bg: 'rgba(99,102,241,0.10)', color: '#6366f1', Icon: Minus },
  }
  const c = cfg[a] ?? cfg.HOLD
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 6,
      padding: '5px 14px', borderRadius: 999,
      background: c.bg, color: c.color,
      fontSize: '0.9rem', fontWeight: 800, letterSpacing: '0.06em',
    }}>
      <c.Icon size={14} strokeWidth={2.5} />
      {a}
    </span>
  )
}

function gradeBadge(grade: string | null, label: string) {
  if (!grade) return null
  const gradeColors: Record<string, { bg: string; color: string }> = {
    A: { bg: 'rgba(5,150,105,0.12)',  color: '#059669' },
    B: { bg: 'rgba(16,185,129,0.10)', color: '#10b981' },
    C: { bg: 'rgba(217,119,6,0.10)',  color: '#d97706' },
    D: { bg: 'rgba(239,68,68,0.10)',  color: '#ef4444' },
    F: { bg: 'rgba(220,38,38,0.12)',  color: '#dc2626' },
  }
  const c = gradeColors[grade.toUpperCase()] ?? gradeColors.C
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 4,
      padding: '2px 8px', borderRadius: 6,
      background: c.bg, color: c.color,
      fontSize: '0.78rem', fontWeight: 700,
    }}>
      {label} {grade}
    </span>
  )
}

function MetricRow({ label, value, sub }: { label: string; value: React.ReactNode; sub?: string }) {
  return (
    <div style={{
      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
      padding: '9px 0',
      borderBottom: '1px solid var(--border)',
    }}>
      <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', fontWeight: 500 }}>
        {label}
        {sub && <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginLeft: 4 }}>{sub}</span>}
      </span>
      <span style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text)' }}>{value}</span>
    </div>
  )
}

function CardSection({ icon, title, accent, children }: {
  icon: React.ReactNode; title: string; accent: string; children: React.ReactNode
}) {
  return (
    <section style={{
      background: 'var(--bg-elevated)',
      borderRadius: 'var(--radius-lg)',
      border: '1px solid var(--border)',
      boxShadow: 'var(--shadow-md)',
      overflow: 'hidden',
    }}>
      <div style={{
        display: 'flex', alignItems: 'center', gap: 10,
        padding: '14px 20px',
        background: `linear-gradient(135deg, ${accent}14 0%, transparent 100%)`,
        borderBottom: '1px solid var(--border)',
      }}>
        <span style={{ color: accent, display: 'flex' }}>{icon}</span>
        <span style={{ fontWeight: 700, fontSize: '0.95rem', letterSpacing: '-0.01em', color: 'var(--text)' }}>
          {title}
        </span>
      </div>
      <div style={{ padding: '4px 20px 16px' }}>
        {children}
      </div>
    </section>
  )
}

// ── component ──────────────────────────────────────────────────────────────

export default function SniperLab() {
  const [symbol, setSymbol] = useState('SPY')
  const [date, setDate] = useState(() => new Date().toISOString().slice(0, 10))
  const [isDays, setIsDays] = useState(180)
  const [oosDays, setOosDays] = useState(90)
  const [dtIs, setDtIs] = useState(1000)
  const [dtOos, setDtOos] = useState(20)
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [opt, setOpt] = useState<SniperOptimizeResponse | null>(null)
  const [sig, setSig] = useState<SniperSignal | null>(null)

  const abortRef = useRef<AbortController | null>(null)

  const SESSION_KEY = 'tradingagents_sniperlab_v1'

  useEffect(() => {
    const saved = sessionStorage.getItem(SESSION_KEY)
    if (!saved) return
    try {
      const s = JSON.parse(saved)
      if (s.symbol) setSymbol(s.symbol)
      if (s.date)   setDate(s.date)
      if (s.isDays) setIsDays(s.isDays)
      if (s.oosDays) setOosDays(s.oosDays)
      if (s.dtIs)   setDtIs(s.dtIs)
      if (s.dtOos)  setDtOos(s.dtOos)
      if (s.opt)    setOpt(s.opt)
      if (s.sig)    setSig(s.sig)
    } catch { /* ignore */ }
  }, [])

  useEffect(() => {
    sessionStorage.setItem(SESSION_KEY, JSON.stringify({ symbol, date, isDays, oosDays, dtIs, dtOos, opt, sig }))
  }, [symbol, date, isDays, oosDays, dtIs, dtOos, opt, sig])

  async function runOptimize() {
    const controller = new AbortController()
    abortRef.current = controller
    setLoading(true)
    setErr(null)
    setOpt(null)
    setSig(null)
    try {
      const { data } = await axios.post<SniperOptimizeResponse>(`${API_BASE}/sniper-optimize`, {
        symbol: symbol.trim().toUpperCase(),
        date,
        is_days: isDays,
        oos_days: oosDays,
        dt_is_days: dtIs,
        dt_oos_days: dtOos,
      }, { signal: controller.signal })
      setOpt(data)
      const sg = await axios.get<SniperSignal>(`${API_BASE}/sniper-signal/${data.symbol}`, {
        params: { date },
        signal: controller.signal,
      })
      setSig(sg.data)
    } catch (e: unknown) {
      if (axios.isCancel(e) || (e instanceof Error && (e.name === 'CanceledError' || e.name === 'AbortError'))) return
      const msg = axios.isAxiosError(e) ? e.response?.data?.detail ?? e.message : String(e)
      setErr(String(msg))
    } finally {
      setLoading(false)
    }
  }

  const stopOptimize = () => { abortRef.current?.abort(); setLoading(false) }

  const fields: Array<{
    label: string
    type?: string
    min?: number
    value: string | number
    onChange: (v: string) => void
  }> = [
    { label: 'Symbol',         value: symbol,  onChange: (v) => setSymbol(v) },
    { label: 'As-of date',     type: 'date',   value: date,   onChange: (v) => setDate(v) },
    { label: 'Classical IS days', type: 'number', min: 60,  value: isDays,  onChange: (v) => setIsDays(Number(v)) },
    { label: 'Classical OOS days',type: 'number', min: 20,  value: oosDays, onChange: (v) => setOosDays(Number(v)) },
    { label: 'DT IS bars',     type: 'number', min: 300, value: dtIs,    onChange: (v) => setDtIs(Number(v)) },
    { label: 'DT OOS step',    type: 'number', min: 5,   value: dtOos,   onChange: (v) => setDtOos(Number(v)) },
  ]

  return (
    <div className="page sniper-lab">
      {/* ── Header ── */}
      <header className="page-header" style={{ marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
          <span style={{
            width: 38, height: 38, borderRadius: 10,
            background: 'linear-gradient(135deg, #6366f1 0%, #818cf8 100%)',
            display: 'grid', placeItems: 'center', color: '#fff',
            boxShadow: '0 4px 14px rgba(99,102,241,0.35)',
            flexShrink: 0,
          }}>
            <Crosshair size={18} />
          </span>
          <h1 style={{ fontSize: 'clamp(1.4rem,2.5vw,1.75rem)', fontWeight: 800, letterSpacing: '-0.03em', color: 'var(--text)', lineHeight: 1.2 }}>
            Precision Sniper Lab
          </h1>
        </div>
        {/* Explainer card */}
        <div style={{
          background: 'var(--bg-elevated)', border: '1px solid var(--border)',
          borderRadius: 'var(--radius-lg)', padding: '1.1rem 1.4rem',
          boxShadow: 'var(--shadow-sm)', marginTop: 8,
        }}>
          <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', lineHeight: 1.65, marginBottom: 12 }}>
            <strong style={{ color: 'var(--text)' }}>Precision Sniper</strong> is a two-layer entry filter designed for high-conviction trades only.
            Layer 1 — <em>Classical WFO</em> — finds the best EMA stack (fast / slow / trend) and minimum score threshold via R-multiple walk-forward optimisation.
            Layer 2 — <em>DT Ensemble</em> — two Decision Tree models (sliding + crash-aware) vote on today's market regime using histogram slope, signal gap, volatility ratio, Bollinger %B and ATR.
            A LONG signal fires <strong>only</strong> when both layers agree. Results are cached to{' '}
            <code style={{ fontFamily: 'var(--font-mono)', fontSize: '0.84em', padding: '0.15em 0.4em', background: 'var(--bg-subtle)', borderRadius: 6, border: '1px solid var(--border)' }}>sniper_cache.db</code> — no recompute on re-run.
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(210px,1fr))', gap: 10 }}>
            {[
              { emoji: '📐', title: 'Classical WFO card', body: 'Shows the winning EMA stack, min-score gate, stop-loss & volatility multipliers, plus IS vs OOS R-Sharpe. LOW confidence means params were unstable across folds — trade with caution.' },
              { emoji: '🤖', title: 'DT Ensemble card', body: 'Two models score today\'s regime. Probs A/B are the raw model probabilities vs their thresholds. Grade veto: if the LLM researcher grades the stock ≤ C, the signal is suppressed regardless of DT output.' },
              { emoji: '🎯', title: 'Latest Signal card', body: 'The combined output: action (BUY/HOLD/SELL), bull/bear LLM grades, today\'s close, the computed stop-loss, and three take-profit targets (TP1 = 1R, TP2 = 2R, TP3 = 3R). Vol regime flags HIGH/NORMAL/LOW volatility.' },
              { emoji: '⚙️', title: 'Parameters', body: 'IS days = how many days to train on. OOS days = held-out test window. DT IS bars = how many daily bars the DT models train on. DT OOS step = re-train frequency. Defaults match the framework\'s standard WFO schedule.' },
            ].map(({ emoji, title, body }) => (
              <div key={title} style={{
                background: 'var(--bg-subtle)', borderRadius: 'var(--radius-sm)',
                border: '1px solid var(--border)', padding: '10px 12px',
              }}>
                <div style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--text)', marginBottom: 4 }}>{emoji} {title}</div>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>{body}</div>
              </div>
            ))}
          </div>
        </div>
      </header>

      {/* ── Form card ── */}
      <section style={{
        background: 'var(--bg-elevated)',
        borderRadius: 'var(--radius-lg)',
        padding: 'clamp(1.25rem,2vw,1.75rem)',
        border: '1px solid var(--border)',
        boxShadow: 'var(--shadow-md)',
        marginBottom: '1.25rem',
      }}>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(170px,1fr))', gap: 14, marginBottom: 20 }}>
          {fields.map((f) => (
            <label key={f.label} style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              <span style={{
                fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase',
                letterSpacing: '0.06em', color: 'var(--text-secondary)',
              }}>
                {f.label}
              </span>
              <input
                type={f.type ?? 'text'}
                min={f.min}
                value={f.value}
                onChange={(e) => f.onChange(e.target.value)}
                style={{
                  padding: '9px 12px',
                  borderRadius: 'var(--radius-sm)',
                  border: '1.5px solid var(--border-strong)',
                  background: 'var(--bg-elevated)',
                  color: 'var(--text)',
                  fontSize: '0.9375rem',
                  fontFamily: 'inherit',
                  outline: 'none',
                  transition: 'border-color 0.15s, box-shadow 0.15s',
                  width: '100%',
                }}
                onFocus={(e) => {
                  e.currentTarget.style.borderColor = 'var(--accent)'
                  e.currentTarget.style.boxShadow = '0 0 0 3px var(--accent-ring)'
                }}
                onBlur={(e) => {
                  e.currentTarget.style.borderColor = 'var(--border-strong)'
                  e.currentTarget.style.boxShadow = 'none'
                }}
              />
            </label>
          ))}
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap' }}>
          <button
            type="button"
            className="btn"
            disabled={loading}
            onClick={runOptimize}
            style={{ gap: 8, paddingLeft: 20, paddingRight: 20 }}
          >
            {loading ? <Loader2 size={16} className="icon-spin" /> : <RefreshCw size={16} />}
            {loading ? 'Running…' : 'Run optimize'}
          </button>
          {loading && (
            <>
              <button onClick={stopOptimize} style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '0.5rem 1rem', background: 'var(--danger)', color: '#fff', border: 'none', borderRadius: 'var(--radius-sm)', cursor: 'pointer', fontWeight: 600, fontSize: '0.875rem' }}>
                <Square size={14} /> Stop
              </button>
              <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                Walk-forward optimization in progress…
              </span>
            </>
          )}
        </div>

        {err && (
          <div style={{
            marginTop: 14, padding: '10px 14px',
            background: 'var(--danger-soft)', borderLeft: '3px solid var(--danger)',
            borderRadius: 'var(--radius-sm)', color: '#991b1b', fontSize: '0.875rem',
          }}>
            {err}
          </div>
        )}
      </section>

      {/* ── Summary banner ── */}
      {opt && (
        <div style={{
          padding: '12px 18px', marginBottom: 16,
          background: 'var(--accent-soft)',
          borderRadius: 'var(--radius-md)',
          border: '1px solid rgba(99,102,241,0.2)',
          fontSize: '0.875rem', color: 'var(--accent-hover)', lineHeight: 1.55,
          display: 'flex', alignItems: 'flex-start', gap: 10,
        }}>
          <Activity size={16} style={{ flexShrink: 0, marginTop: 2 }} />
          <span>{opt.summary}</span>
        </div>
      )}

      {/* ── WFO + DT cards ── */}
      {opt && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
          {/* Classical WFO */}
          <CardSection icon={<BarChart2 size={17} />} title="Classical WFO" accent="#6366f1">
            <MetricRow
              label="EMA stack"
              value={
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.875rem', letterSpacing: '0.02em' }}>
                  {opt.classical.optimal_ema_fast}&thinsp;/&thinsp;{opt.classical.optimal_ema_slow}&thinsp;/&thinsp;{opt.classical.optimal_ema_trend}
                </span>
              }
            />
            <MetricRow label="Min score" value={opt.classical.optimal_min_score} />
            <MetricRow
              label="SL / Vol mult"
              value={
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.875rem' }}>
                  {opt.classical.optimal_sl_mult}&thinsp;/&thinsp;{opt.classical.optimal_vol_mult}
                </span>
              }
            />
            <MetricRow
              label="IS / OOS Sharpe (R)"
              value={
                <span style={{ display: 'flex', gap: 6 }}>
                  {sharpeChip(opt.classical.is_sharpe)}
                  <span style={{ color: 'var(--text-muted)' }}>/</span>
                  {sharpeChip(opt.classical.oos_sharpe)}
                </span>
              }
            />
            <MetricRow
              label="vs default OOS"
              value={sharpeChip(opt.classical.default_oos_sharpe)}
            />
            <MetricRow
              label="Combos tested"
              value={<span style={{ color: 'var(--text-muted)', fontWeight: 500 }}>{opt.classical.combos_tested.toLocaleString()}</span>}
            />
            <div style={{ paddingTop: 10, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', fontWeight: 500 }}>Confidence</span>
              {confidenceBadge(opt.classical.confidence)}
            </div>
          </CardSection>

          {/* DT ensemble */}
          <CardSection icon={<Zap size={17} />} title="DT Ensemble" accent="#f59e0b">
            <MetricRow
              label="OOS Sharpe / hit rate"
              value={
                <span style={{ display: 'flex', gap: 6 }}>
                  {sharpeChip(opt.dt.oos_sharpe)}
                  <span style={{ color: 'var(--text-muted)' }}>/</span>
                  <span style={{ fontWeight: 700, color: 'var(--text)' }}>
                    {(opt.dt.oos_hit_rate * 100).toFixed(1)}%
                  </span>
                </span>
              }
            />
            <MetricRow
              label="Thresholds A / B"
              value={
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.875rem' }}>
                  {opt.dt.threshold_a}&thinsp;/&thinsp;{opt.dt.threshold_b}
                </span>
              }
            />
            <MetricRow
              label="Probs A / B"
              value={
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.875rem' }}>
                  {opt.dt.last_prob_a.toFixed ? opt.dt.last_prob_a.toFixed(4) : opt.dt.last_prob_a}&thinsp;/&thinsp;
                  {opt.dt.last_prob_b.toFixed ? opt.dt.last_prob_b.toFixed(4) : opt.dt.last_prob_b}
                </span>
              }
            />
            <MetricRow
              label="Signal / bull score"
              value={
                <span style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                  <span style={{ fontWeight: 700, color: opt.dt.last_signal > 0 ? '#059669' : 'var(--text-muted)' }}>
                    {opt.dt.last_signal}
                  </span>
                  <span style={{ color: 'var(--text-muted)' }}>/</span>
                  <span style={{ fontWeight: 600, color: 'var(--text)' }}>{opt.dt.bull_score_today}</span>
                  <span style={{
                    fontSize: '0.75rem', fontWeight: 600,
                    padding: '1px 7px', borderRadius: 6,
                    background: opt.dt.grade_veto_ok ? 'rgba(5,150,105,0.1)' : 'rgba(220,38,38,0.1)',
                    color: opt.dt.grade_veto_ok ? '#059669' : '#dc2626',
                  }}>
                    {opt.dt.grade_veto_ok ? 'veto ok' : 'veto no'}
                  </span>
                </span>
              }
            />
            <div style={{ paddingTop: 10, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', fontWeight: 500 }}>Confidence</span>
              {confidenceBadge(opt.dt.confidence)}
            </div>
          </CardSection>
        </div>
      )}

      {/* ── RSI WFO + MACD WFO cards ── */}
      {opt && (opt.rsi || opt.macd) && (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>

          {/* RSI Classical WFO */}
          {opt.rsi && (
            <CardSection icon={<Activity size={17} />} title="RSI Classical WFO" accent="#8b5cf6">
              <div style={{
                fontSize: '0.72rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em',
                color: 'var(--text-muted)', marginTop: 10, marginBottom: 6,
              }}>
                Optimal parameters · used as <code style={{ fontFamily: 'var(--font-mono)', fontSize: '0.9em', background: 'rgba(139,92,246,0.1)', padding: '1px 5px', borderRadius: 4 }}>rsi_opt</code> feature in DT
              </div>
              <MetricRow
                label="Period"
                value={
                  <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 800, fontSize: '1rem', color: '#8b5cf6' }}>
                    {opt.rsi.optimal_period}
                  </span>
                }
              />
              <MetricRow
                label="Upper / Lower thresholds"
                value={
                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.875rem' }}>
                    {opt.rsi.optimal_upper}&thinsp;/&thinsp;{opt.rsi.optimal_lower}
                  </span>
                }
              />
              <MetricRow
                label="IS / OOS Sharpe"
                value={
                  <span style={{ display: 'flex', gap: 6 }}>
                    {sharpeChip(opt.rsi.is_sharpe)}
                    <span style={{ color: 'var(--text-muted)' }}>/</span>
                    {sharpeChip(opt.rsi.oos_sharpe)}
                  </span>
                }
              />
              <MetricRow label="vs default OOS" value={sharpeChip(opt.rsi.default_oos_sharpe)} />
              <MetricRow
                label="Combos tested"
                value={<span style={{ color: 'var(--text-muted)', fontWeight: 500 }}>{opt.rsi.combos_tested.toLocaleString()}</span>}
              />
              <div style={{ paddingTop: 10, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', fontWeight: 500 }}>Confidence</span>
                {confidenceBadge(opt.rsi.confidence)}
              </div>
            </CardSection>
          )}

          {/* MACD Classical WFO */}
          {opt.macd && (
            <CardSection icon={<BarChart2 size={17} />} title="MACD Classical WFO" accent="#e11d48">
              <div style={{
                fontSize: '0.72rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em',
                color: 'var(--text-muted)', marginTop: 10, marginBottom: 6,
              }}>
                Optimal parameters · used as <code style={{ fontFamily: 'var(--font-mono)', fontSize: '0.9em', background: 'rgba(225,29,72,0.1)', padding: '1px 5px', borderRadius: 4 }}>macd_hist_opt</code> feature in DT
              </div>
              <MetricRow
                label="Fast / Slow / Signal"
                value={
                  <span style={{ fontFamily: 'var(--font-mono)', fontWeight: 800, fontSize: '0.95rem', color: '#e11d48' }}>
                    {opt.macd.optimal_fast}&thinsp;/&thinsp;{opt.macd.optimal_slow}&thinsp;/&thinsp;{opt.macd.optimal_signal}
                  </span>
                }
              />
              <MetricRow
                label="IS / OOS Sharpe"
                value={
                  <span style={{ display: 'flex', gap: 6 }}>
                    {sharpeChip(opt.macd.is_sharpe)}
                    <span style={{ color: 'var(--text-muted)' }}>/</span>
                    {sharpeChip(opt.macd.oos_sharpe)}
                  </span>
                }
              />
              <MetricRow label="vs default OOS" value={sharpeChip(opt.macd.default_oos_sharpe)} />
              <MetricRow
                label="Combos tested"
                value={<span style={{ color: 'var(--text-muted)', fontWeight: 500 }}>{opt.macd.combos_tested.toLocaleString()}</span>}
              />
              <div style={{ paddingTop: 10, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', fontWeight: 500 }}>Confidence</span>
                {confidenceBadge(opt.macd.confidence)}
              </div>
            </CardSection>
          )}
        </div>
      )}

      {/* ── Latest signal ── */}
      {sig && !sig.error && (
        <CardSection icon={<Target size={17} />} title="Latest Signal" accent="#059669">
          {/* Action hero row */}
          <div style={{
            display: 'flex', alignItems: 'center', justifyContent: 'space-between',
            padding: '14px 0 10px', borderBottom: '1px solid var(--border)', flexWrap: 'wrap', gap: 12,
          }}>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
              <span style={{ fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)' }}>
                Action
              </span>
              {actionBadge(sig.action)}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4, alignItems: 'flex-end' }}>
              <span style={{ fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)' }}>
                Grades
              </span>
              <div style={{ display: 'flex', gap: 6 }}>
                {gradeBadge(sig.bull_grade, 'Bull')}
                {gradeBadge(sig.bear_grade, 'Bear')}
              </div>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 4, alignItems: 'flex-end' }}>
              <span style={{ fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)' }}>
                Close
              </span>
              <span style={{ fontSize: '1.25rem', fontWeight: 800, letterSpacing: '-0.02em', color: 'var(--text)' }}>
                {sig.latest_close != null ? `$${sig.latest_close.toFixed ? sig.latest_close.toFixed(2) : sig.latest_close}` : '—'}
              </span>
            </div>
          </div>

          {/* Price levels */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 12, padding: '14px 0 6px' }}>
            {[
              { label: 'Stop Loss', value: sig.stop_loss, color: '#dc2626', icon: <ShieldAlert size={13} /> },
              { label: 'TP 1',      value: sig.tp1,       color: '#059669', icon: <ShieldCheck size={13} /> },
              { label: 'TP 2',      value: sig.tp2,       color: '#059669', icon: <ShieldCheck size={13} /> },
              { label: 'TP 3',      value: sig.tp3,       color: '#059669', icon: <ShieldCheck size={13} /> },
            ].map(({ label, value, color, icon }) => (
              <div key={label} style={{
                padding: '10px 12px',
                background: value != null ? `${color}0d` : 'var(--bg-subtle)',
                border: `1px solid ${value != null ? color + '30' : 'var(--border)'}`,
                borderRadius: 'var(--radius-sm)',
                display: 'flex', flexDirection: 'column', gap: 4,
              }}>
                <span style={{ display: 'flex', alignItems: 'center', gap: 4, color: value != null ? color : 'var(--text-muted)', fontSize: '0.72rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                  {icon}{label}
                </span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.9rem', fontWeight: 700, color: value != null ? color : 'var(--text-muted)' }}>
                  {value != null ? (value as number).toFixed ? `$${(value as number).toFixed(2)}` : `$${value}` : '—'}
                </span>
              </div>
            ))}
          </div>

          {/* Vol regime */}
          {sig.vol_regime && (
            <div style={{ paddingTop: 4, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', fontWeight: 500 }}>Vol regime</span>
              <span style={{
                padding: '3px 10px', borderRadius: 999,
                background: sig.vol_regime === 'HIGH' ? 'rgba(239,68,68,0.1)' : sig.vol_regime === 'LOW' ? 'rgba(5,150,105,0.1)' : 'rgba(99,102,241,0.1)',
                color: sig.vol_regime === 'HIGH' ? '#ef4444' : sig.vol_regime === 'LOW' ? '#059669' : '#6366f1',
                fontSize: '0.78rem', fontWeight: 700, letterSpacing: '0.05em',
              }}>
                {sig.vol_regime}
              </span>
            </div>
          )}
        </CardSection>
      )}

      <style>{`
        @media (max-width: 800px) {
          .sniper-lab > div[style*="grid-template-columns: 1fr 1fr"] {
            grid-template-columns: 1fr !important;
          }
          .sniper-lab div[style*="repeat(4,1fr)"] {
            grid-template-columns: 1fr 1fr !important;
          }
        }
      `}</style>
    </div>
  )
}
