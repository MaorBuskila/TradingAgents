/**
 * RSILab.tsx
 * ==========
 * RSI Optimizer Lab — side-by-side comparison of Pure Algo vs LLM-Enhanced optimizer.
 *
 * What this page does:
 *   1. User enters a stock symbol + date + window sizes
 *   2. Clicks "Run Optimization" → API calls both optimizers in parallel
 *   3. Shows results side by side:
 *       - Optimal RSI period, thresholds, Sharpe scores, confidence
 *       - Bar chart: Sharpe by period for both optimizers overlaid
 *       - LLM reasoning panel (regime + explanation)
 *       - Auto-generated comparison summary (who won and why)
 *
 * React concepts used here (good to know as a beginner):
 *   - useState: store data that can change (form values, results, loading state)
 *   - async/await: call the API without freezing the page
 *   - conditional rendering: show loading spinner OR results based on state
 *   - recharts: charting library for the bar chart
 */

import { useState, useEffect, useRef, useCallback } from 'react'
import axios from 'axios'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ReferenceLine, LineChart, Line, Cell, ComposedChart,
} from 'recharts'
import { FlaskConical, TrendingUp, Brain, Database, Trash2, RotateCcw, Zap, TrendingDown, Minus, Square } from 'lucide-react'

const API_BASE = import.meta.env.VITE_API_BASE ?? '/api'

// ── TypeScript types (match the API response models) ─────────────────────────

interface PeriodPoint {
  period: number
  is_sharpe: number
  oos_sharpe: number
}

// Log line type for coloring
type LogLine = { text: string; level: 'info' | 'success' | 'error' | 'warn' | 'data' }

function classifyLog(line: string): LogLine {
  if (line.startsWith('✅') || line.startsWith('🏆')) return { text: line, level: 'success' }
  if (line.startsWith('❌'))                            return { text: line, level: 'error' }
  if (line.startsWith('⚠️') || line.startsWith('⚡'))  return { text: line, level: 'warn' }
  if (line.startsWith('   ') || line.startsWith('📄')) return { text: line, level: 'data' }
  return { text: line, level: 'info' }
}

interface OptimizerResult {
  optimal_period: number
  optimal_upper: number
  optimal_lower: number
  is_sharpe: number
  oos_sharpe: number
  confidence: 'HIGH' | 'MEDIUM' | 'LOW'
  combos_tested: number
  is_days: number
  oos_days: number
  default_is_sharpe: number
  default_oos_sharpe: number
  period_sharpes: PeriodPoint[]
  // LLM-only
  regime?: string
  llm_reasoning?: string
  llm_available?: boolean
  llm_grid?: { period_range: number[]; upper_range: number[]; lower_range: number[] }
  regime_signals?: Record<string, any>
  debug_logs?: string[]
  token_usage?: {
    model: string
    prompt_tokens: number
    completion_tokens: number
    total_tokens: number
    cost_usd: number
  }
}

interface ComparisonSummary {
  period_agreement: boolean
  oos_winner: 'algo' | 'llm' | 'tie'
  oos_winner_sharpe: number
  algo_beats_default: boolean
  llm_beats_default: boolean
  combos_reduction_pct: number
  summary: string
}

interface CachedParam {
  ticker: string
  optimal_period: number
  optimal_upper: number
  optimal_lower: number
  oos_sharpe: number
  confidence: string
  regime: string
  optimizer_provider: string
  optimized_at: string
}

interface RsiHistoryPoint {
  date: string
  rsi: number | null
  close: number
}

interface RsiSignal {
  ticker: string
  as_of_date: string
  action: 'BUY' | 'SELL' | 'HOLD' | null
  current_rsi: number | null
  rsi_zone: string | null
  period: number | null
  overbought: number | null
  oversold: number | null
  using_cached: boolean
  confidence: string | null
  regime: string | null
  optimizer_provider: string | null
  optimized_at: string | null
  latest_close: number | null
  price_change_pct: number | null
  rsi_history: RsiHistoryPoint[]
  error: string | null
}

interface OptimizeResponse {
  symbol: string
  date: string
  algo: OptimizerResult
  llm: OptimizerResult
  comparison: ComparisonSummary
}

// ── OOS Test Types ────────────────────────────────────────────────────────────

interface OosTestDateRow {
  date: string
  fast?: number; slow?: number; signal_period?: number
  period?: number; upper?: number; lower?: number
  is_sharpe?: number; oos_sharpe?: number
  confidence?: string
  pnl_pct?: number; pnl_dollar?: number
  win_rate?: number; max_dd_pct?: number
  n_trades?: number; error?: string
}

interface OosTestSummary {
  total_pnl_dollar: number; win_rate_pct: number
  n_wins: number; n_losses: number
  avg_win_dollar: number; avg_loss_dollar: number
  profit_factor: number
  avg_oos_sharpe: number; avg_is_sharpe: number
  high_conf: number; med_conf: number; low_conf: number
}

interface OosTestResponse {
  symbol: string; oos_start: string; oos_end: string
  is_days: number; oos_days: number; notional: number
  n_requested: number; n_succeeded: number
  rows: OosTestDateRow[]; summary: OosTestSummary; algo_type: string
}

// ── RSI DT Optimizer Types ────────────────────────────────────────────────────

interface RsiDtSlide {
  slide_idx:  number
  is_start:   number
  is_end:     number
  oos_start:  number
  oos_end:    number
  train_acc:  number
  oos_acc:    number
}

interface RsiDtResult {
  symbol:              string
  curr_date:           string
  last_signal:         number
  last_prob_a:         number
  last_prob_b:         number
  last_size_mult:      number
  last_atr_pct:        number
  last_rsi:            number
  avg_oos_acc:         number
  n_slides:            number
  feature_names:       string[]
  rsi_params:          { period: number; upper: number; lower: number }
  wfo_params_used:     boolean
  is_days:             number
  oos_days:            number
  label_horizon:       number
  min_prob_threshold:  number
  slides:              RsiDtSlide[]
}

interface RsiDtPosition {
  symbol:        string
  signal:        number
  prob_a:        number
  prob_b:        number
  atr_pct:       number
  last_rsi:      number
  size_mult:     number
  alloc_pct:     number
  alloc_dollars: number
  avg_oos_acc:   number
}

interface RsiDtPortfolioResult {
  positions:           RsiDtPosition[]
  rejected_liquidity:  string[]
  rejected_no_signal:  string[]
  rejected_errors:     string[]
  total_deployed_pct:  number
  total_deployed_usd:  number
  capital:             number
  curr_date:           string
  n_evaluated:         number
  n_selected:          number
  base_alloc_pct:      number
  max_positions:       number
  min_dollar_vol:      number
}

// ── Helper: confidence badge color ───────────────────────────────────────────

function confidenceStyle(level: string): React.CSSProperties {
  if (level === 'HIGH')   return { color: 'var(--success)', fontWeight: 700 }
  if (level === 'MEDIUM') return { color: 'var(--warning)', fontWeight: 700 }
  return { color: 'var(--danger)', fontWeight: 700 }
}

// ── Helper: Sharpe value coloring ────────────────────────────────────────────

function sharpeStyle(v: number): React.CSSProperties {
  if (v >= 1.0)  return { color: 'var(--success)', fontWeight: 600 }
  if (v >= 0.5)  return { color: 'var(--warning)', fontWeight: 600 }
  if (v >= 0)    return { color: 'var(--text-secondary)' }
  return { color: 'var(--danger)', fontWeight: 600 }
}

// ── Merge chart data: combine both period_sharpes arrays for the chart ────────

function mergeChartData(algo: PeriodPoint[], llm: PeriodPoint[]) {
  // Build a map from period → { period, algo_is, algo_oos, llm_is, llm_oos }
  const map: Record<number, any> = {}

  for (const p of algo) {
    map[p.period] = {
      period: p.period,
      'Algo IS':  p.is_sharpe,
      'Algo OOS': p.oos_sharpe,
    }
  }
  for (const p of llm) {
    if (!map[p.period]) map[p.period] = { period: p.period }
    map[p.period]['LLM IS']  = p.is_sharpe
    map[p.period]['LLM OOS'] = p.oos_sharpe
  }

  return Object.values(map).sort((a: any, b: any) => a.period - b.period)
}

// ── Main component ────────────────────────────────────────────────────────────

export default function RSILab() {
  // Provider → available models (matches llm_catalog.py)
  const LLM_MODELS: Record<string, { label: string; value: string }[]> = {
    google: [
      { label: 'Gemini 2.5 Flash — balanced, stable', value: 'gemini-2.5-flash' },
      { label: 'Gemini 2.5 Flash Lite — fast, low cost', value: 'gemini-2.5-flash-lite' },
      { label: 'Gemini 3 Flash — next-gen fast', value: 'gemini-3-flash-preview' },
      { label: 'Gemini 2.5 Pro — best quality', value: 'gemini-2.5-pro' },
    ],
    openai: [
      { label: 'GPT-5 Mini — balanced speed & cost', value: 'gpt-5-mini' },
      { label: 'GPT-5 Nano — high-throughput', value: 'gpt-5-nano' },
      { label: 'GPT-4.1 — smartest non-reasoning', value: 'gpt-4.1' },
      { label: 'GPT-5.4 — latest frontier', value: 'gpt-5.4' },
    ],
    anthropic: [
      { label: 'Claude Haiku 4.5 — fast, low cost', value: 'claude-haiku-4-5' },
      { label: 'Claude Sonnet 4.6 — balanced', value: 'claude-sonnet-4-6' },
      { label: 'Claude Opus 4.6 — most intelligent', value: 'claude-opus-4-6' },
    ],
    ollama: [
      { label: 'Gemma 4 E4B — 9.6GB, 128K ctx, multimodal (default)', value: 'gemma4:e4b' },
      { label: 'Llama 4 Scout — 67GB, 16 experts, 10M ctx', value: 'llama4:scout' },
      { label: 'Llama 4 Maverick — 245GB, 128 experts, 1M ctx', value: 'llama4:maverick' },
      { label: 'Llama 4 Latest — alias for Scout', value: 'llama4:latest' },
      { label: 'Llama 4 Scout [16x17b] — explicit tag', value: 'llama4:16x17b' },
      { label: 'Llama 4 Maverick [128x17b] — explicit tag', value: 'llama4:128x17b' },
    ],
    heuristic: [],
  }

  // ── Tab state ──────────────────────────────────────────────────────────────
  const [activeTab, setActiveTab] = useState<'classic' | 'dt' | 'oos-test'>('classic')

  // ── Classic WFO state ──────────────────────────────────────────────────────
  const [symbol,      setSymbol]      = useState('FORM')
  const [date,        setDate]        = useState(new Date().toISOString().slice(0, 10))
  const [isDays,      setIsDays]      = useState(1500)
  const [oosDays,     setOosDays]     = useState(500)
  const [llmProvider, setLlmProvider] = useState('ollama')
  const [llmModel,    setLlmModel]    = useState('gemma4:e4b')

  // Result state
  const [result,  setResult]  = useState<OptimizeResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState<string | null>(null)

  // Cache table state
  const [cachedParams,   setCachedParams]   = useState<CachedParam[]>([])
  const [cacheLoading,   setCacheLoading]   = useState(false)
  const [deletingTicker, setDeletingTicker] = useState<string | null>(null)

  // Signal panel state
  const [signals,        setSignals]        = useState<RsiSignal[]>([])
  const [signalsLoading, setSignalsLoading] = useState(false)
  const [expandedTicker, setExpandedTicker] = useState<string | null>(null)

  // ── DT Optimizer state ─────────────────────────────────────────────────────
  const [dtSymbol,       setDtSymbol]       = useState('AAPL')
  const [dtDate,         setDtDate]         = useState(new Date().toISOString().slice(0, 10))
  const [dtIsDays,       setDtIsDays]       = useState(100)
  const [dtOosDays,      setDtOosDays]      = useState(30)
  const [dtLabelHorizon, setDtLabelHorizon] = useState(10)
  const [dtResult,       setDtResult]       = useState<RsiDtResult | null>(null)
  const [dtLoading,      setDtLoading]      = useState(false)
  const [dtError,        setDtError]        = useState<string | null>(null)
  // DT param source
  const [dtParamSource,  setDtParamSource]  = useState<'default' | 'db' | 'manual'>('default')
  const [manualPeriod,   setManualPeriod]   = useState(14)
  const [manualUpper,    setManualUpper]    = useState(70)
  const [manualLower,    setManualLower]    = useState(30)
  const [dbCachedParams, setDbCachedParams] = useState<{
    optimal_period: number; optimal_upper: number; optimal_lower: number;
    confidence: string; oos_sharpe: number
  } | null>(null)
  const [dbFetchStatus,  setDbFetchStatus]  = useState<'idle' | 'loading' | 'found' | 'not_found'>('idle')

  // ── DT Portfolio state ─────────────────────────────────────────────────────
  const [pfSymbols,  setPfSymbols]  = useState('AAPL,MSFT,NVDA,TSLA,AMZN,META,GOOG,NFLX,AMD,INTC')
  const [pfCapital,  setPfCapital]  = useState(100000)
  const [pfResult,   setPfResult]   = useState<RsiDtPortfolioResult | null>(null)
  const [pfLoading,  setPfLoading]  = useState(false)
  const [pfError,    setPfError]    = useState<string | null>(null)

  const loadCache = useCallback(async () => {
    setCacheLoading(true)
    try {
      const res = await axios.get(`${API_BASE}/rsi-cache`)
      setCachedParams(res.data.entries ?? [])
    } catch {
      // silently ignore — cache may be empty
    } finally {
      setCacheLoading(false)
    }
  }, [])

  const loadSignals = useCallback(async () => {
    setSignalsLoading(true)
    try {
      const res = await axios.get(`${API_BASE}/rsi-signals`)
      setSignals(res.data ?? [])
    } catch {
      // silently ignore
    } finally {
      setSignalsLoading(false)
    }
  }, [])

  // Load cache + signals on mount
  useEffect(() => { loadCache(); loadSignals() }, [loadCache, loadSignals])

  // Reset DB fetch state whenever the DT symbol changes
  useEffect(() => {
    setDbFetchStatus('idle')
    setDbCachedParams(null)
  }, [dtSymbol])

  const handleDeleteCache = async (ticker: string) => {
    setDeletingTicker(ticker)
    try {
      await axios.delete(`${API_BASE}/rsi-cache/${ticker}`)
      setCachedParams(prev => prev.filter(r => r.ticker !== ticker))
    } catch {
      // ignore
    } finally {
      setDeletingTicker(null)
    }
  }

  // ── DT API calls ────────────────────────────────────────────────────────
  const fetchDbParams = async (ticker: string) => {
    setDbFetchStatus('loading')
    setDbCachedParams(null)
    try {
      const res = await axios.get(`${API_BASE}/rsi-cache/${ticker.toUpperCase().trim()}`)
      setDbCachedParams(res.data)
      setDbFetchStatus('found')
    } catch (err: any) {
      setDbFetchStatus('not_found')
    }
  }

  const runDtOptimize = async () => {
    const controller = new AbortController()
    abortDtRef.current = controller
    setDtLoading(true); setDtError(null); setDtResult(null)
    try {
      const resolvedParams =
        dtParamSource === 'db' && dbCachedParams
          ? { rsi_period: dbCachedParams.optimal_period, rsi_upper: dbCachedParams.optimal_upper, rsi_lower: dbCachedParams.optimal_lower }
        : dtParamSource === 'manual'
          ? { rsi_period: manualPeriod, rsi_upper: manualUpper, rsi_lower: manualLower }
        : {}
      const res = await axios.post(`${API_BASE}/rsi-dt-optimize`, {
        symbol:        dtSymbol.toUpperCase().trim(),
        date:          dtDate,
        is_days:       dtIsDays,
        oos_days:      dtOosDays,
        label_horizon: dtLabelHorizon,
        ...resolvedParams,
      }, { signal: controller.signal })
      setDtResult(res.data)
    } catch (err: any) {
      if (err.name === 'CanceledError' || err.code === 'ERR_CANCELED') return
      setDtError(err.response?.data?.detail ?? err.message ?? 'Unknown error')
    } finally {
      setDtLoading(false)
    }
  }

  const stopDtOptimize = () => { abortDtRef.current?.abort(); setDtLoading(false) }

  const runPortfolio = async () => {
    setPfLoading(true); setPfError(null); setPfResult(null)
    try {
      const syms = pfSymbols.split(',').map(s => s.trim().toUpperCase()).filter(Boolean)
      const res = await axios.post(`${API_BASE}/rsi-dt-portfolio`, {
        symbols:    syms,
        date:       dtDate,
        capital:    pfCapital,
        is_days:    dtIsDays,
        oos_days:   dtOosDays,
        ...(dtParamSource === 'db' && dbCachedParams
          ? { rsi_period: dbCachedParams.optimal_period, rsi_upper: dbCachedParams.optimal_upper, rsi_lower: dbCachedParams.optimal_lower }
          : dtParamSource === 'manual'
          ? { rsi_period: manualPeriod, rsi_upper: manualUpper, rsi_lower: manualLower }
          : {}),
      })
      setPfResult(res.data)
    } catch (err: any) {
      setPfError(err.response?.data?.detail ?? err.message ?? 'Unknown error')
    } finally {
      setPfLoading(false)
    }
  }

  // ── OOS Test state + handler ──────────────────────────────────────────────
  const [oosSymbol,   setOosSymbol]   = useState('AAPL')
  const [oosEndDate,  setOosEndDate]  = useState(new Date().toISOString().slice(0, 10))
  const [oosNDates,   setOosNDates]   = useState(6)
  const [oosIsDays,   setOosIsDays]   = useState(180)
  const [oosOosDays,  setOosOosDays]  = useState(90)
  const [oosNotional, setOosNotional] = useState(10000)
  const [oosResult,   setOosResult]   = useState<OosTestResponse | null>(null)
  const [oosLoading,  setOosLoading]  = useState(false)
  const [oosError,    setOosError]    = useState<string | null>(null)

  // ── Abort controllers ──────────────────────────────────────────────────────
  const abortOptRef = useRef<AbortController | null>(null)
  const abortDtRef  = useRef<AbortController | null>(null)
  const abortOosRef = useRef<AbortController | null>(null)

  // ── Session state persistence ──────────────────────────────────────────────
  const SESSION_KEY = 'tradingagents_rsilab_v1'

  useEffect(() => {
    const saved = sessionStorage.getItem(SESSION_KEY)
    if (!saved) return
    try {
      const s = JSON.parse(saved)
      if (s.symbol)      setSymbol(s.symbol)
      if (s.date)        setDate(s.date)
      if (s.isDays)      setIsDays(s.isDays)
      if (s.oosDays)     setOosDays(s.oosDays)
      if (s.llmProvider) setLlmProvider(s.llmProvider)
      if (s.llmModel)    setLlmModel(s.llmModel)
      if (s.activeTab)   setActiveTab(s.activeTab)
      if (s.dtSymbol)    setDtSymbol(s.dtSymbol)
      if (s.dtDate)      setDtDate(s.dtDate)
      if (s.oosSymbol)   setOosSymbol(s.oosSymbol)
      if (s.oosEndDate)  setOosEndDate(s.oosEndDate)
      if (s.result)      setResult(s.result)
      if (s.dtResult)    setDtResult(s.dtResult)
      if (s.oosResult)   setOosResult(s.oosResult)
    } catch { /* ignore */ }
  }, [])

  useEffect(() => {
    sessionStorage.setItem(SESSION_KEY, JSON.stringify({
      symbol, date, isDays, oosDays, llmProvider, llmModel, activeTab,
      dtSymbol, dtDate, oosSymbol, oosEndDate,
      result, dtResult, oosResult,
    }))
  }, [symbol, date, isDays, oosDays, llmProvider, llmModel, activeTab,
      dtSymbol, dtDate, oosSymbol, oosEndDate,
      result, dtResult, oosResult])

  const runOosTest = async () => {
    const controller = new AbortController()
    abortOosRef.current = controller
    setOosLoading(true); setOosError(null); setOosResult(null)
    try {
      const res = await axios.post(`${API_BASE}/rsi-oos-test`, {
        symbol:   oosSymbol.toUpperCase().trim(),
        oos_end:  oosEndDate,
        n_dates:  oosNDates,
        is_days:  oosIsDays,
        oos_days: oosOosDays,
        notional: oosNotional,
      }, { signal: abortOosRef.current?.signal })
      setOosResult(res.data)
    } catch (err: any) {
      if (err.name === 'CanceledError' || err.code === 'ERR_CANCELED') return
      setOosError(err.response?.data?.detail ?? err.message ?? 'Unknown error')
    } finally {
      setOosLoading(false)
    }
  }

  const stopOosTest = () => { abortOosRef.current?.abort(); setOosLoading(false) }

  // ── Classic API call ─────────────────────────────────────────────────────
  const runOptimization = async () => {
    const controller = new AbortController()
    abortOptRef.current = controller
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const res = await axios.post(`${API_BASE}/rsi-optimize`, {
        symbol:       symbol.toUpperCase().trim(),
        date,
        is_days:      isDays,
        oos_days:     oosDays,
        llm_provider: llmProvider,
        llm_model:    llmModel,
      }, { signal: controller.signal })
      setResult(res.data)
      loadCache()
      loadSignals()
    } catch (err: any) {
      if (err.name === 'CanceledError' || err.code === 'ERR_CANCELED') return
      setError(err.response?.data?.detail ?? err.message ?? 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  const stopOptimization = () => { abortOptRef.current?.abort(); setLoading(false) }

  // ── Render ──────────────────────────────────────────────────────────────
  return (
    <div className="page-root" style={{ maxWidth: 1200, margin: '0 auto', padding: '2rem 1.5rem' }}>

      {/* ── Page header ── */}
      <div style={{ marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
          <span style={{
            width: 38, height: 38, borderRadius: 10,
            background: 'linear-gradient(135deg, #6366f1 0%, #818cf8 100%)',
            display: 'grid', placeItems: 'center', color: '#fff',
            boxShadow: '0 4px 14px rgba(99,102,241,0.35)', flexShrink: 0,
          }}>
            <FlaskConical size={18} />
          </span>
          <h1 style={{ fontSize: 'clamp(1.35rem,2.5vw,1.7rem)', fontWeight: 800, letterSpacing: '-0.03em', color: 'var(--text)', lineHeight: 1.2 }}>
            RSI Optimizer Lab
          </h1>
        </div>

        {/* Explainer card */}
        <div style={{
          background: 'var(--bg-elevated)', border: '1px solid var(--border)',
          borderRadius: 'var(--radius-lg)', padding: '1.1rem 1.4rem',
          boxShadow: 'var(--shadow-sm)', marginTop: 4,
        }}>
          <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', lineHeight: 1.65, marginBottom: 12 }}>
            <strong style={{ color: 'var(--text)' }}>RSI (Relative Strength Index)</strong> measures the speed and magnitude of recent price changes on a 0–100 scale.
            Values above the <em>upper threshold</em> signal overbought conditions (potential sell); below the <em>lower threshold</em> signal oversold (potential buy).
            The key question is: <em>which period and thresholds work best for this specific ticker right now?</em>
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(210px,1fr))', gap: 10 }}>
            {[
              { emoji: '⚙️', title: 'Classic WFO tab', body: 'Grid-searches every (period, upper, lower) combo using both a pure algo and an LLM-narrowed search. Trains on an in-sample window, validates on a held-out OOS window. Prevents curve-fitting.' },
              { emoji: '🧠', title: 'DT Optimizer tab', body: 'A Decision Tree model learns which RSI parameter combos tend to succeed in regimes similar to today. Acts as a smart filter on top of the WFO grid — fewer combos, faster, regime-aware.' },
              { emoji: '🧪', title: 'OOS Test tab', body: 'Runs a full walk-forward P&L simulation on the best params found above — shows you the equity curve, drawdown, hit-rate and Sharpe on data the optimizer never saw.' },
            ].map(({ emoji, title, body }) => (
              <div key={title} style={{
                background: 'var(--bg-subtle)', borderRadius: 'var(--radius-sm)',
                border: '1px solid var(--border)', padding: '10px 12px',
              }}>
                <div style={{ fontSize: '0.8rem', fontWeight: 700, color: 'var(--text)', marginBottom: 4 }}>
                  {emoji} {title}
                </div>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', lineHeight: 1.5 }}>{body}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Tab Bar ── */}
      <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '1.5rem', borderBottom: '2px solid var(--border)', paddingBottom: '0' }}>
        {([
          { key: 'classic',  label: '⚙️ Classic WFO',   desc: 'Algo + LLM grid search' },
          { key: 'dt',       label: '🧠 DT Optimizer',  desc: 'Decision Tree AI filter' },
          { key: 'oos-test', label: '🧪 OOS Test',       desc: 'Walk-forward P&L validation' },
        ] as const).map(tab => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            style={{
              padding: '0.55rem 1.25rem',
              border: 'none',
              borderBottom: activeTab === tab.key ? '3px solid var(--accent)' : '3px solid transparent',
              background: 'none',
              cursor: 'pointer',
              fontWeight: activeTab === tab.key ? 700 : 400,
              color: activeTab === tab.key ? 'var(--text)' : 'var(--text-muted)',
              fontSize: '0.9rem',
              marginBottom: '-2px',
              transition: 'color 0.15s',
            }}
          >
            {tab.label}
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginLeft: '0.4rem', fontWeight: 400 }}>
              {tab.desc}
            </span>
          </button>
        ))}
      </div>

      {/* ══════════════════════════════════════════════════════════════ */}
      {/*  CLASSIC WFO TAB                                             */}
      {/* ══════════════════════════════════════════════════════════════ */}
      {activeTab === 'classic' && (<>

      {/* ── Input form ── */}
      <div className="card" style={{ padding: '1.5rem', marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'flex-end' }}>

          <label style={labelStyle}>
            Symbol
            <input
              className="field"
              value={symbol}
              onChange={e => setSymbol(e.target.value.toUpperCase())}
              placeholder="e.g. AAPL"
              style={{ width: 120 }}
            />
          </label>

          <label style={labelStyle}>
            As-of Date
            <input
              className="field"
              type="date"
              value={date}
              onChange={e => setDate(e.target.value)}
            />
          </label>

          <label style={labelStyle}>
            Training Days (IS)
            <input
              className="field"
              type="number"
              min={60} max={360}
              value={isDays}
              onChange={e => setIsDays(Number(e.target.value))}
              style={{ width: 120 }}
            />
          </label>

          <label style={labelStyle}>
            Test Days (OOS)
            <input
              className="field"
              type="number"
              min={30} max={180}
              value={oosDays}
              onChange={e => setOosDays(Number(e.target.value))}
              style={{ width: 120 }}
            />
          </label>

          <label style={labelStyle}>
            LLM Provider
            <select
              className="field"
              value={llmProvider}
              onChange={e => {
                const p = e.target.value
                setLlmProvider(p)
                // Auto-select first model for the new provider
                const models = LLM_MODELS[p]
                if (models && models.length > 0) setLlmModel(models[0].value)
                else setLlmModel('')
              }}
              style={{ width: 160 }}
            >
              <option value="ollama">🦙 Ollama (local)</option>
              <option value="google">🟢 Google</option>
              <option value="openai">🔵 OpenAI</option>
              <option value="anthropic">🟠 Anthropic</option>
              <option value="heuristic">⚡ Heuristic (no LLM)</option>
            </select>
          </label>

          {llmProvider !== 'heuristic' && (
            <label style={labelStyle}>
              Model
              <select
                className="field"
                value={llmModel}
                onChange={e => setLlmModel(e.target.value)}
                style={{ width: 240 }}
              >
                {(LLM_MODELS[llmProvider] ?? []).map(m => (
                  <option key={m.value} value={m.value}>{m.label}</option>
                ))}
              </select>
            </label>
          )}

          <button
            className="btn-primary"
            onClick={runOptimization}
            disabled={loading || !symbol || !date}
            style={{ height: 40 }}
          >
            {loading ? 'Running…' : 'Run Optimization'}
          </button>
          {loading && (
            <button onClick={stopOptimization} className="btn btn-stop">
              <Square size={14} /> Stop
            </button>
          )}
        </div>

        {/* Explain IS/OOS to the user */}
        <p style={{ marginTop: '0.75rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
          <strong>Training Days (IS)</strong>: data used to find best RSI parameters.
          &nbsp;<strong>Test Days (OOS)</strong>: unseen data used to validate those parameters don't overfit.
        </p>
      </div>

      {/* ── Loading state ── */}
      {loading && (
        <div className="card" style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
          <div style={{ fontSize: '2rem', marginBottom: '1rem' }}>⚙️</div>
          <p style={{ fontWeight: 600 }}>Running both optimizers in parallel…</p>
          <p style={{ fontSize: '0.85rem', marginTop: '0.5rem' }}>
            Algo: testing ~972 combinations &nbsp;|&nbsp; LLM: classifying regime, then testing fewer combos
          </p>
        </div>
      )}

      {/* ── Error ── */}
      {error && (
        <div className="card" style={{ padding: '1.5rem', borderLeft: '4px solid var(--danger)', background: 'var(--danger-soft)' }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* ── Results ── */}
      {result && !loading && (
        <>
          {/* Comparison banner */}
          <ComparisonBanner comparison={result.comparison} />

          {/* Side-by-side optimizer cards */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '1rem' }}>
            <OptimizerCard
              title="Optimizer A — Pure Algo"
              icon={<TrendingUp size={18} color="var(--accent)" />}
              result={result.algo}
              accentColor="#6366f1"
              isWinner={result.comparison.oos_winner === 'algo'}
            />
            <OptimizerCard
              title="Optimizer B — LLM Enhanced"
              icon={<Brain size={18} color="#059669" />}
              result={result.llm}
              accentColor="#059669"
              isWinner={result.comparison.oos_winner === 'llm'}
            />
          </div>

          {/* Sharpe by Period chart */}
          <SharpeChart algo={result.algo} llm={result.llm} />

          {/* LLM Regime panel */}
          {result.llm.regime && (
            <LLMReasoningPanel llm={result.llm} />
          )}

          {/* Debug log panel — always shown after run */}
          <DebugLogPanel logs={result.llm.debug_logs ?? []} />
        </>
      )}

      {/* ── Current RSI Signals ── */}
      <SignalsPanel
        signals={signals}
        loading={signalsLoading}
        expandedTicker={expandedTicker}
        onToggleExpand={t => setExpandedTicker(prev => prev === t ? null : t)}
        onRefresh={loadSignals}
        onLoadTicker={setSymbol}
      />

      {/* ── Saved RSI Parameters ── */}
      <SavedParamsTable
        entries={cachedParams}
        loading={cacheLoading}
        deletingTicker={deletingTicker}
        onDelete={handleDeleteCache}
        onRefresh={loadCache}
        onLoadTicker={setSymbol}
      />
      </>)}

      {/* ══════════════════════════════════════════════════════════════ */}
      {/*  DT OPTIMIZER TAB                                             */}
      {/* ══════════════════════════════════════════════════════════════ */}
      {activeTab === 'dt' && (
        <RsiDtLabView
          symbol={dtSymbol}          setSymbol={setDtSymbol}
          date={dtDate}              setDate={setDtDate}
          isDays={dtIsDays}          setIsDays={setDtIsDays}
          oosDays={dtOosDays}        setOosDays={setDtOosDays}
          labelHorizon={dtLabelHorizon} setLabelHorizon={setDtLabelHorizon}
          result={dtResult}          loading={dtLoading}  error={dtError}
          onRun={runDtOptimize}      onStop={stopDtOptimize}
          pfSymbols={pfSymbols}      setPfSymbols={setPfSymbols}
          pfCapital={pfCapital}      setPfCapital={setPfCapital}
          pfResult={pfResult}        pfLoading={pfLoading} pfError={pfError}
          onRunPortfolio={runPortfolio}
          dtParamSource={dtParamSource}   setDtParamSource={setDtParamSource}
          manualPeriod={manualPeriod}     setManualPeriod={setManualPeriod}
          manualUpper={manualUpper}       setManualUpper={setManualUpper}
          manualLower={manualLower}       setManualLower={setManualLower}
          dbCachedParams={dbCachedParams}
          dbFetchStatus={dbFetchStatus}
          onFetchDbParams={fetchDbParams}
        />
      )}

      {/* ── OOS Test Tab ── */}
      {activeTab === 'oos-test' && (
        <OosTestPanel
          algo="rsi"
          symbol={oosSymbol}       onSymbol={setOosSymbol}
          endDate={oosEndDate}     onEndDate={setOosEndDate}
          nDates={oosNDates}       onNDates={setOosNDates}
          isDays={oosIsDays}       onIsDays={setOosIsDays}
          oosDays={oosOosDays}     onOosDays={setOosOosDays}
          notional={oosNotional}   onNotional={setOosNotional}
          result={oosResult}       loading={oosLoading}
          error={oosError}         onRun={runOosTest}  onStop={stopOosTest}
        />
      )}

    </div>
  )
}

// ── Comparison Banner ─────────────────────────────────────────────────────────

function ComparisonBanner({ comparison }: { comparison: ComparisonSummary }) {
  const winnerColor = comparison.oos_winner === 'llm' ? '#059669' :
                      comparison.oos_winner === 'algo' ? 'var(--accent)' : 'var(--warning)'

  return (
    <div className="card" style={{
      padding: '1.25rem 1.5rem',
      borderLeft: `4px solid ${winnerColor}`,
      display: 'flex',
      alignItems: 'flex-start',
      gap: '1rem',
    }}>
      <div style={{ flex: 1 }}>
        <p style={{ fontWeight: 600, marginBottom: '0.5rem' }}>⚖️ Comparison Summary</p>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>{comparison.summary}</p>

        <div style={{ display: 'flex', gap: '1.5rem', marginTop: '0.75rem', flexWrap: 'wrap' }}>
          <Pill label="Period Agreement" value={comparison.period_agreement ? '✅ Yes' : '❌ No'} />
          <Pill label="OOS Winner" value={comparison.oos_winner.toUpperCase()} valueColor={winnerColor} />
          <Pill label="Algo beats RSI-14/70/30" value={comparison.algo_beats_default ? '✅' : '❌'} />
          <Pill label="LLM beats RSI-14/70/30" value={comparison.llm_beats_default ? '✅' : '❌'} />
          <Pill label="LLM Grid Reduction" value={`${comparison.combos_reduction_pct}%`} />
        </div>
      </div>
    </div>
  )
}

// ── Optimizer Card ────────────────────────────────────────────────────────────

function OptimizerCard({
  title, icon, result, accentColor, isWinner,
}: {
  title: string
  icon: React.ReactNode
  result: OptimizerResult
  accentColor: string
  isWinner: boolean
}) {
  return (
    <div className="card" style={{
      padding: '1.5rem',
      borderTop: `3px solid ${accentColor}`,
      position: 'relative',
    }}>
      {isWinner && (
        <div style={{
          position: 'absolute', top: 12, right: 12,
          background: accentColor, color: '#fff',
          fontSize: '0.7rem', fontWeight: 700,
          padding: '2px 8px', borderRadius: 20,
        }}>
          OOS WINNER
        </div>
      )}

      <h3 style={{ fontSize: '0.95rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1.25rem' }}>
        {icon} {title}
      </h3>

      {/* Optimal params */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '0.75rem', marginBottom: '1.25rem' }}>
        <StatBox label="Period" value={result.optimal_period} accent={accentColor} />
        <StatBox label="Overbought" value={result.optimal_upper} accent={accentColor} />
        <StatBox label="Oversold" value={result.optimal_lower} accent={accentColor} />
      </div>

      {/* Sharpe scores */}
      <table style={{ width: '100%', fontSize: '0.85rem', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>
            <th style={thStyle}>Metric</th>
            <th style={thStyle}>Optimized</th>
            <th style={thStyle}>Default RSI(14)</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td style={tdStyle}>IS Sharpe</td>
            <td style={{ ...tdStyle, ...sharpeStyle(result.is_sharpe) }}>{result.is_sharpe.toFixed(3)}</td>
            <td style={{ ...tdStyle, ...sharpeStyle(result.default_is_sharpe) }}>{result.default_is_sharpe.toFixed(3)}</td>
          </tr>
          <tr>
            <td style={tdStyle}>OOS Sharpe</td>
            <td style={{ ...tdStyle, ...sharpeStyle(result.oos_sharpe) }}>{result.oos_sharpe.toFixed(3)}</td>
            <td style={{ ...tdStyle, ...sharpeStyle(result.default_oos_sharpe) }}>{result.default_oos_sharpe.toFixed(3)}</td>
          </tr>
          <tr>
            <td style={tdStyle}>Confidence</td>
            <td style={{ ...tdStyle, ...confidenceStyle(result.confidence) }} colSpan={2}>
              {result.confidence}
            </td>
          </tr>
          <tr>
            <td style={tdStyle}>Combos Tested</td>
            <td style={tdStyle} colSpan={2}>{result.combos_tested.toLocaleString()}</td>
          </tr>
        </tbody>
      </table>

      {/* Confidence explanation */}
      <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.75rem' }}>
        {result.confidence === 'HIGH'
          ? '✅ OOS Sharpe held up well — low overfitting risk'
          : result.confidence === 'MEDIUM'
          ? '⚠️ Moderate confidence — some overfitting possible'
          : '❌ Low confidence — OOS Sharpe dropped significantly (overfit)'}
      </p>

      {/* Token cost — only shown for LLM optimizer */}
      {result.token_usage && result.token_usage.model !== 'heuristic' && (
        <div style={{
          marginTop: '1rem',
          padding: '0.65rem 0.85rem',
          background: 'var(--bg-subtle)',
          borderRadius: 'var(--radius-sm)',
          border: '1px solid var(--border)',
          fontSize: '0.78rem',
        }}>
          <div style={{ fontWeight: 600, marginBottom: '0.4rem', color: 'var(--text-secondary)' }}>
            🔢 LLM Token Usage
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.3rem 1rem', color: 'var(--text-muted)' }}>
            <span>Model</span>
            <span style={{ color: 'var(--text)', fontWeight: 500 }}>{result.token_usage.model}</span>
            <span>Prompt tokens</span>
            <span style={{ color: 'var(--text)' }}>{result.token_usage.prompt_tokens.toLocaleString()}</span>
            <span>Response tokens</span>
            <span style={{ color: 'var(--text)' }}>{result.token_usage.completion_tokens.toLocaleString()}</span>
            <span>Total tokens</span>
            <span style={{ color: 'var(--text)', fontWeight: 600 }}>{result.token_usage.total_tokens.toLocaleString()}</span>
            <span>Estimated cost</span>
            <span style={{ color: 'var(--success)', fontWeight: 700 }}>
              ${result.token_usage.cost_usd < 0.001
                ? result.token_usage.cost_usd.toFixed(6)
                : result.token_usage.cost_usd.toFixed(4)}
            </span>
          </div>
        </div>
      )}

      {/* Heuristic fallback notice */}
      {result.token_usage && result.token_usage.model === 'heuristic' && (
        <p style={{ fontSize: '0.75rem', color: 'var(--warning)', marginTop: '0.75rem' }}>
          ⚡ Heuristic fallback used — no LLM API key configured. Token cost: $0.000000
        </p>
      )}
    </div>
  )
}

// ── Sharpe by Period Chart ────────────────────────────────────────────────────

function SharpeChart({ algo, llm }: { algo: OptimizerResult; llm: OptimizerResult }) {
  const chartData = mergeChartData(algo.period_sharpes, llm.period_sharpes)

  return (
    <div className="card" style={{ padding: '1.5rem', marginTop: '1rem' }}>
      <h3 style={{ fontWeight: 700, marginBottom: '0.25rem', fontSize: '0.95rem' }}>
        📊 OOS Sharpe by RSI Period
      </h3>
      <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '1.5rem' }}>
        Each bar = best OOS Sharpe found for that period across all threshold combinations.
        Higher is better. The LLM optimizer only tested periods within its narrowed range.
      </p>

      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis
            dataKey="period"
            tick={{ fontSize: 11, fill: 'var(--text-muted)' }}
            label={{ value: 'RSI Period', position: 'insideBottom', offset: -2, fontSize: 11 }}
          />
          <YAxis
            tick={{ fontSize: 11, fill: 'var(--text-muted)' }}
            label={{ value: 'OOS Sharpe', angle: -90, position: 'insideLeft', fontSize: 11 }}
          />
          <Tooltip
            contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', fontSize: '0.8rem' }}
            formatter={(value: any) => typeof value === 'number' ? value.toFixed(3) : value}
          />
          <Legend wrapperStyle={{ fontSize: '0.8rem' }} />
          {/* Reference line at 0 Sharpe */}
          <ReferenceLine y={0} stroke="var(--danger)" strokeDasharray="4 4" />
          {/* Reference line at 0.5 (acceptable Sharpe) */}
          <ReferenceLine y={0.5} stroke="var(--warning)" strokeDasharray="4 4" label={{ value: '0.5', fontSize: 10 }} />

          <Bar dataKey="Algo OOS" fill="#6366f1" radius={[3, 3, 0, 0]} maxBarSize={20} />
          <Bar dataKey="LLM OOS"  fill="#059669" radius={[3, 3, 0, 0]} maxBarSize={20} />
        </BarChart>
      </ResponsiveContainer>

      {/* Best periods highlighted */}
      <div style={{ display: 'flex', gap: '2rem', marginTop: '1rem', fontSize: '0.82rem' }}>
        <span>
          <span style={{ color: '#6366f1', fontWeight: 700 }}>● Algo</span> best period:{' '}
          <strong>{algo.optimal_period}</strong> (OOS: {algo.oos_sharpe.toFixed(3)})
        </span>
        <span>
          <span style={{ color: '#059669', fontWeight: 700 }}>● LLM</span> best period:{' '}
          <strong>{llm.optimal_period}</strong> (OOS: {llm.oos_sharpe.toFixed(3)})
        </span>
      </div>
    </div>
  )
}

// ── LLM Reasoning Panel ───────────────────────────────────────────────────────

function LLMReasoningPanel({ llm }: { llm: OptimizerResult }) {
  const signals = llm.regime_signals

  return (
    <div className="card" style={{ padding: '1.5rem', marginTop: '1rem' }}>
      <h3 style={{ fontWeight: 700, marginBottom: '1rem', fontSize: '0.95rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
        <Brain size={18} color="#059669" />
        LLM Regime Analysis
        {!llm.llm_available && (
          <span style={{ fontSize: '0.75rem', color: 'var(--warning)', fontWeight: 400 }}>
            (heuristic fallback — no LLM API key)
          </span>
        )}
      </h3>

      {/* Regime badge */}
      <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', marginBottom: '1rem' }}>
        <div style={{
          background: 'var(--accent-soft)', color: 'var(--accent)',
          padding: '4px 14px', borderRadius: 20, fontSize: '0.82rem', fontWeight: 700,
        }}>
          Regime: {llm.regime?.replace(/_/g, ' ').toUpperCase()}
        </div>

        {llm.llm_grid && (
          <>
            <div style={pillStyle}>Periods: {llm.llm_grid.period_range[0]}–{llm.llm_grid.period_range[1]}</div>
            <div style={pillStyle}>Overbought: {llm.llm_grid.upper_range[0]}–{llm.llm_grid.upper_range[1]}</div>
            <div style={pillStyle}>Oversold: {llm.llm_grid.lower_range[0]}–{llm.llm_grid.lower_range[1]}</div>
          </>
        )}
      </div>

      {/* LLM reasoning text */}
      {llm.llm_reasoning && (
        <div style={{
          background: 'var(--bg-subtle)', borderRadius: 'var(--radius-sm)',
          padding: '1rem', fontSize: '0.875rem', color: 'var(--text-secondary)',
          borderLeft: '3px solid #059669', marginBottom: '1rem',
          fontStyle: 'italic',
        }}>
          "{llm.llm_reasoning}"
        </div>
      )}

      {/* Regime signals breakdown */}
      {signals && (
        <>
          <p style={{ fontWeight: 600, fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
            Pre-computed signals passed to the LLM:
          </p>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))', gap: '0.5rem' }}>
            <SignalCard label="Volatility Percentile" value={`${signals.volatility_percentile}%`}
              note="100% = highest ever, 0% = lowest ever" />
            <SignalCard label="Trend Direction" value={signals.trend_direction.toUpperCase()}
              note={`${signals.trend_strength_pct > 0 ? '+' : ''}${signals.trend_strength_pct}% from 50-SMA`} />
            <SignalCard label="Current RSI-14" value={signals.rsi14_current}
              note={signals.rsi14_current > 70 ? 'Overbought zone' : signals.rsi14_current < 30 ? 'Oversold zone' : 'Neutral zone'} />
            <SignalCard label="10-Day Return" value={`${signals.recent_10d_return_pct > 0 ? '+' : ''}${signals.recent_10d_return_pct}%`}
              note="Recent price momentum" />
            <SignalCard label="Above 50-SMA" value={signals.price_above_sma50 ? 'Yes' : 'No'}
              note="Short-term trend" />
            <SignalCard label="Above 200-SMA" value={signals.price_above_sma200 ? 'Yes' : 'No'}
              note="Long-term trend" />
          </div>
        </>
      )}
    </div>
  )
}

// ── Small reusable components ─────────────────────────────────────────────────

function StatBox({ label, value, accent }: { label: string; value: number; accent: string }) {
  return (
    <div style={{
      background: 'var(--bg-subtle)', borderRadius: 'var(--radius-sm)',
      padding: '0.75rem', textAlign: 'center',
    }}>
      <div style={{ fontSize: '1.4rem', fontWeight: 700, color: accent }}>{value}</div>
      <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 2 }}>{label}</div>
    </div>
  )
}

function Pill({ label, value, valueColor }: { label: string; value: string; valueColor?: string }) {
  return (
    <div style={{ fontSize: '0.8rem' }}>
      <span style={{ color: 'var(--text-muted)' }}>{label}: </span>
      <span style={{ fontWeight: 600, color: valueColor ?? 'var(--text)' }}>{value}</span>
    </div>
  )
}

function SignalCard({ label, value, note }: { label: string; value: any; note: string }) {
  return (
    <div style={{
      background: 'var(--bg-elevated)', border: '1px solid var(--border)',
      borderRadius: 'var(--radius-sm)', padding: '0.6rem 0.75rem',
    }}>
      <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>{label}</div>
      <div style={{ fontWeight: 700, fontSize: '1rem', margin: '2px 0' }}>{value}</div>
      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{note}</div>
    </div>
  )
}

// ── Debug Log Panel ───────────────────────────────────────────────────────────

function DebugLogPanel({ logs }: { logs: string[] }) {
  const [open, setOpen] = useState(true)
  const bottomRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom when logs update
  useEffect(() => {
    if (open) bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs, open])

  const logColors: Record<string, string> = {
    success: '#34d399',   // green
    error:   '#f87171',   // red
    warn:    '#fbbf24',   // yellow
    data:    '#94a3b8',   // muted blue-gray
    info:    '#e2e8f0',   // near-white
  }

  return (
    <div style={{ marginTop: '1rem' }}>
      {/* Header — click to toggle */}
      <button
        onClick={() => setOpen(o => !o)}
        style={{
          width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          background: '#0f172a', color: '#94a3b8',
          border: 'none', borderRadius: open ? '10px 10px 0 0' : '10px',
          padding: '0.6rem 1rem', cursor: 'pointer', fontSize: '0.8rem', fontFamily: 'var(--font-mono)',
        }}
      >
        <span>🪵 LLM Debug Log — {logs.length} lines</span>
        <span style={{ fontSize: '0.7rem' }}>{open ? '▲ collapse' : '▼ expand'}</span>
      </button>

      {open && (
        <div style={{
          background: '#0f172a', borderRadius: '0 0 10px 10px',
          padding: '0.75rem 1rem',
          maxHeight: 280, overflowY: 'auto',
          fontFamily: 'var(--font-mono)', fontSize: '0.75rem', lineHeight: 1.7,
        }}>
          {logs.length === 0 ? (
            <span style={{ color: '#475569' }}>No logs yet. Run an optimization first.</span>
          ) : (
            logs.map((line, i) => {
              const { level } = classifyLog(line)
              return (
                <div key={i} style={{ color: logColors[level], whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}>
                  <span style={{ color: '#334155', userSelect: 'none', marginRight: '0.75rem' }}>
                    {String(i + 1).padStart(2, '0')}
                  </span>
                  {line}
                </div>
              )
            })
          )}
          <div ref={bottomRef} />
        </div>
      )}
    </div>
  )
}

// ── Saved Params Table ────────────────────────────────────────────────────────

// ── Action Badge ─────────────────────────────────────────────────────────────

function ActionBadge({ action }: { action: string | null }) {
  if (!action) return <span style={{ color: 'var(--text-muted)' }}>—</span>
  const cfg: Record<string, { bg: string; color: string; icon: React.ReactNode }> = {
    BUY:  { bg: '#dcfce7', color: '#16a34a', icon: <TrendingUp  size={14} /> },
    SELL: { bg: '#fee2e2', color: '#dc2626', icon: <TrendingDown size={14} /> },
    HOLD: { bg: '#fef9c3', color: '#b45309', icon: <Minus       size={14} /> },
  }
  const style = cfg[action] ?? { bg: 'var(--bg-subtle)', color: 'var(--text-secondary)', icon: null }
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 4,
      background: style.bg, color: style.color,
      fontWeight: 800, fontSize: '0.78rem', letterSpacing: '0.04em',
      padding: '3px 10px', borderRadius: 20,
    }}>
      {style.icon}{action}
    </span>
  )
}

// ── RSI Gauge ─────────────────────────────────────────────────────────────────

function RsiGauge({ value, upper, lower }: { value: number; upper: number; lower: number }) {
  const pct = Math.max(0, Math.min(100, value))
  const lowerPct = lower
  const upperPct = upper
  const barColor = value < lower ? '#16a34a' : value > upper ? '#dc2626' : '#f59e0b'

  return (
    <div style={{ position: 'relative', width: '100%', height: 10, background: '#e2e8f0', borderRadius: 5, overflow: 'hidden' }}>
      {/* oversold zone */}
      <div style={{ position: 'absolute', left: 0, width: `${lowerPct}%`, height: '100%', background: '#bbf7d0', opacity: 0.5 }} />
      {/* overbought zone */}
      <div style={{ position: 'absolute', left: `${upperPct}%`, width: `${100 - upperPct}%`, height: '100%', background: '#fecaca', opacity: 0.5 }} />
      {/* cursor */}
      <div style={{
        position: 'absolute', left: `calc(${pct}% - 4px)`, top: -1,
        width: 8, height: 12, background: barColor, borderRadius: 2,
      }} />
    </div>
  )
}

// ── Signals Panel ─────────────────────────────────────────────────────────────

function SignalsPanel({
  signals, loading, expandedTicker, onToggleExpand, onRefresh, onLoadTicker,
}: {
  signals: RsiSignal[]
  loading: boolean
  expandedTicker: string | null
  onToggleExpand: (t: string) => void
  onRefresh: () => void
  onLoadTicker: (t: string) => void
}) {
  const buys  = signals.filter(s => s.action === 'BUY').length
  const sells = signals.filter(s => s.action === 'SELL').length
  const holds = signals.filter(s => s.action === 'HOLD').length

  return (
    <div className="card" style={{ marginTop: '2rem', padding: '1.5rem' }}>

      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
        <h3 style={{ fontSize: '0.95rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.5rem', margin: 0 }}>
          <Zap size={17} color="#f59e0b" />
          Current RSI Signals
          {signals.length > 0 && (
            <span style={{ display: 'flex', gap: 6, marginLeft: 4 }}>
              {buys  > 0 && <span style={{ background: '#dcfce7', color: '#16a34a', fontSize: '0.7rem', fontWeight: 700, padding: '1px 8px', borderRadius: 20 }}>▲ {buys} BUY</span>}
              {sells > 0 && <span style={{ background: '#fee2e2', color: '#dc2626', fontSize: '0.7rem', fontWeight: 700, padding: '1px 8px', borderRadius: 20 }}>▼ {sells} SELL</span>}
              {holds > 0 && <span style={{ background: '#fef9c3', color: '#b45309', fontSize: '0.7rem', fontWeight: 700, padding: '1px 8px', borderRadius: 20 }}>— {holds} HOLD</span>}
            </span>
          )}
        </h3>
        <button
          onClick={onRefresh}
          disabled={loading}
          style={{
            background: 'none', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
            padding: '4px 8px', cursor: 'pointer', color: 'var(--text-muted)',
            display: 'flex', alignItems: 'center', gap: 4, fontSize: '0.78rem',
          }}
        >
          <RotateCcw size={13} style={{ animation: loading ? 'spin 1s linear infinite' : 'none' }} />
          Refresh
        </button>
      </div>

      <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
        Live BUY / SELL / HOLD decisions using each ticker's optimized RSI params.
        Click a row to see the RSI chart.
      </p>

      {loading && signals.length === 0 ? (
        <div style={{ padding: '1.5rem', textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
          Computing signals…
        </div>
      ) : signals.length === 0 ? (
        <div style={{
          padding: '2rem', textAlign: 'center',
          color: 'var(--text-muted)', fontSize: '0.85rem',
          border: '1px dashed var(--border)', borderRadius: 'var(--radius-sm)',
        }}>
          No signals yet — run an optimization above to generate a signal.
        </div>
      ) : (
        <div>
          {signals.map(sig => (
            <div key={sig.ticker} style={{ marginBottom: '0.5rem' }}>

              {/* ── Summary row ── */}
              <div
                onClick={() => onToggleExpand(sig.ticker)}
                style={{
                  display: 'grid',
                  gridTemplateColumns: '100px 90px 90px 110px 110px 110px 1fr',
                  alignItems: 'center', gap: '0.75rem',
                  padding: '0.6rem 0.75rem',
                  background: 'var(--bg-subtle)',
                  borderRadius: 'var(--radius-sm)',
                  cursor: 'pointer',
                  border: expandedTicker === sig.ticker ? '1px solid var(--accent)' : '1px solid transparent',
                }}
              >
                {/* Ticker */}
                <button
                  onClick={e => { e.stopPropagation(); onLoadTicker(sig.ticker) }}
                  style={{
                    background: 'none', border: 'none', cursor: 'pointer',
                    color: 'var(--accent)', fontWeight: 700, fontSize: '0.9rem',
                    textAlign: 'left', padding: 0, textDecoration: 'underline dotted',
                  }}
                >
                  {sig.ticker}
                </button>

                {/* Action badge */}
                <div><ActionBadge action={sig.action} /></div>

                {/* Current RSI */}
                <div style={{ fontSize: '0.82rem' }}>
                  {sig.error ? (
                    <span style={{ color: 'var(--danger)', fontSize: '0.75rem' }}>Error</span>
                  ) : sig.current_rsi != null ? (
                    <span style={{ fontWeight: 600, color: sig.action === 'BUY' ? '#16a34a' : sig.action === 'SELL' ? '#dc2626' : '#b45309' }}>
                      RSI {sig.current_rsi.toFixed(1)}
                    </span>
                  ) : '—'}
                </div>

                {/* Thresholds */}
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                  {sig.period != null ? (
                    <span>Period <strong style={{ color: 'var(--accent)' }}>{sig.period}</strong></span>
                  ) : '—'}
                </div>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                  {sig.oversold != null ? (
                    <span>OB <strong>{sig.overbought}</strong> / OS <strong>{sig.oversold}</strong></span>
                  ) : '—'}
                </div>

                {/* Price */}
                <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                  {sig.latest_close != null ? (
                    <>
                      ${sig.latest_close.toFixed(2)}
                      {sig.price_change_pct != null && (
                        <span style={{ marginLeft: 4, color: sig.price_change_pct >= 0 ? '#16a34a' : '#dc2626', fontSize: '0.75rem' }}>
                          {sig.price_change_pct >= 0 ? '+' : ''}{sig.price_change_pct.toFixed(2)}%
                        </span>
                      )}
                    </>
                  ) : '—'}
                </div>

                {/* RSI gauge */}
                <div style={{ paddingRight: '0.5rem' }}>
                  {sig.current_rsi != null && sig.overbought != null && sig.oversold != null ? (
                    <RsiGauge value={sig.current_rsi} upper={sig.overbought} lower={sig.oversold} />
                  ) : null}
                </div>
              </div>

              {/* ── Expanded RSI chart ── */}
              {expandedTicker === sig.ticker && (
                <div style={{
                  border: '1px solid var(--border)', borderTop: 'none',
                  borderRadius: '0 0 var(--radius-sm) var(--radius-sm)',
                  padding: '1rem 0.75rem',
                  background: 'var(--bg)',
                }}>
                  {sig.error ? (
                    <p style={{ color: 'var(--danger)', fontSize: '0.82rem', margin: 0 }}>⚠ {sig.error}</p>
                  ) : (
                    <>
                      <div style={{ display: 'flex', gap: '2rem', marginBottom: '0.75rem', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                        <span>Regime: <strong>{sig.regime ? sig.regime.replace(/_/g, ' ') : '—'}</strong></span>
                        <span>Confidence: <strong style={confidenceStyle(sig.confidence ?? '')}>{sig.confidence ?? '—'}</strong></span>
                        <span>Source: <strong>{sig.optimizer_provider === 'llm' ? 'LLM' : sig.optimizer_provider === 'algo' ? 'Algo' : sig.using_cached ? 'cached' : 'default'}</strong></span>
                        <span>Optimized: <strong>{sig.optimized_at ? sig.optimized_at.slice(0, 10) : '—'}</strong></span>
                      </div>

                      {/* RSI line chart */}
                      {sig.rsi_history.length > 0 && (
                        <ResponsiveContainer width="100%" height={160}>
                          <LineChart data={sig.rsi_history} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                            <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={d => d.slice(5)} />
                            <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} width={28} />
                            <Tooltip
                              formatter={(v: any, name: string) => [
                                typeof v === 'number' ? v.toFixed(2) : v, name,
                              ]}
                              labelFormatter={l => `Date: ${l}`}
                            />
                            {/* Overbought / oversold bands */}
                            {sig.overbought != null && (
                              <ReferenceLine y={sig.overbought} stroke="#dc2626" strokeDasharray="4 3"
                                label={{ value: `OB ${sig.overbought}`, position: 'insideTopRight', fill: '#dc2626', fontSize: 10 }} />
                            )}
                            {sig.oversold != null && (
                              <ReferenceLine y={sig.oversold} stroke="#16a34a" strokeDasharray="4 3"
                                label={{ value: `OS ${sig.oversold}`, position: 'insideBottomRight', fill: '#16a34a', fontSize: 10 }} />
                            )}
                            <ReferenceLine y={50} stroke="var(--border)" strokeDasharray="2 4" />
                            <Line
                              type="monotone" dataKey="rsi" name={`RSI(${sig.period})`}
                              stroke="var(--accent)" dot={false} strokeWidth={2}
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      )}
                    </>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ── SavedParamsTable ──────────────────────────────────────────────────────────

function SavedParamsTable({
  entries, loading, deletingTicker, onDelete, onRefresh, onLoadTicker,
}: {
  entries: CachedParam[]
  loading: boolean
  deletingTicker: string | null
  onDelete: (ticker: string) => void
  onRefresh: () => void
  onLoadTicker: (ticker: string) => void
}) {
  return (
    <div className="card" style={{ marginTop: '2rem', padding: '1.5rem' }}>

      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
        <h3 style={{ fontSize: '0.95rem', fontWeight: 700, display: 'flex', alignItems: 'center', gap: '0.5rem', margin: 0 }}>
          <Database size={17} color="var(--accent)" />
          Saved RSI Parameters
          <span style={{
            background: 'var(--accent-soft)', color: 'var(--accent)',
            fontSize: '0.72rem', fontWeight: 700, padding: '1px 8px', borderRadius: 20,
          }}>
            {entries.length} ticker{entries.length !== 1 ? 's' : ''}
          </span>
        </h3>

        <button
          onClick={onRefresh}
          disabled={loading}
          title="Refresh"
          style={{
            background: 'none', border: '1px solid var(--border)', borderRadius: 'var(--radius-sm)',
            padding: '4px 8px', cursor: 'pointer', color: 'var(--text-muted)',
            display: 'flex', alignItems: 'center', gap: 4, fontSize: '0.78rem',
          }}
        >
          <RotateCcw size={13} style={{ animation: loading ? 'spin 1s linear infinite' : 'none' }} />
          Refresh
        </button>
      </div>

      <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>
        Each optimization run auto-saves the winning parameters here.
        These values are used by the analysis agents when running RSI calculations for the ticker.
      </p>

      {entries.length === 0 ? (
        <div style={{
          padding: '2rem', textAlign: 'center',
          color: 'var(--text-muted)', fontSize: '0.85rem',
          border: '1px dashed var(--border)', borderRadius: 'var(--radius-sm)',
        }}>
          No saved parameters yet — run an optimization above to save results.
        </div>
      ) : (
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', fontSize: '0.82rem', borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ background: 'var(--bg-subtle)' }}>
                {['Ticker', 'Period', 'Overbought', 'Oversold', 'OOS Sharpe', 'Confidence', 'Regime', 'Source', 'Saved At', ''].map(h => (
                  <th key={h} style={{
                    textAlign: 'left', padding: '0.5rem 0.75rem',
                    borderBottom: '1px solid var(--border)',
                    color: 'var(--text-muted)', fontWeight: 600, fontSize: '0.75rem',
                    whiteSpace: 'nowrap',
                  }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {entries.map((row, i) => (
                <tr
                  key={row.ticker}
                  style={{
                    background: i % 2 === 0 ? 'transparent' : 'var(--bg-subtle)',
                    transition: 'background 0.15s',
                  }}
                >
                  {/* Ticker — click to load into form */}
                  <td style={{ ...tdStyle, fontWeight: 700 }}>
                    <button
                      onClick={() => onLoadTicker(row.ticker)}
                      title="Load this ticker into the form"
                      style={{
                        background: 'none', border: 'none', cursor: 'pointer',
                        color: 'var(--accent)', fontWeight: 700, padding: 0,
                        fontSize: '0.82rem', textDecoration: 'underline dotted',
                      }}
                    >
                      {row.ticker}
                    </button>
                  </td>

                  <td style={{ ...tdStyle, fontWeight: 600, color: 'var(--accent)' }}>{row.optimal_period}</td>
                  <td style={tdStyle}>{row.optimal_upper}</td>
                  <td style={tdStyle}>{row.optimal_lower}</td>

                  {/* OOS Sharpe — colored */}
                  <td style={{ ...tdStyle, ...sharpeStyle(row.oos_sharpe) }}>{row.oos_sharpe.toFixed(3)}</td>

                  {/* Confidence badge */}
                  <td style={{ ...tdStyle, ...confidenceStyle(row.confidence) }}>{row.confidence}</td>

                  {/* Regime */}
                  <td style={{ ...tdStyle, color: 'var(--text-secondary)' }}>
                    {row.regime ? row.regime.replace(/_/g, ' ') : '—'}
                  </td>

                  {/* Optimizer provider */}
                  <td style={{ ...tdStyle, color: 'var(--text-muted)' }}>
                    <span style={{
                      background: row.optimizer_provider === 'llm' ? '#059669' : 'var(--accent)',
                      color: '#fff', fontSize: '0.68rem', fontWeight: 700,
                      padding: '1px 7px', borderRadius: 20,
                    }}>
                      {row.optimizer_provider === 'llm' ? 'LLM' : 'ALGO'}
                    </span>
                  </td>

                  {/* Timestamp */}
                  <td style={{ ...tdStyle, color: 'var(--text-muted)', fontSize: '0.75rem', whiteSpace: 'nowrap' }}>
                    {row.optimized_at ? row.optimized_at.replace('T', ' ').slice(0, 16) : '—'}
                  </td>

                  {/* Delete */}
                  <td style={{ ...tdStyle, textAlign: 'right' }}>
                    <button
                      onClick={() => onDelete(row.ticker)}
                      disabled={deletingTicker === row.ticker}
                      title={`Remove ${row.ticker} from cache`}
                      style={{
                        background: 'none', border: 'none', cursor: 'pointer',
                        color: 'var(--danger)', padding: '2px 4px',
                        opacity: deletingTicker === row.ticker ? 0.4 : 1,
                      }}
                    >
                      <Trash2 size={14} />
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

// ── Style constants ───────────────────────────────────────────────────────────

const labelStyle: React.CSSProperties = {
  display: 'flex', flexDirection: 'column', gap: '0.35rem',
  fontSize: '0.82rem', fontWeight: 600, color: 'var(--text-secondary)',
}

const thStyle: React.CSSProperties = {
  textAlign: 'left', padding: '0.3rem 0.5rem',
  borderBottom: '1px solid var(--border)',
}

const tdStyle: React.CSSProperties = {
  padding: '0.35rem 0.5rem',
  borderBottom: '1px solid var(--border)',
  fontSize: '0.85rem',
}

const pillStyle: React.CSSProperties = {
  background: 'var(--bg-subtle)', border: '1px solid var(--border)',
  padding: '3px 12px', borderRadius: 20, fontSize: '0.8rem',
}

// ═════════════════════════════════════════════════════════════════════════════
// RSI DT OPTIMIZER — full component tree
// ═════════════════════════════════════════════════════════════════════════════

interface RsiDtLabViewProps {
  symbol: string;       setSymbol: (v: string) => void
  date: string;         setDate: (v: string) => void
  isDays: number;       setIsDays: (v: number) => void
  oosDays: number;      setOosDays: (v: number) => void
  labelHorizon: number; setLabelHorizon: (v: number) => void
  result: RsiDtResult | null
  loading: boolean
  error: string | null
  onRun: () => void
  onStop?: () => void
  pfSymbols: string;    setPfSymbols: (v: string) => void
  pfCapital: number;    setPfCapital: (v: number) => void
  pfResult: RsiDtPortfolioResult | null
  pfLoading: boolean
  pfError: string | null
  onRunPortfolio: () => void
  dtParamSource: 'default' | 'db' | 'manual'
  setDtParamSource: (v: 'default' | 'db' | 'manual') => void
  manualPeriod: number;   setManualPeriod: (v: number) => void
  manualUpper: number;    setManualUpper: (v: number) => void
  manualLower: number;    setManualLower: (v: number) => void
  dbCachedParams: { optimal_period: number; optimal_upper: number; optimal_lower: number; confidence: string; oos_sharpe: number } | null
  dbFetchStatus: 'idle' | 'loading' | 'found' | 'not_found'
  onFetchDbParams: (ticker: string) => void
}

function RsiDtLabView({
  symbol, setSymbol, date, setDate,
  isDays, setIsDays, oosDays, setOosDays,
  labelHorizon, setLabelHorizon,
  result, loading, error, onRun, onStop,
  pfSymbols, setPfSymbols, pfCapital, setPfCapital,
  pfResult, pfLoading, pfError, onRunPortfolio,
  dtParamSource, setDtParamSource,
  manualPeriod, setManualPeriod,
  manualUpper, setManualUpper,
  manualLower, setManualLower,
  dbCachedParams, dbFetchStatus, onFetchDbParams,
}: RsiDtLabViewProps) {
  return (
    <>
      {/* ── Explainer ── */}
      <div className="card" style={{
        padding: '1rem 1.25rem', marginBottom: '1.25rem',
        borderLeft: '4px solid #8b5cf6',
        background: 'var(--bg-subtle)',
        fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.7,
      }}>
        <strong style={{ color: 'var(--text)', display: 'block', marginBottom: '0.4rem' }}>
          🧠 How the RSI DT Optimizer works
        </strong>
        RSI is the <em>feature</em>, not the decision-maker. A Decision Tree classifies whether each oversold condition
        is worth trading based on: <strong>RSI(14)</strong> · <strong>EMA Ratio (50/200)</strong> ·
        {' '}<strong>MACD Histogram</strong> · <strong>Bollinger %B</strong> · <strong>ATR-14</strong> ·
        {' '}<strong>Volume Ratio (30d/60d)</strong>.
        Two models run in parallel — <strong>Model A</strong> (Sliding WFO, retrained every {oosDays} bars) captures
        current alpha, <strong>Model B</strong> (Fixed 10yr) is crash-aware (2008/2020).
        A Long signal fires only when <em>both</em> agree (P &gt; 0.60). Position size is halved when ATR percentile &gt; 80th.
      </div>

      {/* ── Config Form ── */}
      <div className="card" style={{ padding: '1.5rem', marginBottom: '1.5rem' }}>
        <h3 style={{ fontWeight: 700, fontSize: '0.9rem', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          ⚙️ Single-Symbol RSI DT Optimizer
        </h3>
        <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'flex-end' }}>
          <label style={labelStyle}>
            Symbol
            <input className="field" value={symbol}
              onChange={e => setSymbol(e.target.value.toUpperCase())}
              placeholder="e.g. AAPL" style={{ width: 110 }} />
          </label>
          <label style={labelStyle}>
            As-of Date
            <input className="field" type="date" value={date}
              onChange={e => setDate(e.target.value)} />
          </label>
          <label style={labelStyle}>
            IS Window (bars)
            <input className="field" type="number" min={60} max={500}
              value={isDays} onChange={e => setIsDays(Number(e.target.value))}
              style={{ width: 120 }} />
          </label>
          <label style={labelStyle}>
            OOS / Retrain (bars)
            <input className="field" type="number" min={10} max={90}
              value={oosDays} onChange={e => setOosDays(Number(e.target.value))}
              style={{ width: 120 }} />
          </label>
          <label style={labelStyle}>
            Label Horizon (bars)
            <input className="field" type="number" min={3} max={30}
              value={labelHorizon} onChange={e => setLabelHorizon(Number(e.target.value))}
              style={{ width: 120 }} />
          </label>
          <button className="btn-primary" onClick={onRun}
            disabled={loading || !symbol || !date || (dtParamSource === 'db' && dbFetchStatus !== 'found')}
            style={{ height: 40 }}>
            {loading ? '⏳ Training…' : '▶ Run RSI DT Optimizer'}
          </button>
          {loading && onStop && (
            <button onClick={onStop} style={{ height: 40, display: 'flex', alignItems: 'center', gap: 6, padding: '0 1rem', background: 'var(--danger)', color: '#fff', border: 'none', borderRadius: 'var(--radius-sm)', cursor: 'pointer', fontWeight: 600, fontSize: '0.875rem' }}>
              <Square size={14} /> Stop
            </button>
          )}
          {dtParamSource === 'db' && dbFetchStatus !== 'found' && !loading && (
            <span style={{ fontSize: '0.76rem', color: 'var(--warning)', alignSelf: 'center' }}>
              Fetch DB params first
            </span>
          )}
        </div>

        {/* ── RSI Param Source selector ── */}
        <div style={{
          marginTop: '1rem', padding: '0.75rem 1rem',
          borderRadius: 8, border: '1px solid var(--border)',
          background: 'var(--bg-subtle)',
        }}>
          <div style={{ fontSize: '0.82rem', fontWeight: 600, marginBottom: '0.6rem', color: 'var(--text)' }}>
            RSI Parameters Source
          </div>

          {/* Radio group */}
          <div style={{ display: 'flex', gap: '1.25rem', flexWrap: 'wrap', marginBottom: '0.75rem' }}>
            {(['default', 'db', 'manual'] as const).map(src => (
              <label key={src} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', cursor: 'pointer', fontSize: '0.83rem' }}>
                <input
                  type="radio"
                  name="dtParamSource"
                  value={src}
                  checked={dtParamSource === src}
                  onChange={() => setDtParamSource(src)}
                />
                {src === 'default' ? 'Default (14 / 70 / 30)' : src === 'db' ? 'From Cache (DB)' : 'Manual'}
              </label>
            ))}
          </div>

          {/* From Cache panel */}
          {dtParamSource === 'db' && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap', marginBottom: '0.5rem' }}>
              <button
                className="btn-primary"
                style={{ fontSize: '0.78rem', padding: '4px 12px', height: 32 }}
                disabled={dbFetchStatus === 'loading' || !symbol}
                onClick={() => onFetchDbParams(symbol)}
              >
                {dbFetchStatus === 'loading' ? '⏳ Fetching…' : `Fetch for ${symbol || '—'}`}
              </button>

              {dbFetchStatus === 'found' && dbCachedParams && (
                <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', alignItems: 'center' }}>
                  {[
                    { label: 'Period', value: dbCachedParams.optimal_period },
                    { label: 'Upper',  value: dbCachedParams.optimal_upper  },
                    { label: 'Lower',  value: dbCachedParams.optimal_lower  },
                  ].map(p => (
                    <span key={p.label} style={{
                      fontSize: '0.75rem', padding: '2px 10px', borderRadius: 20,
                      background: '#16a34a22', border: '1px solid #16a34a',
                      color: '#15803d', fontWeight: 600,
                    }}>
                      {p.label}: {p.value}
                    </span>
                  ))}
                  <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                    confidence: {dbCachedParams.confidence} · OOS Sharpe: {dbCachedParams.oos_sharpe.toFixed(2)}
                  </span>
                </div>
              )}

              {dbFetchStatus === 'not_found' && (
                <span style={{ fontSize: '0.78rem', color: 'var(--warning)', fontStyle: 'italic' }}>
                  No cached params for {symbol}. Run Classic WFO first.
                </span>
              )}
            </div>
          )}

          {/* Manual inputs panel */}
          {dtParamSource === 'manual' && (
            <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'flex-end', marginBottom: '0.5rem' }}>
              <label style={{ display: 'flex', flexDirection: 'column', fontSize: '0.78rem', gap: 2 }}>
                Period (2–50)
                <input className="field" type="number" min={2} max={50}
                  value={manualPeriod}
                  onChange={e => setManualPeriod(Math.max(2, Math.min(50, Number(e.target.value))))}
                  style={{ width: 80 }} />
              </label>
              <label style={{ display: 'flex', flexDirection: 'column', fontSize: '0.78rem', gap: 2 }}>
                Upper (50–95)
                <input className="field" type="number" min={50} max={95}
                  value={manualUpper}
                  onChange={e => setManualUpper(Math.max(50, Math.min(95, Number(e.target.value))))}
                  style={{ width: 80 }} />
              </label>
              <label style={{ display: 'flex', flexDirection: 'column', fontSize: '0.78rem', gap: 2 }}>
                Lower (5–50)
                <input className="field" type="number" min={5} max={50}
                  value={manualLower}
                  onChange={e => setManualLower(Math.max(5, Math.min(50, Number(e.target.value))))}
                  style={{ width: 80 }} />
              </label>
            </div>
          )}

          {/* Resolved params summary — always visible */}
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
            Will use: &nbsp;
            {dtParamSource === 'default' && <strong>Period 14 / Upper 70 / Lower 30</strong>}
            {dtParamSource === 'db' && dbFetchStatus === 'found' && dbCachedParams &&
              <strong>Period {dbCachedParams.optimal_period} / Upper {dbCachedParams.optimal_upper} / Lower {dbCachedParams.optimal_lower}</strong>}
            {dtParamSource === 'db' && dbFetchStatus !== 'found' &&
              <em>fetch params above first</em>}
            {dtParamSource === 'manual' &&
              <strong>Period {manualPeriod} / Upper {manualUpper} / Lower {manualLower}</strong>}
          </div>
        </div>

        <p style={{ marginTop: '0.6rem', fontSize: '0.78rem', color: 'var(--text-muted)' }}>
          IS window = training set per slide · OOS = re-train every N bars · Label horizon = "profitable N bars later?"
        </p>
      </div>

      {/* ── Loading ── */}
      {loading && (
        <div className="card" style={{ padding: '3rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
          <div style={{ fontSize: '2.5rem', marginBottom: '0.75rem' }}>🧠</div>
          <p style={{ fontWeight: 600, fontSize: '1rem' }}>Training Decision Trees on RSI features…</p>
          <p style={{ fontSize: '0.82rem', marginTop: '0.5rem', color: 'var(--text-muted)' }}>
            Model A: sliding WFO ({isDays}-bar IS, retrain every {oosDays} bars)
            &nbsp;·&nbsp; Model B: fixed 10-year crash-aware tree
          </p>
        </div>
      )}

      {/* ── Error ── */}
      {error && !loading && (
        <div className="card" style={{ padding: '1.25rem', borderLeft: '4px solid var(--danger)', background: 'var(--danger-soft)', marginBottom: '1rem' }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* ── Results ── */}
      {result && !loading && (
        <>
          <RsiDtSignalBanner result={result} />
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '1rem' }}>
            <RsiDtModelCard
              title="Model A — Sliding WFO"
              subtitle={`Retrained every ${result.oos_days} bars · IS=${result.is_days} bars`}
              prob={result.last_prob_a}
              accent="#8b5cf6"
              badge="Current regime"
            />
            <RsiDtModelCard
              title="Model B — Fixed (10yr)"
              subtitle="Trained on full history incl. 2008 / 2020 crashes"
              prob={result.last_prob_b}
              accent="#f59e0b"
              badge="Crash-aware"
            />
          </div>
          <RsiDtSlidesChart slides={result.slides} />
          <RsiDtFeatureList features={result.feature_names} />
          <RsiDtSlidesTable slides={result.slides} />
        </>
      )}

      {/* ── Portfolio Optimizer ── */}
      <RsiDtPortfolioPanel
        symbols={pfSymbols}   setSymbols={setPfSymbols}
        capital={pfCapital}   setCapital={setPfCapital}
        date={date}
        isDays={isDays}       oosDays={oosDays}
        result={pfResult}     loading={pfLoading} error={pfError}
        onRun={onRunPortfolio}
      />
    </>
  )
}

// ── Signal Banner ─────────────────────────────────────────────────────────────

function RsiDtSignalBanner({ result }: { result: RsiDtResult }) {
  const isLong     = result.last_signal === 1
  const isHalfSize = result.last_size_mult < 1.0
  const signalColor = isLong ? '#16a34a' : '#64748b'
  const signalBg    = isLong ? '#dcfce7' : 'var(--bg-subtle)'
  const allocPct    = (result.last_size_mult * 5).toFixed(1)
  const rsiColor    = result.last_rsi < 30 ? '#16a34a' : result.last_rsi > 70 ? '#ef4444' : 'var(--text-secondary)'

  return (
    <div className="card" style={{
      padding: '1.5rem', borderTop: `4px solid ${signalColor}`,
      display: 'grid', gridTemplateColumns: 'auto 1fr', gap: '1.5rem', alignItems: 'center',
    }}>
      {/* Big signal badge */}
      <div style={{
        background: signalBg, border: `2px solid ${signalColor}`,
        borderRadius: 12, padding: '1rem 2rem', textAlign: 'center', minWidth: 160,
      }}>
        <div style={{ fontSize: '0.7rem', color: signalColor, fontWeight: 700, letterSpacing: '0.1em', marginBottom: 6 }}>
          HYBRID SIGNAL
        </div>
        <div style={{ fontSize: '2rem', fontWeight: 900, color: signalColor, lineHeight: 1 }}>
          {isLong ? '▲ LONG' : '— FLAT'}
        </div>
        <div style={{ fontSize: '0.78rem', color: signalColor, marginTop: 6, fontWeight: 600 }}>
          {allocPct}% of capital
          {isHalfSize && <span style={{ color: '#f59e0b', marginLeft: 4 }}>(½ size — high vol)</span>}
        </div>
      </div>

      {/* Metrics grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.75rem' }}>
        <RsiDtMetricBox label="Model A Confidence" value={`${(result.last_prob_a * 100).toFixed(1)}%`}
          sub="Sliding WFO" color={result.last_prob_a >= 0.6 ? '#16a34a' : '#ef4444'} />
        <RsiDtMetricBox label="Model B Confidence" value={`${(result.last_prob_b * 100).toFixed(1)}%`}
          sub="Fixed 10yr" color={result.last_prob_b >= 0.6 ? '#16a34a' : '#ef4444'} />
        <RsiDtMetricBox label="Threshold" value={`${(result.min_prob_threshold * 100).toFixed(0)}%`}
          sub="Both must exceed" color="var(--text-secondary)" />
        <RsiDtMetricBox label="ATR Percentile" value={`${result.last_atr_pct.toFixed(1)}th`}
          sub={result.last_atr_pct > 80 ? '⚠️ High-vol regime' : '✓ Normal vol'}
          color={result.last_atr_pct > 80 ? '#f59e0b' : '#16a34a'} />
        <RsiDtMetricBox
          label={`Current RSI(${result.rsi_params.period})`}
          value={result.last_rsi.toFixed(1)}
          sub={result.last_rsi < result.rsi_params.lower ? '🟢 Oversold zone'
             : result.last_rsi > result.rsi_params.upper ? '🔴 Overbought' : '⚪ Neutral'}
          color={rsiColor}
        />
        <RsiDtMetricBox label="Avg OOS Accuracy" value={`${(result.avg_oos_acc * 100).toFixed(1)}%`}
          sub={`across ${result.n_slides} slide windows`} color="var(--text-secondary)" />
      </div>

      {/* WFO params badge */}
      {result.wfo_params_used && (
        <div style={{
          gridColumn: '1 / -1', marginTop: '0.75rem',
          display: 'flex', alignItems: 'center', gap: '0.6rem', flexWrap: 'wrap',
          padding: '0.5rem 0.75rem', borderRadius: 8,
          background: '#f0fdf4', border: '1px solid #86efac',
        }}>
          <span style={{ fontSize: '0.75rem', fontWeight: 700, color: '#15803d' }}>
            🎯 WFO-Optimized RSI params injected:
          </span>
          {[
            { label: 'Period', value: result.rsi_params.period },
            { label: 'Upper',  value: result.rsi_params.upper  },
            { label: 'Lower',  value: result.rsi_params.lower  },
          ].map(p => (
            <span key={p.label} style={{
              fontSize: '0.72rem', fontWeight: 700, padding: '2px 10px',
              borderRadius: 20, background: '#16a34a22', border: '1px solid #16a34a',
              color: '#15803d',
            }}>
              {p.label}: {p.value}
            </span>
          ))}
          <span style={{ fontSize: '0.72rem', color: '#16a34a', marginLeft: 4 }}>
            — RSI period &amp; distance features computed with these values
          </span>
        </div>
      )}
    </div>
  )
}

function RsiDtMetricBox({ label, value, sub, color }: { label: string; value: string; sub: string; color: string }) {
  return (
    <div style={{
      background: 'var(--bg-subtle)', borderRadius: 8, padding: '0.65rem 0.85rem',
      border: '1px solid var(--border)',
    }}>
      <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginBottom: 4, fontWeight: 500 }}>{label}</div>
      <div style={{ fontSize: '1.1rem', fontWeight: 800, color, lineHeight: 1.1 }}>{value}</div>
      <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', marginTop: 3 }}>{sub}</div>
    </div>
  )
}

// ── Model A / B Cards ─────────────────────────────────────────────────────────

function RsiDtModelCard({
  title, subtitle, prob, accent, badge,
}: { title: string; subtitle: string; prob: number; accent: string; badge: string }) {
  const pct    = Math.round(prob * 100)
  const passes = prob >= 0.6
  return (
    <div className="card" style={{ padding: '1.25rem', borderTop: `3px solid ${accent}` }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '0.75rem' }}>
        <div>
          <div style={{ fontWeight: 700, fontSize: '0.9rem' }}>{title}</div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 2 }}>{subtitle}</div>
        </div>
        <div style={{
          background: accent + '22', color: accent,
          fontSize: '0.65rem', fontWeight: 700, padding: '2px 8px', borderRadius: 20, whiteSpace: 'nowrap',
        }}>{badge}</div>
      </div>
      {/* Probability gauge */}
      <div style={{ marginBottom: '0.5rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>P(profitable)</span>
          <span style={{ fontSize: '0.85rem', fontWeight: 800, color: passes ? '#16a34a' : '#ef4444' }}>
            {pct}%
          </span>
        </div>
        <div style={{ height: 10, background: 'var(--bg-subtle)', borderRadius: 5, overflow: 'hidden', position: 'relative' }}>
          <div style={{ position: 'absolute', left: '60%', top: 0, width: 2, height: '100%', background: '#f59e0b', zIndex: 2 }} />
          <div style={{
            width: `${pct}%`, height: '100%',
            background: passes ? '#16a34a' : '#ef4444',
            borderRadius: 5, transition: 'width 0.4s ease',
          }} />
        </div>
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 2 }}>
          <span style={{ fontSize: '0.65rem', color: '#f59e0b' }}>│ 60% threshold</span>
        </div>
      </div>
      <div style={{
        marginTop: '0.5rem', padding: '0.4rem 0.6rem',
        borderRadius: 6, fontSize: '0.75rem', fontWeight: 600,
        background: passes ? '#dcfce7' : '#fee2e2',
        color: passes ? '#16a34a' : '#ef4444',
      }}>
        {passes ? '✅ Passes threshold — votes LONG' : '❌ Below threshold — vetoes signal'}
      </div>
    </div>
  )
}

// ── Sliding Windows OOS Accuracy Chart ───────────────────────────────────────

function RsiDtSlidesChart({ slides }: { slides: RsiDtSlide[] }) {
  if (!slides.length) return null
  const data = slides.map(s => ({
    window:     `W${s.slide_idx + 1}`,
    'OOS Acc':  parseFloat((s.oos_acc  * 100).toFixed(1)),
    'Train Acc': parseFloat((s.train_acc * 100).toFixed(1)),
    bars:       `bars ${s.oos_start}–${s.oos_end}`,
  }))
  const avgOos = data.reduce((a, d) => a + d['OOS Acc'], 0) / data.length

  return (
    <div className="card" style={{ padding: '1.5rem', marginTop: '1rem' }}>
      <h3 style={{ fontWeight: 700, fontSize: '0.95rem', marginBottom: '0.25rem' }}>
        📈 Sliding Window OOS Accuracy — Model A
      </h3>
      <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginBottom: '1.25rem' }}>
        Each bar = one IS/OOS fold. <strong>OOS Acc</strong> = % correct on unseen bars.
        Dashed line = 60% baseline (random ≈ 50%). Avg OOS: <strong>{avgOos.toFixed(1)}%</strong>
      </p>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ top: 5, right: 20, left: -10, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis dataKey="window" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
          <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: 'var(--text-muted)' }}
            tickFormatter={(v: number) => `${v}%`} />
          <Tooltip
            contentStyle={{ background: 'var(--bg-elevated)', border: '1px solid var(--border)', fontSize: '0.78rem' }}
            formatter={(v: any, name: string) => [`${v}%`, name]}
            labelFormatter={(label: string, payload: any[]) => `${label} · ${payload?.[0]?.payload?.bars ?? ''}`}
          />
          <ReferenceLine y={60} stroke="#f59e0b" strokeDasharray="5 3"
            label={{ value: '60% threshold', fontSize: 9, fill: '#f59e0b', position: 'insideTopRight' }} />
          <ReferenceLine y={50} stroke="#ef4444" strokeDasharray="3 2"
            label={{ value: '50% random', fontSize: 9, fill: '#ef4444', position: 'insideBottomRight' }} />
          <Bar dataKey="OOS Acc" fill="#8b5cf6" radius={[4, 4, 0, 0]} maxBarSize={40}>
            {data.map((d, i) => (
              <Cell key={i} fill={d['OOS Acc'] >= 60 ? '#8b5cf6' : d['OOS Acc'] >= 50 ? '#f59e0b' : '#ef4444'} />
            ))}
          </Bar>
          <Bar dataKey="Train Acc" fill="#c4b5fd" radius={[4, 4, 0, 0]} maxBarSize={40} opacity={0.5} />
        </BarChart>
      </ResponsiveContainer>
      <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
        🟣 OOS Accuracy &nbsp;·&nbsp; 🔵 Train Accuracy (faded) &nbsp;·&nbsp;
        Purple = above threshold · Yellow = above random · Red = below random
      </p>
    </div>
  )
}

// ── Feature List ──────────────────────────────────────────────────────────────

function RsiDtFeatureList({ features }: { features: string[] }) {
  const descriptions: Record<string, string> = {
    rsi14:     'RSI(14) — core mean-reversion signal',
    ema_ratio: 'EMA50 / EMA200 — trend direction',
    macd_hist: 'MACD Histogram — momentum confirmation',
    vol_ratio: 'Avg Volume 30d / 60d — "Big Money" confirmation',
    bb_pct_b:  'Bollinger %B — overbought / oversold position',
    atr14:     'ATR-14 — volatility regime normaliser',
  }
  return (
    <div className="card" style={{ padding: '1.25rem', marginTop: '1rem' }}>
      <h3 style={{ fontWeight: 700, fontSize: '0.9rem', marginBottom: '0.75rem' }}>🔬 RSI DT Input Features</h3>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(260px, 1fr))', gap: '0.5rem' }}>
        {features.map((f, i) => (
          <div key={f} style={{
            display: 'flex', alignItems: 'center', gap: '0.6rem',
            background: 'var(--bg-subtle)', borderRadius: 8,
            padding: '0.5rem 0.75rem', border: '1px solid var(--border)',
          }}>
            <span style={{
              background: '#8b5cf6', color: '#fff', borderRadius: '50%',
              width: 20, height: 20, display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: '0.65rem', fontWeight: 700, flexShrink: 0,
            }}>{i + 1}</span>
            <div>
              <div style={{ fontSize: '0.78rem', fontWeight: 700, fontFamily: 'var(--font-mono)' }}>{f}</div>
              <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>{descriptions[f] ?? ''}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── Slides Detail Table ───────────────────────────────────────────────────────

function RsiDtSlidesTable({ slides }: { slides: RsiDtSlide[] }) {
  const [open, setOpen] = useState(false)
  if (!slides.length) return null
  return (
    <div style={{ marginTop: '1rem' }}>
      <button onClick={() => setOpen(o => !o)} style={{
        width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        background: 'var(--bg-subtle)', color: 'var(--text-muted)',
        border: '1px solid var(--border)', borderRadius: open ? '10px 10px 0 0' : 10,
        padding: '0.6rem 1rem', cursor: 'pointer', fontSize: '0.8rem',
      }}>
        <span>📋 Sliding Window Detail — {slides.length} folds</span>
        <span>{open ? '▲' : '▼'}</span>
      </button>
      {open && (
        <div style={{ border: '1px solid var(--border)', borderTop: 'none', borderRadius: '0 0 10px 10px', overflowX: 'auto' }}>
          <table style={{ width: '100%', fontSize: '0.8rem', borderCollapse: 'collapse' }}>
            <thead style={{ background: 'var(--bg-subtle)' }}>
              <tr>
                {['Window', 'IS bars', 'OOS bars', 'Train Acc', 'OOS Acc', 'Status'].map(h => (
                  <th key={h} style={{ ...thStyle, padding: '0.5rem 0.75rem', fontWeight: 600 }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {slides.map(s => {
                const oosOk = s.oos_acc >= 0.6
                return (
                  <tr key={s.slide_idx} style={{ borderTop: '1px solid var(--border)' }}>
                    <td style={{ padding: '0.45rem 0.75rem', fontWeight: 600 }}>W{s.slide_idx + 1}</td>
                    <td style={{ padding: '0.45rem 0.75rem', color: 'var(--text-muted)' }}>{s.is_start}–{s.is_end}</td>
                    <td style={{ padding: '0.45rem 0.75rem', color: 'var(--text-muted)' }}>{s.oos_start}–{s.oos_end}</td>
                    <td style={{ padding: '0.45rem 0.75rem' }}>{(s.train_acc * 100).toFixed(1)}%</td>
                    <td style={{ padding: '0.45rem 0.75rem', fontWeight: 700, color: oosOk ? '#16a34a' : s.oos_acc >= 0.5 ? '#f59e0b' : '#ef4444' }}>
                      {(s.oos_acc * 100).toFixed(1)}%
                    </td>
                    <td style={{ padding: '0.45rem 0.75rem' }}>
                      <span style={{
                        fontSize: '0.7rem', fontWeight: 700, padding: '2px 8px', borderRadius: 20,
                        background: oosOk ? '#dcfce7' : '#fee2e2',
                        color: oosOk ? '#16a34a' : '#ef4444',
                      }}>
                        {oosOk ? '✓ Above threshold' : '✗ Below threshold'}
                      </span>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

// ═════════════════════════════════════════════════════════════════════════════
// RSI DT PORTFOLIO PANEL
// ═════════════════════════════════════════════════════════════════════════════

interface RsiDtPortfolioPanelProps {
  symbols: string;  setSymbols: (v: string) => void
  capital: number;  setCapital: (v: number) => void
  date: string
  isDays: number;   oosDays: number
  result: RsiDtPortfolioResult | null
  loading: boolean; error: string | null
  onRun: () => void
}

function RsiDtPortfolioPanel({
  symbols, setSymbols, capital, setCapital,
  date, isDays, oosDays,
  result, loading, error, onRun,
}: RsiDtPortfolioPanelProps) {
  return (
    <div style={{ marginTop: '2rem' }}>
      {/* Divider */}
      <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1.25rem' }}>
        <div style={{ flex: 1, height: 1, background: 'var(--border)' }} />
        <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontWeight: 600, whiteSpace: 'nowrap' }}>
          💼 PORTFOLIO OPTIMIZER — 5% Rule · Max 20 Positions · Liquidity Filter
        </span>
        <div style={{ flex: 1, height: 1, background: 'var(--border)' }} />
      </div>

      {/* Config */}
      <div className="card" style={{ padding: '1.5rem', marginBottom: '1.5rem' }}>
        <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', alignItems: 'flex-end' }}>
          <label style={{ ...labelStyle, flex: 1, minWidth: 280 }}>
            Symbol Universe (comma-separated)
            <input className="field" value={symbols}
              onChange={e => setSymbols(e.target.value)}
              placeholder="AAPL,MSFT,NVDA,TSLA,…"
              style={{ width: '100%' }} />
          </label>
          <label style={labelStyle}>
            Capital ($)
            <input className="field" type="number" min={1000} step={1000}
              value={capital} onChange={e => setCapital(Number(e.target.value))}
              style={{ width: 140 }} />
          </label>
          <button className="btn-primary" onClick={onRun}
            disabled={loading || !symbols.trim()} style={{ height: 40 }}>
            {loading ? '⏳ Scanning…' : '▶ Run Portfolio Scan'}
          </button>
        </div>
        <p style={{ marginTop: '0.6rem', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          Uses date <strong>{date}</strong> · IS={isDays} bars · OOS={oosDays} bars ·
          Liquidity gate: $5M+ 30-day avg dollar volume · Allocation: 5% full / 2.5% high-vol
        </p>
      </div>

      {loading && (
        <div className="card" style={{ padding: '2.5rem', textAlign: 'center', color: 'var(--text-secondary)' }}>
          <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>💼</div>
          <p style={{ fontWeight: 600 }}>Running RSI DT optimizer on each symbol…</p>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.4rem' }}>
            {symbols.split(',').filter(Boolean).length} symbols · may take a minute
          </p>
        </div>
      )}

      {error && !loading && (
        <div className="card" style={{ padding: '1.25rem', borderLeft: '4px solid var(--danger)', background: 'var(--danger-soft)' }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {result && !loading && <RsiDtPortfolioResults result={result} />}
    </div>
  )
}

// ── Portfolio Results ─────────────────────────────────────────────────────────

function RsiDtPortfolioResults({ result }: { result: RsiDtPortfolioResult }) {
  const deployedPct = (result.total_deployed_pct * 100).toFixed(1)
  const cashPct     = (100 - result.total_deployed_pct * 100).toFixed(1)
  const cashUsd     = result.capital - result.total_deployed_usd

  return (
    <>
      {/* Summary banner */}
      <div className="card" style={{
        padding: '1.25rem 1.5rem', marginBottom: '1rem',
        display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', gap: '1rem',
        borderTop: '3px solid #16a34a',
      }}>
        <RsiDtMetricBox label="Positions Selected"   value={String(result.n_selected)}
          sub={`of ${result.n_evaluated} evaluated`} color="#16a34a" />
        <RsiDtMetricBox label="Capital Deployed"     value={`$${result.total_deployed_usd.toLocaleString()}`}
          sub={`${deployedPct}% of total`} color="#8b5cf6" />
        <RsiDtMetricBox label="Cash Reserve"         value={`$${cashUsd.toLocaleString()}`}
          sub={`${cashPct}% undeployed`} color="var(--text-secondary)" />
        <RsiDtMetricBox label="Rejected (Liquidity)" value={String(result.rejected_liquidity.length)}
          sub="Below $5M dollar vol" color="#f59e0b" />
        <RsiDtMetricBox label="Rejected (No Signal)" value={String(result.rejected_no_signal.length)}
          sub="DT threshold not met" color="#64748b" />
        <RsiDtMetricBox label="Max Positions"        value={String(result.max_positions)}
          sub="5% each, capped at 20" color="var(--text-secondary)" />
      </div>

      {/* Positions table */}
      {result.positions.length > 0 && (
        <div className="card" style={{ padding: '1.25rem', marginBottom: '1rem' }}>
          <h3 style={{ fontWeight: 700, fontSize: '0.9rem', marginBottom: '0.75rem' }}>
            ✅ Selected Positions ({result.positions.length})
          </h3>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', fontSize: '0.8rem', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ color: 'var(--text-muted)', fontSize: '0.72rem' }}>
                  {['Symbol', 'Model A P', 'Model B P', 'RSI(14)', 'ATR Pct', 'Vol Regime', 'Alloc %', 'Alloc $', 'OOS Acc'].map(h => (
                    <th key={h} style={{ ...thStyle, padding: '0 0.75rem 0.5rem 0' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {result.positions.map((p, i) => {
                  const rsiColor = p.last_rsi < 30 ? '#16a34a' : p.last_rsi > 70 ? '#ef4444' : 'var(--text-muted)'
                  return (
                    <tr key={p.symbol} style={{ borderTop: '1px solid var(--border)', background: i % 2 ? 'var(--bg-subtle)' : 'transparent' }}>
                      <td style={{ padding: '0.45rem 0.75rem 0.45rem 0', fontWeight: 800, color: '#8b5cf6' }}>{p.symbol}</td>
                      <td style={{ padding: '0.45rem 0.75rem 0.45rem 0', color: p.prob_a >= 0.6 ? '#16a34a' : '#ef4444', fontWeight: 700 }}>
                        {(p.prob_a * 100).toFixed(1)}%
                      </td>
                      <td style={{ padding: '0.45rem 0.75rem 0.45rem 0', color: p.prob_b >= 0.6 ? '#16a34a' : '#ef4444', fontWeight: 700 }}>
                        {(p.prob_b * 100).toFixed(1)}%
                      </td>
                      <td style={{ padding: '0.45rem 0.75rem 0.45rem 0', color: rsiColor, fontWeight: 700 }}>
                        {p.last_rsi.toFixed(1)}
                      </td>
                      <td style={{ padding: '0.45rem 0.75rem 0.45rem 0', color: p.atr_pct > 80 ? '#f59e0b' : 'var(--text-muted)' }}>
                        {p.atr_pct.toFixed(1)}th
                      </td>
                      <td style={{ padding: '0.45rem 0.75rem 0.45rem 0' }}>
                        <span style={{
                          fontSize: '0.68rem', fontWeight: 700, padding: '2px 7px', borderRadius: 20,
                          background: p.size_mult < 1 ? '#fef9c3' : '#dcfce7',
                          color: p.size_mult < 1 ? '#b45309' : '#16a34a',
                        }}>
                          {p.size_mult < 1 ? '⚡ Half' : '✓ Full'}
                        </span>
                      </td>
                      <td style={{ padding: '0.45rem 0.75rem 0.45rem 0', fontWeight: 700 }}>
                        {(p.alloc_pct * 100).toFixed(1)}%
                      </td>
                      <td style={{ padding: '0.45rem 0.75rem 0.45rem 0', fontWeight: 700, color: '#16a34a' }}>
                        ${p.alloc_dollars.toLocaleString()}
                      </td>
                      <td style={{ padding: '0.45rem 0.75rem 0.45rem 0', color: p.avg_oos_acc >= 0.6 ? '#16a34a' : 'var(--text-muted)' }}>
                        {(p.avg_oos_acc * 100).toFixed(1)}%
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Rejected lists */}
      {(result.rejected_liquidity.length > 0 || result.rejected_no_signal.length > 0 || result.rejected_errors.length > 0) && (
        <div className="card" style={{ padding: '1.25rem' }}>
          <h3 style={{ fontWeight: 700, fontSize: '0.9rem', marginBottom: '0.75rem', color: 'var(--text-muted)' }}>
            ❌ Rejected Symbols
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(220px, 1fr))', gap: '0.75rem' }}>
            {result.rejected_liquidity.length > 0 && (
              <div>
                <div style={{ fontSize: '0.72rem', fontWeight: 700, color: '#f59e0b', marginBottom: 4 }}>
                  Liquidity gate ({result.rejected_liquidity.length})
                </div>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                  {result.rejected_liquidity.join(' · ')}
                </div>
              </div>
            )}
            {result.rejected_no_signal.length > 0 && (
              <div>
                <div style={{ fontSize: '0.72rem', fontWeight: 700, color: '#64748b', marginBottom: 4 }}>
                  No DT signal ({result.rejected_no_signal.length})
                </div>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                  {result.rejected_no_signal.join(' · ')}
                </div>
              </div>
            )}
            {result.rejected_errors.length > 0 && (
              <div>
                <div style={{ fontSize: '0.72rem', fontWeight: 700, color: '#ef4444', marginBottom: 4 }}>
                  Errors ({result.rejected_errors.length})
                </div>
                <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                  {result.rejected_errors.join(' · ')}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

    </>
  )
}

// ── OOS Test Panel (reused in RSI Lab) ────────────────────────────────────────

interface OosTestPanelProps {
  algo: 'macd' | 'rsi'
  symbol: string; onSymbol: (v: string) => void
  endDate: string; onEndDate: (v: string) => void
  nDates: number; onNDates: (v: number) => void
  isDays: number; onIsDays: (v: number) => void
  oosDays: number; onOosDays: (v: number) => void
  notional: number; onNotional: (v: number) => void
  result: OosTestResponse | null
  loading: boolean; error: string | null
  onRun: () => void
  onStop?: () => void
}

function OosTestPanel({
  algo, symbol, onSymbol, endDate, onEndDate, nDates, onNDates,
  isDays, onIsDays, oosDays, onOosDays, notional, onNotional,
  result, loading, error, onRun, onStop,
}: OosTestPanelProps) {
  const accentColor = algo === 'rsi' ? '#7c3aed' : '#e11d48'
  const labelStyle: React.CSSProperties = {
    display: 'flex', flexDirection: 'column', gap: '0.35rem',
    fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 500,
  }
  const inputStyle: React.CSSProperties = {
    padding: '0.45rem 0.7rem', borderRadius: 'var(--radius-sm)',
    border: '1px solid var(--border)', background: 'var(--bg)',
    color: 'var(--text)', fontSize: '0.9rem',
  }

  const pnlColor = (v: number) => v > 0 ? 'var(--success)' : v < 0 ? 'var(--danger)' : 'var(--text-muted)'
  const confColor = (c?: string) =>
    c === 'HIGH' ? 'var(--success)' : c === 'MEDIUM' ? 'var(--warning)' : 'var(--danger)'
  const sharpeColor = (v?: number) => {
    if (v === undefined || v === null) return 'var(--text-muted)'
    if (v >= 1.0) return 'var(--success)'
    if (v >= 0.5) return 'var(--warning)'
    if (v >= 0)   return 'var(--text-secondary)'
    return 'var(--danger)'
  }

  const equityCurveData = result?.rows
    .filter(r => !r.error && r.pnl_dollar !== undefined)
    .reduce<{ date: string; cumPnl: number }[]>((acc, r, i) => {
      const prev = i === 0 ? 0 : acc[i - 1].cumPnl
      acc.push({ date: r.date, cumPnl: parseFloat((prev + (r.pnl_dollar ?? 0)).toFixed(2)) })
      return acc
    }, []) ?? []

  const tdStyle: React.CSSProperties = { padding: '0.45rem 0.6rem', borderTop: '1px solid var(--border)', fontSize: '0.82rem' }
  const thStyle: React.CSSProperties = { padding: '0.35rem 0.6rem', fontWeight: 600, fontSize: '0.75rem', color: 'var(--text-muted)', textAlign: 'left' as const }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>

      {/* Config card */}
      <div className="card" style={{ padding: '1.5rem' }}>
        <div style={{ fontWeight: 700, fontSize: '1rem', marginBottom: '1.1rem', display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ color: accentColor }}>🧪</span> OOS Walk-Forward Integration Test
          <span style={{ fontSize: '0.78rem', fontWeight: 400, color: 'var(--text-muted)', marginLeft: 6 }}>
            — Run the optimizer across N historical dates and see actual dollar P&amp;L per OOS window
          </span>
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', gap: '1rem', marginBottom: '1.2rem' }}>
          <label style={labelStyle}>Symbol<input style={inputStyle} value={symbol} onChange={e => onSymbol(e.target.value.toUpperCase())} placeholder="AAPL" /></label>
          <label style={labelStyle}>Sweep End Date<input type="date" style={inputStyle} value={endDate} onChange={e => onEndDate(e.target.value)} /></label>
          <label style={labelStyle}>N Dates<input type="number" style={inputStyle} min={2} max={20} value={nDates} onChange={e => onNDates(Number(e.target.value))} /></label>
          <label style={labelStyle}>IS Days<input type="number" style={inputStyle} min={30} value={isDays} onChange={e => onIsDays(Number(e.target.value))} /></label>
          <label style={labelStyle}>OOS Days<input type="number" style={inputStyle} min={20} value={oosDays} onChange={e => onOosDays(Number(e.target.value))} /></label>
          <label style={labelStyle}>Notional ($)<input type="number" style={inputStyle} min={100} step={1000} value={notional} onChange={e => onNotional(Number(e.target.value))} /></label>
        </div>
        <button
          className="btn-primary"
          onClick={onRun}
          disabled={loading}
          style={{ background: accentColor, border: 'none', borderRadius: 'var(--radius-sm)', padding: '0.6rem 1.4rem', color: '#fff', fontWeight: 700, cursor: loading ? 'not-allowed' : 'pointer', opacity: loading ? 0.7 : 1 }}
        >
          {loading ? '⏳ Running sweep…' : `▶ Run ${algo.toUpperCase()} OOS Sweep`}
        </button>
        {loading && onStop && (
          <button onClick={onStop} style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '0.6rem 1rem', background: 'var(--danger)', color: '#fff', border: 'none', borderRadius: 'var(--radius-sm)', cursor: 'pointer', fontWeight: 600, fontSize: '0.875rem' }}>
            <Square size={14} /> Stop
          </button>
        )}
        {result && !loading && (
          <span style={{ marginLeft: 12, fontSize: '0.8rem', color: 'var(--text-muted)' }}>
            ✅ {result.n_succeeded}/{result.n_requested} dates OK · sweep {result.oos_start} → {result.oos_end}
          </span>
        )}
      </div>

      {error && (
        <div className="card" style={{ padding: '1rem', background: 'rgba(239,68,68,0.08)', borderLeft: '4px solid var(--danger)' }}>
          <strong style={{ color: 'var(--danger)' }}>Error:</strong> {error}
        </div>
      )}

      {loading && (
        <div className="card" style={{ padding: '2rem', textAlign: 'center', color: 'var(--text-muted)' }}>
          <div style={{ fontSize: '1.8rem', marginBottom: 8 }}>⏳</div>
          Running optimizer for {nDates} dates — this may take a minute…
        </div>
      )}

      {result && !loading && (
        <>
          {/* Summary panel */}
          <div className="card" style={{ padding: '1.25rem', borderLeft: `4px solid ${accentColor}` }}>
            <div style={{ fontWeight: 700, marginBottom: '0.9rem', fontSize: '0.9rem' }}>📊 Sweep Summary</div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: '0.75rem' }}>
              {[
                { label: 'Total P&L', value: `$${result.summary.total_pnl_dollar >= 0 ? '+' : ''}${result.summary.total_pnl_dollar.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`, color: pnlColor(result.summary.total_pnl_dollar) },
                { label: 'Win Rate', value: `${result.summary.win_rate_pct}%  (${result.summary.n_wins}W / ${result.summary.n_losses}L)`, color: result.summary.win_rate_pct >= 50 ? 'var(--success)' : 'var(--danger)' },
                { label: 'Avg Win', value: `$${result.summary.avg_win_dollar.toFixed(2)}`, color: 'var(--success)' },
                { label: 'Avg Loss', value: `$${result.summary.avg_loss_dollar.toFixed(2)}`, color: 'var(--danger)' },
                { label: 'Profit Factor', value: result.summary.profit_factor >= 9999 ? '∞' : result.summary.profit_factor.toFixed(2), color: result.summary.profit_factor >= 1.5 ? 'var(--success)' : result.summary.profit_factor >= 1 ? 'var(--warning)' : 'var(--danger)' },
                { label: 'Avg OOS Sharpe', value: result.summary.avg_oos_sharpe.toFixed(4), color: sharpeColor(result.summary.avg_oos_sharpe) },
                { label: 'Avg IS Sharpe', value: result.summary.avg_is_sharpe.toFixed(4), color: sharpeColor(result.summary.avg_is_sharpe) },
                { label: 'Confidence', value: `H:${result.summary.high_conf} M:${result.summary.med_conf} L:${result.summary.low_conf}`, color: 'var(--text-secondary)' },
              ].map(({ label, value, color }) => (
                <div key={label} style={{ background: 'var(--bg-subtle)', borderRadius: 'var(--radius-sm)', padding: '0.6rem 0.8rem', border: '1px solid var(--border)' }}>
                  <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginBottom: 3 }}>{label}</div>
                  <div style={{ fontWeight: 700, fontSize: '0.9rem', color }}>{value}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Cumulative P&L chart */}
          {equityCurveData.length > 1 && (
            <div className="card" style={{ padding: '1.25rem' }}>
              <div style={{ fontWeight: 700, marginBottom: '0.9rem', fontSize: '0.9rem' }}>📈 Cumulative P&amp;L Across Sweep Dates</div>
              <ResponsiveContainer width="100%" height={200}>
                <ComposedChart data={equityCurveData} margin={{ top: 8, right: 16, left: 8, bottom: 8 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                  <XAxis dataKey="date" tick={{ fontSize: 11, fill: 'var(--text-muted)' }} />
                  <YAxis tick={{ fontSize: 11, fill: 'var(--text-muted)' }} tickFormatter={v => `$${v}`} />
                  <Tooltip formatter={(v: number) => [`$${v.toFixed(2)}`, 'Cum. P&L']} labelStyle={{ color: 'var(--text)' }} contentStyle={{ background: 'var(--bg)', border: '1px solid var(--border)', borderRadius: 6 }} />
                  <ReferenceLine y={0} stroke="var(--border)" strokeDasharray="4 4" />
                  <Line type="monotone" dataKey="cumPnl" stroke={accentColor} strokeWidth={2} dot={{ r: 4, fill: accentColor }} activeDot={{ r: 6 }} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Per-date results table */}
          <div className="card" style={{ padding: '1.25rem', overflowX: 'auto' }}>
            <div style={{ fontWeight: 700, marginBottom: '0.9rem', fontSize: '0.9rem' }}>📋 Per-Date Results</div>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.82rem' }}>
              <thead>
                <tr style={{ background: 'var(--bg-subtle)' }}>
                  <th style={thStyle}>Date</th>
                  <th style={thStyle}>Params (p / hi / lo)</th>
                  <th style={thStyle}>IS Sharpe</th>
                  <th style={thStyle}>OOS Sharpe</th>
                  <th style={thStyle}>Confidence</th>
                  <th style={thStyle}>OOS Return</th>
                  <th style={thStyle}>P&amp;L ($)</th>
                  <th style={thStyle}>Win Rate</th>
                  <th style={thStyle}>Max DD</th>
                </tr>
              </thead>
              <tbody>
                {result.rows.map(row => (
                  <tr key={row.date} style={{ background: row.pnl_dollar && row.pnl_dollar > 0 ? 'rgba(16,185,129,0.04)' : row.pnl_dollar && row.pnl_dollar < 0 ? 'rgba(239,68,68,0.04)' : 'transparent' }}>
                    <td style={{ ...tdStyle, fontWeight: 600 }}>{row.date}</td>
                    <td style={{ ...tdStyle, fontFamily: 'monospace', color: 'var(--text-secondary)' }}>
                      {row.error ? '—' : `${row.period} / ${row.upper} / ${row.lower}`}
                    </td>
                    <td style={{ ...tdStyle, color: sharpeColor(row.is_sharpe) }}>{row.is_sharpe?.toFixed(4) ?? '—'}</td>
                    <td style={{ ...tdStyle, color: sharpeColor(row.oos_sharpe) }}>{row.oos_sharpe?.toFixed(4) ?? '—'}</td>
                    <td style={{ ...tdStyle, fontWeight: 700, color: confColor(row.confidence) }}>{row.confidence ?? '—'}</td>
                    <td style={{ ...tdStyle, color: pnlColor(row.pnl_pct ?? 0) }}>{row.pnl_pct !== undefined ? `${row.pnl_pct >= 0 ? '+' : ''}${row.pnl_pct.toFixed(2)}%` : row.error ? <span style={{ color: 'var(--danger)', fontSize: '0.75rem' }}>ERR</span> : '—'}</td>
                    <td style={{ ...tdStyle, fontWeight: 700, color: pnlColor(row.pnl_dollar ?? 0) }}>{row.pnl_dollar !== undefined ? `$${row.pnl_dollar >= 0 ? '+' : ''}${row.pnl_dollar.toFixed(2)}` : '—'}</td>
                    <td style={tdStyle}>{row.win_rate !== undefined ? `${row.win_rate.toFixed(1)}%` : '—'}</td>
                    <td style={{ ...tdStyle, color: (row.max_dd_pct ?? 0) > 10 ? 'var(--danger)' : (row.max_dd_pct ?? 0) > 5 ? 'var(--warning)' : 'var(--success)' }}>{row.max_dd_pct !== undefined ? `${row.max_dd_pct.toFixed(2)}%` : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}
