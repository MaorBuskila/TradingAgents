import { useState, useRef, useEffect } from 'react'
import axios from 'axios'
import { useTabLogger } from '../hooks/useTabLogger'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  ReferenceLine, Cell,
} from 'recharts'
import { FlaskConical, Zap, Square, ChevronDown, ChevronUp, Trophy } from 'lucide-react'

const API_BASE = import.meta.env.VITE_API_BASE ?? '/api'

// ── Types ─────────────────────────────────────────────────────────────────────

interface FeatureLabResultRow {
  run_id: string
  symbol: string
  date: string
  rule_set_name: string
  features_used: string[]
  veto_rules: string[]
  oos_sharpe: number
  hit_rate: number
  trade_count: number
  confidence: 'HIGH' | 'MEDIUM' | 'LOW'
  n_slides: number
  avg_oos_acc: number
  ran_at: string
}

interface FeatureLabResponse {
  run_id: string
  symbol: string
  date: string
  results: FeatureLabResultRow[]
  winner: string | null
  ran_at: string
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function confidenceBadge(c: string) {
  const colors: Record<string, string> = {
    HIGH: '#16a34a', MEDIUM: '#d97706', LOW: '#dc2626',
  }
  return (
    <span style={{
      background: colors[c] ?? '#6b7280', color: '#fff',
      borderRadius: 4, padding: '2px 8px', fontSize: 11, fontWeight: 600,
    }}>{c}</span>
  )
}

function barColor(v: number) {
  if (v > 0.5) return '#16a34a'
  if (v > 0)   return '#d97706'
  return '#dc2626'
}

const RULE_SET_DESCRIPTIONS: Record<string, string> = {
  BASE:   'RSI · EMA ratio · MACD hist · ATR — baseline 4 features',
  VOLUME: 'BASE + volume ratio feature + volume burst veto (vol > 1.2× SMA)',
  TREND:  'BASE + ADX-14 feature + ADX > 20 veto (trending market filter)',
  FULL:   'All features + all veto rules',
  CUSTOM: 'User-defined feature columns and veto rules',
}

// ── Component ─────────────────────────────────────────────────────────────────

export default function RSIFeatureLab() {
  const log = useTabLogger('RSIFeatureLab')
  const [symbol, setSymbol] = useState('SPY')
  const [date, setDate] = useState(() => new Date().toISOString().slice(0, 10))
  const [selectedRuleSets, setSelectedRuleSets] = useState<string[]>(['BASE', 'VOLUME', 'TREND', 'FULL'])
  const [isDays, setIsDays] = useState(1000)
  const [oosDays, setOosDays] = useState(20)
  const [labelHorizon, setLabelHorizon] = useState(10)
  const [customFeatures, setCustomFeatures] = useState('')
  const [customVetos, setCustomVetos] = useState('')

  const [response, setResponse] = useState<FeatureLabResponse | null>(null)
  const [history, setHistory] = useState<FeatureLabResultRow[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [historyOpen, setHistoryOpen] = useState(false)

  const abortRef = useRef<AbortController | null>(null)

  const PREDEFINED = ['BASE', 'VOLUME', 'TREND', 'FULL']

  function toggleRuleSet(name: string) {
    setSelectedRuleSets(prev =>
      prev.includes(name) ? prev.filter(n => n !== name) : [...prev, name]
    )
  }

  async function loadHistory(sym: string) {
    try {
      const { data } = await axios.get<FeatureLabResultRow[]>(
        `${API_BASE}/rsi-dt-feature-lab/${sym.toUpperCase()}`
      )
      setHistory(data)
    } catch {
      setHistory([])
    }
  }

  useEffect(() => { loadHistory(symbol) }, [symbol])

  async function handleRun() {
    if (!symbol.trim()) return
    setLoading(true)
    setError(null)
    setResponse(null)
    abortRef.current = new AbortController()

    const ruleSets = [...selectedRuleSets]
    if (ruleSets.includes('CUSTOM') && !customFeatures.trim()) {
      setError('CUSTOM selected but no feature columns provided.')
      setLoading(false)
      return
    }

    log('action:run', { symbol: symbol.toUpperCase().trim(), ruleSets })
    try {
      log('api:start', { endpoint: 'rsi-dt-feature-lab' })
      const { data } = await axios.post<FeatureLabResponse>(
        `${API_BASE}/rsi-dt-feature-lab`,
        {
          symbol: symbol.toUpperCase().trim(),
          date,
          rule_sets: ruleSets,
          custom_features: ruleSets.includes('CUSTOM') && customFeatures.trim()
            ? customFeatures.split(',').map(s => s.trim()).filter(Boolean)
            : null,
          custom_vetos: ruleSets.includes('CUSTOM') && customVetos.trim()
            ? customVetos.split(',').map(s => s.trim()).filter(Boolean)
            : null,
          is_days: isDays,
          oos_days: oosDays,
          label_horizon: labelHorizon,
        },
        { signal: abortRef.current.signal }
      )
      setResponse(data)
      log('api:success')
      loadHistory(symbol)
    } catch (err: any) {
      if (!axios.isCancel(err)) {
        log('api:error', err)
        setError(err.response?.data?.detail ?? String(err))
      }
    } finally {
      setLoading(false)
    }
  }

  function handleStop() {
    abortRef.current?.abort()
    setLoading(false)
  }

  function saveWinner(row: FeatureLabResultRow) {
    localStorage.setItem(
      `rsi_dt_lab_winner_${symbol.toUpperCase()}`,
      JSON.stringify({ rule_set_name: row.rule_set_name, features_used: row.features_used, veto_rules: row.veto_rules, saved_at: new Date().toISOString() })
    )
    alert(`Saved "${row.rule_set_name}" as active config for ${symbol.toUpperCase()}.`)
  }

  // Group history by run_id
  const historyByRun: Record<string, FeatureLabResultRow[]> = {}
  for (const r of history) {
    if (!historyByRun[r.run_id]) historyByRun[r.run_id] = []
    historyByRun[r.run_id].push(r)
  }
  const recentRuns = Object.entries(historyByRun).slice(0, 10)

  return (
    <div style={{ maxWidth: 960, margin: '0 auto', padding: '24px 16px' }}>
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
        <FlaskConical size={22} />
        <h2 style={{ margin: 0 }}>RSI DT Feature Lab</h2>
      </div>
      <p style={{ color: 'var(--text-muted)', marginTop: 4, marginBottom: 20, fontSize: 13 }}>
        Compare named feature rule sets for the XGBoost optimizer — pick the best config for your symbol.
      </p>

      {/* Input form */}
      <div style={{ background: 'var(--card-bg)', border: '1px solid var(--border)', borderRadius: 8, padding: 16, marginBottom: 20 }}>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginBottom: 16 }}>
          <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 12 }}>
            Symbol
            <input value={symbol} onChange={e => setSymbol(e.target.value.toUpperCase())}
              style={{ width: 90, padding: '6px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--input-bg)', color: 'var(--text)' }} />
          </label>
          <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 12 }}>
            As-of Date
            <input type="date" value={date} onChange={e => setDate(e.target.value)}
              style={{ padding: '6px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--input-bg)', color: 'var(--text)' }} />
          </label>
          <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 12 }}>
            IS Window (bars)
            <input type="number" value={isDays} onChange={e => setIsDays(+e.target.value)} min={500}
              style={{ width: 110, padding: '6px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--input-bg)', color: 'var(--text)' }} />
          </label>
          <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 12 }}>
            OOS / Retrain (bars)
            <input type="number" value={oosDays} onChange={e => setOosDays(+e.target.value)} min={5}
              style={{ width: 110, padding: '6px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--input-bg)', color: 'var(--text)' }} />
          </label>
          <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 12 }}>
            Label Horizon (bars)
            <input type="number" value={labelHorizon} onChange={e => setLabelHorizon(+e.target.value)} min={3}
              style={{ width: 110, padding: '6px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--input-bg)', color: 'var(--text)' }} />
          </label>
        </div>

        {/* Rule set checkboxes */}
        <div style={{ marginBottom: 12 }}>
          <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>Rule Sets to Compare</div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 10 }}>
            {[...PREDEFINED, 'CUSTOM'].map(name => (
              <label key={name} style={{ display: 'flex', alignItems: 'flex-start', gap: 6, cursor: 'pointer', fontSize: 13 }}>
                <input type="checkbox" checked={selectedRuleSets.includes(name)}
                  onChange={() => toggleRuleSet(name)} style={{ marginTop: 2 }} />
                <div>
                  <div style={{ fontWeight: 600 }}>{name}</div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', maxWidth: 200 }}>{RULE_SET_DESCRIPTIONS[name]}</div>
                </div>
              </label>
            ))}
          </div>
        </div>

        {/* Custom fields */}
        {selectedRuleSets.includes('CUSTOM') && (
          <div style={{ display: 'flex', gap: 12, marginBottom: 12, flexWrap: 'wrap' }}>
            <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 12, flex: 1, minWidth: 200 }}>
              Custom feature columns (comma-separated)
              <input value={customFeatures} onChange={e => setCustomFeatures(e.target.value)}
                placeholder="rsi_14, adx_14, volume_ratio"
                style={{ padding: '6px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--input-bg)', color: 'var(--text)' }} />
            </label>
            <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 12, flex: 1, minWidth: 200 }}>
              Custom veto rules (comma-separated)
              <input value={customVetos} onChange={e => setCustomVetos(e.target.value)}
                placeholder="ema200, sentiment, adx_gt_20"
                style={{ padding: '6px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--input-bg)', color: 'var(--text)' }} />
            </label>
          </div>
        )}

        {/* Action buttons */}
        <div style={{ display: 'flex', gap: 8 }}>
          <button onClick={handleRun} disabled={loading || selectedRuleSets.length === 0}
            style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '8px 16px', borderRadius: 6, border: 'none', background: '#6366f1', color: '#fff', cursor: 'pointer', fontWeight: 600 }}>
            {loading ? <span style={{ display: 'inline-block', width: 14, height: 14, border: '2px solid #fff', borderTopColor: 'transparent', borderRadius: '50%', animation: 'spin 0.7s linear infinite' }} />
              : <Zap size={14} />}
            {loading ? 'Running…' : 'Run Lab'}
          </button>
          {loading && (
            <button onClick={handleStop}
              style={{ display: 'flex', alignItems: 'center', gap: 6, padding: '8px 14px', borderRadius: 6, border: '1px solid var(--border)', background: 'var(--card-bg)', color: 'var(--text)', cursor: 'pointer' }}>
              <Square size={14} /> Stop
            </button>
          )}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div style={{ background: '#fee2e2', border: '1px solid #fca5a5', borderRadius: 6, padding: '10px 14px', color: '#991b1b', marginBottom: 16, fontSize: 13 }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      {/* Results */}
      {response && response.results.length > 0 && (
        <>
          {/* Winner banner */}
          {response.winner && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, background: '#f0fdf4', border: '1px solid #86efac', borderRadius: 6, padding: '10px 14px', marginBottom: 16 }}>
              <Trophy size={16} color="#16a34a" />
              <span style={{ fontSize: 13 }}>
                Best OOS Sharpe: <strong>{response.winner}</strong> wins for <strong>{response.symbol}</strong> on {response.date}
              </span>
            </div>
          )}

          {/* Comparison table */}
          <div style={{ background: 'var(--card-bg)', border: '1px solid var(--border)', borderRadius: 8, marginBottom: 20, overflow: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border)', background: 'var(--table-header-bg)' }}>
                  {['Rule Set', 'OOS Sharpe', 'Hit Rate', 'Trades', 'Confidence', 'Slides', 'Avg OOS Acc', 'Features', ''].map(h => (
                    <th key={h} style={{ padding: '10px 12px', textAlign: 'left', fontWeight: 600, whiteSpace: 'nowrap' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {response.results.map(row => {
                  const isWinner = row.rule_set_name === response.winner
                  return (
                    <tr key={row.rule_set_name}
                      style={{ borderBottom: '1px solid var(--border)', background: isWinner ? 'rgba(234,179,8,0.08)' : undefined }}>
                      <td style={{ padding: '10px 12px', fontWeight: isWinner ? 700 : 400 }}>
                        {isWinner && <Trophy size={12} color="#ca8a04" style={{ marginRight: 4, verticalAlign: 'middle' }} />}
                        {row.rule_set_name}
                      </td>
                      <td style={{ padding: '10px 12px', fontWeight: 600, color: row.oos_sharpe > 0.5 ? '#16a34a' : row.oos_sharpe > 0 ? '#d97706' : '#dc2626' }}>
                        {row.oos_sharpe.toFixed(3)}
                      </td>
                      <td style={{ padding: '10px 12px' }}>{(row.hit_rate * 100).toFixed(1)}%</td>
                      <td style={{ padding: '10px 12px' }}>{row.trade_count}</td>
                      <td style={{ padding: '10px 12px' }}>{confidenceBadge(row.confidence)}</td>
                      <td style={{ padding: '10px 12px' }}>{row.n_slides}</td>
                      <td style={{ padding: '10px 12px' }}>{(row.avg_oos_acc * 100).toFixed(1)}%</td>
                      <td style={{ padding: '10px 12px', maxWidth: 220 }}>
                        <div style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.4 }}>
                          <div><strong>Features:</strong> {row.features_used.join(', ')}</div>
                          <div><strong>Vetos:</strong> {row.veto_rules.join(', ')}</div>
                        </div>
                      </td>
                      <td style={{ padding: '10px 12px' }}>
                        <button onClick={() => saveWinner(row)}
                          style={{ fontSize: 11, padding: '4px 8px', borderRadius: 4, border: '1px solid var(--border)', background: 'var(--card-bg)', color: 'var(--text)', cursor: 'pointer', whiteSpace: 'nowrap' }}>
                          Use this
                        </button>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          {/* OOS Sharpe bar chart */}
          <div style={{ background: 'var(--card-bg)', border: '1px solid var(--border)', borderRadius: 8, padding: 16, marginBottom: 20 }}>
            <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 12 }}>OOS Sharpe by Rule Set</div>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={response.results.map(r => ({ name: r.rule_set_name, sharpe: r.oos_sharpe }))}
                margin={{ top: 4, right: 16, left: 0, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v: number) => v.toFixed(4)} />
                <ReferenceLine y={0} stroke="var(--text-muted)" />
                <ReferenceLine y={0.5} stroke="#16a34a" strokeDasharray="4 4" label={{ value: '0.5', fontSize: 10, fill: '#16a34a' }} />
                <Bar dataKey="sharpe">
                  {response.results.map(r => (
                    <Cell key={r.rule_set_name} fill={barColor(r.oos_sharpe)} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </>
      )}

      {/* History panel */}
      {recentRuns.length > 0 && (
        <div style={{ background: 'var(--card-bg)', border: '1px solid var(--border)', borderRadius: 8 }}>
          <button onClick={() => setHistoryOpen(o => !o)}
            style={{ width: '100%', display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '12px 16px', background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text)', fontSize: 13, fontWeight: 600 }}>
            Past Runs for {symbol.toUpperCase()} ({recentRuns.length} shown)
            {historyOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </button>
          {historyOpen && (
            <div style={{ overflow: 'auto', borderTop: '1px solid var(--border)' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid var(--border)', background: 'var(--table-header-bg)' }}>
                    {['Date', 'Rule Set', 'OOS Sharpe', 'Hit Rate', 'Confidence', 'Ran At'].map(h => (
                      <th key={h} style={{ padding: '8px 12px', textAlign: 'left', fontWeight: 600 }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {recentRuns.flatMap(([runId, rows]) =>
                    rows.map((r, i) => (
                      <tr key={`${runId}-${r.rule_set_name}`} style={{ borderBottom: '1px solid var(--border)' }}>
                        <td style={{ padding: '7px 12px' }}>{i === 0 ? r.date : ''}</td>
                        <td style={{ padding: '7px 12px', fontWeight: 600 }}>{r.rule_set_name}</td>
                        <td style={{ padding: '7px 12px', color: r.oos_sharpe > 0.5 ? '#16a34a' : r.oos_sharpe > 0 ? '#d97706' : '#dc2626' }}>
                          {r.oos_sharpe.toFixed(3)}
                        </td>
                        <td style={{ padding: '7px 12px' }}>{(r.hit_rate * 100).toFixed(1)}%</td>
                        <td style={{ padding: '7px 12px' }}>{confidenceBadge(r.confidence)}</td>
                        <td style={{ padding: '7px 12px', color: 'var(--text-muted)' }}>{r.ran_at.slice(0, 16).replace('T', ' ')}</td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      <style>{`@keyframes spin { to { transform: rotate(360deg) } }`}</style>
    </div>
  )
}
