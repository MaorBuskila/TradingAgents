import { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import axios from 'axios'
import { RefreshCw, SlidersHorizontal, ChevronDown, ChevronRight } from 'lucide-react'
import { PieChart, Pie, Cell, Tooltip as ReTooltip, ResponsiveContainer } from 'recharts'
import PositionDrawer from './PositionDrawer'
import TickerAutocomplete from './TickerAutocomplete'

const API_BASE = import.meta.env.VITE_API_BASE ?? '/api'

const STORAGE_KEY = 'portfolio_hidden_exchanges'
const STORAGE_KEY_CAT = 'portfolio_hidden_categories'

const PIE_COLORS = [
  '#6366f1', '#059669', '#f59e0b', '#3b82f6', '#ec4899',
  '#14b8a6', '#f97316', '#8b5cf6', '#dc2626', '#64748b',
]
const getCategoryColor = (i: number) => PIE_COLORS[i % PIE_COLORS.length]

const UNCATEGORIZED = 'Uncategorized'

const CATEGORY_SUGGESTIONS = [
  'Tech', 'US Tech', 'Semiconductors',
  'Biotech', 'Medtech', 'Healthcare',
  'Global Equity', 'US Equity',
  'Defense',
  'Nuclear Energy', 'Energy', 'Clean Energy',
  'Data Centers', 'Real Estate',
  'Financials', 'Consumer',
  'Bonds', 'Cash', 'Crypto',
]

const STORAGE_KEY_CURRENCY = 'portfolio_display_currency'
const STORAGE_KEY_CAT_TARGETS = 'portfolio_category_targets'

// Default target % per category — single number, must sum ≤ 100
const DEFAULT_CATEGORY_TARGETS: Record<string, number> = {
  'US Equity':      35,
  'Global Equity':  20,
  'Bonds':          15,
  'Tech':            5,
  'US Tech':         5,
  'Semiconductors':  5,
  'Financials':      5,
  'Healthcare':      5,
  'Energy':          5,
  'Real Estate':     5,
  'Cash':            3,
  'Biotech':         3,
  'Medtech':         2,
  'Nuclear Energy':  2,
  'Clean Energy':    2,
  'Defense':         2,
  'Data Centers':    2,
  'Consumer':        2,
  'Crypto':          2,
}

interface PendingBuyItem {
  ticker: string; name: string | null; category: string | null
  category_label: string | null; current_price: number | null; target_pct: number | null
}

export default function PortfolioPage() {
  const [positions, setPositions] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [newPos, setNewPos] = useState({ ticker: '', quantity: '', cost_basis: '', category: '' })
  const [drawerPos, setDrawerPos] = useState<any | null>(null)
  const [hiddenExchanges, setHiddenExchanges] = useState<Set<string>>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY)
      return saved ? new Set(JSON.parse(saved)) : new Set()
    } catch { return new Set() }
  })
  const [hiddenCategories, setHiddenCategories] = useState<Set<string>>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY_CAT)
      return saved ? new Set(JSON.parse(saved)) : new Set()
    } catch { return new Set() }
  })
  const [showChart, setShowChart] = useState(true)
  // 'USD' = show everything in USD (convert ILS→USD), 'ILS' = show everything in ILS (convert USD→ILS)
  const [displayCurrency, setDisplayCurrency] = useState<'USD' | 'ILS'>(() => {
    return (localStorage.getItem(STORAGE_KEY_CURRENCY) as 'USD' | 'ILS') ?? 'USD'
  })
  const [usdToIls, setUsdToIls] = useState<number>(3.7) // fallback rate
  // category → target % (single number), stored in localStorage
  const [categoryTargets, setCategoryTargets] = useState<Record<string, number>>(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY_CAT_TARGETS)
      if (!saved) return DEFAULT_CATEGORY_TARGETS
      const parsed = JSON.parse(saved)
      // Migrate old range format {min,max} → midpoint number
      const migrated: Record<string, number> = {}
      for (const [k, v] of Object.entries(parsed)) {
        if (typeof v === 'number') migrated[k] = v
        else if (v && typeof v === 'object' && 'min' in v && 'max' in v) {
          migrated[k] = ((v as any).min + (v as any).max) / 2
        }
      }
      return migrated
    } catch { return DEFAULT_CATEGORY_TARGETS }
  })
  const [editingCatTarget, setEditingCatTarget] = useState<string | null>(null)
  const [catTargetDraft, setCatTargetDraft] = useState('')

  const [pendingBuys, setPendingBuys] = useState<PendingBuyItem[]>([])
  const [editingMarkTicker, setEditingMarkTicker] = useState<string | null>(null)
  const [markTargetDraft, setMarkTargetDraft] = useState('')

  useEffect(() => { fetchPositions(); fetchFxRate(); fetchPendingBuys() }, [])

  // Prune stale category targets + hidden categories whenever positions change
  useEffect(() => {
    if (positions.length === 0) return
    const liveCategories = new Set(
      positions.map(p => p.category?.trim() || UNCATEGORIZED)
    )
    // Remove targets for categories no longer in portfolio
    const staleTargetKeys = Object.keys(categoryTargets).filter(k => !liveCategories.has(k))
    if (staleTargetKeys.length > 0) {
      setCategoryTargets(prev => {
        const updated = { ...prev }
        staleTargetKeys.forEach(k => delete updated[k])
        localStorage.setItem(STORAGE_KEY_CAT_TARGETS, JSON.stringify(updated))
        return updated
      })
    }
    // Remove hidden-category filters for categories no longer in portfolio
    setHiddenCategories(prev => {
      const stale = [...prev].filter(k => !liveCategories.has(k))
      if (stale.length === 0) return prev
      const next = new Set(prev)
      stale.forEach(k => next.delete(k))
      localStorage.setItem(STORAGE_KEY_CAT, JSON.stringify([...next]))
      return next
    })
  }, [positions])

  const fetchPositions = async () => {
    try {
      const res = await axios.get(`${API_BASE}/portfolio/positions`)
      setPositions(res.data)
    } catch (err) {
      console.error(err)
    }
  }

  const fetchPendingBuys = async () => {
    try {
      const res = await axios.get<PendingBuyItem[]>(`${API_BASE}/catalog/marks`)
      setPendingBuys(res.data)
    } catch (err) {
      console.error(err)
    }
  }

  const saveMarkTarget = async (ticker: string, val: string) => {
    const num = parseFloat(val)
    const target_pct = isNaN(num) ? null : Math.max(0, Math.min(100, num))
    await axios.patch(`${API_BASE}/catalog/marks/${encodeURIComponent(ticker)}`, { target_pct })
    setEditingMarkTicker(null)
    fetchPendingBuys()
  }

  const removeMark = async (ticker: string) => {
    await axios.post(`${API_BASE}/catalog/mark/${encodeURIComponent(ticker)}`, { mark: false })
    fetchPendingBuys()
  }

  const fetchFxRate = async () => {
    try {
      const res = await axios.get(`${API_BASE}/portfolio/fx-rate`)
      if (res.data?.usd_to_ils) setUsdToIls(res.data.usd_to_ils)
    } catch { /* keep fallback */ }
  }

  const toggleDisplayCurrency = () => {
    setDisplayCurrency(prev => {
      const next = prev === 'USD' ? 'ILS' : 'USD'
      localStorage.setItem(STORAGE_KEY_CURRENCY, next)
      return next
    })
  }

  /** Convert a raw value (native currency) to the display currency. */
  const toDisplay = (val: number | null | undefined, posExchange?: string | null): number | null => {
    if (val == null) return null
    const posIsIls = posExchange === 'Israeli (TASE)'
    if (displayCurrency === 'USD' && posIsIls) return val / usdToIls
    if (displayCurrency === 'ILS' && !posIsIls) return val * usdToIls
    return val
  }

  const handleAdd = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      await axios.post(`${API_BASE}/portfolio/positions`, {
        ticker: newPos.ticker,
        quantity: parseFloat(newPos.quantity),
        cost_basis: parseFloat(newPos.cost_basis),
        category: newPos.category.trim() || undefined,
      })
      setNewPos({ ticker: '', quantity: '', cost_basis: '', category: '' })
      fetchPositions()
    } catch (err) {
      console.error(err)
    }
  }

  const handleDelete = async (id: number) => {
    try {
      await axios.delete(`${API_BASE}/portfolio/positions/${id}`)
      fetchPositions()
    } catch (err) {
      console.error(err)
    }
  }

  const handleRefresh = async () => {
    try {
      setLoading(true)
      await axios.post(`${API_BASE}/portfolio/refresh-prices`)
      await fetchPositions()
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const toggleExchange = (exchange: string) => {
    setHiddenExchanges(prev => {
      const next = new Set(prev)
      if (next.has(exchange)) next.delete(exchange)
      else next.add(exchange)
      localStorage.setItem(STORAGE_KEY, JSON.stringify([...next]))
      return next
    })
  }

  const toggleCategory = (cat: string) => {
    setHiddenCategories(prev => {
      const next = new Set(prev)
      if (next.has(cat)) next.delete(cat)
      else next.add(cat)
      localStorage.setItem(STORAGE_KEY_CAT, JSON.stringify([...next]))
      return next
    })
  }

  const [targetError, setTargetError] = useState<string | null>(null)

  const saveCategoryTarget = (cat: string, val: string) => {
    const num = Math.max(0, Math.min(100, parseFloat(val) || 0))
    const otherSum = Object.entries(categoryTargets)
      .filter(([k]) => k !== cat)
      .reduce((s, [, v]) => s + v, 0)
    if (otherSum + num > 100) {
      const remaining = Math.max(0, 100 - otherSum)
      setTargetError(`Total would exceed 100%. Max allowed for ${cat}: ${remaining.toFixed(1)}%`)
      return
    }
    setTargetError(null)
    const updated = { ...categoryTargets, [cat]: num }
    setCategoryTargets(updated)
    localStorage.setItem(STORAGE_KEY_CAT_TARGETS, JSON.stringify(updated))
    setEditingCatTarget(null)
  }

  const openDrawer = (p: any) => setDrawerPos(p)
  const closeDrawer = () => {
    setDrawerPos(null)
    fetchPositions()
  }

  /** Format a raw native-currency value in the chosen display currency. */
  const fmtCurrency = (val: number | null | undefined, exchange?: string | null) => {
    const converted = toDisplay(val, exchange)
    if (converted == null) return '—'
    return new Intl.NumberFormat(displayCurrency === 'ILS' ? 'he-IL' : 'en-US', {
      style: 'currency',
      currency: displayCurrency,
      maximumFractionDigits: displayCurrency === 'ILS' ? 0 : 2,
    }).format(converted)
  }

  /** Format an already-display-currency value (totals already converted). */
  const fmtTotal = (val: number) =>
    new Intl.NumberFormat(displayCurrency === 'ILS' ? 'he-IL' : 'en-US', {
      style: 'currency',
      currency: displayCurrency,
      maximumFractionDigits: displayCurrency === 'ILS' ? 0 : 2,
    }).format(val)

  const fmtPct = (val: number | null) => {
    if (val === null || val === undefined) return '—'
    return `${val > 0 ? '+' : ''}${val.toFixed(2)}%`
  }

  // Exchange-filtered positions
  const visiblePositions = positions.filter(p => !hiddenExchanges.has(p.exchange ?? 'US'))
  const allExchanges = [...new Set(positions.map(p => p.exchange ?? 'US'))]

  // Category list (from all positions, not just visible)
  const allCategories = [...new Set(
    positions.map(p => p.category?.trim() || UNCATEGORIZED)
  )]

  // Category-filtered positions (applied on top of exchange filter)
  const filteredPositions = visiblePositions.filter(p => {
    const cat = p.category?.trim() || UNCATEGORIZED
    return !hiddenCategories.has(cat)
  })

  // Actual portfolio values by category
  const categoryMap: Record<string, number> = {}
  for (const p of visiblePositions) {
    const cat = p.category?.trim() || UNCATEGORIZED
    categoryMap[cat] = (categoryMap[cat] ?? 0) + (toDisplay(p.market_value, p.exchange) ?? 0)
  }

  const totalValue = filteredPositions.reduce((s, p) => s + (toDisplay(p.market_value, p.exchange) ?? 0), 0)
  const totalPnl   = filteredPositions.reduce((s, p) => s + (toDisplay(p.unrealized_pnl, p.exchange) ?? 0), 0)

  // Pending buys: dollar delta per ticker
  const pendingBuysWithDelta = pendingBuys.map(pb => ({
    ...pb,
    delta_usd: pb.target_pct != null && totalValue > 0
      ? (pb.target_pct / 100) * totalValue : null,
  }))
  const totalPendingTargetPct = pendingBuys.reduce((s, pb) => s + (pb.target_pct ?? 0), 0)

  // Pending categories set (for rendering distinctions in pie legend)
  const pendingCategorySet = new Set(
    pendingBuys
      .filter(pb => pb.target_pct != null)
      .map(pb => pb.category_label || pb.category || 'Pending')
  )

  // Merge pending slices into the same category map so one pie shows everything
  const mergedCategoryMap: Record<string, number> = { ...categoryMap }
  for (const pb of pendingBuys) {
    if (pb.target_pct != null && totalValue > 0) {
      const cat = pb.category_label || pb.category || 'Pending'
      mergedCategoryMap[cat] = (mergedCategoryMap[cat] ?? 0) + (pb.target_pct / 100) * totalValue
    }
  }

  // pieData is now the merged (actual + pending) set — drives the single donut and all derived fields
  const pieData = Object.entries(mergedCategoryMap)
    .filter(([, v]) => v > 0)
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value)
  const totalPieValue = pieData.reduce((s, d) => s + d.value, 0)

  // Actual-only value for per-category pct in legend (pending-only cats show 0 actual)
  const actualCategoryMap: Record<string, number> = categoryMap
  const totalActualValue = Object.values(actualCategoryMap).reduce((s, v) => s + v, 0)

  return (
    <div>
      <header className="page-header toolbar">
        <div>
          <h1 className="page-title">Portfolio</h1>
          <p className="page-desc" style={{ marginTop: '0.5rem' }}>
            Track positions, cost basis, and live marks via yfinance refresh.
          </p>
        </div>
        <button type="button" className="btn" onClick={handleRefresh} disabled={loading}>
          <RefreshCw size={18} className={loading ? 'icon-spin' : undefined} />
          {loading ? 'Refreshing…' : 'Refresh prices'}
        </button>
      </header>

      {/* Exchange filter chips */}
      {allExchanges.length > 1 && (
        <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.5rem', flexWrap: 'wrap', alignItems: 'center' }}>
          <span style={{ fontSize: '0.75rem', color: '#6b7280', fontWeight: 600 }}>Exchange:</span>
          {allExchanges.map(ex => {
            const hidden = hiddenExchanges.has(ex)
            const isIL = ex === 'Israeli (TASE)'
            return (
              <button
                key={ex}
                onClick={() => toggleExchange(ex)}
                style={{
                  padding: '3px 12px', borderRadius: 20, border: '1.5px solid',
                  fontSize: '0.78rem', fontWeight: 600, cursor: 'pointer',
                  borderColor: hidden ? '#d1d5db' : isIL ? '#2563eb' : '#16a34a',
                  background: hidden ? '#f3f4f6' : isIL ? '#dbeafe' : '#dcfce7',
                  color: hidden ? '#9ca3af' : isIL ? '#1d4ed8' : '#15803d',
                  textDecoration: hidden ? 'line-through' : 'none',
                  transition: 'all 0.15s',
                }}
              >
                {ex === 'Israeli (TASE)' ? 'TASE' : ex}
                {hidden ? ' (hidden)' : ''}
              </button>
            )
          })}
        </div>
      )}

      {/* Category filter chips */}
      {allCategories.length > 1 && (
        <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.75rem', flexWrap: 'wrap', alignItems: 'center' }}>
          <span style={{ fontSize: '0.75rem', color: '#6b7280', fontWeight: 600 }}>Category:</span>
          {allCategories.map((cat) => {
            const hidden = hiddenCategories.has(cat)
            const idx = pieData.findIndex(d => d.name === cat)
            const color = getCategoryColor(idx >= 0 ? idx : allCategories.indexOf(cat))
            return (
              <button
                key={cat}
                onClick={() => toggleCategory(cat)}
                style={{
                  padding: '3px 12px', borderRadius: 20, border: '1.5px solid',
                  fontSize: '0.78rem', fontWeight: 600, cursor: 'pointer',
                  borderColor: hidden ? '#d1d5db' : color,
                  background: hidden ? '#f3f4f6' : `${color}18`,
                  color: hidden ? '#9ca3af' : color,
                  textDecoration: hidden ? 'line-through' : 'none',
                  transition: 'all 0.15s',
                }}
              >
                {cat}{hidden ? ' (hidden)' : ''}
              </button>
            )
          })}
        </div>
      )}

      {/* Summary bar */}
      {positions.length > 0 && (
        <div style={{
          display: 'flex', gap: '1.5rem', padding: '0.85rem 1.25rem',
          background: '#1e3a5f', color: '#fff', borderRadius: 10,
          marginBottom: '1.25rem', flexWrap: 'wrap', alignItems: 'center',
        }}>
          <div>
            <div style={{ fontSize: '0.7rem', opacity: 0.7, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Positions</div>
            <div style={{ fontWeight: 700, fontSize: '1.1rem' }}>
              {filteredPositions.length}
              {filteredPositions.length !== positions.length && (
                <span style={{ fontSize: '0.75rem', opacity: 0.6, marginLeft: 4 }}>/ {positions.length}</span>
              )}
            </div>
          </div>
          <div>
            <div style={{ fontSize: '0.7rem', opacity: 0.7, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Total Market Value</div>
            <div style={{ fontWeight: 700, fontSize: '1.1rem' }}>{fmtTotal(totalValue)}</div>
          </div>
          <div>
            <div style={{ fontSize: '0.7rem', opacity: 0.7, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Total Unrealized P&L</div>
            <div style={{ fontWeight: 700, fontSize: '1.1rem', color: totalPnl >= 0 ? '#86efac' : '#fca5a5' }}>
              {fmtTotal(totalPnl)}
            </div>
          </div>
          {/* Currency toggle */}
          <div style={{ marginLeft: 'auto', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
            <div style={{ fontSize: '0.65rem', opacity: 0.7, textTransform: 'uppercase', letterSpacing: '0.05em' }}>Display in</div>
            <button
              type="button"
              onClick={toggleDisplayCurrency}
              style={{
                display: 'flex', alignItems: 'center', gap: 0,
                background: 'rgba(255,255,255,0.12)', border: '1.5px solid rgba(255,255,255,0.25)',
                borderRadius: 20, padding: '3px 5px', cursor: 'pointer', userSelect: 'none',
              }}
            >
              {(['USD', 'ILS'] as const).map(cur => (
                <span
                  key={cur}
                  style={{
                    padding: '3px 12px', borderRadius: 16, fontWeight: 700, fontSize: '0.8rem',
                    background: displayCurrency === cur ? '#fff' : 'transparent',
                    color: displayCurrency === cur ? '#1e3a5f' : 'rgba(255,255,255,0.65)',
                    transition: 'all 0.15s',
                  }}
                >
                  {cur}
                </span>
              ))}
            </button>
            <div style={{ fontSize: '0.6rem', opacity: 0.5 }}>1 USD = ₪{usdToIls.toFixed(2)}</div>
          </div>
        </div>
      )}

      {/* Allocation pie chart */}
      {pieData.length > 0 && (
        <div className="card" style={{ marginBottom: '1.25rem' }}>
          <div className="card-header" style={{
            marginBottom: showChart ? undefined : 0,
            borderBottom: showChart ? undefined : 'none',
            paddingBottom: showChart ? undefined : 0,
          }}>
            <h2 className="card-title">Allocation by Category</h2>
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={() => setShowChart(v => !v)}
              style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}
            >
              {showChart ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
              {showChart ? 'Hide' : 'Show'}
            </button>
          </div>

          {showChart && (
            <div style={{ display: 'flex', gap: '2rem', alignItems: 'center', flexWrap: 'wrap' }}>
              {/* Donut chart */}
              <div style={{ flex: '0 0 240px', height: 240 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={pieData}
                      cx="50%"
                      cy="50%"
                      innerRadius={68}
                      outerRadius={105}
                      paddingAngle={2}
                      dataKey="value"
                    >
                      {pieData.map((entry, index) => {
                        const isPending = pendingCategorySet.has(entry.name) && !categoryMap[entry.name]
                        return (
                          <Cell
                            key={index}
                            fill={getCategoryColor(index)}
                            fillOpacity={isPending ? 0.4 : 1}
                            stroke={isPending ? getCategoryColor(index) : 'none'}
                            strokeWidth={isPending ? 1.5 : 0}
                            strokeDasharray={isPending ? '4 2' : undefined}
                          />
                        )
                      })}
                    </Pie>
                    <ReTooltip
                      formatter={(value: number, name: string) => {
                        const isPending = pendingCategorySet.has(name) && !categoryMap[name]
                        const label = isPending ? `${name} (pending)` : name
                        return [
                          `${fmtTotal(value)} (${totalPieValue > 0 ? ((value / totalPieValue) * 100).toFixed(1) : '0'}%)`,
                          label,
                        ]
                      }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              {/* Legend table with single-number targets */}
              <div style={{ flex: 1, minWidth: 260 }}>
                {targetError && (
                  <div style={{ fontSize: '0.75rem', color: '#dc2626', background: '#fef2f2', border: '1px solid #fecaca', borderRadius: 6, padding: '4px 10px', marginBottom: 8 }}>
                    ⚠️ {targetError}
                  </div>
                )}
                {/* Total target budget indicator */}
                {(() => {
                  const totalTarget = Object.values(categoryTargets).reduce((s, v) => s + v, 0)
                  const totalCommitted = totalTarget + totalPendingTargetPct
                  const remaining = 100 - totalCommitted
                  return (
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '0 0.5rem 6px', fontSize: '0.72rem', color: remaining < 0 ? '#dc2626' : remaining === 0 ? '#16a34a' : '#6b7280' }}>
                      <span>
                        Targets: <strong>{totalTarget.toFixed(1)}%</strong>
                        {totalPendingTargetPct > 0 && (
                          <span style={{ color: 'var(--accent, #4f8ef7)', marginLeft: 4 }}>
                            + {totalPendingTargetPct.toFixed(1)}% pending
                          </span>
                        )}
                      </span>
                      <span style={{ marginLeft: 'auto' }}>
                        {remaining > 0 ? `${remaining.toFixed(1)}% unallocated` : remaining === 0 ? '✓ fully allocated' : '⚠️ over 100%'}
                      </span>
                    </div>
                  )
                })()}
                <div style={{ display: 'flex', fontSize: '0.7rem', color: '#9ca3af', fontWeight: 600, padding: '0 0.5rem 4px', gap: '0.5rem' }}>
                  <span style={{ flex: 1 }}>Category</span>
                  <span style={{ minWidth: 44, textAlign: 'right' }}>Actual</span>
                  <span style={{ minWidth: 64, textAlign: 'right' }}>Target</span>
                  <span style={{ minWidth: 80, textAlign: 'right' }}>Value</span>
                </div>
                {pieData.map((entry, index) => {
                  // isPendingOnly: category exists only from pending buys, not in actual portfolio
                  const isPendingOnly = pendingCategorySet.has(entry.name) && !categoryMap[entry.name]
                  // actualValue for this category (0 if pending-only)
                  const actualValue = actualCategoryMap[entry.name] ?? 0
                  // "Actual" % is actual value vs total actual portfolio (not merged), so it stays meaningful
                  const pct = totalActualValue > 0 ? (actualValue / totalActualValue) * 100 : 0
                  const isHidden = hiddenCategories.has(entry.name)
                  const target = categoryTargets[entry.name] ?? null
                  const diff = target != null ? pct - target : null
                  const isOver = !isPendingOnly && diff != null && diff > 2
                  const isUnder = !isPendingOnly && diff != null && diff < -2
                  const isEditing = editingCatTarget === entry.name
                  return (
                    <div
                      key={entry.name}
                      style={{
                        display: 'flex', alignItems: 'center', gap: '0.5rem',
                        padding: '0.3rem 0.5rem', borderRadius: 6,
                        opacity: isHidden ? 0.4 : 1,
                        transition: 'opacity 0.15s',
                        marginBottom: 2,
                        background: isPendingOnly
                          ? 'rgba(79,142,247,0.05)'
                          : isOver ? 'rgba(220,38,38,0.04)' : isUnder ? 'rgba(37,99,235,0.04)' : target != null ? 'rgba(22,163,74,0.04)' : 'transparent',
                      }}
                    >
                      <div
                        onClick={() => !isPendingOnly && toggleCategory(entry.name)}
                        style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flex: 1, cursor: isPendingOnly ? 'default' : 'pointer' }}
                      >
                        {/* Dot: solid for actual, dashed circle for pending-only */}
                        {isPendingOnly ? (
                          <div style={{
                            width: 11, height: 11, borderRadius: '50%', flexShrink: 0,
                            border: `2px dashed ${getCategoryColor(index)}`,
                            background: 'transparent',
                          }} />
                        ) : (
                          <div style={{ width: 11, height: 11, borderRadius: '50%', background: getCategoryColor(index), flexShrink: 0 }} />
                        )}
                        <span style={{ fontSize: '0.85rem', fontWeight: 500, color: isPendingOnly ? 'var(--text-muted, #6b7280)' : 'inherit' }}>
                          {entry.name}
                          {isPendingOnly && <span style={{ fontSize: '0.7rem', marginLeft: 4, opacity: 0.7 }}>pending</span>}
                        </span>
                      </div>
                      {/* Actual % — shows 0% for pending-only rows */}
                      <span style={{
                        fontSize: '0.8rem', minWidth: 44, textAlign: 'right', fontWeight: 600,
                        color: isPendingOnly ? '#9ca3af' : isOver ? '#dc2626' : isUnder ? '#2563eb' : target != null ? '#16a34a' : '#6b7280',
                      }}>
                        {!isPendingOnly && (isOver ? '▲ ' : isUnder ? '▼ ' : target != null ? '✓ ' : '')}
                        {isPendingOnly ? '—' : `${pct.toFixed(1)}%`}
                      </span>
                      {/* Target — click to edit (allowed for pending-only too, so user can pre-set) */}
                      <div style={{ minWidth: 64, textAlign: 'right' }}>
                        {isEditing ? (
                          <span style={{ display: 'inline-flex', alignItems: 'center', gap: 2 }}>
                            <input
                              type="number" min="0" max="100" step="1"
                              value={catTargetDraft}
                              onChange={e => { setTargetError(null); setCatTargetDraft(e.target.value) }}
                              onBlur={() => saveCategoryTarget(entry.name, catTargetDraft)}
                              onKeyDown={e => {
                                if (e.key === 'Enter') saveCategoryTarget(entry.name, catTargetDraft)
                                if (e.key === 'Escape') { setEditingCatTarget(null); setTargetError(null) }
                              }}
                              autoFocus
                              style={{ width: 44, padding: '2px 4px', fontSize: '0.78rem', border: '1px solid #6366f1', borderRadius: 4, textAlign: 'right' }}
                            />
                            <span style={{ fontSize: '0.7rem', color: '#9ca3af' }}>%</span>
                          </span>
                        ) : (
                          <span
                            onClick={() => { setEditingCatTarget(entry.name); setCatTargetDraft(target != null ? String(target) : '') }}
                            title="Click to set target %"
                            style={{
                              fontSize: '0.78rem', cursor: 'pointer', borderRadius: 4,
                              padding: '2px 6px', fontWeight: target != null ? 700 : 400,
                              background: target != null ? 'rgba(99,102,241,0.1)' : '#f3f4f6',
                              color: target != null ? '#4f46e5' : '#9ca3af',
                            }}
                          >
                            {target != null ? `${target}%` : '+ target'}
                            {!isPendingOnly && diff != null && (
                              <span style={{ marginLeft: 4, fontSize: '0.68rem', color: isOver ? '#dc2626' : isUnder ? '#2563eb' : '#16a34a' }}>
                                {diff > 0 ? `+${diff.toFixed(1)}` : diff.toFixed(1)}
                              </span>
                            )}
                          </span>
                        )}
                      </div>
                      <span style={{ fontSize: '0.8rem', color: isPendingOnly ? '#9ca3af' : '#374151', minWidth: 80, textAlign: 'right', fontStyle: isPendingOnly ? 'italic' : 'normal' }}>
                        {isPendingOnly ? `+${fmtTotal(entry.value)}` : fmtTotal(entry.value)}
                      </span>
                    </div>
                  )
                })}
              </div>
            </div>

          )}
        </div>
      )}

      {/* Rebalance Advisor — shown outside chart card so always visible */}
      {pieData.length > 0 && (() => {
        const rebalanceRows: { cat: string; pct: number; target: number; diffPct: number; diffVal: number; status: 'over' | 'under' | 'ok' }[] = []

        // Only show categories that actually exist in the portfolio right now
        const allTargetCats = pieData.map(d => d.name)

        allTargetCats.forEach(cat => {
          const target = categoryTargets[cat]
          if (target == null) return
          const pieEntry = pieData.find(d => d.name === cat)
          const pct = pieEntry && totalPieValue > 0 ? (pieEntry.value / totalPieValue) * 100 : 0
          const diffPct = pct - target
          const diffVal = (diffPct / 100) * totalPieValue
          const status: 'over' | 'under' | 'ok' = Math.abs(diffPct) <= 2 ? 'ok' : diffPct > 0 ? 'over' : 'under'
          rebalanceRows.push({ cat, pct, target, diffPct, diffVal, status })
        })

        const unders = rebalanceRows.filter(r => r.status === 'under').sort((a, b) => a.diffPct - b.diffPct)
        const overs  = rebalanceRows.filter(r => r.status === 'over').sort((a, b) => b.diffPct - a.diffPct)
        const oks    = rebalanceRows.filter(r => r.status === 'ok')

        if (rebalanceRows.length === 0) return null

        return (
          <div className="card" style={{ marginBottom: '1.25rem' }}>
            <div className="card-header">
              <h2 className="card-title">Rebalance Advisor</h2>
              <span style={{ fontSize: '0.8rem', color: '#6b7280' }}>
                vs. midpoint targets · total {fmtTotal(totalPieValue)}
              </span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '0.75rem' }}>

              {unders.length > 0 && (
                <div style={{ background: '#eff6ff', borderRadius: 10, padding: '0.85rem 1rem', border: '1px solid #bfdbfe' }}>
                  <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#1d4ed8', marginBottom: '0.6rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                    📈 Buy / Increase
                  </div>
                  {unders.map(r => (
                    <div key={r.cat} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 7, gap: 8 }}>
                      <span style={{ fontSize: '0.85rem', fontWeight: 600, color: '#1e40af', flex: 1 }}>{r.cat}</span>
                      <span style={{ fontSize: '0.78rem', color: '#374151', whiteSpace: 'nowrap' }}>
                        {r.pct.toFixed(1)}% → {r.target}%
                      </span>
                      <span style={{ fontSize: '0.85rem', fontWeight: 700, color: '#1d4ed8', whiteSpace: 'nowrap', minWidth: 70, textAlign: 'right' }}>
                        +{fmtTotal(Math.abs(r.diffVal))}
                      </span>
                    </div>
                  ))}
                </div>
              )}

              {overs.length > 0 && (
                <div style={{ background: '#fef2f2', borderRadius: 10, padding: '0.85rem 1rem', border: '1px solid #fecaca' }}>
                  <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#dc2626', marginBottom: '0.6rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                    ✂️ Trim / Reduce
                  </div>
                  {overs.map(r => (
                    <div key={r.cat} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 7, gap: 8 }}>
                      <span style={{ fontSize: '0.85rem', fontWeight: 600, color: '#991b1b', flex: 1 }}>{r.cat}</span>
                      <span style={{ fontSize: '0.78rem', color: '#374151', whiteSpace: 'nowrap' }}>
                        {r.pct.toFixed(1)}% → {r.target}%
                      </span>
                      <span style={{ fontSize: '0.85rem', fontWeight: 700, color: '#dc2626', whiteSpace: 'nowrap', minWidth: 70, textAlign: 'right' }}>
                        -{fmtTotal(Math.abs(r.diffVal))}
                      </span>
                    </div>
                  ))}
                </div>
              )}

              {oks.length > 0 && (
                <div style={{ background: '#f0fdf4', borderRadius: 10, padding: '0.85rem 1rem', border: '1px solid #bbf7d0' }}>
                  <div style={{ fontSize: '0.75rem', fontWeight: 700, color: '#16a34a', marginBottom: '0.6rem', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                    ✅ On target
                  </div>
                  {oks.map(r => (
                    <div key={r.cat} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 5, gap: 8 }}>
                      <span style={{ fontSize: '0.85rem', fontWeight: 600, color: '#15803d', flex: 1 }}>{r.cat}</span>
                      <span style={{ fontSize: '0.78rem', color: '#374151' }}>
                        {r.pct.toFixed(1)}% · target {r.target}%
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
            <p style={{ marginTop: '0.75rem', marginBottom: 0, fontSize: '0.72rem', color: '#9ca3af' }}>
              * Dollar amounts = approximate cash to move. Not financial advice. Click any target % in the Allocation legend above to edit.
            </p>
          </div>
        )
      })()}

      {/* ── Pending Buys ── */}
      {pendingBuys.length > 0 && (
        <div className="card" style={{ marginBottom: '1.5rem' }}>
          <div className="card-header">
            <h2 className="card-title">Pending Buys</h2>
            {totalPendingTargetPct > 0 && (
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                {totalPendingTargetPct.toFixed(1)}% of portfolio targeted
              </span>
            )}
          </div>

          {/* Column headers */}
          <div style={{
            display: 'grid',
            gridTemplateColumns: '90px 1fr 180px 90px 90px 110px 32px',
            gap: '0 1rem',
            padding: '0.4rem 0.75rem',
            fontSize: '0.72rem',
            fontWeight: 700,
            color: 'var(--text-muted)',
            textTransform: 'uppercase',
            letterSpacing: '0.05em',
            borderBottom: '1px solid var(--border)',
            marginBottom: '0.25rem',
          }}>
            <span>Ticker</span>
            <span>Name</span>
            <span>Category</span>
            <span style={{ textAlign: 'right' }}>Price</span>
            <span style={{ textAlign: 'right' }}>Target %</span>
            <span style={{ textAlign: 'right' }}>$ to Buy</span>
            <span />
          </div>

          {/* Rows */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.15rem' }}>
            {pendingBuysWithDelta.map(pb => {
              const catIdx = pieData.findIndex(d => d.name === (pb.category_label || pb.category || 'Pending'))
              const catColor = getCategoryColor(catIdx >= 0 ? catIdx : 0)
              return (
                <div
                  key={pb.ticker}
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '90px 1fr 180px 90px 90px 110px 32px',
                    gap: '0 1rem',
                    alignItems: 'center',
                    padding: '0.55rem 0.75rem',
                    borderRadius: 8,
                    background: 'var(--bg-subtle)',
                    transition: 'background 0.12s',
                  }}
                  onMouseEnter={e => (e.currentTarget as HTMLDivElement).style.background = 'var(--bg-elevated)'}
                  onMouseLeave={e => (e.currentTarget as HTMLDivElement).style.background = 'var(--bg-subtle)'}
                >
                  {/* Ticker */}
                  <Link
                    to="/watchlist"
                    style={{
                      fontWeight: 700, fontSize: '0.95rem',
                      color: 'var(--accent, #4f8ef7)',
                      textDecoration: 'none',
                    }}
                    onMouseEnter={e => ((e.currentTarget as HTMLAnchorElement).style.textDecoration = 'underline')}
                    onMouseLeave={e => ((e.currentTarget as HTMLAnchorElement).style.textDecoration = 'none')}
                  >
                    {pb.ticker}
                  </Link>

                  {/* Name */}
                  <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {pb.name ?? '—'}
                  </span>

                  {/* Category pill */}
                  <span>
                    {(pb.category_label || pb.category) ? (
                      <span style={{
                        display: 'inline-block',
                        fontSize: '0.72rem', fontWeight: 700,
                        padding: '3px 10px', borderRadius: 20,
                        background: `${catColor}1a`, color: catColor,
                        border: `1px solid ${catColor}55`,
                        whiteSpace: 'nowrap', letterSpacing: '0.03em',
                        textTransform: 'uppercase',
                      }}>
                        {pb.category_label || pb.category}
                      </span>
                    ) : (
                      <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>—</span>
                    )}
                  </span>

                  {/* Price */}
                  <span style={{ fontSize: '0.88rem', color: 'var(--text)', textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>
                    {pb.current_price != null ? `$${pb.current_price.toFixed(2)}` : '—'}
                  </span>

                  {/* Target % — editable */}
                  <div style={{ textAlign: 'right' }}>
                    {editingMarkTicker === pb.ticker ? (
                      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 2, justifyContent: 'flex-end' }}>
                        <input
                          type="number" step="0.1" min="0" max="100"
                          value={markTargetDraft}
                          autoFocus
                          style={{
                            width: 56, padding: '3px 6px',
                            fontSize: '0.82rem', textAlign: 'right',
                            border: '1.5px solid var(--accent, #6366f1)',
                            borderRadius: 6, background: 'var(--bg-base)',
                            color: 'var(--text)',
                          }}
                          onChange={e => setMarkTargetDraft(e.target.value)}
                          onBlur={() => saveMarkTarget(pb.ticker, markTargetDraft)}
                          onKeyDown={e => {
                            if (e.key === 'Enter') saveMarkTarget(pb.ticker, markTargetDraft)
                            if (e.key === 'Escape') setEditingMarkTicker(null)
                          }}
                        />
                        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>%</span>
                      </span>
                    ) : (
                      <span
                        onClick={() => { setEditingMarkTicker(pb.ticker); setMarkTargetDraft(pb.target_pct?.toString() ?? '') }}
                        title="Click to edit target %"
                        style={{
                          cursor: 'pointer',
                          display: 'inline-block',
                          fontSize: '0.85rem', fontWeight: pb.target_pct != null ? 700 : 400,
                          padding: '3px 8px', borderRadius: 6,
                          background: pb.target_pct != null ? 'rgba(99,102,241,0.1)' : 'var(--bg-base)',
                          color: pb.target_pct != null ? 'var(--accent, #4f46e5)' : 'var(--text-muted)',
                          border: '1px solid',
                          borderColor: pb.target_pct != null ? 'rgba(99,102,241,0.3)' : 'var(--border)',
                        }}
                      >
                        {pb.target_pct != null ? `${pb.target_pct.toFixed(1)}%` : '+ set'}
                      </span>
                    )}
                  </div>

                  {/* $ to Buy */}
                  <div style={{ textAlign: 'right' }}>
                    {pb.delta_usd != null ? (
                      <span style={{
                        fontSize: '0.95rem', fontWeight: 700,
                        color: 'var(--text)',
                        fontVariantNumeric: 'tabular-nums',
                      }}>
                        ${pb.delta_usd.toFixed(0)}
                      </span>
                    ) : (
                      <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>—</span>
                    )}
                  </div>

                  {/* Remove */}
                  <button
                    type="button"
                    onClick={() => removeMark(pb.ticker)}
                    title="Remove from pending buys"
                    style={{
                      background: 'none', border: 'none', cursor: 'pointer',
                      color: 'var(--text-muted)', fontSize: '1rem', lineHeight: 1,
                      padding: '2px 4px', borderRadius: 4,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      transition: 'color 0.12s',
                    }}
                    onMouseEnter={e => ((e.currentTarget as HTMLButtonElement).style.color = 'var(--danger, #dc2626)')}
                    onMouseLeave={e => ((e.currentTarget as HTMLButtonElement).style.color = 'var(--text-muted)')}
                  >×</button>
                </div>
              )
            })}
          </div>

          <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.85rem', marginBottom: 0, textAlign: 'right' }}>
            $ to Buy = target % × current portfolio value. Not financial advice.
          </p>
        </div>
      )}

      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Add position</h2>
        </div>
        <form onSubmit={handleAdd} className="form-row-inline" style={{ marginBottom: '1.5rem' }}>
          <div className="form-field-inline" style={{ flex: '1 1 160px', maxWidth: '220px' }}>
            <label htmlFor="np-ticker">Ticker</label>
            <TickerAutocomplete
              id="np-ticker"
              value={newPos.ticker}
              onChange={(val: string) => setNewPos({ ...newPos, ticker: val })}
              required
            />
          </div>
          <div className="form-field-inline" style={{ flex: '1 1 100px', maxWidth: '140px' }}>
            <label htmlFor="np-qty">Quantity</label>
            <input
              id="np-qty" type="number" step="any" className="form-control"
              value={newPos.quantity}
              onChange={(e) => setNewPos({ ...newPos, quantity: e.target.value })}
              required
            />
          </div>
          <div className="form-field-inline" style={{ flex: '1 1 120px', maxWidth: '160px' }}>
            <label htmlFor="np-cost">Cost basis ($)</label>
            <input
              id="np-cost" type="number" step="any" className="form-control"
              value={newPos.cost_basis}
              onChange={(e) => setNewPos({ ...newPos, cost_basis: e.target.value })}
              required
            />
          </div>
          <div className="form-field-inline" style={{ flex: '1 1 120px', maxWidth: '160px' }}>
            <label htmlFor="np-cat">Category</label>
            <select
              id="np-cat"
              className="form-control"
              value={newPos.category}
              onChange={(e) => setNewPos({ ...newPos, category: e.target.value })}
            >
              <option value="">— None —</option>
              {CATEGORY_SUGGESTIONS.map(c => <option key={c} value={c}>{c}</option>)}
            </select>
          </div>
          <button type="submit" className="btn" style={{ alignSelf: 'flex-end' }}>Add</button>
        </form>

        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th style={{ width: '100px' }}></th>
                <th>Ticker</th>
                <th>Exchange</th>
                <th>Category</th>
                <th>Qty</th>
                <th>Avg Cost</th>
                <th>Last Price</th>
                <th>Market Value</th>
                <th>Unrealized P&amp;L</th>
              </tr>
            </thead>
            <tbody>
              {filteredPositions.map((p) => {
                const catIdx = pieData.findIndex(d => d.name === (p.category?.trim() || UNCATEGORIZED))
                const catColor = getCategoryColor(catIdx >= 0 ? catIdx : allCategories.indexOf(p.category?.trim() || UNCATEGORIZED))
                return (
                  <tr key={p.id}>
                    <td style={{ whiteSpace: 'nowrap' }}>
                      <button
                        type="button"
                        className="btn btn-sm"
                        onClick={() => openDrawer(p)}
                        style={{ marginRight: 4, display: 'inline-flex', alignItems: 'center', gap: 3 }}
                      >
                        <SlidersHorizontal size={12} /> Edit
                      </button>
                      <button
                        type="button"
                        className="btn btn-danger btn-sm"
                        onClick={() => handleDelete(p.id)}
                      >
                        ✕
                      </button>
                    </td>
                    <td>
                      <Link
                        to={`/ticker/${encodeURIComponent(p.ticker)}`}
                        style={{ fontWeight: 700, textDecoration: 'none', color: 'var(--accent, #4f8ef7)' }}
                        onMouseEnter={e => ((e.currentTarget as HTMLAnchorElement).style.textDecoration = 'underline')}
                        onMouseLeave={e => ((e.currentTarget as HTMLAnchorElement).style.textDecoration = 'none')}
                      >
                        {p.ticker}
                      </Link>
                      {p.has_lots && (
                        <span style={{
                          marginLeft: 6, fontSize: '0.68rem', background: '#dbeafe',
                          color: '#1d4ed8', borderRadius: 4, padding: '1px 5px',
                        }}>lots</span>
                      )}
                    </td>
                    <td style={{ color: '#6b7280', fontSize: '0.85rem' }}>{p.exchange ?? '—'}</td>
                    <td>
                      {p.category ? (
                        <span style={{
                          fontSize: '0.75rem', fontWeight: 600, padding: '2px 8px',
                          borderRadius: 12, background: `${catColor}18`, color: catColor,
                          whiteSpace: 'nowrap',
                        }}>
                          {p.category}
                        </span>
                      ) : (
                        <span style={{ color: '#9ca3af', fontSize: '0.8rem' }}>—</span>
                      )}
                    </td>
                    <td>{p.quantity}</td>
                    <td>{fmtCurrency(p.cost_basis, p.exchange)}</td>
                    <td>
                      {fmtCurrency(p.current_price, p.exchange)}
                      {p.manual_price != null && (
                        <span title="Manual price" style={{
                          marginLeft: 5, fontSize: '0.65rem', background: '#fef3c7',
                          color: '#92400e', borderRadius: 4, padding: '1px 5px', verticalAlign: 'middle',
                        }}>manual</span>
                      )}
                    </td>
                    <td>{fmtCurrency(p.market_value, p.exchange)}</td>
                    <td>
                      <span className={p.unrealized_pnl > 0 ? 'text-success' : p.unrealized_pnl < 0 ? 'text-danger' : ''}>
                        {fmtCurrency(p.unrealized_pnl, p.exchange)}
                      </span>{' '}
                      <span className="muted" style={{ fontSize: '0.8125rem' }}>
                        ({fmtPct(p.unrealized_pnl_pct)})
                      </span>
                    </td>
                  </tr>
                )
              })}
              {filteredPositions.length === 0 && (
                <tr>
                  <td colSpan={9} className="empty-state" style={{ textAlign: 'center', padding: '2.5rem 1rem' }}>
                    {positions.length === 0
                      ? <><strong>No positions</strong><br />Add a row above to get started.</>
                      : <><strong>All positions are hidden</strong><br />Click the filter chips above to show them.</>
                    }
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      <PositionDrawer
        position={drawerPos}
        onClose={closeDrawer}
        onRefresh={fetchPositions}
      />
    </div>
  )
}
