import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'
import { X, Plus, Trash2, Pencil, Check, Ban } from 'lucide-react'

const API_BASE = import.meta.env.VITE_API_BASE ?? '/api'

interface Position {
  id: number
  ticker: string
  quantity: number
  cost_basis: number
  current_price?: number
  market_value?: number
  unrealized_pnl?: number
  unrealized_pnl_pct?: number
  exchange?: string
  target_weight?: number
  notes?: string
  follow_ticker?: string
  manual_price?: number
  category?: string
}

interface Lot {
  id: number
  position_id: number
  purchased_at: string
  price_per_share: number
  quantity: number
  notes?: string
}

interface Props {
  position: Position | null
  onClose: () => void
  onRefresh: () => void
}

const EXCHANGES = ['US', 'Israeli (TASE)', 'Other']

const makeFmt$ = (exchange?: string | null) => (v?: number | null) => {
  if (v == null) return '—'
  const isILS = exchange === 'Israeli (TASE)'
  return new Intl.NumberFormat(isILS ? 'he-IL' : 'en-US', {
    style: 'currency',
    currency: isILS ? 'ILS' : 'USD',
  }).format(v)
}

const fmtPct = (v?: number | null) =>
  v == null ? '—' : `${v > 0 ? '+' : ''}${v.toFixed(2)}%`

const today = () => new Date().toISOString().slice(0, 10)

const emptyLotForm = () => ({ purchased_at: today(), price_per_share: '', quantity: '', notes: '' })

export default function PositionDrawer({ position, onClose, onRefresh }: Props) {
  const [lots, setLots] = useState<Lot[]>([])
  const [loadingLots, setLoadingLots] = useState(false)

  // Lot add form
  const [showAddForm, setShowAddForm] = useState(false)
  const [addForm, setAddForm] = useState(emptyLotForm())
  const [addSaving, setAddSaving] = useState(false)

  // Lot inline edit
  const [editingLotId, setEditingLotId] = useState<number | null>(null)
  const [editForm, setEditForm] = useState(emptyLotForm())

  // Meta fields
  const [meta, setMeta] = useState({ exchange: 'US', notes: '', follow_ticker: '', manual_price: '', category: '' })
  const [metaDirty, setMetaDirty] = useState(false)
  const [metaSaving, setMetaSaving] = useState(false)

  const fetchLots = useCallback(async () => {
    if (!position) return
    setLoadingLots(true)
    try {
      const res = await axios.get(`${API_BASE}/portfolio/positions/${position.id}/lots`)
      setLots(res.data)
    } finally {
      setLoadingLots(false)
    }
  }, [position])

  useEffect(() => {
    if (!position) return
    fetchLots()
    setMeta({
      exchange: position.exchange ?? 'US',
      notes: position.notes ?? '',
      follow_ticker: position.follow_ticker ?? '',
      manual_price: position.manual_price != null ? String(position.manual_price) : '',
      category: position.category ?? '',
    })
    setShowAddForm(false)
    setEditingLotId(null)
    setMetaDirty(false)
  }, [position, fetchLots])

  if (!position) return null

  const fmt$ = makeFmt$(meta.exchange)

  // Weighted avg from lots (or fallback to position cost_basis)
  const avgCost = lots.length > 0
    ? lots.reduce((s, l) => s + l.price_per_share * l.quantity, 0) /
      lots.reduce((s, l) => s + l.quantity, 0)
    : position.cost_basis
  const totalShares = lots.length > 0 ? lots.reduce((s, l) => s + l.quantity, 0) : position.quantity
  const totalCost = avgCost * totalShares

  // P&L
  const mv = position.current_price != null ? position.current_price * totalShares : null
  const pnl = mv != null ? mv - totalCost : null
  const pnlPct = pnl != null && totalCost > 0 ? (pnl / totalCost) * 100 : null

  // ── Lot handlers ──────────────────────────────────────────────────────────
  const handleAddLot = async () => {
    if (!addForm.price_per_share || !addForm.quantity) return
    setAddSaving(true)
    try {
      await axios.post(`${API_BASE}/portfolio/positions/${position.id}/lots`, {
        purchased_at: addForm.purchased_at,
        price_per_share: parseFloat(addForm.price_per_share),
        quantity: parseFloat(addForm.quantity),
        notes: addForm.notes || null,
      })
      setAddForm(emptyLotForm())
      setShowAddForm(false)
      await fetchLots()
      onRefresh()
    } finally {
      setAddSaving(false)
    }
  }

  const startEdit = (lot: Lot) => {
    setEditingLotId(lot.id)
    setEditForm({
      purchased_at: lot.purchased_at,
      price_per_share: String(lot.price_per_share),
      quantity: String(lot.quantity),
      notes: lot.notes ?? '',
    })
  }

  const handleEditLot = async (lot: Lot) => {
    await axios.put(`${API_BASE}/portfolio/positions/${position.id}/lots/${lot.id}`, {
      purchased_at: editForm.purchased_at,
      price_per_share: parseFloat(editForm.price_per_share),
      quantity: parseFloat(editForm.quantity),
      notes: editForm.notes || null,
    })
    setEditingLotId(null)
    await fetchLots()
    onRefresh()
  }

  const handleDeleteLot = async (lotId: number) => {
    if (!confirm('Delete this lot?')) return
    await axios.delete(`${API_BASE}/portfolio/positions/${position.id}/lots/${lotId}`)
    await fetchLots()
    onRefresh()
  }

  // ── Meta save ─────────────────────────────────────────────────────────────
  const handleSaveMeta = async () => {
    setMetaSaving(true)
    try {
      await axios.patch(`${API_BASE}/portfolio/positions/${position.id}`, {
        exchange: meta.exchange || null,
        notes: meta.notes || null,
        follow_ticker: meta.follow_ticker.trim() || null,
        manual_price: meta.manual_price !== '' ? parseFloat(meta.manual_price) : null,
        category: meta.category.trim() || null,
      })
      setMetaDirty(false)
      onRefresh()
    } finally {
      setMetaSaving(false)
    }
  }

  const pnlColor = pnl == null ? '' : pnl >= 0 ? '#16a34a' : '#dc2626'

  return (
    <>
      {/* Backdrop */}
      <div
        onClick={onClose}
        style={{
          position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.35)',
          zIndex: 200, animation: 'fadeIn 0.15s ease',
        }}
      />

      {/* Drawer */}
      <div style={{
        position: 'fixed', top: 0, right: 0, bottom: 0, width: 'min(620px, 100vw)',
        background: '#fff', boxShadow: '-4px 0 24px rgba(0,0,0,0.15)',
        zIndex: 201, display: 'flex', flexDirection: 'column',
        animation: 'slideIn 0.2s ease',
        fontFamily: 'Arial, sans-serif',
      }}>

        {/* Header */}
        <div style={{
          padding: '1.25rem 1.5rem', borderBottom: '1px solid #e5e7eb',
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
          background: '#1e3a5f', color: '#fff',
        }}>
          <div>
            <div style={{ fontSize: '1.4rem', fontWeight: 700 }}>{position.ticker}</div>
            <div style={{ fontSize: '0.85rem', opacity: 0.8, marginTop: 2 }}>
              {meta.exchange} &nbsp;·&nbsp; {totalShares.toFixed(2)} shares
            </div>
          </div>
          <button onClick={onClose} style={{ background: 'none', border: 'none', color: '#fff', cursor: 'pointer', padding: 4 }}>
            <X size={22} />
          </button>
        </div>

        {/* Snapshot bar */}
        <div style={{
          display: 'grid', gridTemplateColumns: 'repeat(4,1fr)',
          gap: '0.75rem', padding: '1rem 1.5rem',
          background: '#f8fafc', borderBottom: '1px solid #e5e7eb',
        }}>
          {[
            ['Avg Cost', fmt$(avgCost)],
            ['Last Price', fmt$(position.current_price)],
            ['Market Value', fmt$(mv)],
            ['Unrealized P&L', pnl != null ? `${fmt$(pnl)} (${fmtPct(pnlPct)})` : '—'],
          ].map(([label, value]) => (
            <div key={label}>
              <div style={{ fontSize: '0.7rem', color: '#6b7280', textTransform: 'uppercase', letterSpacing: '0.05em' }}>{label}</div>
              <div style={{ fontWeight: 600, fontSize: '0.95rem', color: label === 'Unrealized P&L' ? pnlColor : '#111' }}>{value}</div>
            </div>
          ))}
        </div>

        {/* Scrollable body */}
        <div style={{ flex: 1, overflowY: 'auto', padding: '1.25rem 1.5rem' }}>

          {/* ── Buy Lots ── */}
          <div style={{ marginBottom: '2rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
              <h3 style={{ margin: 0, fontSize: '1rem', fontWeight: 700 }}>Buy Lots</h3>
              <button
                onClick={() => { setShowAddForm(v => !v); setAddForm(emptyLotForm()) }}
                style={{
                  display: 'flex', alignItems: 'center', gap: 4, fontSize: '0.82rem',
                  background: '#1e3a5f', color: '#fff', border: 'none', borderRadius: 6,
                  padding: '5px 12px', cursor: 'pointer',
                }}
              >
                <Plus size={14} /> Add lot
              </button>
            </div>

            {/* Add lot form */}
            {showAddForm && (
              <div style={{
                background: '#f0f7ff', border: '1px solid #bfdbfe', borderRadius: 8,
                padding: '0.9rem', marginBottom: '0.75rem',
                display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr auto', gap: '0.5rem', alignItems: 'end',
              }}>
                {[
                  { label: 'Date', key: 'purchased_at', type: 'date' },
                  { label: `Price / Share (${meta.exchange === 'Israeli (TASE)' ? '₪' : '$'})`, key: 'price_per_share', type: 'number' },
                  { label: 'Quantity', key: 'quantity', type: 'number' },
                  { label: 'Notes', key: 'notes', type: 'text' },
                ].map(({ label, key, type }) => (
                  <div key={key}>
                    <label style={{ fontSize: '0.72rem', color: '#374151', display: 'block', marginBottom: 3 }}>{label}</label>
                    <input
                      type={type}
                      step="any"
                      value={(addForm as any)[key]}
                      onChange={e => setAddForm(f => ({ ...f, [key]: e.target.value }))}
                      style={{ width: '100%', padding: '5px 8px', border: '1px solid #d1d5db', borderRadius: 5, fontSize: '0.85rem', boxSizing: 'border-box' }}
                    />
                  </div>
                ))}
                <button
                  onClick={handleAddLot}
                  disabled={addSaving}
                  style={{
                    background: '#16a34a', color: '#fff', border: 'none', borderRadius: 6,
                    padding: '6px 14px', cursor: 'pointer', fontSize: '0.82rem', whiteSpace: 'nowrap',
                  }}
                >
                  {addSaving ? '…' : 'Save'}
                </button>
              </div>
            )}

            {/* Lots table */}
            {loadingLots ? (
              <div style={{ color: '#6b7280', fontSize: '0.875rem' }}>Loading lots…</div>
            ) : lots.length === 0 ? (
              <div style={{
                textAlign: 'center', padding: '1.5rem', border: '1px dashed #d1d5db',
                borderRadius: 8, color: '#9ca3af', fontSize: '0.875rem',
              }}>
                No buy lots yet — add one above to track individual purchases.
              </div>
            ) : (
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.875rem' }}>
                <thead>
                  <tr style={{ background: '#f9fafb' }}>
                    {['Date', 'Price / Share', 'Qty', 'Total Cost', 'Notes', ''].map(h => (
                      <th key={h} style={{ textAlign: 'left', padding: '6px 10px', color: '#6b7280', fontWeight: 600, fontSize: '0.75rem', textTransform: 'uppercase', borderBottom: '1px solid #e5e7eb' }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {lots.map(lot => (
                    <tr key={lot.id} style={{ borderBottom: '1px solid #f3f4f6' }}>
                      {editingLotId === lot.id ? (
                        <>
                          {(['purchased_at', 'price_per_share', 'quantity', 'notes'] as const).map((key, i) => (
                            <td key={key} style={{ padding: '4px 6px' }} colSpan={i === 3 ? 2 : 1}>
                              <input
                                type={key === 'purchased_at' ? 'date' : key === 'notes' ? 'text' : 'number'}
                                step="any"
                                value={(editForm as any)[key]}
                                onChange={e => setEditForm(f => ({ ...f, [key]: e.target.value }))}
                                style={{ width: '100%', padding: '4px 7px', border: '1px solid #93c5fd', borderRadius: 4, fontSize: '0.82rem', boxSizing: 'border-box' }}
                              />
                            </td>
                          ))}
                          <td style={{ padding: '4px 6px', whiteSpace: 'nowrap' }}>
                            <button onClick={() => handleEditLot(lot)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#16a34a', marginRight: 4 }}><Check size={15} /></button>
                            <button onClick={() => setEditingLotId(null)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#9ca3af' }}><Ban size={15} /></button>
                          </td>
                        </>
                      ) : (
                        <>
                          <td style={{ padding: '8px 10px' }}>{lot.purchased_at}</td>
                          <td style={{ padding: '8px 10px' }}>{fmt$(lot.price_per_share)}</td>
                          <td style={{ padding: '8px 10px' }}>{lot.quantity}</td>
                          <td style={{ padding: '8px 10px' }}>{fmt$(lot.price_per_share * lot.quantity)}</td>
                          <td style={{ padding: '8px 10px', color: '#6b7280' }}>{lot.notes ?? '—'}</td>
                          <td style={{ padding: '8px 10px', whiteSpace: 'nowrap' }}>
                            <button onClick={() => startEdit(lot)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#3b82f6', marginRight: 6 }}><Pencil size={14} /></button>
                            <button onClick={() => handleDeleteLot(lot.id)} style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#ef4444' }}><Trash2 size={14} /></button>
                          </td>
                        </>
                      )}
                    </tr>
                  ))}
                  {/* Summary row */}
                  <tr style={{ background: '#f9fafb', fontWeight: 600 }}>
                    <td style={{ padding: '8px 10px', color: '#374151' }}>Avg / Total</td>
                    <td style={{ padding: '8px 10px' }}>{fmt$(avgCost)}</td>
                    <td style={{ padding: '8px 10px' }}>{totalShares.toFixed(2)}</td>
                    <td style={{ padding: '8px 10px' }}>{fmt$(totalCost)}</td>
                    <td colSpan={2} />
                  </tr>
                </tbody>
              </table>
            )}
          </div>

          {/* ── Position Metadata ── */}
          <div style={{ borderTop: '1px solid #e5e7eb', paddingTop: '1.25rem' }}>
            <h3 style={{ margin: '0 0 0.75rem', fontSize: '1rem', fontWeight: 700 }}>Position Details</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.75rem' }}>
              <div>
                <label style={{ fontSize: '0.78rem', color: '#374151', display: 'block', marginBottom: 4, fontWeight: 600 }}>Exchange</label>
                <select
                  value={meta.exchange}
                  onChange={e => { setMeta(m => ({ ...m, exchange: e.target.value })); setMetaDirty(true) }}
                  style={{ width: '100%', padding: '7px 10px', border: '1px solid #d1d5db', borderRadius: 6, fontSize: '0.875rem' }}
                >
                  {EXCHANGES.map(ex => <option key={ex}>{ex}</option>)}
                </select>
              </div>
              <div>
                <label style={{ fontSize: '0.78rem', color: '#374151', display: 'block', marginBottom: 4, fontWeight: 600 }}>Category</label>
                <select
                  value={meta.category}
                  onChange={e => { setMeta(m => ({ ...m, category: e.target.value })); setMetaDirty(true) }}
                  style={{ width: '100%', padding: '7px 10px', border: '1px solid #d1d5db', borderRadius: 6, fontSize: '0.875rem', boxSizing: 'border-box', background: '#fff' }}
                >
                  <option value="">— None —</option>
                  {[
                    'Tech', 'US Tech', 'Semiconductors',
                    'Biotech', 'Medtech', 'Healthcare',
                    'Global Equity', 'US Equity',
                    'Defense',
                    'Nuclear Energy', 'Energy', 'Clean Energy',
                    'Data Centers', 'Real Estate',
                    'Financials', 'Consumer',
                    'Bonds', 'Cash', 'Crypto',
                  ].map(c => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
              <div>
                <label style={{ fontSize: '0.78rem', color: '#374151', display: 'block', marginBottom: 4, fontWeight: 600 }}>
                  Follow Ticker
                </label>
                <input
                  type="text"
                  value={meta.follow_ticker}
                  onChange={e => { setMeta(m => ({ ...m, follow_ticker: e.target.value.toUpperCase() })); setMetaDirty(true) }}
                  placeholder="e.g. SPXS.L or SPY"
                  style={{ width: '100%', padding: '7px 10px', border: '1px solid #d1d5db', borderRadius: 6, fontSize: '0.875rem', boxSizing: 'border-box' }}
                />
                <div style={{ fontSize: '0.7rem', color: '#6b7280', marginTop: 3 }}>
                  TASE positions: price = follow × USD/ILS rate
                </div>
              </div>
              <div>
                <label style={{ fontSize: '0.78rem', color: '#374151', display: 'block', marginBottom: 4, fontWeight: 600 }}>
                  Manual Price {meta.exchange === 'Israeli (TASE)' ? '(₪)' : '($)'}
                </label>
                <input
                  type="number"
                  step="any"
                  min="0"
                  value={meta.manual_price}
                  onChange={e => { setMeta(m => ({ ...m, manual_price: e.target.value })); setMetaDirty(true) }}
                  placeholder="Leave blank for auto-fetch"
                  style={{ width: '100%', padding: '7px 10px', border: `1px solid ${meta.manual_price ? '#f59e0b' : '#d1d5db'}`, borderRadius: 6, fontSize: '0.875rem', boxSizing: 'border-box' }}
                />
                <div style={{ fontSize: '0.7rem', color: meta.manual_price ? '#b45309' : '#6b7280', marginTop: 3 }}>
                  {meta.manual_price ? 'Refresh is skipped — price managed manually' : 'Auto-fetched on refresh'}
                </div>
              </div>
              <div style={{ gridColumn: '1 / -1' }}>
                <label style={{ fontSize: '0.78rem', color: '#374151', display: 'block', marginBottom: 4, fontWeight: 600 }}>Notes</label>
                <textarea
                  rows={3}
                  value={meta.notes}
                  onChange={e => { setMeta(m => ({ ...m, notes: e.target.value })); setMetaDirty(true) }}
                  placeholder="Strategy notes, thesis, reminders…"
                  style={{ width: '100%', padding: '7px 10px', border: '1px solid #d1d5db', borderRadius: 6, fontSize: '0.875rem', resize: 'vertical', boxSizing: 'border-box' }}
                />
              </div>
            </div>

            {metaDirty && (
              <button
                onClick={handleSaveMeta}
                disabled={metaSaving}
                style={{
                  marginTop: '0.75rem', background: '#1e3a5f', color: '#fff',
                  border: 'none', borderRadius: 6, padding: '8px 20px',
                  fontSize: '0.875rem', cursor: 'pointer', fontWeight: 600,
                }}
              >
                {metaSaving ? 'Saving…' : 'Save details'}
              </button>
            )}
          </div>
        </div>
      </div>

      <style>{`
        @keyframes fadeIn { from { opacity: 0 } to { opacity: 1 } }
        @keyframes slideIn { from { transform: translateX(100%) } to { transform: translateX(0) } }
      `}</style>
    </>
  )
}
