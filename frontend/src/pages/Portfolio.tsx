import { useState, useEffect } from 'react'
import axios from 'axios'
import { RefreshCw } from 'lucide-react'

const API_BASE = import.meta.env.VITE_API_BASE ?? '/api'

export default function Portfolio() {
  const [positions, setPositions] = useState<any[]>([])
  const [loading, setLoading] = useState(false)
  const [newPos, setNewPos] = useState({ ticker: '', quantity: '', cost_basis: '' })

  useEffect(() => {
    fetchPositions()
  }, [])

  const fetchPositions = async () => {
    try {
      const res = await axios.get(`${API_BASE}/portfolio/positions`)
      setPositions(res.data)
    } catch (err) {
      console.error(err)
    }
  }

  const handleAdd = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      await axios.post(`${API_BASE}/portfolio/positions`, {
        ticker: newPos.ticker,
        quantity: parseFloat(newPos.quantity),
        cost_basis: parseFloat(newPos.cost_basis),
      })
      setNewPos({ ticker: '', quantity: '', cost_basis: '' })
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

  const formatCurrency = (val: number | null) => {
    if (val === null || val === undefined) return '—'
    return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(val)
  }

  const formatPct = (val: number | null) => {
    if (val === null || val === undefined) return '—'
    return `${val > 0 ? '+' : ''}${val.toFixed(2)}%`
  }

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

      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Add position</h2>
        </div>
        <form onSubmit={handleAdd} className="form-row-inline" style={{ marginBottom: '1.5rem' }}>
          <div className="form-field-inline" style={{ flex: '1 1 120px', maxWidth: '160px' }}>
            <label htmlFor="np-ticker">Ticker</label>
            <input
              id="np-ticker"
              type="text"
              className="form-control"
              value={newPos.ticker}
              onChange={(e) => setNewPos({ ...newPos, ticker: e.target.value })}
              required
            />
          </div>
          <div className="form-field-inline" style={{ flex: '1 1 100px', maxWidth: '140px' }}>
            <label htmlFor="np-qty">Quantity</label>
            <input
              id="np-qty"
              type="number"
              step="any"
              className="form-control"
              value={newPos.quantity}
              onChange={(e) => setNewPos({ ...newPos, quantity: e.target.value })}
              required
            />
          </div>
          <div className="form-field-inline" style={{ flex: '1 1 120px', maxWidth: '160px' }}>
            <label htmlFor="np-cost">Cost basis ($)</label>
            <input
              id="np-cost"
              type="number"
              step="any"
              className="form-control"
              value={newPos.cost_basis}
              onChange={(e) => setNewPos({ ...newPos, cost_basis: e.target.value })}
              required
            />
          </div>
          <button type="submit" className="btn" style={{ alignSelf: 'flex-end' }}>
            Add
          </button>
        </form>

        <div className="table-wrap">
          <table className="data-table">
            <thead>
              <tr>
                <th>Ticker</th>
                <th>Qty</th>
                <th>Cost basis</th>
                <th>Last price</th>
                <th>Market value</th>
                <th>Unrealized P&amp;L</th>
                <th style={{ width: '100px' }}></th>
              </tr>
            </thead>
            <tbody>
              {positions.map((p) => (
                <tr key={p.id}>
                  <td>
                    <strong>{p.ticker}</strong>
                  </td>
                  <td>{p.quantity}</td>
                  <td>{formatCurrency(p.cost_basis)}</td>
                  <td>{formatCurrency(p.current_price)}</td>
                  <td>{formatCurrency(p.market_value)}</td>
                  <td>
                    <span
                      className={
                        p.unrealized_pnl > 0
                          ? 'text-success'
                          : p.unrealized_pnl < 0
                            ? 'text-danger'
                            : ''
                      }
                    >
                      {formatCurrency(p.unrealized_pnl)}
                    </span>{' '}
                    <span className="muted" style={{ fontSize: '0.8125rem' }}>
                      ({formatPct(p.unrealized_pnl_pct)})
                    </span>
                  </td>
                  <td>
                    <button type="button" className="btn btn-danger btn-sm" onClick={() => handleDelete(p.id)}>
                      Remove
                    </button>
                  </td>
                </tr>
              ))}
              {positions.length === 0 && (
                <tr>
                  <td colSpan={7} className="empty-state" style={{ textAlign: 'center', padding: '2.5rem 1rem' }}>
                    <strong>No positions</strong>
                    Add a row above to get started.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

    </div>
  )
}
