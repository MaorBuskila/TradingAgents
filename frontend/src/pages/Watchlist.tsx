import { useState, useEffect, useCallback } from 'react'
import { Link } from 'react-router-dom'
import axios from 'axios'
import { Star } from 'lucide-react'

const API_BASE = import.meta.env.VITE_API_BASE ?? '/api'

type Cat = { slug: string; label: string; count: number }
type Row = {
  id: number
  ticker: string
  name: string | null
  category: string
  category_label: string | null
  asset_type: string
  source: string
  is_favorite: boolean
}

export default function Watchlist() {
  const [categories, setCategories] = useState<Cat[]>([])
  const [rows, setRows] = useState<Row[]>([])
  const [category, setCategory] = useState<string>('')
  const [q, setQ] = useState('')
  const [favoritesOnly, setFavoritesOnly] = useState(false)
  const [sort, setSort] = useState('ticker')
  const [order, setOrder] = useState<'asc' | 'desc'>('asc')
  const [loading, setLoading] = useState(false)
  const [newItem, setNewItem] = useState({ ticker: '', name: '', category: 'green-energy', asset_type: 'etf' })

  const loadCategories = useCallback(async () => {
    const res = await axios.get<Cat[]>(`${API_BASE}/catalog/categories`)
    setCategories(res.data)
  }, [])

  const loadItems = useCallback(async () => {
    setLoading(true)
    try {
      const params: Record<string, string | boolean> = { sort, order }
      if (category) params.category = category
      if (q.trim()) params.q = q.trim()
      if (favoritesOnly) params.favorites_only = true
      const res = await axios.get<Row[]>(`${API_BASE}/catalog/items`, { params })
      setRows(res.data)
    } finally {
      setLoading(false)
    }
  }, [category, q, favoritesOnly, sort, order])

  useEffect(() => {
    loadCategories()
  }, [loadCategories])

  useEffect(() => {
    if (categories.length === 0) return
    setNewItem((prev) => {
      if (categories.some((c) => c.slug === prev.category)) return prev
      const green = categories.find((c) => c.slug === 'green-energy')
      return { ...prev, category: green?.slug ?? categories[0].slug }
    })
  }, [categories])

  useEffect(() => {
    const t = setTimeout(() => loadItems(), 200)
    return () => clearTimeout(t)
  }, [loadItems])

  const toggleFavorite = async (ticker: string, next: boolean) => {
    await axios.post(`${API_BASE}/catalog/favorites/${encodeURIComponent(ticker)}`, {
      favorite: next,
    })
    loadItems()
  }

  const addCustom = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      await axios.post(`${API_BASE}/catalog/items`, {
        ticker: newItem.ticker,
        name: newItem.name,
        category: newItem.category,
        asset_type: newItem.asset_type,
      })
      setNewItem((prev) => ({ ...prev, ticker: '', name: '' }))
      loadCategories()
      loadItems()
    } catch (err: unknown) {
      if (axios.isAxiosError(err) && err.response?.status === 409) {
        alert('That ticker is already in the catalog.')
      } else {
        console.error(err)
      }
    }
  }

  return (
    <div>
      <header className="page-header">
        <h1 className="page-title">Symbol catalog</h1>
        <p className="page-desc">
          Filter by theme, search tickers or names, star favorites, and sort. Data is local; extend the seed list in{' '}
          <code>api/catalog_data.json</code> if needed.
        </p>
      </header>

      <div className="card" style={{ marginBottom: '1.25rem' }}>
        <div className="form-row-inline" style={{ alignItems: 'flex-end', gap: '1rem 1.25rem' }}>
          <div className="form-field-inline" style={{ flex: '0 1 240px' }}>
            <label htmlFor="wl-cat">Category</label>
            <select
              id="wl-cat"
              className="form-control"
              value={category}
              onChange={(e) => setCategory(e.target.value)}
            >
              <option value="">All categories</option>
              {categories.map((c) => (
                <option key={c.slug} value={c.slug}>
                  {c.label} ({c.count})
                </option>
              ))}
            </select>
          </div>
          <div className="form-field-inline" style={{ flex: '1 1 220px', minWidth: '180px' }}>
            <label htmlFor="wl-q">Search</label>
            <input
              id="wl-q"
              className="form-control"
              placeholder="Ticker or name…"
              value={q}
              onChange={(e) => setQ(e.target.value)}
            />
          </div>
          <label
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              cursor: 'pointer',
              fontSize: '0.875rem',
              fontWeight: 500,
              color: 'var(--text-secondary)',
              paddingBottom: '0.35rem',
            }}
          >
            <input type="checkbox" checked={favoritesOnly} onChange={(e) => setFavoritesOnly(e.target.checked)} />
            Favorites only
          </label>
          <div className="form-field-inline" style={{ flex: '0 1 160px' }}>
            <label htmlFor="wl-sort">Sort by</label>
            <select id="wl-sort" className="form-control" value={sort} onChange={(e) => setSort(e.target.value)}>
              <option value="ticker">Ticker</option>
              <option value="name">Name</option>
              <option value="category">Category</option>
              <option value="asset_type">Stock / ETF</option>
              <option value="favorite">Favorites first</option>
            </select>
          </div>
          <div className="form-field-inline" style={{ flex: '0 1 130px' }}>
            <label htmlFor="wl-order">Order</label>
            <select
              id="wl-order"
              className="form-control"
              value={order}
              onChange={(e) => setOrder(e.target.value as 'asc' | 'desc')}
            >
              <option value="asc">Ascending</option>
              <option value="desc">Descending</option>
            </select>
          </div>
        </div>
        <p className="muted" style={{ marginTop: '1rem', fontSize: '0.875rem', lineHeight: 1.55 }}>
          <strong style={{ color: 'var(--text-secondary)' }}>Tip:</strong> “Favorites first” + Asc puts starred rows on
          top; Desc flips to non-favorites first. Other fields sort alphabetically (or by type for stock/ETF).
        </p>
      </div>

      <div className="card">
        <form onSubmit={addCustom}>
          {loading && <p className="loading-hint">Updating list…</p>}
          <div className="table-wrap" style={{ border: 'none' }}>
            <table className="watchlist-table">
              <thead>
                <tr>
                  <th style={{ width: '48px' }} aria-label="Favorite">
                    <Star size={14} style={{ opacity: 0.5, verticalAlign: 'middle' }} />
                  </th>
                  <th>Ticker</th>
                  <th>Name</th>
                  <th>Category</th>
                  <th>Type</th>
                  <th style={{ width: '100px' }}></th>
                </tr>
              </thead>
              <tbody>
                <tr className="add-row">
                  <td className="muted" style={{ fontSize: '0.75rem', fontWeight: 600 }} title="New row">
                    +
                  </td>
                  <td>
                    <input
                      className="form-control"
                      required
                      placeholder="e.g. QQQ"
                      aria-label="New ticker"
                      value={newItem.ticker}
                      onChange={(e) => setNewItem({ ...newItem, ticker: e.target.value })}
                    />
                  </td>
                  <td>
                    <input
                      className="form-control"
                      placeholder="Optional name"
                      aria-label="New name"
                      value={newItem.name}
                      onChange={(e) => setNewItem({ ...newItem, name: e.target.value })}
                    />
                  </td>
                  <td>
                    <select
                      className="form-control"
                      aria-label="Category"
                      value={newItem.category}
                      onChange={(e) => setNewItem({ ...newItem, category: e.target.value })}
                      disabled={categories.length === 0}
                    >
                      {categories.length === 0 ? (
                        <option value="green-energy">Loading…</option>
                      ) : (
                        categories.map((c) => (
                          <option key={c.slug} value={c.slug}>
                            {c.label}
                          </option>
                        ))
                      )}
                    </select>
                  </td>
                  <td>
                    <select
                      className="form-control"
                      aria-label="Asset type"
                      value={newItem.asset_type}
                      onChange={(e) => setNewItem({ ...newItem, asset_type: e.target.value })}
                    >
                      <option value="stock">Stock</option>
                      <option value="etf">ETF</option>
                    </select>
                  </td>
                  <td>
                    <button type="submit" className="btn btn-sm">
                      Add
                    </button>
                  </td>
                </tr>
                {rows.map((r) => (
                  <tr key={r.id}>
                    <td>
                      <button
                        type="button"
                        className="fav-btn"
                        aria-label={r.is_favorite ? 'Remove favorite' : 'Add favorite'}
                        onClick={() => toggleFavorite(r.ticker, !r.is_favorite)}
                      >
                        <Star
                          size={20}
                          strokeWidth={2}
                          fill={r.is_favorite ? 'var(--warning)' : 'none'}
                          color={r.is_favorite ? 'var(--warning)' : 'var(--text-muted)'}
                        />
                      </button>
                    </td>
                    <td>
                      <strong>{r.ticker}</strong>
                    </td>
                    <td>{r.name || '—'}</td>
                    <td>{r.category_label || r.category}</td>
                    <td>
                      <span
                        className="badge"
                        style={{
                          background: 'var(--bg-subtle)',
                          color: 'var(--text-secondary)',
                          border: '1px solid var(--border)',
                          textTransform: 'uppercase',
                          fontSize: '0.65rem',
                          letterSpacing: '0.06em',
                        }}
                      >
                        {r.asset_type}
                      </span>
                    </td>
                    <td>
                      <Link className="link-analyze" to={`/?ticker=${encodeURIComponent(r.ticker)}`}>
                        Analyze
                      </Link>
                    </td>
                  </tr>
                ))}
                {!loading && rows.length === 0 && (
                  <tr>
                    <td colSpan={6} style={{ padding: '1.5rem', textAlign: 'center' }} className="muted">
                      No symbols match your filters. Add one in the row above or widen filters.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </form>
      </div>
    </div>
  )
}
