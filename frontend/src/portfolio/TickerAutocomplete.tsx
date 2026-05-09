import { useState, useEffect, useRef, useCallback } from 'react'
import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_BASE ?? '/api'

interface CatalogItem {
  ticker: string
  name?: string
  category?: string
  asset_type?: string
}

interface Props {
  value: string
  onChange: (ticker: string) => void
  onSelect?: (item: CatalogItem) => void
  placeholder?: string
  id?: string
  className?: string
  required?: boolean
}

const ASSET_BADGE: Record<string, { bg: string; color: string; label: string }> = {
  etf:   { bg: '#dbeafe', color: '#1d4ed8', label: 'ETF' },
  stock: { bg: '#dcfce7', color: '#15803d', label: 'Stock' },
  index: { bg: '#fef9c3', color: '#854d0e', label: 'Index' },
}

export default function TickerAutocomplete({ value, onChange, onSelect, placeholder = 'e.g. SPY', id, className, required }: Props) {
  const [results, setResults]     = useState<CatalogItem[]>([])
  const [open, setOpen]           = useState(false)
  const [highlighted, setHighlighted] = useState(-1)
  const [loading, setLoading]     = useState(false)
  const debounceRef               = useRef<ReturnType<typeof setTimeout> | null>(null)
  const containerRef              = useRef<HTMLDivElement>(null)
  const inputRef                  = useRef<HTMLInputElement>(null)

  const search = useCallback(async (q: string) => {
    if (!q.trim()) { setResults([]); setOpen(false); return }
    setLoading(true)
    try {
      const res = await axios.get(`${API_BASE}/catalog/items`, { params: { q: q.trim() } })
      setResults(res.data.slice(0, 10))
      setOpen(res.data.length > 0)
      setHighlighted(-1)
    } catch {
      setResults([])
      setOpen(false)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => search(value), 200)
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current) }
  }, [value, search])

  // Close on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const pick = (item: CatalogItem) => {
    onChange(item.ticker)
    onSelect?.(item)
    setOpen(false)
    setHighlighted(-1)
    inputRef.current?.blur()
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!open || results.length === 0) return
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setHighlighted(h => Math.min(h + 1, results.length - 1))
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setHighlighted(h => Math.max(h - 1, 0))
    } else if (e.key === 'Enter' && highlighted >= 0) {
      e.preventDefault()
      pick(results[highlighted])
    } else if (e.key === 'Escape') {
      setOpen(false)
    }
  }

  return (
    <div ref={containerRef} style={{ position: 'relative' }}>
      <div style={{ position: 'relative' }}>
        <input
          ref={inputRef}
          id={id}
          type="text"
          className={className ?? 'form-control'}
          value={value}
          placeholder={placeholder}
          required={required}
          autoComplete="off"
          onChange={e => onChange(e.target.value.toUpperCase())}
          onFocus={() => { if (results.length > 0) setOpen(true) }}
          onKeyDown={handleKeyDown}
          style={{ paddingRight: loading ? '2rem' : undefined }}
        />
        {loading && (
          <div style={{
            position: 'absolute', right: 9, top: '50%', transform: 'translateY(-50%)',
            width: 14, height: 14, border: '2px solid #d1d5db',
            borderTopColor: '#3b82f6', borderRadius: '50%',
            animation: 'spin 0.6s linear infinite',
          }} />
        )}
      </div>

      {open && results.length > 0 && (
        <ul style={{
          position: 'absolute', top: 'calc(100% + 4px)', left: 0, right: 0,
          background: '#fff', border: '1px solid #e5e7eb', borderRadius: 8,
          boxShadow: '0 8px 24px rgba(0,0,0,0.12)', zIndex: 300,
          margin: 0, padding: '4px 0', listStyle: 'none',
          maxHeight: 280, overflowY: 'auto',
        }}>
          {results.map((item, idx) => {
            const badge = ASSET_BADGE[item.asset_type ?? ''] ?? null
            const isActive = idx === highlighted
            return (
              <li
                key={item.ticker}
                onMouseDown={() => pick(item)}
                onMouseEnter={() => setHighlighted(idx)}
                style={{
                  padding: '8px 12px', cursor: 'pointer', display: 'flex',
                  alignItems: 'center', gap: 8,
                  background: isActive ? '#eff6ff' : 'transparent',
                  borderLeft: isActive ? '3px solid #3b82f6' : '3px solid transparent',
                }}
              >
                {/* Ticker */}
                <span style={{ fontWeight: 700, fontSize: '0.9rem', minWidth: 56, color: '#111' }}>
                  {item.ticker}
                </span>
                {/* Name */}
                <span style={{ fontSize: '0.8rem', color: '#6b7280', flex: 1 }}>
                  {item.name ?? ''}
                </span>
                {/* Type badge */}
                {badge && (
                  <span style={{
                    fontSize: '0.68rem', fontWeight: 600, padding: '1px 6px', borderRadius: 4,
                    background: badge.bg, color: badge.color, flexShrink: 0,
                  }}>
                    {badge.label}
                  </span>
                )}
              </li>
            )
          })}
        </ul>
      )}

      <style>{`@keyframes spin { to { transform: translateY(-50%) rotate(360deg); } }`}</style>
    </div>
  )
}
