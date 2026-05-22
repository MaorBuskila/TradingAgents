import { useState, useEffect, useMemo } from 'react'
import axios from 'axios'
import { useTabLogger } from '../hooks/useTabLogger'
import ReactMarkdown from 'react-markdown'
import { FileSearch, RefreshCw } from 'lucide-react'

const API_BASE = import.meta.env.VITE_API_BASE ?? '/api'

export default function Reports() {
  const log = useTabLogger('Reports')
  const [reports, setReports] = useState<{ id: string; ticker: string; date: string }[]>([])
  const [selectedReport, setSelectedReport] = useState<string | null>(null)
  const [reportContent, setReportContent] = useState<string>('')
  const [loadingList, setLoadingList] = useState(true)
  const [loadingDoc, setLoadingDoc] = useState(false)
  const [query, setQuery] = useState('')

  useEffect(() => {
    fetchReports()
  }, [])

  const fetchReports = async () => {
    setLoadingList(true)
    try {
      log('api:start', { endpoint: 'reports' })
      const res = await axios.get(`${API_BASE}/reports`)
      setReports(res.data)
      log('api:success')
    } catch (err) {
      log('api:error', err)
    } finally {
      setLoadingList(false)
    }
  }

  const loadReport = async (id: string) => {
    setLoadingDoc(true)
    if (id !== selectedReport) setReportContent('')
    log('action:load-report', { id })
    try {
      log('api:start', { endpoint: `reports/${id}` })
      const res = await axios.get(`${API_BASE}/reports/${id}`)
      setReportContent(res.data.content)
      setSelectedReport(id)
      log('api:success')
    } catch (err) {
      log('api:error', err)
    } finally {
      setLoadingDoc(false)
    }
  }

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return reports
    return reports.filter(
      (r) =>
        r.ticker.toLowerCase().includes(q) ||
        String(r.date).toLowerCase().includes(q) ||
        r.id.toLowerCase().includes(q)
    )
  }, [reports, query])

  return (
    <div>
      <header className="page-header">
        <h1 className="page-title">Analysis reports</h1>
        <p className="page-desc">
          Open a saved markdown report. Typography is tuned for long reads: headings, lists, tables, and code blocks.
        </p>
      </header>

      <div className="reports-layout">
        <div className="card" style={{ marginBottom: 0, position: 'sticky', top: '1rem' }}>
          <div className="card-header">
            <h2 className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <FileSearch size={20} strokeWidth={2} aria-hidden />
              Library
            </h2>
            <button
              type="button"
              className="btn btn-ghost btn-sm"
              onClick={() => fetchReports()}
              disabled={loadingList}
              title="Refresh list"
            >
              <RefreshCw size={16} className={loadingList ? 'icon-spin' : ''} />
            </button>
          </div>
          <div className="form-group" style={{ marginBottom: '0.75rem' }}>
            <label htmlFor="report-search">Search</label>
            <input
              id="report-search"
              className="form-control"
              placeholder="Ticker, date, id…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </div>
          {loadingList && <p className="loading-hint">Loading reports…</p>}
          <ul className="report-list">
            {filtered.map((r) => (
              <li key={r.id}>
                <button
                  type="button"
                  className={`report-list-item${selectedReport === r.id ? ' selected' : ''}`}
                  onClick={() => loadReport(r.id)}
                >
                  <div className="report-list-ticker">{r.ticker}</div>
                  <div className="report-list-date">{r.date}</div>
                </button>
              </li>
            ))}
            {!loadingList && filtered.length === 0 && (
              <li className="empty-state" style={{ padding: '1.5rem 0.5rem' }}>
                {reports.length === 0 ? (
                  <>
                    <strong>No reports yet</strong>
                    Run an analysis to generate one.
                  </>
                ) : (
                  <>
                    <strong>No matches</strong>
                    Try a different search.
                  </>
                )}
              </li>
            )}
          </ul>
        </div>

        <div className="card" style={{ marginBottom: 0 }}>
          {selectedReport ? (
            <>
              <div className="card-header">
                <h2 className="card-title">Report</h2>
                {loadingDoc && <span className="muted" style={{ fontSize: '0.875rem' }}>Loading…</span>}
              </div>
              <div className="report-viewport report-prose">
                {loadingDoc ? (
                  <p className="muted" style={{ padding: '1rem 0' }}>
                    Loading document…
                  </p>
                ) : (
                  <ReactMarkdown>{reportContent}</ReactMarkdown>
                )}
              </div>
            </>
          ) : (
            <div className="empty-state report-viewport">
              <strong>Select a report</strong>
              Choose an item from the library to render markdown here.
            </div>
          )}
        </div>
      </div>

    </div>
  )
}
