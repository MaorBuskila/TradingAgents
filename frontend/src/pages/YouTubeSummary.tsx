import { useState, useEffect, useCallback } from 'react'
import axios from 'axios'
import ReactMarkdown from 'react-markdown'
import { Clapperboard, Loader2 } from 'lucide-react'

const API_BASE = import.meta.env.VITE_API_BASE ?? '/api'

type SummaryResult = {
  id: number
  video_id: string
  url: string
  title: string | null
  transcript_chars: number
  summary_en: string
  summary_he: string
  model: string
  provider?: string
  created_at?: string
}

type ListItem = {
  id: number
  video_id: string
  url: string
  title: string | null
  transcript_chars: number | null
  provider: string | null
  model: string | null
  created_at: string
}

export default function YouTubeSummary() {
  const [url, setUrl] = useState('https://www.youtube.com/watch?v=Vov0WMe3Dvc')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<SummaryResult | null>(null)
  const [tab, setTab] = useState<'en' | 'he'>('en')
  const [saved, setSaved] = useState<ListItem[]>([])
  const [loadingList, setLoadingList] = useState(false)
  const [retranslateLoading, setRetranslateLoading] = useState(false)

  const refreshList = useCallback(async () => {
    setLoadingList(true)
    try {
      const res = await axios.get<ListItem[]>(`${API_BASE}/youtube/summaries`)
      setSaved(res.data)
    } catch {
      setSaved([])
    } finally {
      setLoadingList(false)
    }
  }, [])

  useEffect(() => {
    refreshList()
  }, [refreshList])

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    setError(null)
    setResult(null)
    setLoading(true)
    try {
      const res = await axios.post<SummaryResult>(`${API_BASE}/youtube/summarize`, { url: url.trim() })
      setResult(res.data)
      setTab('en')
      refreshList()
    } catch (err: unknown) {
      if (axios.isAxiosError(err)) {
        const raw = err.response?.data
        let msg = err.message
        if (typeof raw === 'string') {
          msg = raw
        } else if (raw && typeof raw === 'object' && !Array.isArray(raw) && 'detail' in raw) {
          const det = (raw as { detail: unknown }).detail
          msg = Array.isArray(det) ? det.map((x) => JSON.stringify(x)).join(' ') : String(det)
        }
        setError(msg)
      } else {
        setError('Request failed')
      }
    } finally {
      setLoading(false)
    }
  }

  const loadSaved = async (id: number) => {
    setError(null)
    setLoading(true)
    try {
      const res = await axios.get<SummaryResult>(`${API_BASE}/youtube/summaries/${id}`)
      setResult(res.data)
      setTab('en')
    } catch (err: unknown) {
      if (axios.isAxiosError(err)) {
        const raw = err.response?.data
        let msg = err.message
        if (typeof raw === 'string') {
          msg = raw
        } else if (raw && typeof raw === 'object' && !Array.isArray(raw) && 'detail' in raw) {
          const det = (raw as { detail: unknown }).detail
          msg = Array.isArray(det) ? det.map((x) => JSON.stringify(x)).join(' ') : String(det)
        }
        setError(msg)
      }
    } finally {
      setLoading(false)
    }
  }

  const retranslateHebrew = async () => {
    if (!result?.id) return
    setError(null)
    setRetranslateLoading(true)
    try {
      const res = await axios.post<SummaryResult>(
        `${API_BASE}/youtube/summaries/${result.id}/retranslate-he`
      )
      setResult(res.data)
    } catch (err: unknown) {
      if (axios.isAxiosError(err)) {
        const raw = err.response?.data
        let msg = err.message
        if (typeof raw === 'string') {
          msg = raw
        } else if (raw && typeof raw === 'object' && !Array.isArray(raw) && 'detail' in raw) {
          const det = (raw as { detail: unknown }).detail
          msg = Array.isArray(det) ? det.map((x) => JSON.stringify(x)).join(' ') : String(det)
        }
        setError(msg)
      } else {
        setError('Re-translate failed')
      }
    } finally {
      setRetranslateLoading(false)
    }
  }

  const md = tab === 'en' ? result?.summary_en : result?.summary_he

  return (
    <div>
      <header className="page-header">
        <h1 className="page-title">YouTube summary</h1>
        <p className="page-desc">
          LLM summarizes captions with TL;DR, key points, and—when the video is about markets—an{' '}
          <strong>actionable angle</strong> (speaker stance, what to watch; not personal advice).{' '}
          <strong>Hebrew</strong> is produced with <strong>Google Translate</strong> via{' '}
          <code>deep-translator</code> (not an LLM). Both languages are stored in SQLite (
          <code>portfolio.db</code> → <code>youtube_summaries</code>).
        </p>
      </header>

      <div className="card">
        <div className="card-header">
          <h2 className="card-title" style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Clapperboard size={20} aria-hidden />
            Video
          </h2>
        </div>
        <form onSubmit={submit}>
          <div className="form-group">
            <label htmlFor="yt-url">YouTube URL or video ID</label>
            <input
              id="yt-url"
              className="form-control"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              placeholder="https://www.youtube.com/watch?v=..."
            />
          </div>
          <button type="submit" className="btn" disabled={loading}>
            {loading ? (
              <>
                <Loader2 size={18} className="icon-spin" />
                Summarizing…
              </>
            ) : (
              'Summarize & save'
            )}
          </button>
        </form>
      </div>

      {error && (
        <div className="card alert alert-error">
          <strong>Error</strong>
          <p>{error}</p>
        </div>
      )}

      {result && (
        <div className="card">
          <div className="result-meta">
            {result.title && (
              <>
                <strong style={{ color: 'var(--text)', fontSize: '1rem' }}>{result.title}</strong>
                <br />
              </>
            )}
            <a href={result.url} target="_blank" rel="noreferrer">
              {result.url}
            </a>
            <br />
            <span className="muted">
              DB id: {result.id}
              {result.created_at ? ` · Saved ${result.created_at}` : ''} · Transcript:{' '}
              {result.transcript_chars.toLocaleString()} chars · Provider: {result.provider ?? '—'} · Model:{' '}
              {result.model}
            </span>
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', marginBottom: '1rem', alignItems: 'center' }}>
            <button
              type="button"
              className={tab === 'en' ? 'btn' : 'btn btn-secondary'}
              onClick={() => setTab('en')}
            >
              English
            </button>
            <button
              type="button"
              className={tab === 'he' ? 'btn' : 'btn btn-secondary'}
              onClick={() => setTab('he')}
            >
              עברית (machine translation)
            </button>
            {result.summary_en?.trim() && (
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => void retranslateHebrew()}
                disabled={retranslateLoading || loading}
                title="Re-run Google Translate on the English text only (no new LLM summary)"
              >
                {retranslateLoading ? (
                  <>
                    <Loader2 size={16} className="icon-spin" aria-hidden />
                    Translating…
                  </>
                ) : (
                  'Re-translate Hebrew'
                )}
              </button>
            )}
          </div>
          <div className="youtube-summary-md report-prose" dir={tab === 'he' ? 'rtl' : 'ltr'}>
            <ReactMarkdown>{md ?? ''}</ReactMarkdown>
          </div>
        </div>
      )}

      <div className="card">
        <div className="card-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h2 className="card-title">Saved summaries</h2>
          <button type="button" className="btn btn-secondary" onClick={() => refreshList()} disabled={loadingList}>
            {loadingList ? '…' : 'Refresh'}
          </button>
        </div>
        {saved.length === 0 && !loadingList && <p className="muted">No saved summaries yet.</p>}
        {saved.length > 0 && (
          <table className="data-table">
            <thead>
              <tr>
                <th>ID</th>
                <th>Title</th>
                <th>Video</th>
                <th>When</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {saved.map((s) => (
                <tr key={s.id}>
                  <td>{s.id}</td>
                  <td>{s.title || '—'}</td>
                  <td>
                    <code>{s.video_id}</code>
                  </td>
                  <td className="muted" style={{ fontSize: '0.85rem' }}>
                    {s.created_at}
                  </td>
                  <td>
                    <button type="button" className="btn btn-secondary" onClick={() => loadSaved(s.id)}>
                      Open
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
