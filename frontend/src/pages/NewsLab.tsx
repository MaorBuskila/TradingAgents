import React, { useEffect, useRef, useState } from 'react'
import axios from 'axios'
import { useTabLogger } from '../hooks/useTabLogger'
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine,
} from 'recharts'
import { RefreshCw, Square } from 'lucide-react'

const API_BASE = import.meta.env.VITE_API_BASE ?? '/api'
const SESSION_KEY = 'tradingagents_newslab_v1'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface TrendingStock {
  symbol: string
  display_name: string
  price: number
  change_pct: number
  volume: number
  market_cap: number | null
}

interface RedditTrending {
  symbol: string
  mentions: number
  posts: { title: string; score: number; url: string }[]
}

interface EarningsRow {
  ticker: string
  next_earnings_date: string | null
  all_earnings_dates: string[]
  eps_estimate_avg: number | null
  eps_estimate_low: number | null
  eps_estimate_high: number | null
  revenue_estimate_low: number | null
  revenue_estimate_high: number | null
}

interface FGPoint {
  date: string
  score: number
  label: string
}

interface NewsArticle {
  title: string
  publisher: string
  link: string
  pub_date: string | null
  summary: string
}

interface RedditPost {
  subreddit: string
  title: string
  score: number
  num_comments: number
  upvote_ratio: number
  flair: string
  url: string
  comments: string[]
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function today(): string {
  return new Date().toISOString().slice(0, 10)
}
function daysAgo(n: number): string {
  const d = new Date()
  d.setDate(d.getDate() - n)
  return d.toISOString().slice(0, 10)
}
function daysUntil(dateStr: string | null): number | null {
  if (!dateStr) return null
  const diff = new Date(dateStr).getTime() - new Date(today()).getTime()
  return Math.ceil(diff / (1000 * 60 * 60 * 24))
}
function fmtVol(v: number): string {
  if (v >= 1e9) return `${(v / 1e9).toFixed(1)}B`
  if (v >= 1e6) return `${(v / 1e6).toFixed(1)}M`
  if (v >= 1e3) return `${(v / 1e3).toFixed(0)}K`
  return String(v)
}
function fgColor(score: number): string {
  if (score <= 25) return 'var(--color-sell, #ef4444)'
  if (score <= 45) return '#f97316'
  if (score <= 55) return '#eab308'
  if (score <= 75) return '#84cc16'
  return 'var(--color-buy, #22c55e)'
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Shared card components
// ---------------------------------------------------------------------------

const CARD_STYLE: React.CSSProperties = {
  border: '1px solid var(--border)',
  borderRadius: '0.45rem',
  padding: '0.65rem 0.75rem',
  background: 'var(--bg-card, var(--bg-subtle))',
  display: 'flex',
  flexDirection: 'column',
  gap: '0.3rem',
}

function ArticleCard({ article }: { article: NewsArticle }) {
  return (
    <div style={CARD_STYLE}>
      <a
        href={article.link}
        target="_blank"
        rel="noopener noreferrer"
        style={{ fontWeight: 600, fontSize: '0.85rem', color: 'var(--color-primary, #6366f1)', textDecoration: 'none', lineHeight: 1.35 }}
      >
        {article.title}
      </a>
      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
        <span>{article.publisher}</span>
        {article.pub_date && <span>· {article.pub_date}</span>}
      </div>
      {article.summary && (
        <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: 1.45 }}>
          {article.summary.length > 220 ? article.summary.slice(0, 220) + '…' : article.summary}
        </div>
      )}
    </div>
  )
}

function RedditPostCard({ post }: { post: RedditPost }) {
  const [expanded, setExpanded] = useState(false)
  const ratioPct = Math.round(post.upvote_ratio * 100)
  const scoreColor = post.score > 500 ? 'var(--color-buy,#22c55e)' : post.score > 100 ? '#eab308' : 'var(--text-secondary)'
  return (
    <div style={CARD_STYLE}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: '0.5rem', alignItems: 'flex-start' }}>
        <a
          href={post.url}
          target="_blank"
          rel="noopener noreferrer"
          style={{ fontWeight: 600, fontSize: '0.85rem', color: 'var(--color-primary, #6366f1)', textDecoration: 'none', lineHeight: 1.35, flex: 1 }}
        >
          {post.title}
        </a>
        <span style={{ fontWeight: 700, fontSize: '0.82rem', color: scoreColor, whiteSpace: 'nowrap' }}>
          ↑ {post.score.toLocaleString()}
        </span>
      </div>
      <div style={{ fontSize: '0.73rem', color: 'var(--text-muted)', display: 'flex', gap: '0.6rem', flexWrap: 'wrap' }}>
        <span>r/{post.subreddit}</span>
        <span>💬 {post.num_comments} comments</span>
        <span>{ratioPct}% upvoted</span>
        {post.flair && <span style={{ background: 'var(--bg-subtle)', padding: '0 0.3rem', borderRadius: '0.25rem' }}>{post.flair}</span>}
      </div>
      {post.comments.length > 0 && (
        <div>
          <button
            onClick={() => setExpanded(x => !x)}
            style={{ background: 'none', border: 'none', cursor: 'pointer', fontSize: '0.75rem', color: 'var(--text-muted)', padding: 0 }}
          >
            {expanded ? '▲ Hide comments' : `▼ Top ${post.comments.length} comment${post.comments.length > 1 ? 's' : ''}`}
          </button>
          {expanded && (
            <div style={{ marginTop: '0.3rem', display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
              {post.comments.map((c, i) => (
                <div key={i} style={{ fontSize: '0.77rem', color: 'var(--text-secondary)', padding: '0.3rem 0.5rem', background: 'var(--bg-page,#f9f9f9)', borderRadius: '0.3rem', borderLeft: '2px solid var(--border)' }}>
                  {c}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function EarningsBadge({ days }: { days: number | null }) {
  if (days === null) return null
  if (days < 0) return <span style={{ color: 'var(--text-muted)', fontSize: '0.75rem' }}>Past</span>
  if (days <= 2) return <span className="badge badge-danger">Imminent</span>
  if (days <= 7) return <span className="badge badge-warning">Soon</span>
  return <span style={{ color: 'var(--text-secondary)', fontSize: '0.75rem' }}>{days}d</span>
}

function StopButton({ onClick }: { onClick: () => void }) {
  return (
    <button className="btn btn-sm btn-danger" onClick={onClick} title="Stop">
      <Square size={14} /> Stop
    </button>
  )
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function NewsLab() {
  const log = useTabLogger('NewsLab')
  // ── Trending (market movers) ──
  const [screener, setScreener] = useState('most_actives')
  const [trending, setTrending] = useState<TrendingStock[]>([])
  const [trendingLoading, setTrendingLoading] = useState(false)

  // ── Reddit hot tickers ──
  const [redditTrending, setRedditTrending] = useState<RedditTrending[]>([])
  const [redditTrendingLoading, setRedditTrendingLoading] = useState(false)

  // ── Shared ticker drill-down input ──
  const [ticker, setTicker] = useState('')

  // ── Fear & Greed ──
  const [fgHistory, setFgHistory] = useState<FGPoint[]>([])
  const [fgCurrent, setFgCurrent] = useState<{ score: number; label: string } | null>(null)
  const [fgLoading, setFgLoading] = useState(false)

  // ── Earnings calendar ──
  const [earningsTickers, setEarningsTickers] = useState('')
  const [earnings, setEarnings] = useState<EarningsRow[]>([])
  const [earningsLoading, setEarningsLoading] = useState(false)

  // ── Ticker news ──
  const [newsStartDate, setNewsStartDate] = useState(daysAgo(7))
  const [newsEndDate, setNewsEndDate] = useState(today())
  const [newsArticles, setNewsArticles] = useState<NewsArticle[]>([])
  const [newsLoading, setNewsLoading] = useState(false)
  const [newsError, setNewsError] = useState('')
  const newsAbortRef = useRef<AbortController | null>(null)

  // ── Reddit sentiment ──
  const [redditDays, setRedditDays] = useState(3)
  const [redditPosts, setRedditPosts] = useState<RedditPost[]>([])
  const [redditLoading, setRedditLoading] = useState(false)
  const [redditError, setRedditError] = useState('')
  const redditAbortRef = useRef<AbortController | null>(null)

  // ── Influencer mentions ──
  const [influencerPerson, setInfluencerPerson] = useState('Trump')
  const [influencerCustom, setInfluencerCustom] = useState('')
  const [influencerArticles, setInfluencerArticles] = useState<NewsArticle[]>([])
  const [influencerLoading, setInfluencerLoading] = useState(false)
  const influencerAbortRef = useRef<AbortController | null>(null)

  // ── Restore session ──
  useEffect(() => {
    try {
      const saved = sessionStorage.getItem(SESSION_KEY)
      if (saved) {
        const s = JSON.parse(saved)
        if (s.screener) setScreener(s.screener)
        if (s.ticker) setTicker(s.ticker)
        if (s.earningsTickers) setEarningsTickers(s.earningsTickers)
        if (s.newsStartDate) setNewsStartDate(s.newsStartDate)
        if (s.newsEndDate) setNewsEndDate(s.newsEndDate)
        if (s.redditDays) setRedditDays(s.redditDays)
        if (s.influencerPerson) setInfluencerPerson(s.influencerPerson)
      }
    } catch { /* ignore */ }

    // Auto-load on mount
    fetchTrending()
    fetchRedditTrending()
    fetchFearGreed()
  }, [])

  // ── Persist session ──
  useEffect(() => {
    sessionStorage.setItem(SESSION_KEY, JSON.stringify({
      screener, ticker, earningsTickers, newsStartDate, newsEndDate, redditDays, influencerPerson,
    }))
  }, [screener, ticker, earningsTickers, newsStartDate, newsEndDate, redditDays, influencerPerson])

  // ── Pill click: fill ticker in all sections ──
  function drillInto(symbol: string) {
    setTicker(symbol)
  }

  // ── Market movers ──
  async function fetchTrending() {
    setTrendingLoading(true)
    log('action:fetch', { section: 'trending', screener })
    try {
      log('api:start', { endpoint: 'news/trending' })
      const res = await axios.get(`${API_BASE}/news/trending`, { params: { screener, count: 25 } })
      setTrending(res.data)
      log('api:success')
    } catch (err) {
      log('api:error', err)
    } finally {
      setTrendingLoading(false)
    }
  }

  // ── Reddit hot tickers ──
  async function fetchRedditTrending() {
    setRedditTrendingLoading(true)
    try {
      const res = await axios.get(`${API_BASE}/news/reddit-trending`, { params: { limit: 100, top_n: 30 } })
      setRedditTrending(res.data)
    } catch { /* ignore */ } finally {
      setRedditTrendingLoading(false)
    }
  }

  // ── Fear & Greed ──
  async function fetchFearGreed() {
    setFgLoading(true)
    try {
      const res = await axios.get(`${API_BASE}/news/fear-greed`, { params: { days: 14 } })
      setFgHistory(res.data.history ?? [])
      if (res.data.current_score !== null) {
        setFgCurrent({ score: res.data.current_score, label: res.data.current_label })
      }
    } catch { /* ignore */ } finally {
      setFgLoading(false)
    }
  }

  // ── Earnings ──
  async function fetchEarnings() {
    if (!earningsTickers.trim()) return
    setEarningsLoading(true)
    setEarnings([])
    try {
      const res = await axios.get(`${API_BASE}/news/earnings`, { params: { tickers: earningsTickers } })
      setEarnings(res.data)
    } catch { /* ignore */ } finally {
      setEarningsLoading(false)
    }
  }

  // ── Ticker news ──
  async function fetchTickerNews() {
    if (!ticker.trim()) return
    newsAbortRef.current?.abort()
    const ctrl = new AbortController()
    newsAbortRef.current = ctrl
    setNewsLoading(true)
    setNewsArticles([])
    setNewsError('')
    try {
      const res = await axios.get(`${API_BASE}/news/ticker-articles`, {
        params: { ticker: ticker.trim().toUpperCase(), start_date: newsStartDate, end_date: newsEndDate },
        signal: ctrl.signal,
      })
      setNewsArticles(res.data ?? [])
      if ((res.data ?? []).length === 0) setNewsError('No articles found for this date range.')
    } catch (e) {
      if (!axios.isCancel(e)) setNewsError('Error fetching news.')
    } finally {
      setNewsLoading(false)
    }
  }

  // ── Reddit sentiment ──
  async function fetchRedditSentiment() {
    if (!ticker.trim()) return
    redditAbortRef.current?.abort()
    const ctrl = new AbortController()
    redditAbortRef.current = ctrl
    setRedditLoading(true)
    setRedditPosts([])
    setRedditError('')
    try {
      const res = await axios.get(`${API_BASE}/news/reddit-posts`, {
        params: { ticker: ticker.trim().toUpperCase(), days: redditDays },
        signal: ctrl.signal,
      })
      setRedditPosts(res.data ?? [])
      if ((res.data ?? []).length === 0) setRedditError('No Reddit posts found.')
    } catch (e) {
      if (!axios.isCancel(e)) setRedditError('Error fetching Reddit posts.')
    } finally {
      setRedditLoading(false)
    }
  }

  // ── Influencer mentions ──
  async function fetchInfluencerMentions() {
    if (!ticker.trim()) return
    const person = influencerCustom.trim() || influencerPerson
    influencerAbortRef.current?.abort()
    const ctrl = new AbortController()
    influencerAbortRef.current = ctrl
    setInfluencerLoading(true)
    setInfluencerArticles([])
    try {
      const res = await axios.get(`${API_BASE}/news/influencer-mentions`, {
        params: { person, ticker: ticker.trim().toUpperCase(), days: 14 },
        signal: ctrl.signal,
      })
      setInfluencerArticles(res.data)
    } catch (e) {
      if (!axios.isCancel(e)) setInfluencerArticles([])
    } finally {
      setInfluencerLoading(false)
    }
  }

  // ── Derived ──
  const activePerson = influencerCustom.trim() || influencerPerson

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------
  return (
    <div className="page-content">
      {/* Header */}
      <div className="page-header">
        <h1 className="page-title">News Lab</h1>
        <p className="page-subtitle">Market movers, social buzz, earnings & influencer mentions</p>
      </div>

      {/* ══════════ ROW 1: TRENDING ══════════ */}
      <div className="card" style={{ marginBottom: '1rem' }}>
        <div className="card-header" style={{ display: 'flex', gap: '0.75rem', alignItems: 'center', flexWrap: 'wrap' }}>
          <span style={{ fontWeight: 600 }}>Market Movers</span>
          <select
            className="form-control"
            style={{ width: 'auto' }}
            value={screener}
            onChange={e => setScreener(e.target.value)}
          >
            <option value="most_actives">Most Active</option>
            <option value="day_gainers">Day Gainers</option>
            <option value="day_losers">Day Losers</option>
            <option value="most_shorted_stocks">Most Shorted</option>
            <option value="growth_technology_stocks">Growth Tech</option>
          </select>
          <button className="btn btn-sm btn-primary" onClick={fetchTrending} disabled={trendingLoading}>
            <RefreshCw size={14} /> {trendingLoading ? 'Loading…' : 'Refresh'}
          </button>
        </div>

        {/* Market mover pills */}
        <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap', padding: '0.5rem 0' }}>
          {trending.length === 0 && !trendingLoading && (
            <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No data — click Refresh</span>
          )}
          {trending.map(s => (
            <button
              key={s.symbol}
              onClick={() => drillInto(s.symbol)}
              style={{
                border: '1px solid var(--border)',
                borderRadius: '1rem',
                padding: '0.25rem 0.65rem',
                background: 'var(--bg-subtle)',
                cursor: 'pointer',
                fontSize: '0.8rem',
                color: s.change_pct >= 0 ? 'var(--color-buy, #22c55e)' : 'var(--color-sell, #ef4444)',
                fontWeight: 600,
              }}
              title={`${s.display_name} — Vol: ${fmtVol(s.volume)}`}
            >
              {s.symbol} {s.change_pct >= 0 ? '+' : ''}{s.change_pct.toFixed(2)}%
            </button>
          ))}
        </div>

        {/* Reddit hot tickers */}
        <div style={{ borderTop: '1px solid var(--border)', paddingTop: '0.75rem', marginTop: '0.25rem' }}>
          <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center', marginBottom: '0.5rem' }}>
            <span style={{ fontWeight: 600 }}>Reddit Hot Tickers</span>
            <button className="btn btn-sm btn-primary" onClick={fetchRedditTrending} disabled={redditTrendingLoading}>
              <RefreshCw size={14} /> {redditTrendingLoading ? 'Scanning…' : 'Refresh'}
            </button>
            <span style={{ color: 'var(--text-muted)', fontSize: '0.78rem' }}>r/wallstreetbets · r/stocks · r/options</span>
          </div>
          <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
            {redditTrending.length === 0 && !redditTrendingLoading && (
              <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No data — click Refresh</span>
            )}
            {redditTrending.slice(0, 30).map(r => {
              const intensity = r.mentions > 20 ? '#f97316' : r.mentions > 8 ? '#eab308' : 'var(--text-secondary)'
              return (
                <button
                  key={r.symbol}
                  onClick={() => drillInto(r.symbol)}
                  style={{
                    border: '1px solid var(--border)',
                    borderRadius: '1rem',
                    padding: '0.25rem 0.65rem',
                    background: 'var(--bg-subtle)',
                    cursor: 'pointer',
                    fontSize: '0.8rem',
                    color: intensity,
                    fontWeight: 600,
                  }}
                  title={r.posts[0]?.title ?? ''}
                >
                  {r.symbol} 🔥{r.mentions}
                </button>
              )
            })}
          </div>
        </div>
      </div>

      {/* ══════════ ROW 2: FEAR & GREED ══════════ */}
      <div className="card" style={{ marginBottom: '1rem' }}>
        <div className="card-header" style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
          <span style={{ fontWeight: 600 }}>Fear &amp; Greed Index</span>
          <button className="btn btn-sm btn-primary" onClick={fetchFearGreed} disabled={fgLoading}>
            <RefreshCw size={14} /> {fgLoading ? 'Loading…' : 'Refresh'}
          </button>
        </div>
        <div style={{ display: 'flex', gap: '2rem', alignItems: 'center', flexWrap: 'wrap', padding: '0.5rem 0' }}>
          {fgCurrent && (
            <div style={{ textAlign: 'center', minWidth: '8rem' }}>
              <div style={{
                fontSize: '3rem',
                fontWeight: 700,
                color: fgColor(fgCurrent.score),
                lineHeight: 1,
              }}>{fgCurrent.score}</div>
              <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                {fgCurrent.label}
              </div>
            </div>
          )}
          {fgHistory.length > 0 && (
            <div style={{ flex: 1, minWidth: '260px', height: '100px' }}>
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={[...fgHistory].reverse()}>
                  <XAxis dataKey="date" tick={{ fontSize: 10 }} tickFormatter={d => d.slice(5)} />
                  <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} width={28} />
                  <Tooltip
                    formatter={(v: number, _n: string, p: any) => [`${v} — ${p.payload.label}`, 'Score']}
                    labelFormatter={l => l}
                  />
                  <ReferenceLine y={50} stroke="var(--border)" strokeDasharray="3 3" />
                  <Line type="monotone" dataKey="score" dot={false} strokeWidth={2} stroke="#6366f1" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
          {!fgCurrent && !fgLoading && (
            <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No data — click Refresh</span>
          )}
        </div>
      </div>

      {/* ══════════ ROW 3: EARNINGS CALENDAR ══════════ */}
      <div className="card" style={{ marginBottom: '1rem' }}>
        <div className="card-header" style={{ fontWeight: 600, marginBottom: '0.5rem' }}>Earnings Calendar</div>
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', flexWrap: 'wrap', marginBottom: '0.75rem' }}>
          <input
            className="form-control"
            style={{ flex: 1, minWidth: '200px', maxWidth: '400px' }}
            placeholder="Tickers: AAPL, NVDA, TSLA, MSFT"
            value={earningsTickers}
            onChange={e => setEarningsTickers(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && fetchEarnings()}
          />
          <button className="btn btn-primary" onClick={fetchEarnings} disabled={earningsLoading}>
            {earningsLoading ? 'Loading…' : 'Load Earnings'}
          </button>
        </div>
        {earnings.length > 0 && (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid var(--border)', textAlign: 'left' }}>
                  <th style={{ padding: '0.4rem 0.75rem' }}>Ticker</th>
                  <th style={{ padding: '0.4rem 0.75rem' }}>Next Date</th>
                  <th style={{ padding: '0.4rem 0.75rem' }}>Days Away</th>
                  <th style={{ padding: '0.4rem 0.75rem' }}>EPS Low</th>
                  <th style={{ padding: '0.4rem 0.75rem' }}>EPS Avg</th>
                  <th style={{ padding: '0.4rem 0.75rem' }}>EPS High</th>
                </tr>
              </thead>
              <tbody>
                {earnings.map(row => {
                  const d = daysUntil(row.next_earnings_date)
                  return (
                    <tr
                      key={row.ticker}
                      style={{ borderBottom: '1px solid var(--border)', cursor: 'pointer' }}
                      onClick={() => drillInto(row.ticker)}
                    >
                      <td style={{ padding: '0.4rem 0.75rem', fontWeight: 600 }}>{row.ticker}</td>
                      <td style={{ padding: '0.4rem 0.75rem' }}>{row.next_earnings_date ?? '—'}</td>
                      <td style={{ padding: '0.4rem 0.75rem' }}><EarningsBadge days={d} /></td>
                      <td style={{ padding: '0.4rem 0.75rem' }}>{row.eps_estimate_low ?? '—'}</td>
                      <td style={{ padding: '0.4rem 0.75rem', fontWeight: 500 }}>{row.eps_estimate_avg ?? '—'}</td>
                      <td style={{ padding: '0.4rem 0.75rem' }}>{row.eps_estimate_high ?? '—'}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* ══════════ ROW 4: TICKER DEEP-DIVE ══════════ */}
      <div className="card" style={{ marginBottom: '1rem' }}>
        <div className="card-header" style={{ fontWeight: 600, marginBottom: '0.75rem' }}>Ticker Deep-Dive</div>
        {/* Shared ticker input */}
        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', marginBottom: '1rem', flexWrap: 'wrap' }}>
          <input
            className="form-control"
            style={{ maxWidth: '140px', fontWeight: 600, textTransform: 'uppercase' }}
            placeholder="Ticker, e.g. NVDA"
            value={ticker}
            onChange={e => setTicker(e.target.value.toUpperCase())}
          />
          <span style={{ color: 'var(--text-muted)', fontSize: '0.82rem' }}>
            Click any trending pill above to auto-fill
          </span>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '1rem' }}>

          {/* ── News Feed ── */}
          <div style={{ border: '1px solid var(--border)', borderRadius: '0.5rem', padding: '0.75rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            <div style={{ fontWeight: 600 }}>News Feed</div>
            <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
              <input
                type="date"
                className="form-control"
                style={{ flex: 1, minWidth: '120px' }}
                value={newsStartDate}
                onChange={e => setNewsStartDate(e.target.value)}
              />
              <input
                type="date"
                className="form-control"
                style={{ flex: 1, minWidth: '120px' }}
                value={newsEndDate}
                onChange={e => setNewsEndDate(e.target.value)}
              />
            </div>
            <div style={{ display: 'flex', gap: '0.4rem' }}>
              <button className="btn btn-primary btn-sm" onClick={fetchTickerNews} disabled={newsLoading || !ticker}>
                {newsLoading ? 'Fetching…' : 'Fetch News'}
              </button>
              {newsLoading && <StopButton onClick={() => { newsAbortRef.current?.abort(); setNewsLoading(false) }} />}
            </div>
            {newsError && <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{newsError}</div>}
            {newsArticles.length > 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', maxHeight: '420px', overflowY: 'auto' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{newsArticles.length} article{newsArticles.length !== 1 ? 's' : ''}</div>
                {newsArticles.map((a, i) => <ArticleCard key={i} article={a} />)}
              </div>
            )}
          </div>

          {/* ── Reddit Sentiment ── */}
          <div style={{ border: '1px solid var(--border)', borderRadius: '0.5rem', padding: '0.75rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
            <div style={{ fontWeight: 600 }}>Reddit / Social</div>
            <div style={{ display: 'flex', gap: '0.4rem', alignItems: 'center' }}>
              <label style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', whiteSpace: 'nowrap' }}>Look back:</label>
              <select
                className="form-control"
                style={{ width: 'auto' }}
                value={redditDays}
                onChange={e => setRedditDays(Number(e.target.value))}
              >
                <option value={1}>1 day</option>
                <option value={3}>3 days</option>
                <option value={7}>7 days</option>
              </select>
            </div>
            <div style={{ display: 'flex', gap: '0.4rem' }}>
              <button className="btn btn-primary btn-sm" onClick={fetchRedditSentiment} disabled={redditLoading || !ticker}>
                {redditLoading ? 'Fetching…' : 'Fetch Reddit'}
              </button>
              {redditLoading && <StopButton onClick={() => { redditAbortRef.current?.abort(); setRedditLoading(false) }} />}
            </div>
            {redditError && <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{redditError}</div>}
            {redditPosts.length > 0 && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', maxHeight: '420px', overflowY: 'auto' }}>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{redditPosts.length} post{redditPosts.length !== 1 ? 's' : ''} from r/wallstreetbets, r/stocks, r/options</div>
                {redditPosts.map((p, i) => <RedditPostCard key={i} post={p} />)}
              </div>
            )}
          </div>

          {/* ── Influencer Mentions ── */}
          <div style={{ border: '1px solid var(--border)', borderRadius: '0.5rem', padding: '0.75rem' }}>
            <div style={{ fontWeight: 600, marginBottom: '0.5rem' }}>Influencer Mentions</div>
            {/* Person presets */}
            <div style={{ display: 'flex', gap: '0.3rem', flexWrap: 'wrap', marginBottom: '0.5rem' }}>
              {['Trump', 'Musk', 'Buffett', 'Pelosi'].map(p => (
                <button
                  key={p}
                  className={`btn btn-sm ${influencerPerson === p && !influencerCustom ? 'btn-primary' : 'btn-outline'}`}
                  onClick={() => { setInfluencerPerson(p); setInfluencerCustom('') }}
                >
                  {p}
                </button>
              ))}
            </div>
            <input
              className="form-control"
              style={{ marginBottom: '0.5rem' }}
              placeholder="Custom name, e.g. Cathie Wood"
              value={influencerCustom}
              onChange={e => setInfluencerCustom(e.target.value)}
            />
            <div style={{ display: 'flex', gap: '0.4rem', marginBottom: '0.5rem' }}>
              <button
                className="btn btn-primary btn-sm"
                onClick={fetchInfluencerMentions}
                disabled={influencerLoading || !ticker}
              >
                {influencerLoading ? 'Searching…' : `Search "${activePerson}" + ${ticker || '?'}`}
              </button>
              {influencerLoading && <StopButton onClick={() => { influencerAbortRef.current?.abort(); setInfluencerLoading(false) }} />}
            </div>
            {influencerArticles.length === 0 && !influencerLoading && ticker && (
              <span style={{ color: 'var(--text-muted)', fontSize: '0.82rem' }}>No articles found</span>
            )}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', maxHeight: '360px', overflowY: 'auto' }}>
              {influencerArticles.map((a, i) => (
                <div key={i} style={{
                  border: '1px solid var(--border)',
                  borderRadius: '0.4rem',
                  padding: '0.5rem',
                  background: 'var(--bg-subtle)',
                }}>
                  <a
                    href={a.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{ fontWeight: 600, fontSize: '0.83rem', color: 'var(--color-primary, #6366f1)', textDecoration: 'none' }}
                  >
                    {a.title}
                  </a>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>
                    {a.publisher}{a.pub_date ? ` · ${a.pub_date}` : ''}
                  </div>
                  {a.summary && (
                    <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>
                      {a.summary.slice(0, 200)}{a.summary.length > 200 ? '…' : ''}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
