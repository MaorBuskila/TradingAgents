import { useState, useEffect, useMemo, useRef } from 'react'
import { useSearchParams, Link, useLocation } from 'react-router-dom'
import axios from 'axios'
import { Loader2, Sparkles } from 'lucide-react'

const API_BASE = import.meta.env.VITE_API_BASE ?? '/api'

const ANALYST_KEY_TO_AGENT: Record<string, string> = {
  market: 'Market Analyst',
  social: 'Social Analyst',
  news: 'News Analyst',
  fundamentals: 'Fundamentals Analyst',
}

const PROGRESS_TEAMS: [string, string[]][] = [
  ['Analyst Team', ['Market Analyst', 'Social Analyst', 'News Analyst', 'Fundamentals Analyst']],
  ['Research Team', ['Bull Researcher', 'Bear Researcher', 'Research Manager']],
  ['Trading Team', ['Trader']],
  ['Risk Management', ['Aggressive Analyst', 'Neutral Analyst', 'Conservative Analyst']],
  ['Portfolio Management', ['Portfolio Manager']],
]

const REPORT_SECTION_DEFS: { section: string; analystKey: string | null }[] = [
  { section: 'market_report', analystKey: 'market' },
  { section: 'sentiment_report', analystKey: 'social' },
  { section: 'news_report', analystKey: 'news' },
  { section: 'fundamentals_report', analystKey: 'fundamentals' },
  { section: 'investment_plan', analystKey: null },
  { section: 'trader_investment_plan', analystKey: null },
  { section: 'final_trade_decision', analystKey: null },
]

const FINALIZING_AGENT: Record<string, string> = {
  market_report: 'Market Analyst',
  sentiment_report: 'Social Analyst',
  news_report: 'News Analyst',
  fundamentals_report: 'Fundamentals Analyst',
  investment_plan: 'Research Manager',
  trader_investment_plan: 'Trader',
  final_trade_decision: 'Portfolio Manager',
}

type LlmModelOption = { label: string; value: string }
type LlmProviderEntry = {
  id: string
  label: string
  backend_url: string
  shallow_models: LlmModelOption[]
  deep_models: LlmModelOption[]
}

type WizardStep = { id: string; title: string; description: string }
type LabeledValue = { label: string; value: string }
type LabeledInt = { label: string; value: number }

type AnalysisWizardSpec = {
  steps: WizardStep[]
  step7_by_provider: Record<string, { title: string; description: string; prompt: string }>
  prompts: Record<string, string>
  ticker_examples: string
  research_depth_options: LabeledInt[]
  analyst_options: LabeledValue[]
  thinking_options_by_provider: Record<string, LabeledValue[]>
}

const FALLBACK_WIZARD: AnalysisWizardSpec = {
  steps: [
    {
      id: 'ticker',
      title: 'Step 1: Ticker Symbol',
      description:
        'Enter the exact ticker symbol to analyze, including exchange suffix when needed (examples: SPY, CNC.TO, 7203.T, 0700.HK)',
    },
    {
      id: 'analysis_date',
      title: 'Step 2: Analysis Date',
      description: 'Enter the analysis date (YYYY-MM-DD)',
    },
    {
      id: 'analysts',
      title: 'Step 3: Analysts Team',
      description: 'Select your LLM analyst agents for the analysis',
    },
    {
      id: 'research_depth',
      title: 'Step 4: Research Depth',
      description: 'Select your research depth level',
    },
    {
      id: 'llm_provider',
      title: 'Step 5: OpenAI backend',
      description: 'Select which service to talk to',
    },
    {
      id: 'thinking_models',
      title: 'Step 6: Thinking Agents',
      description: 'Select your thinking agents for analysis',
    },
  ],
  step7_by_provider: {
    google: {
      title: 'Step 7: Thinking Mode',
      description: 'Configure Gemini thinking mode',
      prompt: 'Select Thinking Mode:',
    },
    openai: {
      title: 'Step 7: Reasoning Effort',
      description: 'Configure OpenAI reasoning effort level',
      prompt: 'Select Reasoning Effort:',
    },
    anthropic: {
      title: 'Step 7: Effort Level',
      description: 'Configure Claude effort level',
      prompt: 'Select Effort Level:',
    },
  },
  prompts: {
    ticker: 'Enter the exact ticker symbol to analyze',
    research_depth: 'Select Your [Research Depth]:',
    analysts: 'Select Your [Analysts Team]:',
    llm_provider: 'Select your LLM Provider:',
    quick_thinker: 'Select Your [Quick-Thinking LLM Engine]:',
    deep_thinker: 'Select Your [Deep-Thinking LLM Engine]:',
  },
  ticker_examples: 'Examples: SPY, CNC.TO, 7203.T, 0700.HK',
  research_depth_options: [
    {
      label: 'Shallow - Quick research, few debate and strategy discussion rounds',
      value: 1,
    },
    {
      label: 'Medium - Middle ground, moderate debate rounds and strategy discussion',
      value: 3,
    },
    {
      label: 'Deep - Comprehensive research, in depth debate and strategy discussion',
      value: 5,
    },
  ],
  analyst_options: [
    { label: 'Market Analyst', value: 'market' },
    { label: 'Social Media Analyst', value: 'social' },
    { label: 'News Analyst', value: 'news' },
    { label: 'Fundamentals Analyst', value: 'fundamentals' },
  ],
  thinking_options_by_provider: {
    google: [
      { label: 'Enable Thinking (recommended)', value: 'high' },
      { label: 'Minimal/Disable Thinking', value: 'minimal' },
    ],
    openai: [
      { label: 'Medium (Default)', value: 'medium' },
      { label: 'High (More thorough)', value: 'high' },
      { label: 'Low (Faster)', value: 'low' },
    ],
    anthropic: [
      { label: 'High (recommended)', value: 'high' },
      { label: 'Medium (balanced)', value: 'medium' },
      { label: 'Low (faster, cheaper)', value: 'low' },
    ],
  },
}

type LlmCatalogPayload = {
  providers: LlmProviderEntry[]
  wizard?: AnalysisWizardSpec
}

type MsgToolRow = { id: number; time: string; kind: string; text: string }

const RUN_ANALYSIS_STORAGE_KEY = 'tradingagents_run_analysis_v2'
const MAX_STORED_EVENTS = 200

type FormDataState = {
  ticker: string
  analysis_date: string
  analysts: string[]
  research_depth: number
  llm_provider: string
  backend_url: string
  shallow_thinker: string
  deep_thinker: string
}

/** Session snapshot; v1 used geminiMinimalThinking, v2 uses google_thinking_level */
type PersistedRunSnapshot = {
  v?: number
  jobId: string
  status: 'running' | 'completed' | 'error'
  formData: FormDataState
  customBackendUrl: boolean
  google_thinking_level?: string
  geminiMinimalThinking?: boolean
  openaiReasoningEffort: string
  anthropicEffort: string
  events: any[]
  agentStatus: Record<string, string>
  reportSections: Record<string, string | null>
  msgToolRows: MsgToolRow[]
  msgIdMax: number
  jobStartedAt: number | null
  lastEventAt: number | null
  savedReportId?: string | null
  reportSaveError?: string | null
}

const LEGACY_RUN_STORAGE_KEY = 'tradingagents_run_analysis_v1'

function readPersistedRun(): PersistedRunSnapshot | null {
  if (typeof window === 'undefined') return null
  for (const key of [RUN_ANALYSIS_STORAGE_KEY, LEGACY_RUN_STORAGE_KEY]) {
    try {
      const raw = sessionStorage.getItem(key)
      if (!raw) continue
      const p = JSON.parse(raw) as PersistedRunSnapshot
      if (typeof p.jobId !== 'string' || !p.jobId) continue
      if (p.status !== 'running' && p.status !== 'completed' && p.status !== 'error') continue
      return p
    } catch {
      continue
    }
  }
  return null
}

const persistRunRef = { current: null as PersistedRunSnapshot | null }

function pickValidOrFirst(current: string, options: LlmModelOption[]): string {
  if (options.some((o) => o.value === current)) return current
  return options[0]?.value ?? current
}

function wizardStep(w: AnalysisWizardSpec, id: string): WizardStep | undefined {
  return w.steps.find((s) => s.id === id)
}

function normalizeResearchDepth(d: number, w: AnalysisWizardSpec): number {
  const allowed = w.research_depth_options.map((o) => o.value)
  return allowed.includes(d) ? d : (allowed[0] ?? 1)
}

function formatDurationMs(ms: number): string {
  if (ms < 0 || !Number.isFinite(ms)) return '0s'
  const s = Math.floor(ms / 1000)
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  const r = s % 60
  if (m < 60) return `${m}m ${r}s`
  const h = Math.floor(m / 60)
  return `${h}h ${m % 60}m`
}

function clockStr(d = new Date()): string {
  return d.toTimeString().slice(0, 8)
}

function truncate(s: string, n: number): string {
  if (s.length <= n) return s
  return `${s.slice(0, n - 3)}...`
}

function formatToolArgs(args: unknown): string {
  if (args == null) return ''
  try {
    return typeof args === 'string' ? args : JSON.stringify(args)
  } catch {
    return String(args)
  }
}

function buildInitialAgentStatus(selectedAnalysts: string[]): Record<string, string> {
  const m: Record<string, string> = {}
  for (const key of selectedAnalysts) {
    const agent = ANALYST_KEY_TO_AGENT[key]
    if (agent) m[agent] = 'pending'
  }
  const fixed = [
    'Bull Researcher',
    'Bear Researcher',
    'Research Manager',
    'Trader',
    'Aggressive Analyst',
    'Neutral Analyst',
    'Conservative Analyst',
    'Portfolio Manager',
  ]
  for (const a of fixed) m[a] = 'pending'
  return m
}

function initReportSections(selectedAnalysts: string[]): Record<string, string | null> {
  const m: Record<string, string | null> = {}
  for (const { section, analystKey } of REPORT_SECTION_DEFS) {
    if (analystKey === null || selectedAnalysts.includes(analystKey)) {
      m[section] = null
    }
  }
  return m
}

function completedReportCount(
  sections: Record<string, string | null>,
  agentStatus: Record<string, string>,
): number {
  let n = 0
  for (const [sec, content] of Object.entries(sections)) {
    const fin = FINALIZING_AGENT[sec]
    if (!fin) continue
    if (content != null && agentStatus[fin] === 'completed') n++
  }
  return n
}

function statusClass(st: string): string {
  if (st === 'completed') return 'cli-status cli-status-done'
  if (st === 'in_progress') return 'cli-status cli-status-run'
  if (st === 'error') return 'cli-status cli-status-err'
  return 'cli-status cli-status-pend'
}

export default function RunAnalysis() {
  const [searchParams] = useSearchParams()
  const location = useLocation()
  const tickerFromUrl = searchParams.get('ticker')?.trim()

  const persistedOnMount = useRef<PersistedRunSnapshot | null | undefined>(undefined)
  if (persistedOnMount.current === undefined) {
    persistedOnMount.current = readPersistedRun()
  }
  const p0 = persistedOnMount.current

  const defaultForm = (): FormDataState => ({
    ticker: (tickerFromUrl && tickerFromUrl.toUpperCase()) || 'SPY',
    analysis_date: new Date().toISOString().split('T')[0],
    analysts: FALLBACK_WIZARD.analyst_options.map((o) => o.value),
    research_depth: 1,
    llm_provider: 'openai',
    backend_url: 'https://api.openai.com/v1',
    shallow_thinker: 'gpt-4o-mini',
    deep_thinker: 'gpt-4o',
  })

  const [formData, setFormData] = useState<FormDataState>(() => {
    const base = p0?.formData ?? defaultForm()
    const rd = normalizeResearchDepth(base.research_depth, FALLBACK_WIZARD)
    const urlT = tickerFromUrl?.trim()
    if (urlT) return { ...base, ticker: urlT.toUpperCase(), research_depth: rd }
    return { ...base, research_depth: rd }
  })

  const [customBackendUrl, setCustomBackendUrl] = useState(() => p0?.customBackendUrl ?? false)
  const [googleThinkingLevel, setGoogleThinkingLevel] = useState<'high' | 'minimal'>(() => {
    const g = p0?.google_thinking_level
    if (g === 'high' || g === 'minimal') return g
    if (p0?.geminiMinimalThinking === true) return 'minimal'
    return 'high'
  })
  const [openaiReasoningEffort, setOpenaiReasoningEffort] = useState(
    () => p0?.openaiReasoningEffort ?? 'medium',
  )
  const [anthropicEffort, setAnthropicEffort] = useState(() => p0?.anthropicEffort ?? 'high')

  const [jobId, setJobId] = useState<string | null>(() => p0?.jobId ?? null)
  const [events, setEvents] = useState<any[]>(() => p0?.events ?? [])
  const [agentStatus, setAgentStatus] = useState<Record<string, string>>(() => p0?.agentStatus ?? {})
  const [reportSections, setReportSections] = useState<Record<string, string | null>>(
    () => p0?.reportSections ?? {},
  )
  const [msgToolRows, setMsgToolRows] = useState<MsgToolRow[]>(() => p0?.msgToolRows ?? [])
  const msgIdRef = useRef(p0?.msgIdMax ?? 0)

  const statusRef = useRef<'idle' | 'running' | 'completed' | 'error'>('idle')
  const sseActiveRef = useRef(false)
  const sseStreamEndedRef = useRef(false)
  const sseRetryCountRef = useRef(0)
  const MAX_SSE_RETRIES = 3
  const [sseReconnectNonce, setSseReconnectNonce] = useState(0)
  const userEditedTicker = useRef(false)
  const lastNavKeyForTicker = useRef<string | undefined>(undefined)
  const lastTickerFromUrl = useRef<string | undefined>(undefined)

  const [status, setStatus] = useState<'idle' | 'running' | 'completed' | 'error'>(() => {
    if (p0?.jobId && p0.status) return p0.status
    return 'idle'
  })
  statusRef.current = status

  const [jobStartedAt, setJobStartedAt] = useState<number | null>(() => p0?.jobStartedAt ?? null)
  const [lastEventAt, setLastEventAt] = useState<number | null>(() => p0?.lastEventAt ?? null)
  const [savedReportId, setSavedReportId] = useState<string | null>(() => p0?.savedReportId ?? null)
  const [reportSaveError, setReportSaveError] = useState<string | null>(
    () => p0?.reportSaveError ?? null,
  )
  const [now, setNow] = useState(() => Date.now())
  const [llmCatalog, setLlmCatalog] = useState<LlmCatalogPayload | null>(null)
  const [llmCatalogReady, setLlmCatalogReady] = useState(false)

  useEffect(() => {
    msgIdRef.current = p0?.msgIdMax ?? 0
    // eslint-disable-next-line react-hooks/exhaustive-deps -- sync once from persisted snapshot p0
  }, [])

  useEffect(() => {
    const param = tickerFromUrl?.trim()
    if (!param) {
      lastNavKeyForTicker.current = location.key
      lastTickerFromUrl.current = tickerFromUrl
      return
    }
    if (status === 'running') {
      lastNavKeyForTicker.current = location.key
      lastTickerFromUrl.current = tickerFromUrl
      return
    }

    const upper = param.toUpperCase()
    const keyChanged = lastNavKeyForTicker.current !== location.key
    const tickerParamChanged = lastTickerFromUrl.current !== tickerFromUrl
    lastNavKeyForTicker.current = location.key
    lastTickerFromUrl.current = tickerFromUrl

    if (keyChanged || tickerParamChanged) {
      userEditedTicker.current = false
      setFormData((prev) => ({ ...prev, ticker: upper }))
    }
  }, [location.key, tickerFromUrl, status])

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const { data } = await axios.get<{
          llm_provider: string
          backend_url: string
          shallow_thinker: string
          deep_thinker: string
          research_depth?: number
          google_thinking_level?: string | null
          openai_reasoning_effort?: string | null
          anthropic_effort?: string | null
          catalog?: LlmCatalogPayload
        }>(`${API_BASE}/analysis-defaults`)
        if (cancelled) return
        if (data.catalog?.providers?.length) {
          setLlmCatalog(data.catalog)
        }
        if (!p0?.jobId) {
          const w = data.catalog?.wizard ?? FALLBACK_WIZARD
          const rdRaw =
            typeof data.research_depth === 'number' && Number.isFinite(data.research_depth)
              ? data.research_depth
              : 1
          const rd = normalizeResearchDepth(rdRaw, w)
          setFormData((prev) => ({
            ...prev,
            llm_provider: data.llm_provider,
            backend_url: data.backend_url,
            shallow_thinker: data.shallow_thinker,
            deep_thinker: data.deep_thinker,
            research_depth: rd,
          }))
          const gl = data.google_thinking_level
          if (gl === 'high' || gl === 'minimal') {
            setGoogleThinkingLevel(gl)
          }
          if (data.openai_reasoning_effort) {
            setOpenaiReasoningEffort(data.openai_reasoning_effort)
          }
          if (data.anthropic_effort) {
            setAnthropicEffort(data.anthropic_effort)
          }
        }
      } catch {
        /* keep static defaults */
      } finally {
        if (!cancelled) setLlmCatalogReady(true)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [])

  const currentLlmProvider = llmCatalog?.providers.find(
    (p) => p.id === formData.llm_provider.toLowerCase(),
  )

  const wizard = useMemo(
    () => llmCatalog?.wizard ?? FALLBACK_WIZARD,
    [llmCatalog],
  )

  useEffect(() => {
    setFormData((prev) => {
      const rd = normalizeResearchDepth(prev.research_depth, wizard)
      return rd === prev.research_depth ? prev : { ...prev, research_depth: rd }
    })
  }, [wizard])

  useEffect(() => {
    if (jobId) return
    if (!llmCatalog) return
    const p = llmCatalog.providers.find((x) => x.id === formData.llm_provider.toLowerCase())
    if (!p) return
    setFormData((prev) => {
      const s = pickValidOrFirst(prev.shallow_thinker, p.shallow_models)
      const d = pickValidOrFirst(prev.deep_thinker, p.deep_models)
      if (s === prev.shallow_thinker && d === prev.deep_thinker) return prev
      return { ...prev, shallow_thinker: s, deep_thinker: d }
    })
  }, [llmCatalog, formData.llm_provider, jobId])

  const handleLlmProviderChange = (providerId: string) => {
    const p = llmCatalog?.providers.find((x) => x.id === providerId)
    if (!p) {
      setFormData((prev) => ({ ...prev, llm_provider: providerId }))
      return
    }
    setCustomBackendUrl(false)
    setFormData((prev) => ({
      ...prev,
      llm_provider: p.id,
      backend_url: p.backend_url,
      shallow_thinker: pickValidOrFirst(prev.shallow_thinker, p.shallow_models),
      deep_thinker: pickValidOrFirst(prev.deep_thinker, p.deep_models),
    }))
  }

  const handleAnalystToggle = (analyst: string) => {
    setFormData((prev) => ({
      ...prev,
      analysts: prev.analysts.includes(analyst)
        ? prev.analysts.filter((a) => a !== analyst)
        : [...prev.analysts, analyst],
    }))
  }

  const buildAnalyzePayload = () => {
    const p = formData.llm_provider.toLowerCase()
    return {
      ...formData,
      llm_provider: p,
          google_thinking_level: p === 'google' ? googleThinkingLevel : null,
      openai_reasoning_effort: p === 'openai' ? openaiReasoningEffort : null,
      anthropic_effort: p === 'anthropic' ? anthropicEffort : null,
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    try {
      setJobId(null)
      sessionStorage.removeItem(RUN_ANALYSIS_STORAGE_KEY)
      sessionStorage.removeItem(LEGACY_RUN_STORAGE_KEY)
      setStatus('running')
      setEvents([])
      setJobStartedAt(null)
      setLastEventAt(null)
      const analysts = [...formData.analysts]
      setAgentStatus(buildInitialAgentStatus(analysts))
      setReportSections(initReportSections(analysts))
      setMsgToolRows([])
      msgIdRef.current = 0
      sseRetryCountRef.current = 0
      sseStreamEndedRef.current = false
      setSseReconnectNonce(0)
      setSavedReportId(null)
      setReportSaveError(null)

      const res = await axios.post(`${API_BASE}/analyze`, buildAnalyzePayload())
      const t = Date.now()
      setJobStartedAt(t)
      setLastEventAt(t)
      setJobId(res.data.job_id)
    } catch (err) {
      console.error(err)
      setStatus('error')
    }
  }

  useEffect(() => {
    if (status !== 'running') return
    const id = window.setInterval(() => setNow(Date.now()), 1000)
    return () => window.clearInterval(id)
  }, [status])

  useEffect(() => {
    if (!jobId || statusRef.current !== 'running' || sseActiveRef.current) return

    sseActiveRef.current = true
    sseStreamEndedRef.current = false
    const eventSource = new EventSource(`${API_BASE}/jobs/${jobId}/events`)

    eventSource.onmessage = (e) => {
      const data = JSON.parse(e.data)
      setLastEventAt(Date.now())
      setEvents((prev) => {
        const next = [...prev, data]
        return next.length > MAX_STORED_EVENTS ? next.slice(-MAX_STORED_EVENTS) : next
      })

      if (data.type === 'agent_status') {
        setAgentStatus((prev) => ({ ...prev, [data.agent_name]: data.status }))
      }
      if (data.type === 'report_section' && data.section_name) {
        setReportSections((prev) => ({
          ...prev,
          [data.section_name]: data.content ?? '',
        }))
      }
      if (data.type === 'message') {
        const raw =
          typeof data.content === 'string' ? data.content : String(data.content ?? '')
        const id = ++msgIdRef.current
        setMsgToolRows((prev) =>
          [
            {
              id,
              time: clockStr(),
              kind: String(data.msg_type ?? 'Message'),
              text: truncate(raw, 220),
            },
            ...prev,
          ].slice(0, 80),
        )
      }
      if (data.type === 'tool_call') {
        const id = ++msgIdRef.current
        const text = `${data.tool_name}: ${formatToolArgs(data.args)}`
        setMsgToolRows((prev) =>
          [{ id, time: clockStr(), kind: 'Tool', text: truncate(text, 220) }, ...prev].slice(
            0,
            80,
          ),
        )
      }

      if (data.type === 'complete') {
        sseStreamEndedRef.current = true
        sseActiveRef.current = false
        setStatus('completed')
        if (data.report_saved === true && data.report_id) {
          setSavedReportId(String(data.report_id))
          setReportSaveError(null)
        } else if (data.report_saved === false) {
          setSavedReportId(null)
          setReportSaveError(
            typeof data.report_save_error === 'string'
              ? data.report_save_error
              : 'Could not save report to disk',
          )
        }
        eventSource.close()
      } else if (data.type === 'error') {
        sseStreamEndedRef.current = true
        sseActiveRef.current = false
        setStatus('error')
        eventSource.close()
      }
    }

    eventSource.onerror = () => {
      eventSource.close()
      sseActiveRef.current = false
      if (sseStreamEndedRef.current || statusRef.current !== 'running') return
      if (sseRetryCountRef.current < MAX_SSE_RETRIES) {
        sseRetryCountRef.current += 1
        window.setTimeout(() => {
          if (statusRef.current === 'running') setSseReconnectNonce((n) => n + 1)
        }, 1500)
      } else {
        setStatus('error')
      }
    }

    return () => {
      sseActiveRef.current = false
      eventSource.close()
    }
  }, [jobId, sseReconnectNonce])

  useEffect(() => {
    if (!jobId) {
      sessionStorage.removeItem(RUN_ANALYSIS_STORAGE_KEY)
      sessionStorage.removeItem(LEGACY_RUN_STORAGE_KEY)
      persistRunRef.current = null
      return
    }
    const persistedStatus: PersistedRunSnapshot['status'] =
      status === 'completed' || status === 'error' ? status : 'running'
    const payload: PersistedRunSnapshot = {
      v: 2,
      jobId,
      status: persistedStatus,
      formData,
      customBackendUrl,
      google_thinking_level: googleThinkingLevel,
      openaiReasoningEffort,
      anthropicEffort,
      events: events.slice(-MAX_STORED_EVENTS),
      agentStatus,
      reportSections,
      msgToolRows,
      msgIdMax: msgIdRef.current,
      jobStartedAt,
      lastEventAt,
      savedReportId,
      reportSaveError,
    }
    persistRunRef.current = payload
    try {
      sessionStorage.setItem(RUN_ANALYSIS_STORAGE_KEY, JSON.stringify(payload))
    } catch {
      try {
        sessionStorage.setItem(
          RUN_ANALYSIS_STORAGE_KEY,
          JSON.stringify({ ...payload, events: events.slice(-80) }),
        )
      } catch {
        /* quota */
      }
    }
  }, [
    jobId,
    status,
    formData,
    customBackendUrl,
    googleThinkingLevel,
    openaiReasoningEffort,
    anthropicEffort,
    events,
    agentStatus,
    reportSections,
    msgToolRows,
    jobStartedAt,
    lastEventAt,
    savedReportId,
    reportSaveError,
  ])

  useEffect(() => {
    const flush = () => {
      const x = persistRunRef.current
      if (!x?.jobId) return
      try {
        sessionStorage.setItem(RUN_ANALYSIS_STORAGE_KEY, JSON.stringify(x))
      } catch {
        /* ignore */
      }
    }
    const onVis = () => {
      if (document.visibilityState === 'hidden') flush()
    }
    window.addEventListener('pagehide', flush)
    document.addEventListener('visibilitychange', onVis)
    return () => {
      window.removeEventListener('pagehide', flush)
      document.removeEventListener('visibilitychange', onVis)
      flush()
    }
  }, [])

  const progressRows = useMemo(() => {
    const rows: { team: string; agent: string; status: string; sep?: boolean }[] = []
    for (const [teamName, agents] of PROGRESS_TEAMS) {
      const active =
        teamName === 'Analyst Team'
          ? agents.filter((a) =>
              formData.analysts.some((k) => ANALYST_KEY_TO_AGENT[k] === a),
            )
          : agents
      if (active.length === 0) continue
      active.forEach((agent, idx) => {
        rows.push({
          team: idx === 0 ? teamName : '',
          agent,
          status: agentStatus[agent] ?? 'pending',
        })
      })
      rows.push({ team: '', agent: '', status: '', sep: true })
    }
    return rows
  }, [agentStatus, formData.analysts])

  const agentsCompleted = useMemo(
    () => Object.values(agentStatus).filter((s) => s === 'completed').length,
    [agentStatus],
  )
  const agentsTotal = useMemo(() => Object.keys(agentStatus).length, [agentStatus])
  const reportsCompleted = useMemo(
    () => completedReportCount(reportSections, agentStatus),
    [reportSections, agentStatus],
  )
  const reportsTotal = useMemo(() => Object.keys(reportSections).length, [reportSections])

  const providerLower = formData.llm_provider.toLowerCase()
  const catalogUrlsKnown =
    llmCatalog?.providers.some((p) => p.backend_url === formData.backend_url) ?? false

  return (
    <div>
      <header className="page-header">
        <h1 className="page-title">Run analysis</h1>
        <p className="page-desc">
          Configure ticker, date, and analyst modules. Progress streams here when a job is running.
          If you leave this page and come back, the current job (running or finished) is restored from
          this browser tab&apos;s session storage until you start a new analysis.
        </p>
      </header>

      <div className="card">
        <div className="card-header">
          <h2 className="card-title">Job parameters</h2>
        </div>
        <form onSubmit={handleSubmit}>
          <p className="page-desc" style={{ margin: '0 0 1rem', fontSize: '0.9rem' }}>
            Same steps and option text as the CLI wizard; catalog and copy come from the API when
            available.
          </p>

          <div className="form-group run-wizard-step">
            <h3 className="run-wizard-step-title">{wizardStep(wizard, 'ticker')?.title}</h3>
            <p className="page-desc" style={{ margin: '0 0 0.75rem', fontSize: '0.85rem' }}>
              {wizardStep(wizard, 'ticker')?.description}
            </p>
            <label htmlFor="ticker">{wizard.prompts.ticker}</label>
            <p className="page-desc" style={{ margin: '0.15rem 0 0.35rem', fontSize: '0.8rem' }}>
              {wizard.ticker_examples}
            </p>
            <input
              id="ticker"
              type="text"
              className="form-control"
              style={{ maxWidth: '280px' }}
              value={formData.ticker}
              onChange={(e) => {
                userEditedTicker.current = true
                setFormData({ ...formData, ticker: e.target.value })
              }}
            />
          </div>

          <div className="form-group run-wizard-step">
            <h3 className="run-wizard-step-title">{wizardStep(wizard, 'analysis_date')?.title}</h3>
            <p className="page-desc" style={{ margin: '0 0 0.75rem', fontSize: '0.85rem' }}>
              {wizardStep(wizard, 'analysis_date')?.description}
            </p>
            <label htmlFor="analysis_date">YYYY-MM-DD</label>
            <input
              id="analysis_date"
              type="date"
              className="form-control"
              style={{ maxWidth: '220px' }}
              value={formData.analysis_date}
              onChange={(e) => setFormData({ ...formData, analysis_date: e.target.value })}
            />
          </div>

          <div className="form-group run-wizard-step">
            <h3 className="run-wizard-step-title">{wizardStep(wizard, 'analysts')?.title}</h3>
            <p className="page-desc" style={{ margin: '0 0 0.75rem', fontSize: '0.85rem' }}>
              {wizardStep(wizard, 'analysts')?.description}
            </p>
            <span className="label">{wizard.prompts.analysts}</span>
            <div className="chip-group">
              {wizard.analyst_options.map((o) => (
                <label key={o.value} className="chip">
                  <input
                    type="checkbox"
                    checked={formData.analysts.includes(o.value)}
                    onChange={() => handleAnalystToggle(o.value)}
                  />
                  {o.label}
                </label>
              ))}
            </div>
          </div>

          <div className="form-group run-wizard-step">
            <h3 className="run-wizard-step-title">{wizardStep(wizard, 'research_depth')?.title}</h3>
            <p className="page-desc" style={{ margin: '0 0 0.75rem', fontSize: '0.85rem' }}>
              {wizardStep(wizard, 'research_depth')?.description}
            </p>
            <label htmlFor="research_depth">{wizard.prompts.research_depth}</label>
            <select
              id="research_depth"
              className="form-control"
              style={{ maxWidth: '100%' }}
              value={formData.research_depth}
              onChange={(e) =>
                setFormData({ ...formData, research_depth: Number(e.target.value) })
              }
            >
              {wizard.research_depth_options.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label} (rounds: {o.value})
                </option>
              ))}
            </select>
          </div>

          <div className="form-group run-wizard-step">
            <h3 className="run-wizard-step-title">{wizardStep(wizard, 'llm_provider')?.title}</h3>
            <p className="page-desc" style={{ margin: '0 0 0.75rem', fontSize: '0.85rem' }}>
              {wizardStep(wizard, 'llm_provider')?.description}
            </p>
            <span className="label">{wizard.prompts.llm_provider}</span>
            {!llmCatalogReady ? (
              <p className="page-desc" style={{ margin: 0, fontSize: '0.85rem' }}>
                Loading provider list…
              </p>
            ) : llmCatalog ? (
              <select
                className="form-control"
                style={{ maxWidth: '100%' }}
                value={formData.llm_provider.toLowerCase()}
                onChange={(e) => handleLlmProviderChange(e.target.value)}
              >
                {llmCatalog.providers.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.label}
                  </option>
                ))}
              </select>
            ) : (
              <input
                type="text"
                className="form-control"
                value={formData.llm_provider}
                onChange={(e) => setFormData({ ...formData, llm_provider: e.target.value })}
                placeholder="e.g. google, openai"
              />
            )}
          </div>

          <div className="form-group run-wizard-step">
            <span className="label">Backend URL</span>
            {llmCatalog && !customBackendUrl ? (
              <select
                className="form-control"
                style={{ maxWidth: '100%' }}
                value={formData.backend_url}
                onChange={(e) => {
                  const url = e.target.value
                  const p = llmCatalog.providers.find((x) => x.backend_url === url)
                  if (p) handleLlmProviderChange(p.id)
                  else setFormData((prev) => ({ ...prev, backend_url: url }))
                }}
              >
                {!catalogUrlsKnown && (
                  <option value={formData.backend_url}>
                    Current URL (not in catalog list)
                  </option>
                )}
                {llmCatalog.providers.map((p) => (
                  <option key={p.id} value={p.backend_url}>
                    {p.label} — {p.backend_url}
                  </option>
                ))}
              </select>
            ) : (
              <input
                id="backend_url"
                type="text"
                className="form-control"
                value={formData.backend_url}
                onChange={(e) => setFormData({ ...formData, backend_url: e.target.value })}
                placeholder="https://..."
              />
            )}
            <label className="run-analysis-check" style={{ marginTop: '0.5rem' }}>
              <input
                type="checkbox"
                checked={customBackendUrl}
                onChange={(e) => {
                  const on = e.target.checked
                  setCustomBackendUrl(on)
                  if (!on && currentLlmProvider) {
                    setFormData((prev) => ({ ...prev, backend_url: currentLlmProvider.backend_url }))
                  }
                }}
              />
              Custom backend URL (text)
            </label>
          </div>

          <div className="form-group run-wizard-step">
            <h3 className="run-wizard-step-title">{wizardStep(wizard, 'thinking_models')?.title}</h3>
            <p className="page-desc" style={{ margin: '0 0 0.75rem', fontSize: '0.85rem' }}>
              {wizardStep(wizard, 'thinking_models')?.description}
            </p>
            <div className="form-row-inline" style={{ marginBottom: '0.5rem', flexWrap: 'wrap', gap: '1rem' }}>
              <div className="form-field-inline" style={{ flex: '1 1 260px', minWidth: '220px' }}>
                <label htmlFor="shallow_thinker">{wizard.prompts.quick_thinker}</label>
                {currentLlmProvider ? (
                  <select
                    id="shallow_thinker"
                    className="form-control"
                    value={formData.shallow_thinker}
                    onChange={(e) => setFormData({ ...formData, shallow_thinker: e.target.value })}
                  >
                    {currentLlmProvider.shallow_models.map((m) => (
                      <option key={m.value} value={m.value}>
                        {m.label}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    id="shallow_thinker"
                    type="text"
                    className="form-control"
                    value={formData.shallow_thinker}
                    onChange={(e) => setFormData({ ...formData, shallow_thinker: e.target.value })}
                  />
                )}
              </div>
              <div className="form-field-inline" style={{ flex: '1 1 260px', minWidth: '220px' }}>
                <label htmlFor="deep_thinker">{wizard.prompts.deep_thinker}</label>
                {currentLlmProvider ? (
                  <select
                    id="deep_thinker"
                    className="form-control"
                    value={formData.deep_thinker}
                    onChange={(e) => setFormData({ ...formData, deep_thinker: e.target.value })}
                  >
                    {currentLlmProvider.deep_models.map((m) => (
                      <option key={m.value} value={m.value}>
                        {m.label}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    id="deep_thinker"
                    type="text"
                    className="form-control"
                    value={formData.deep_thinker}
                    onChange={(e) => setFormData({ ...formData, deep_thinker: e.target.value })}
                  />
                )}
              </div>
            </div>
          </div>

          {providerLower === 'google' && wizard.step7_by_provider.google && (
            <div className="form-group run-wizard-step">
              <h3 className="run-wizard-step-title">{wizard.step7_by_provider.google.title}</h3>
              <p className="page-desc" style={{ margin: '0 0 0.75rem', fontSize: '0.85rem' }}>
                {wizard.step7_by_provider.google.description}
              </p>
              <span className="label">{wizard.step7_by_provider.google.prompt}</span>
              <div className="chip-group" style={{ flexDirection: 'column', alignItems: 'flex-start' }}>
                {wizard.thinking_options_by_provider.google?.map((o) => (
                  <label key={o.value} className="run-analysis-check" style={{ display: 'flex', gap: '0.5rem' }}>
                    <input
                      type="radio"
                      name="google_thinking"
                      checked={googleThinkingLevel === o.value}
                      onChange={() => setGoogleThinkingLevel(o.value as 'high' | 'minimal')}
                    />
                    {o.label}
                  </label>
                ))}
              </div>
            </div>
          )}

          {providerLower === 'openai' && wizard.step7_by_provider.openai && (
            <div className="form-group run-wizard-step">
              <h3 className="run-wizard-step-title">{wizard.step7_by_provider.openai.title}</h3>
              <p className="page-desc" style={{ margin: '0 0 0.75rem', fontSize: '0.85rem' }}>
                {wizard.step7_by_provider.openai.description}
              </p>
              <label htmlFor="openai_effort">{wizard.step7_by_provider.openai.prompt}</label>
              <select
                id="openai_effort"
                className="form-control"
                style={{ maxWidth: '100%' }}
                value={openaiReasoningEffort}
                onChange={(e) => setOpenaiReasoningEffort(e.target.value)}
              >
                {wizard.thinking_options_by_provider.openai?.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
            </div>
          )}

          {providerLower === 'anthropic' && wizard.step7_by_provider.anthropic && (
            <div className="form-group run-wizard-step">
              <h3 className="run-wizard-step-title">{wizard.step7_by_provider.anthropic.title}</h3>
              <p className="page-desc" style={{ margin: '0 0 0.75rem', fontSize: '0.85rem' }}>
                {wizard.step7_by_provider.anthropic.description}
              </p>
              <label htmlFor="anthropic_effort">{wizard.step7_by_provider.anthropic.prompt}</label>
              <select
                id="anthropic_effort"
                className="form-control"
                style={{ maxWidth: '100%' }}
                value={anthropicEffort}
                onChange={(e) => setAnthropicEffort(e.target.value)}
              >
                {wizard.thinking_options_by_provider.anthropic?.map((o) => (
                  <option key={o.value} value={o.value}>
                    {o.label}
                  </option>
                ))}
              </select>
            </div>
          )}

          <button type="submit" className="btn" disabled={status === 'running'}>
            {status === 'running' ? (
              <>
                <Loader2 size={18} className="icon-spin" />
                Running…
              </>
            ) : (
              <>
                <Sparkles size={18} />
                Start analysis
              </>
            )}
          </button>
        </form>
      </div>

      {jobId && (
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Live progress</h2>
            {status === 'completed' && (
              <span className="badge badge-pos" style={{ textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                Complete
              </span>
            )}
            {status === 'error' && (
              <span className="badge badge-neg" style={{ textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                Error
              </span>
            )}
          </div>
          {status === 'running' && jobStartedAt != null && (
            <p
              className="page-desc"
              style={{ margin: '0 0 0.75rem', fontSize: '0.9rem', opacity: 0.9 }}
            >
              Job running · {formatDurationMs(now - jobStartedAt)} elapsed
              {lastEventAt != null ? (
                <> · last stream update {formatDurationMs(now - lastEventAt)} ago</>
              ) : null}
              . Long pauses after <code>Data</code> rows are normal while the LLM works (often minutes).
            </p>
          )}

          {status === 'completed' && savedReportId && (
            <p className="page-desc" style={{ margin: '0 0 0.75rem', fontSize: '0.9rem' }}>
              Report saved under <code>reports/{savedReportId}</code> (same layout as CLI).{' '}
              <Link to="/reports">View in Reports</Link>
            </p>
          )}
          {status === 'completed' && reportSaveError && (
            <p
              className="page-desc"
              style={{ margin: '0 0 0.75rem', fontSize: '0.9rem', color: 'var(--cli-err, #c44)' }}
            >
              Report was not saved: {reportSaveError}
            </p>
          )}

          <div className="run-analysis-summary">
            Agents: {agentsCompleted}/{agentsTotal} · Reports: {reportsCompleted}/{reportsTotal}
          </div>

          <div className="run-analysis-panels">
            <div className="run-analysis-panel">
              <h3 className="run-analysis-panel-title">Progress</h3>
              <div className="cli-table-wrap">
                <table className="cli-table">
                  <thead>
                    <tr>
                      <th>Team</th>
                      <th>Agent</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {progressRows.map((row, i) =>
                      row.sep ? (
                        <tr key={`sep-${i}`} className="cli-table-sep">
                          <td colSpan={3} />
                        </tr>
                      ) : (
                        <tr key={`${row.team}-${row.agent}-${i}`}>
                          <td>{row.team}</td>
                          <td>{row.agent}</td>
                          <td>
                            <span className={statusClass(row.status)}>
                              {row.status === 'in_progress' ? (
                                <span className="cli-spinner" aria-hidden />
                              ) : null}{' '}
                              {row.status}
                            </span>
                          </td>
                        </tr>
                      ),
                    )}
                  </tbody>
                </table>
              </div>
            </div>
            <div className="run-analysis-panel">
              <h3 className="run-analysis-panel-title">Messages &amp; Tools</h3>
              <div className="cli-table-wrap">
                <table className="cli-table cli-table-messages">
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>Type</th>
                      <th>Content</th>
                    </tr>
                  </thead>
                  <tbody>
                    {msgToolRows.length === 0 ? (
                      <tr>
                        <td colSpan={3} className="cli-table-empty">
                          Waiting for stream…
                        </td>
                      </tr>
                    ) : (
                      msgToolRows.map((r) => (
                        <tr key={r.id}>
                          <td className="cli-mono">{r.time}</td>
                          <td>{r.kind}</td>
                          <td className="cli-msg-content">{r.text}</td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {events.length > 0 && (
            <details className="run-analysis-raw" style={{ marginTop: '1rem' }}>
              <summary>Raw event log ({events.length})</summary>
              <div className="progress-panel" role="log" style={{ marginTop: '0.5rem' }}>
                {events.map((e, i) => (
                  <div key={i} className="progress-line">
                    {e.type === 'message' && (
                      <span className="progress-msg">
                        [{e.msg_type}] {String(e.content ?? '').slice(0, 500)}
                        {String(e.content ?? '').length > 500 ? '…' : ''}
                      </span>
                    )}
                    {e.type === 'tool_call' && (
                      <span className="progress-msg">
                        [Tool] {e.tool_name}
                        {e.args != null ? ` ${JSON.stringify(e.args)}` : ''}
                      </span>
                    )}
                    {e.type === 'agent_status' && (
                      <span className="progress-agent">
                        Agent {e.agent_name} → {e.status}
                      </span>
                    )}
                    {e.type === 'report_section' && (
                      <span className="progress-section">Section: {e.section_name}</span>
                    )}
                    {e.type === 'complete' && (
                      <span className="progress-done">Complete — decision: {e.decision}</span>
                    )}
                    {e.type === 'error' && <span className="progress-err">Error: {e.error}</span>}
                  </div>
                ))}
              </div>
            </details>
          )}
        </div>
      )}
    </div>
  )
}
