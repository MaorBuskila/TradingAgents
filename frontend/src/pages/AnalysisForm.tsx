/**
 * AnalysisForm.tsx
 * Wizard form extracted from RunAnalysis. Handles all step inputs
 * and calls onSubmit with the fully assembled payload.
 */
import { Loader2, Sparkles } from 'lucide-react'

// ── Types (re-exported so RunAnalysis can import them) ─────────────────────────
export type LlmModelOption = { label: string; value: string }
export type LlmProviderEntry = {
  id: string
  label: string
  backend_url: string
  shallow_models: LlmModelOption[]
  deep_models: LlmModelOption[]
}

export type WizardStep = { id: string; title: string; description: string }
export type LabeledValue = { label: string; value: string }
export type LabeledInt = { label: string; value: number }

export type AnalysisWizardSpec = {
  steps: WizardStep[]
  step7_by_provider: Record<string, { title: string; description: string; prompt: string }>
  prompts: Record<string, string>
  ticker_examples: string
  research_depth_options: LabeledInt[]
  analyst_options: LabeledValue[]
  thinking_options_by_provider: Record<string, LabeledValue[]>
}

export type LlmCatalogPayload = {
  providers: LlmProviderEntry[]
  wizard?: AnalysisWizardSpec
}

export type FormDataState = {
  ticker: string
  analysis_date: string
  analysts: string[]
  research_depth: number
  llm_provider: string
  backend_url: string
  shallow_thinker: string
  deep_thinker: string
}

// ── Helpers ────────────────────────────────────────────────────────────────────
export function pickValidOrFirst(current: string, options: LlmModelOption[]): string {
  if (options.some((o) => o.value === current)) return current
  return options[0]?.value ?? current
}

function wizardStep(w: AnalysisWizardSpec, id: string): WizardStep | undefined {
  return w.steps.find((s) => s.id === id)
}

// ── Props ──────────────────────────────────────────────────────────────────────
interface AnalysisFormProps {
  formData: FormDataState
  setFormData: React.Dispatch<React.SetStateAction<FormDataState>>
  customBackendUrl: boolean
  setCustomBackendUrl: React.Dispatch<React.SetStateAction<boolean>>
  googleThinkingLevel: 'high' | 'minimal'
  setGoogleThinkingLevel: React.Dispatch<React.SetStateAction<'high' | 'minimal'>>
  openaiReasoningEffort: string
  setOpenaiReasoningEffort: React.Dispatch<React.SetStateAction<string>>
  anthropicEffort: string
  setAnthropicEffort: React.Dispatch<React.SetStateAction<string>>
  llmCatalog: LlmCatalogPayload | null
  llmCatalogReady: boolean
  wizard: AnalysisWizardSpec
  currentLlmProvider: LlmProviderEntry | undefined
  onLlmProviderChange: (providerId: string) => void
  onAnalystToggle: (analyst: string) => void
  onSubmit: (e: React.FormEvent) => void
  isRunning: boolean
}

export default function AnalysisForm({
  formData,
  setFormData,
  customBackendUrl,
  setCustomBackendUrl,
  googleThinkingLevel,
  setGoogleThinkingLevel,
  openaiReasoningEffort,
  setOpenaiReasoningEffort,
  anthropicEffort,
  setAnthropicEffort,
  llmCatalog,
  llmCatalogReady,
  wizard,
  currentLlmProvider,
  onLlmProviderChange,
  onAnalystToggle,
  onSubmit,
  isRunning,
}: AnalysisFormProps) {
  const providerLower = formData.llm_provider.toLowerCase()
  const catalogUrlsKnown =
    llmCatalog?.providers.some((p) => p.backend_url === formData.backend_url) ?? false

  return (
    <form onSubmit={onSubmit}>
      {/* Step 1 — Ticker */}
      <div className="form-group run-wizard-step">
        <h3 className="run-wizard-step-title">{wizardStep(wizard, 'ticker')?.title}</h3>
        <p className="page-desc" style={{ margin: '0 0 0.75rem', fontSize: 'var(--text-sm)' }}>
          {wizardStep(wizard, 'ticker')?.description}
        </p>
        <label htmlFor="ticker">{wizard.prompts.ticker}</label>
        <p style={{ margin: '0.1rem 0 0.3rem', fontSize: 'var(--text-xs)', color: 'var(--text-muted)' }}>
          {wizard.ticker_examples}
        </p>
        <input
          id="ticker"
          type="text"
          className="form-control"
          style={{ maxWidth: '280px' }}
          value={formData.ticker}
          onChange={(e) => setFormData((p) => ({ ...p, ticker: e.target.value }))}
        />
      </div>

      {/* Step 2 — Date */}
      <div className="form-group run-wizard-step">
        <h3 className="run-wizard-step-title">{wizardStep(wizard, 'analysis_date')?.title}</h3>
        <p className="page-desc" style={{ margin: '0 0 0.75rem', fontSize: 'var(--text-sm)' }}>
          {wizardStep(wizard, 'analysis_date')?.description}
        </p>
        <label htmlFor="analysis_date">YYYY-MM-DD</label>
        <input
          id="analysis_date"
          type="date"
          className="form-control"
          style={{ maxWidth: '220px' }}
          value={formData.analysis_date}
          onChange={(e) => setFormData((p) => ({ ...p, analysis_date: e.target.value }))}
        />
      </div>

      {/* Step 3 — Analysts */}
      <div className="form-group run-wizard-step">
        <h3 className="run-wizard-step-title">{wizardStep(wizard, 'analysts')?.title}</h3>
        <p className="page-desc" style={{ margin: '0 0 0.75rem', fontSize: 'var(--text-sm)' }}>
          {wizardStep(wizard, 'analysts')?.description}
        </p>
        <span className="label">{wizard.prompts.analysts}</span>
        <div className="chip-group">
          {wizard.analyst_options.map((o) => (
            <label key={o.value} className="chip">
              <input
                type="checkbox"
                checked={formData.analysts.includes(o.value)}
                onChange={() => onAnalystToggle(o.value)}
              />
              {o.label}
            </label>
          ))}
        </div>
      </div>

      {/* Step 4 — Research depth */}
      <div className="form-group run-wizard-step">
        <h3 className="run-wizard-step-title">{wizardStep(wizard, 'research_depth')?.title}</h3>
        <p className="page-desc" style={{ margin: '0 0 0.75rem', fontSize: 'var(--text-sm)' }}>
          {wizardStep(wizard, 'research_depth')?.description}
        </p>
        <label htmlFor="research_depth">{wizard.prompts.research_depth}</label>
        <select
          id="research_depth"
          className="form-control"
          value={formData.research_depth}
          onChange={(e) => setFormData((p) => ({ ...p, research_depth: Number(e.target.value) }))}
        >
          {wizard.research_depth_options.map((o) => (
            <option key={o.value} value={o.value}>
              {o.label} (rounds: {o.value})
            </option>
          ))}
        </select>
      </div>

      {/* Step 5 — LLM provider */}
      <div className="form-group run-wizard-step">
        <h3 className="run-wizard-step-title">{wizardStep(wizard, 'llm_provider')?.title}</h3>
        <p className="page-desc" style={{ margin: '0 0 0.75rem', fontSize: 'var(--text-sm)' }}>
          {wizardStep(wizard, 'llm_provider')?.description}
        </p>
        <span className="label">{wizard.prompts.llm_provider}</span>
        {!llmCatalogReady ? (
          <p className="page-desc" style={{ margin: 0, fontSize: 'var(--text-sm)' }}>Loading provider list…</p>
        ) : llmCatalog ? (
          <select
            className="form-control"
            value={formData.llm_provider.toLowerCase()}
            onChange={(e) => onLlmProviderChange(e.target.value)}
          >
            {llmCatalog.providers.map((p) => (
              <option key={p.id} value={p.id}>{p.label}</option>
            ))}
          </select>
        ) : (
          <input
            type="text"
            className="form-control"
            value={formData.llm_provider}
            onChange={(e) => setFormData((p) => ({ ...p, llm_provider: e.target.value }))}
            placeholder="e.g. google, openai"
          />
        )}
      </div>

      {/* Backend URL */}
      <div className="form-group run-wizard-step">
        <span className="label">Backend URL</span>
        {llmCatalog && !customBackendUrl ? (
          <select
            className="form-control"
            value={formData.backend_url}
            onChange={(e) => {
              const url = e.target.value
              const p = llmCatalog.providers.find((x) => x.backend_url === url)
              if (p) onLlmProviderChange(p.id)
              else setFormData((prev) => ({ ...prev, backend_url: url }))
            }}
          >
            {!catalogUrlsKnown && (
              <option value={formData.backend_url}>Current URL (not in catalog)</option>
            )}
            {llmCatalog.providers.map((p) => (
              <option key={p.id} value={p.backend_url}>
                {p.label}{p.backend_url ? ` — ${p.backend_url}` : ' — (SDK default)'}
              </option>
            ))}
          </select>
        ) : (
          <input
            type="text"
            className="form-control"
            value={formData.backend_url}
            onChange={(e) => setFormData((p) => ({ ...p, backend_url: e.target.value }))}
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
          Custom backend URL (text input)
        </label>
      </div>

      {/* Step 6 — Thinking models */}
      <div className="form-group run-wizard-step">
        <h3 className="run-wizard-step-title">{wizardStep(wizard, 'thinking_models')?.title}</h3>
        <p className="page-desc" style={{ margin: '0 0 0.75rem', fontSize: 'var(--text-sm)' }}>
          {wizardStep(wizard, 'thinking_models')?.description}
        </p>
        <div className="form-row-inline" style={{ gap: '1rem' }}>
          <div className="form-field-inline" style={{ flex: '1 1 260px', minWidth: '220px' }}>
            <label htmlFor="shallow_thinker">{wizard.prompts.quick_thinker}</label>
            {currentLlmProvider ? (
              <select
                id="shallow_thinker"
                className="form-control"
                value={formData.shallow_thinker}
                onChange={(e) => setFormData((p) => ({ ...p, shallow_thinker: e.target.value }))}
              >
                {currentLlmProvider.shallow_models.map((m) => (
                  <option key={m.value} value={m.value}>{m.label}</option>
                ))}
              </select>
            ) : (
              <input
                id="shallow_thinker"
                type="text"
                className="form-control"
                value={formData.shallow_thinker}
                onChange={(e) => setFormData((p) => ({ ...p, shallow_thinker: e.target.value }))}
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
                onChange={(e) => setFormData((p) => ({ ...p, deep_thinker: e.target.value }))}
              >
                {currentLlmProvider.deep_models.map((m) => (
                  <option key={m.value} value={m.value}>{m.label}</option>
                ))}
              </select>
            ) : (
              <input
                id="deep_thinker"
                type="text"
                className="form-control"
                value={formData.deep_thinker}
                onChange={(e) => setFormData((p) => ({ ...p, deep_thinker: e.target.value }))}
              />
            )}
          </div>
        </div>
      </div>

      {/* Step 7 — Provider-specific thinking config */}
      {providerLower === 'google' && wizard.step7_by_provider.google && (
        <div className="form-group run-wizard-step">
          <h3 className="run-wizard-step-title">{wizard.step7_by_provider.google.title}</h3>
          <p className="page-desc" style={{ margin: '0 0 0.75rem', fontSize: 'var(--text-sm)' }}>
            {wizard.step7_by_provider.google.description}
          </p>
          <span className="label">{wizard.step7_by_provider.google.prompt}</span>
          <div className="chip-group" style={{ flexDirection: 'column', alignItems: 'flex-start' }}>
            {wizard.thinking_options_by_provider.google?.map((o) => (
              <label key={o.value} className="run-analysis-check" style={{ gap: '0.5rem' }}>
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
          <p className="page-desc" style={{ margin: '0 0 0.75rem', fontSize: 'var(--text-sm)' }}>
            {wizard.step7_by_provider.openai.description}
          </p>
          <label htmlFor="openai_effort">{wizard.step7_by_provider.openai.prompt}</label>
          <select
            id="openai_effort"
            className="form-control"
            value={openaiReasoningEffort}
            onChange={(e) => setOpenaiReasoningEffort(e.target.value)}
          >
            {wizard.thinking_options_by_provider.openai?.map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </div>
      )}

      {providerLower === 'anthropic' && wizard.step7_by_provider.anthropic && (
        <div className="form-group run-wizard-step">
          <h3 className="run-wizard-step-title">{wizard.step7_by_provider.anthropic.title}</h3>
          <p className="page-desc" style={{ margin: '0 0 0.75rem', fontSize: 'var(--text-sm)' }}>
            {wizard.step7_by_provider.anthropic.description}
          </p>
          <label htmlFor="anthropic_effort">{wizard.step7_by_provider.anthropic.prompt}</label>
          <select
            id="anthropic_effort"
            className="form-control"
            value={anthropicEffort}
            onChange={(e) => setAnthropicEffort(e.target.value)}
          >
            {wizard.thinking_options_by_provider.anthropic?.map((o) => (
              <option key={o.value} value={o.value}>{o.label}</option>
            ))}
          </select>
        </div>
      )}

      <div className="run-action-row">
        <button type="submit" className="btn" disabled={isRunning}>
          {isRunning ? (
            <><Loader2 size={17} className="icon-spin" /> Running…</>
          ) : (
            <><Sparkles size={17} /> Start analysis</>
          )}
        </button>
      </div>
    </form>
  )
}
