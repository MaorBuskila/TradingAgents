# TradingAgents Framework — Claude Context

## Project Purpose
Multi-agent LLM framework for equity trading analysis. Agents collaborate in a
LangGraph pipeline: Analysts → Researchers (bull/bear debate) → Trader →
Risk Management → Portfolio Manager → BUY/HOLD/SELL decision.
Research-only; not financial advice.

## Architecture Overview
- **Orchestration:** LangGraph (`tradingagents/graph/`) — stateful agent pipeline with conditional routing
- **Agents:** `tradingagents/agents/` — analysts, researchers, trader, risk mgmt, managers
- **Data:** `tradingagents/dataflows/` — yfinance (primary), Alpha Vantage (fallback), routed via `interface.py`
- **LLM Clients:** `tradingagents/llm_clients/factory.py` — OpenAI, Anthropic, Google, xAI, OpenRouter, Ollama
- **Quant:** `tradingagents/quant_ml/` — RSI/MACD walk-forward optimizers, Decision Tree ensembles
- **API:** FastAPI (`api/main.py`), SSE for streaming analysis progress
- **CLI:** Rich/Typer (`cli/main.py`)
- **DB:** SQLite via `database/` — portfolio, rsi_cache, macd_cache
- **Memory:** ChromaDB (`<project_dir>/chroma_db/`) — persistent semantic embeddings for agent memory (replaces BM25)
- **Bot:** Optional Telegram bot (`telegram_bot/`)
- **Frontend:** React/Vite (`frontend/`)

## Key Entry Points
- `TradingAgentsGraph.propagate(ticker, date)` — main analysis run
- `uvicorn api.main:app --host 127.0.0.1 --port 8000` — REST API server
- `tradingagents` — CLI command (installed via pyproject.toml)
- `tradingagents/services/analysis_runner.py` — `AnalysisRunner` class

## Agent Pipeline (in order)
1. **RSI Optimizer** — walk-forward optimize RSI params, cache to DB
2. **Market Analyst** — technical indicators (MACD, RSI, Bollinger, ATR)
3. **Social Media Analyst** — sentiment from news/social data
4. **News Analyst** — macro news impact
5. **Fundamentals Analyst** — P/E, debt, growth metrics
6. **Bull Researcher** — argues bullish case
7. **Bear Researcher** — argues bearish case
8. **Research Manager** — synthesizes debate → investment_plan
9. **Trader** — composes trader_investment_plan with position sizing
10. **Risk Team** (Aggressive/Neutral/Conservative) — risk debate
11. **Portfolio Manager** — final APPROVED/REJECTED + BUY/HOLD/SELL

## Agent State
Defined in `tradingagents/agents/utils/agent_states.py`:
- `AgentState` — main shared state (all reports, plans, final decision)
- `InvestDebateState` — bull/bear debate transcript + judge decision
- `RiskDebateState` — risk team debate transcript

## Configuration
- Default config: `tradingagents/default_config.py`
- API keys & secrets: `.env` (never commit, never read aloud)
- LLM model registry: `tradingagents/llm_catalog.py`
- Runtime overrides via `AnalysisRunner` merge DEFAULT_CONFIG with user selections

## Coding Rules
- **One method per code generation request** — generate a single focused method, not a full class rewrite
- All new agents follow the factory function pattern in `tradingagents/agents/`
- Keep `dataflows/interface.py` as the single routing point for data vendors — never call yfinance or Alpha Vantage directly from agents
- New optimizers must include walk-forward validation with out-of-sample Sharpe reporting
- Cache optimizer results to SQLite — never recompute if a valid cache hit exists (`rsi_cache.db`, `macd_cache.db`)
- No mocks in integration tests — use real data fixtures or live yfinance calls
- LangGraph state updates: always return partial dicts, never mutate state in-place
- LLM tool calls must be bound via `.bind_tools()` on the LLM client, not injected in prompts
- **GUI + Stop Button Requirement:** Every GUI feature that runs a long-running process (analysis, tests, optimizers) MUST include a stop/cancel button to halt execution
- **Test Stop Button Requirement:** Every new test that can run for extended periods MUST include stop button capability (async cancellation or timeout interrupt handler)
- **Result Persistence:** All test runs and analysis results MUST be saved to SQLite (`data/test_results.db` or appropriate DB) with: test_id, ticker, timestamp, status, results, error_log
- **Tab Switching State Preservation:** Long-running processes in the frontend MUST persist their state to DB on each progress update, enabling resumption if user switches tabs mid-execution

## Testing
```bash
pytest tests/ -v
```
Key test files:
- `tests/test_integration_oos.py` — optimizer out-of-sample validation
- `tests/test_ticker_symbol_handling.py` — exchange suffix parsing (.TO, .L, .HK, .T)
- `tests/test_analysis_selections.py` — analysis selection validation

## Output Locations
- Analysis logs: `eval_results/{ticker}/TradingAgentsStrategy_logs/`
- Optimizer cache: `data/rsi_cache.db`, `data/macd_cache.db`
- Portfolio state: `data/portfolio.db`
- Agent memory: `<project_dir>/chroma_db/` (ChromaDB persistent vector store)

## API Routes (FastAPI)
- `POST /api/analyze` — run full analysis (SSE streaming)
- `POST /api/rsi-optimize` — RSI parameter optimization
- `POST /api/macd-optimize` — MACD parameter optimization
- `POST /api/macd-dt-optimize` — MACD Decision Tree ensemble optimization
- `GET/POST /api/portfolio/*` — portfolio management
- `GET /api/catalog/*` — watchlist management
- `POST /api/youtube/summarize` — YouTube transcript summarization
- `POST /api/obsidian/export/{report_id}` — export report to Obsidian vault
- `POST /api/obsidian/import` — import vault notes into ChromaDB memory
- `GET /api/obsidian/status` — Obsidian integration status

## Quantitative Optimizers
- **RSI WFO:** 180-day train / 90-day OOS / retrain every 30 days; optimizes period + thresholds; algo or LLM-narrowed grid
- **MACD DT Ensemble:** Two-model ensemble (sliding WFO + fixed crash-aware); features: histogram slope, signal gap, volatility ratio, Bollinger %B, ATR; LONG only when both models agree (prob > 0.60)
- **Confidence scoring:** HIGH/MEDIUM/LOW based on OOS parameter stability

## LLM Provider Config
Set `llm_provider` in DEFAULT_CONFIG or at runtime:
- `openai` — gpt-5.x, supports `reasoning_effort`
- `anthropic` — claude-4.x, supports extended thinking + `effort`
- `google` — gemini-3.x, supports `thinking_level`
- `xai` — grok-4.x
- `openrouter` — multi-model routing
- `ollama` — local inference (default: gemma4:e4b)

## Tech Stack
Python 3.11+, LangGraph, LangChain, FastAPI, SQLite, ChromaDB, yfinance, stockstats,
scikit-learn, pandas, numpy, Rich, Typer, python-telegram-bot, uvicorn, Pydantic v2

## Relevant MCP Tools (connected)

- `notion` — trading journal, watchlist research notes

## Useful Skills (Cowork)
- `engineering:code-review` — review agent prompts, LangGraph nodes, optimizer logic
- `engineering:system-design` — design new agents or data vendor integrations
- `engineering:testing-strategy` — WFO coverage, agent regression tests
- `engineering:architecture` — ADRs for LLM provider changes, DB migrations
- `engineering:tech-debt` — audit duplicate optimizer patterns
- `xlsx` — export backtest/portfolio results to Excel
- `schedule` — schedule daily watchlist analysis runs

## Claude Code Skills — When to Invoke
- `/simplify` — after editing `graph/setup.py` or any analyst file (checks for over-complexity)
- `/review` — before merging any new agent or optimizer to main
- `/security-review` — after touching `api/`, `.env.*`, or `database.py`
- `/update-config` — to add hooks, permissions, or env vars to `.claude/settings.json`
- `/schedule` — to set up daily watchlist analysis cron jobs
- `/init` — if CLAUDE.md becomes stale after major refactors

## Readability & Documentation Standards

### Agent Node Functions — Required Docstring
Every `node(state)` function inside an agent factory MUST have a one-line docstring:
```python
def node(state: AgentState) -> dict:
    """Runs <role> analysis and returns {<key>: <value>} state update."""
```

### Prompt Templates — Extraction Rule
All LLM prompt strings longer than 20 lines MUST be extracted to a module-level constant
before the factory function:
```python
MARKET_ANALYST_PROMPT = """..."""

def create_market_analyst(llm, toolkit):
    ...
```

### Graph Node Naming Convention
Node IDs in `graph/setup.py` must exactly match the agent file and factory function name:
- File: `market_analyst.py` → Factory: `create_market_analyst()` → Node ID: `"market_analyst"`

### State Key Contract
When adding a new key to `AgentState` in `agent_states.py`, update it in THREE places:
1. TypedDict field definition (with type annotation + `Optional` if not always present)
2. `graph/propagation.py` — set the initial/default value
3. The "Agent State" section of this CLAUDE.md

### Complexity Budget
- No function in `graph/setup.py` may exceed 60 lines — split by concern:
  `_add_analyst_nodes()`, `_add_researcher_nodes()`, `_add_risk_nodes()`, `_add_edges()`
- No inline prompt string longer than 15 lines — extract to a module-level constant

### Dataflow Discipline
Never import yfinance, alpha_vantage, or any data vendor directly into agent files.
All data access must go through `tradingagents/dataflows/interface.py → route_to_vendor()`.

## Reading Order for New Developers
To understand the system, read files in this order:
1. `tradingagents/default_config.py` — what is configurable and why
2. `tradingagents/agents/utils/agent_states.py` — the shared data contract between all agents
3. `tradingagents/graph/trading_graph.py` — the public entry point
4. `tradingagents/graph/setup.py` — how the LangGraph is wired together
5. `tradingagents/agents/analysts/market_analyst.py` — canonical example of the agent pattern
6. `tradingagents/dataflows/interface.py` — how data vendors are routed and abstracted
