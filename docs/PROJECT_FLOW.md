# TradingAgents — project flow

High-level architecture, agent workflow, how the GUI/API runs an analysis, and where the indicator labs live.

---

## 1. System overview

```mermaid
flowchart TB
  subgraph clients["Clients"]
    CLI["CLI (Typer)<br/>cli/main.py"]
    WEB["Frontend (React/Vite)<br/>frontend/src/"]
    TGAPP["telegram_bot/<br/>optional bot"]
  end

  subgraph api_layer["API layer (routes + schemas only)"]
    FAST["FastAPI<br/>api/main.py"]
    JOBS["In-memory jobs + SSE<br/>/api/jobs/{id}/events"]
    PF["Portfolio router<br/>api/portfolio_routes.py → /api/portfolio/*"]
    LAB["RSI/MACD/WFO/DT endpoints<br/>api/lab_routes.py → /api/rsi-* /api/macd-*"]
    CAT["Catalog + YouTube<br/>api/catalog_routes.py · api/youtube_routes.py"]
  end

  subgraph db_layer["Database (centralized)"]
    DBC["database/core/ — engine, session, base"]
    DBP["database/portfolio/ — positions, lots CRUD"]
    DBCAT["database/catalog/ — watchlist, tickers, favorites"]
    DBLC["database/lab_cache/ — RSI/MACD param caches"]
    DBJ["database/jobs/ — job history"]
    DBF[("data/*.db — SQLite files<br/>portfolio.db · rsi_cache.db · macd_cache.db")]
  end

  subgraph core["Core engine"]
    FLOW["run_cli_style_analysis<br/>tradingagents/services/analysis_flow.py"]
    RUN["AnalysisRunner<br/>tradingagents/services/analysis_runner.py"]
    TG["TradingAgentsGraph<br/>tradingagents/core_graph/trading_graph.py"]
    GS["GraphSetup + LangGraph<br/>tradingagents/core_graph/setup.py"]
  end

  subgraph llm["LLM layer"]
    FAC["create_llm_client<br/>tradingagents/llm_clients/factory.py"]
    CAT2["model_catalog.py"]
    P1["OpenAI / Anthropic / Google / xAI / OpenRouter / Ollama"]
  end

  subgraph ingestion["Data ingestion"]
    DF["tradingagents/dataflows/<br/>yfinance · alpha_vantage · news_scrapers<br/>reddit_sentiment · fear_greed"]
  end

  subgraph quant["Quant / ML"]
    OPT["quant_ml/optimizers/<br/>RSI+MACD algo+LLM+DT optimizers"]
    WFO["quant_ml/walk_forward/<br/>wfo_analyzer"]
    IND["quant_ml/indicators/<br/>stockstats_utils"]
    SIG["quant_ml/signals/<br/>rsi_signal · macd_signal"]
    LLI["quant_ml/llm_invoke.py"]
  end

  subgraph tools["Agent tools"]
    TLS["tradingagents/agents/utils/<br/>core_stock · fundamental · news · technical<br/>sentiment_tools · (ETF: etf_holdings, etf_sector_weights)"]
  end

  CLI --> FLOW
  TGAPP -.->|HTTP if used| FAST
  WEB -->|HTTP :8000| FAST
  FAST -->|thread pool| FLOW
  FAST --> JOBS
  FAST --> PF
  FAST --> LAB
  FAST --> CAT
  PF --> DBP
  CAT --> DBCAT
  LAB --> OPT
  LAB --> SIG
  LAB --> DBLC
  FLOW --> RUN
  RUN --> TG
  TG --> GS
  TG --> FAC
  FAC --> CAT2
  FAC --> P1
  TG --> DF
  TG --> TLS
  TLS --> DF
  OPT --> IND
  OPT --> WFO
  OPT --> LLI
  DBC --> DBF
  DBLC --> DBF
  DBP --> DBF
  FAST -.->|writes| FS["results/ · reports/"]
  RUN -.->|writes via graph| FS
```

---

## 2. LangGraph agent pipeline

The pipeline begins with the RSI Optimizer, then analysts run **in user-selected order**, each with a **tool loop** (analyst → tools → analyst) then message clear. After the last analyst, fixed downstream teams run.

```mermaid
flowchart TB
  START([START]) --> RSI["RSI Optimizer<br/>(walk-forward, cached to rsi_cache.db)"]
  RSI --> CHAIN["Selected analysts in order<br/>each: Analyst ⇄ ToolNode → Msg Clear"]
  CHAIN --> BR["Bull Researcher"]
  BR <-->|continue debate| BE["Bear Researcher"]
  BR --> RM["Research Manager"]
  BE --> RM
  RM --> TR["Trader"]
  TR --> AG["Aggressive Analyst"]
  AG -->|or end risk| PM["Portfolio Manager"]
  AG --> CO["Conservative Analyst"]
  CO -->|or end risk| PM
  CO --> NE["Neutral Analyst"]
  NE -->|or loop| AG
  NE -->|or end risk| PM
  PM --> END([END])
```

**Analyst tool sets:**

| Analyst | Tools | Notes |
|---------|-------|-------|
| Market (Technical) | `get_stock_data`, `get_indicators`, `get_rsi_signal` | RSI via cached WFO params — 1 call replaces 9 |
| Sentiment (Social) | `get_reddit_sentiment`, `get_market_fear_greed` | Reddit public API + alternative.me Fear & Greed |
| News | `get_news`, `get_global_news`, `get_insider_transactions` | Ticker-specific + macro news |
| Fundamentals | `get_fundamentals`, `get_balance_sheet`, `get_cashflow`, `get_income_statement`, `get_etf_holdings`, `get_etf_sector_weights` | ETF-aware: instrument context flags ETFs and directs agent to holdings tools |

**Implementation:** `tradingagents/graph/setup.py` (`GraphSetup`), **orchestration class:** `tradingagents/graph/trading_graph.py`.

---

## 3. Analysis run (shared by CLI and API)

```mermaid
sequenceDiagram
  participant U as User / UI
  participant API as FastAPI
  participant T as Thread pool
  participant F as analysis_flow
  participant R as AnalysisRunner
  participant G as TradingAgentsGraph
  participant L as LLM + tools

  alt GUI path
    U->>API: POST /api/analyze
    API->>API: create job_id + asyncio.Queue
    API->>T: run_job()
    T->>F: run_cli_style_analysis(callbacks → queue)
  else CLI path
    U->>F: run_cli_style_analysis / runner.run with Rich callbacks
  end

  F->>R: AnalysisRunner.run()
  R->>R: merge selections into DEFAULT_CONFIG (tradingagents/config/)
  R->>G: build graph + stream(init_state)
  loop stream chunks
    G->>L: agents + ToolNode
    L-->>G: state updates
    G-->>R: chunk
    R->>R: on_message / on_tool_call / on_agent_status / on_report_section
    opt API
      R->>API: queue.put(SSE events)
    end
  end
  R->>R: process_signal → decision
  R-->>U: final_state, decision (or SSE complete)
```

**Entry:** `tradingagents/services/analysis_flow.py` wraps `AnalysisRunner` so API and CLI stay aligned.

**GUI streaming:** `GET /api/jobs/{job_id}/events` returns **Server-Sent Events** until `complete` or `error`.

**Optional guard:** `POST /api/analyze` can require `X-TradingAgents-Bot-Secret` when `BOT_API_SECRET` is set.

---

## 4. Frontend routes → API (reference)

| UI page | Route | Typical API usage |
|---------|-------|-------------------|
| Run analysis | `/` | `POST /api/analyze`, SSE `GET /api/jobs/{id}/events`; defaults `GET /api/analysis-defaults`, models `GET /api/llm-catalog` |
| Reports | `/reports` | `GET /api/reports`, `GET /api/reports/{id}`, `GET /api/reports/by-ticker/{ticker}/latest-decision` |
| Portfolio | `/portfolio` | `GET/POST/PUT/PATCH/DELETE /api/portfolio/positions…`, lots under `…/lots`, `POST /api/portfolio/refresh-prices` |
| Watchlist | `/watchlist` | `GET /api/catalog/categories`, `GET /api/catalog/items`, favorites `POST /api/catalog/favorites/{ticker}`, `POST /api/catalog/items` |
| YouTube | `/youtube` | `POST /api/youtube/summarize`, list/detail/retranslate under `/api/youtube/summaries…` |
| RSI Lab | `/rsi-lab` | `POST /api/rsi-optimize`, signals `GET /api/rsi-signal/{ticker}` / `GET /api/rsi-signals`, cache `GET/DELETE /api/rsi-cache…`, DT `POST /api/rsi-dt-optimize`, `POST /api/rsi-dt-portfolio` |
| MACD Lab | `/macd-lab` | `POST /api/macd-optimize`, signals `GET /api/macd-signal/{ticker}` / `GET /api/macd-signals`, cache `GET/DELETE /api/macd-cache…`, `POST /api/macd-wfo-analyze`, DT `POST /api/macd-dt-optimize`, `POST /api/macd-dt-portfolio` |

**Frontend layout:** shared portfolio UI lives under `frontend/src/portfolio/` (`PortfolioPage`, `PositionDrawer`, `TickerAutocomplete`); page shells under `frontend/src/pages/`.

---

## 5. Repository layout (top level)

```
TradingAgents/
├── api/                        # FastAPI web layer — routes & Pydantic schemas only
│   ├── portfolio_routes.py
│   ├── lab_routes.py
│   ├── catalog_routes.py
│   └── youtube_routes.py
├── cli/                        # Typer CLI entrypoint
├── frontend/                   # React/Vite SPA
├── telegram_bot/               # Optional bot entrypoints
├── database/                   # CENTRALIZED DATABASE MODULE
│   ├── core/                   # engine.py, session maker, declarative base
│   ├── portfolio/              # positions, lots CRUD + repository
│   ├── catalog/                # watchlist, tickers, favorites + seed JSON
│   ├── lab_cache/              # RSI/MACD optimization param caches
│   └── jobs/                   # job history (SSE live state stays in-memory)
├── tradingagents/              # CORE APPLICATION LIBRARY
│   ├── agents/
│   │   ├── analysts/           # market_analyst, social_media_analyst, news_analyst, fundamentals_analyst
│   │   ├── risk_mgmt/          # aggressive, neutral, conservative debaters
│   │   └── utils/
│   │       ├── sentiment_tools.py      # get_reddit_sentiment, get_market_fear_greed (@tools)
│   │       ├── fundamental_data_tools.py  # + get_etf_holdings, get_etf_sector_weights
│   │       ├── technical_indicators_tools.py  # get_rsi_signal, get_macd_signal, get_macd_dt_signal
│   │       ├── news_data_tools.py
│   │       └── agent_utils.py          # build_instrument_context (ETF-aware)
│   ├── graph/                  # LangGraph state, node/edge wiring, signal processing
│   ├── dataflows/              # Pure data ingestion (agents only)
│   │   ├── y_finance.py        # + is_etf(), get_etf_holdings(), get_etf_sector_weights()
│   │   ├── reddit_sentiment.py # Reddit public JSON API (no auth)
│   │   ├── fear_greed.py       # CNN Fear & Greed via alternative.me
│   │   ├── alpha_vantage*.py
│   │   ├── yfinance_news.py
│   │   └── interface.py        # route_to_vendor() — single routing point for all vendors
│   ├── quant_ml/               # All ML & heavy math
│   │   ├── optimizers/         # RSI+MACD algo, LLM, DT optimizers
│   │   ├── walk_forward/       # WFO analyzer
│   │   ├── indicators/         # Pure stockstats math
│   │   └── signals/            # rsi_signal, macd_signal, sniper_signal (output layer)
│   ├── llm_clients/            # Provider factories + model_catalog
│   └── services/               # AnalysisRunner, analysis_flow, report_save
├── data/                       # Runtime SQLite files (gitignored)
│   ├── portfolio.db
│   ├── rsi_cache.db
│   └── macd_cache.db
├── tests/
│   └── test_sentiment_tools.py # 11 mock-based tests for Reddit + Fear & Greed dataflows
├── docs/
├── results/                    # Run outputs (gitignored)
├── reports/                    # Markdown reports (gitignored)
├── run_gui.sh
├── pyproject.toml
└── README.md
```

---

## 6. Package map

| Area | Role |
|------|------|
| `api/` | FastAPI routes + Pydantic schemas — **no DB logic** |
| `database/core/` | SQLAlchemy engine, session factory, declarative base |
| `database/portfolio/` | Portfolio tables, positions & lots CRUD |
| `database/catalog/` | Watchlist, ticker catalog, favorites CRUD + seed JSON |
| `database/lab_cache/` | RSI/MACD SQLite param cache read/write |
| `database/jobs/` | Job run history (not live SSE state) |
| `tradingagents/graph/` | LangGraph wiring, propagation, reflection, conditional logic, signal processing |
| `tradingagents/agents/analysts/` | Market (technical), Sentiment (Reddit+FG), News, Fundamentals (ETF-aware) analysts |
| `tradingagents/agents/utils/` | Agent-callable tools: stock, fundamental, news, technical, sentiment, ETF |
| `tradingagents/dataflows/` | Pure market/news/fundamentals/sentiment data ingestion — **all access via `interface.py`** |
| `tradingagents/dataflows/reddit_sentiment.py` | Reddit public API — r/wsb, r/stocks, r/options; no auth |
| `tradingagents/dataflows/fear_greed.py` | CNN Fear & Greed via alternative.me; no auth |
| `tradingagents/dataflows/y_finance.py` | yfinance wrapper + `is_etf()`, ETF holdings/sector tools |
| `tradingagents/quant_ml/` | RSI/MACD algo+LLM optimizers, WFO, decision-tree, signals, stockstats |
| `tradingagents/llm_clients/` | Provider-specific LLM factories + model catalog |
| `tradingagents/services/` | `analysis_flow`, `analysis_runner`, `report_save` |
