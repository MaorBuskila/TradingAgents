# TradingAgents — project flow

High-level architecture, agent workflow, and how the GUI/API path runs an analysis.

---

## 1. System overview

```mermaid
flowchart TB
  subgraph clients["Clients"]
    CLI["CLI (Typer)<br/>cli/main.py"]
    WEB["Frontend (React/Vite)<br/>frontend/"]
  end

  subgraph api_layer["API"]
    FAST["FastAPI<br/>api/main.py"]
    JOBS["In-memory jobs + SSE<br/>/api/jobs/{id}/events"]
  end

  subgraph core["Core engine"]
    RUN["AnalysisRunner<br/>tradingagents/services/"]
    TG["TradingAgentsGraph<br/>tradingagents/graph/"]
    GS["GraphSetup + LangGraph<br/>StateGraph AgentState"]
  end

  subgraph llm["LLM layer"]
    FAC["create_llm_client<br/>tradingagents/llm_clients/"]
    P1["OpenAI / Anthropic / Google"]
  end

  subgraph data["Data & persistence"]
    DF["Dataflows<br/>yfinance, Alpha Vantage, …"]
    DB[("SQLite<br/>portfolio, catalog")]
    FS["results/ · reports/<br/>markdown reports"]
  end

  CLI --> RUN
  WEB -->|HTTP :8000| FAST
  FAST -->|thread pool| RUN
  FAST --> JOBS
  RUN --> TG
  TG --> GS
  TG --> FAC
  FAC --> P1
  TG --> DF
  FAST --> DB
  FAST --> FS
  RUN -.->|writes via graph| FS
```

---

## 2. LangGraph agent pipeline

Analysts run **in user-selected order**, each with **tool loop** (analyst → tools → analyst) then message clear, then the next analyst. After the last analyst, fixed downstream teams run.

```mermaid
flowchart TB
  START([START]) --> CHAIN["Selected analysts in order<br/>each: Analyst ⇄ ToolNode → Msg Clear"]
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

**Implementation:** `tradingagents/graph/setup.py` (`setup_graph`), **orchestration class:** `tradingagents/graph/trading_graph.py`.

---

## 3. Analysis run (shared by CLI and API)

```mermaid
sequenceDiagram
  participant U as User / UI
  participant API as FastAPI
  participant T as Thread pool
  participant R as AnalysisRunner
  participant G as TradingAgentsGraph
  participant L as LLM + tools

  alt GUI path
    U->>API: POST /api/analyze
    API->>API: create job_id + asyncio.Queue
    API->>T: run_job()
    T->>R: run(selections, callbacks → queue)
  else CLI path
    U->>R: run() directly with Rich callbacks
  end

  R->>R: merge selections into DEFAULT_CONFIG
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

**GUI streaming:** `GET /api/jobs/{job_id}/events` returns **Server-Sent Events** until `complete` or `error`.

---

## 4. Frontend routes → API (reference)

| UI page            | Typical API usage                                      |
|-------------------|--------------------------------------------------------|
| Run Analysis      | `POST /api/analyze`, SSE job events                    |
| Reports           | `GET /api/reports`, `GET /api/reports/{id}`            |
| Portfolio         | `GET/POST/PUT/DELETE /api/portfolio/...`, refresh prices |
| Watchlist         | Catalog: `GET /api/catalog/...`, favorites             |
| YouTube summary   | `POST /api/youtube/summarize`                          |

---

## 5. Package map (short)

| Area | Role |
|------|------|
| `cli/` | Interactive Typer CLI, optional stats callbacks |
| `api/` | FastAPI app, SQLite init, job streaming |
| `tradingagents/graph/` | LangGraph wiring, propagation, reflection, signals |
| `tradingagents/agents/` | Analysts, researchers, trader, risk debators, portfolio manager |
| `tradingagents/dataflows/` | Market/news/fundamental data adapters |
| `tradingagents/llm_clients/` | Provider-specific LLM construction |
