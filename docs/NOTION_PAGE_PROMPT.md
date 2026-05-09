# Notion Page Prompt — TradingAgents Framework

> Copy this entire document as the prompt to Notion AI ("Create page from prompt") or paste it directly as page content. Each `---` is a Notion divider. Use the H1/H2/H3 headings as-is — Notion renders them natively.

---

# 🤖 TradingAgents — LLM Multi-Agent Trading Framework

> **Purpose of this page:** Track all features, algorithms, research concepts, and development decisions for the TradingAgents project built on LangGraph + multi-provider LLMs.

**Status:** `Active Development`  
**Version:** `v0.2.2`  
**Owner:** Maor  
**Last Updated:** April 8, 2026

---

## 📌 Project Summary

TradingAgents is a **multi-agent LLM framework** that mirrors real-world trading firm dynamics. Specialized AI agents — analysts, researchers, a trader, risk managers, and a portfolio manager — collaborate through structured debates to produce final BUY / SELL / HOLD decisions.

Built on **LangGraph** (StateGraph) with streaming support via FastAPI SSE. Supports **OpenAI, Anthropic, Google, xAI, OpenRouter, and Ollama** as LLM backends.

**Entry points:**
- CLI → `tradingagents` or `python -m cli.main`
- GUI → React/Vite frontend + FastAPI on `:8000`
- Python API → `TradingAgentsGraph().propagate("NVDA", "2026-01-15")`

---

## 🏗️ Architecture Overview

| Layer | Component | Notes |
|---|---|---|
| Clients | CLI (Typer), React/Vite frontend | Both share the same AnalysisRunner |
| API | FastAPI + SSE job streaming | `/api/analyze` → `GET /api/jobs/{id}/events` |
| Core Engine | TradingAgentsGraph + LangGraph | `tradingagents/graph/` |
| Agents | 10 specialized agents | See Agent Pipeline below |
| LLM Layer | Factory pattern | `tradingagents/llm_clients/` |
| Data | yfinance, Alpha Vantage, SQLite | `tradingagents/dataflows/` |

---

## 🔁 Agent Pipeline

```
START
  └─► [Selected Analysts in order — each: Analyst ⇄ ToolNode → Msg Clear]
        └─► Bull Researcher ⇄ Bear Researcher (debate)
              └─► Research Manager
                    └─► Trader
                          └─► Aggressive / Conservative / Neutral Risk Debators (loop)
                                └─► Portfolio Manager
                                      └─► END (BUY / SELL / HOLD + size)
```

### Agent Roles

| Agent | File | Role |
|---|---|---|
| Fundamentals Analyst | `agents/analysts/fundamentals_analyst.py` | P/E, revenue, financials |
| Market Analyst | `agents/analysts/market_analyst.py` | Technical indicators (RSI, MACD…) |
| News Analyst | `agents/analysts/news_analyst.py` | Macro events, global news |
| Social Media Analyst | `agents/analysts/social_media_analyst.py` | Sentiment scoring |
| Bull Researcher | `agents/researchers/bull_researcher.py` | Argues for upside |
| Bear Researcher | `agents/researchers/bear_researcher.py` | Argues for downside |
| Research Manager | `agents/managers/research_manager.py` | Synthesizes debate |
| Trader | `agents/trader/trader.py` | Makes trade proposal |
| Risk Debators (x3) | `agents/risk_mgmt/` | Aggressive / Conservative / Neutral |
| Portfolio Manager | `agents/managers/portfolio_manager.py` | Final approve/reject |

---

## ✅ Feature Tracker

### Core Features

| Feature | Status | Notes |
|---|---|---|
| Multi-provider LLM support | ✅ Done | OpenAI, Anthropic, Google, xAI, OpenRouter, Ollama |
| LangGraph StateGraph pipeline | ✅ Done | Full streaming, conditional edges |
| CLI interactive interface | ✅ Done | Typer + Rich callbacks |
| React/FastAPI GUI | ✅ Done | SSE streaming, job tracking |
| yfinance data adapter | ✅ Done | Prices, indicators, news |
| Alpha Vantage data adapter | ✅ Done | Premium data source |
| SQLite portfolio persistence | ✅ Done | portfolio.db |
| Report generation (Markdown) | ✅ Done | `reports/` + `results/` |
| Telegram bot | ✅ Done | `telegram_bot/` |
| YouTube summarization endpoint | ✅ Done | `/api/youtube/summarize` |
| Five-tier rating scale | ✅ Done | v0.2.2 |
| Anthropic effort control | ✅ Done | `anthropic_effort: high/medium/low` |

### Planned / In Progress

| Feature | Status | Priority | Notes |
|---|---|---|---|
| RSI parameter optimization (dynamic period) | 🔄 Planned | High | See Algorithms section |
| MACD adaptive fast/slow EMA | 🔄 Planned | High | |
| Volatility-adjusted position sizing | 🔄 Planned | High | ATR-based |
| Backtesting harness | 🔄 Planned | Medium | |
| Walk-forward optimization | 🔄 Planned | Medium | |
| Regime detection (trending vs ranging) | 🔄 Planned | Medium | |
| Multi-timeframe analysis | 🔄 Planned | Low | |
| Options flow integration | 💡 Idea | Low | |

---

## 📈 Core Trading Concepts

### RSI — Relative Strength Index

**What it is:**
RSI measures the speed and magnitude of recent price changes to detect overbought/oversold conditions. It outputs a value 0–100.

**Formula:**
```
RS  = Average Gain over N periods / Average Loss over N periods
RSI = 100 − (100 / (1 + RS))
```

**Standard thresholds:**
- `> 70` → Overbought (potential sell signal)
- `< 30` → Oversold (potential buy signal)
- `50` → Momentum midline (trend confirmation)

**Standard period:** 14 (Wilder's original)

---

### How to Optimize RSI

**Problem with static RSI(14):** Markets shift between trending (RSI stays overbought/oversold) and ranging (RSI oscillates normally). Fixed parameters cause false signals.

**Optimization strategies:**

| Strategy | How | When to use |
|---|---|---|
| **Adaptive Period (ARSI)** | Adjust period with Kaufman's Efficiency Ratio: short period when trending, long when choppy | Always — replaces static 14 |
| **Dynamic thresholds** | Use percentile of RSI over trailing 252 days instead of fixed 70/30 | Stocks with structural bias |
| **Divergence detection** | Price makes new high but RSI doesn't → bearish divergence (strongest RSI signal) | Reversal setups |
| **Multi-timeframe RSI** | Daily RSI for direction, hourly RSI for entry timing | Swing trading |
| **RSI of RSI** | Apply RSI again on the RSI line to smooth noise | High-volatility assets |

**Code pattern for adaptive RSI to add to the agent:**
```python
def adaptive_rsi(close: pd.Series, min_period=5, max_period=25) -> pd.Series:
    """Kaufman Efficiency Ratio drives the RSI period."""
    direction = abs(close.diff(max_period))
    volatility = close.diff(1).abs().rolling(max_period).sum()
    er = direction / volatility.replace(0, np.nan)  # 0=choppy, 1=trending
    period = (min_period + (1 - er) * (max_period - min_period)).round().astype(int)
    return pd.Series(
        [ta.RSI(close[:i+1], timeperiod=int(period.iloc[i])).iloc[-1]
         for i in range(len(close))],
        index=close.index
    )
```

---

## 🧮 Algorithm Library for the Agent

### Currently Used

| Indicator | Tool | Parameters |
|---|---|---|
| RSI | `get_indicators(symbol, "rsi", date, lookback)` | Default 14 period |
| MACD | `get_indicators(symbol, "macd", date, lookback)` | 12/26/9 EMA |
| Others | Passed as string to `get_indicators` | yfinance / Alpha Vantage |

---

### Recommended Algorithms to Add

#### 1. ATR — Average True Range (Volatility)
Measures market volatility — essential for position sizing and stop-loss placement.
```
ATR(14) = EMA of max(High-Low, |High-PrevClose|, |Low-PrevClose|)
```
**Agent use:** Scale position size inversely to ATR. High ATR = smaller position.

---

#### 2. Bollinger Bands (Mean Reversion)
Price channel: 20-period SMA ± 2 standard deviations.
- Price touches upper band + RSI > 70 → strong sell
- Price touches lower band + RSI < 30 → strong buy
- Band squeeze (low bandwidth) → breakout imminent

---

#### 3. VWAP — Volume Weighted Average Price
Institutional benchmark. Price above VWAP = bullish bias. Below = bearish.
Best used intraday. Add as a filter: only take buy signals above VWAP.

---

#### 4. Stochastic RSI
RSI applied within a Stochastic formula — more sensitive than standard RSI. Good for short-term entries.
```
StochRSI = (RSI - RSI_min(14)) / (RSI_max(14) - RSI_min(14))
```
Thresholds: > 0.8 overbought, < 0.2 oversold.

---

#### 5. Ichimoku Cloud
All-in-one indicator: trend direction, support/resistance, momentum.
- Price above cloud → bullish
- Cloud twist ahead → trend change signal
- Chikou span crossing price → confirmation

---

#### 6. On-Balance Volume (OBV)
Cumulative volume pressure. OBV rising with flat price → accumulation (bullish divergence).
Lightweight signal to add to the sentiment analyst.

---

#### 7. Supertrend (Trending Markets)
ATR-based trend follower. Binary: above line = bullish, below = bearish.
Combine with RSI for filtered entries: Supertrend bullish + RSI < 60 → buy.

---

#### 8. Williams %R (Overbought/Oversold)
Similar to RSI but faster. Range: −100 to 0.
- > −20 → Overbought
- < −80 → Oversold

---

### Signal Combination Matrix (for Market Analyst agent)

| RSI | MACD | Bollinger | Supertrend | Signal Strength |
|---|---|---|---|---|
| < 30 | Bullish cross | Lower band touch | Bullish | 🟢 Strong Buy |
| < 40 | Bullish cross | — | Bullish | 🟢 Buy |
| 30–60 | — | — | Bullish | 🟡 Neutral-Bullish |
| > 60 | Bearish cross | — | Bearish | 🔴 Sell |
| > 70 | Bearish cross | Upper band touch | Bearish | 🔴 Strong Sell |

---

## ⚙️ Configuration Reference

```python
config = DEFAULT_CONFIG.copy()
config["llm_provider"]        = "anthropic"       # openai | anthropic | google | xai | openrouter | ollama
config["deep_think_llm"]      = "claude-opus-4-6" # Complex reasoning (researchers, portfolio manager)
config["quick_think_llm"]     = "claude-haiku-4-5-20251001" # Fast tasks (data fetching tools)
config["max_debate_rounds"]   = 2                 # Bull/Bear researcher debate rounds
config["max_risk_discuss_rounds"] = 2             # Risk team debate rounds
config["anthropic_effort"]    = "high"            # high | medium | low
config["data_vendors"] = {
    "core_stock_apis":      "yfinance",
    "technical_indicators": "yfinance",   # swap to alpha_vantage for premium
    "fundamental_data":     "yfinance",
    "news_data":            "yfinance",
}
```

---

## 📂 Key File Map

```
tradingagents/
├── graph/
│   ├── trading_graph.py       ← Main entry point (TradingAgentsGraph)
│   ├── setup.py               ← LangGraph wiring (setup_graph)
│   ├── propagation.py         ← .propagate() logic
│   ├── signal_processing.py   ← BUY/SELL/HOLD signal extraction
│   └── conditional_logic.py   ← Edge conditions (debate termination etc.)
├── agents/
│   ├── analysts/              ← 4 analyst agents
│   ├── researchers/           ← Bull + Bear
│   ├── managers/              ← Research Manager + Portfolio Manager
│   ├── risk_mgmt/             ← Aggressive / Conservative / Neutral
│   ├── trader/
│   └── utils/
│       ├── technical_indicators_tools.py  ← get_indicators() LangChain tool
│       └── agent_states.py    ← AgentState (LangGraph state schema)
├── dataflows/
│   ├── interface.py           ← route_to_vendor() dispatcher
│   ├── y_finance.py           ← yfinance adapter
│   └── alpha_vantage_indicator.py ← Alpha Vantage adapter
└── default_config.py          ← DEFAULT_CONFIG dict
```

---

## 🔗 Resources

- [arXiv Paper](https://arxiv.org/abs/2412.20138)
- [Trading-R1 Technical Report](https://arxiv.org/abs/2509.11420)
- [Tauric Research Community](https://tauric.ai/)
- [GitHub](https://github.com/TauricResearch/TradingAgents)
- [Demo Video](https://www.youtube.com/watch?v=90gr5lwjIho)

---

*Page generated: 2026-04-08*
