# MACD Decision Tree Optimization

## Overview

The MACD DT system is a two-model ensemble that uses Decision Trees to decide **when** to follow MACD signals. Rather than acting on every MACD crossover, it predicts 10-bar forward profitability from engineered features and only fires a LONG when both models agree with high confidence.

---

## Architecture

```
Symbol + Date
    │
    ├─ _load_ohlcv()          → 15 years OHLCV (yfinance, cached)
    ├─ _build_features()      → [macd_hist, macd_slope, signal_gap,
    │                             vol_ratio, bb_pct_b, atr14]
    ├─ _build_labels()        → binary: 1 if Close[t+10] > Close[t]
    │
    ├─ Model A: _run_sliding_wfo()
    │   └─ DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, balanced)
    │      IS=100 days sliding, retrains every 30 days
    │
    ├─ Model B: _train_fixed_model()
    │   └─ DecisionTreeClassifier(max_depth=3)
    │      Trained on 10+ years — encodes crash regimes (2008, 2020)
    │
    └─ _hybrid_signal(prob_a, prob_b, atr_pct)
        └─ LONG only when BOTH > 0.60
           size_mult = 1.0 if ATR_pct ≤ 80 else 0.5
```

---

## Files

| File | Role |
|------|------|
| `tradingagents/dataflows/macd_dt_optimizer.py` | Core DT optimizer — entry: `run_dt_optimizer()` |
| `tradingagents/dataflows/macd_dt_portfolio.py` | Portfolio layer — entry: `run_portfolio_optimizer()` |
| `tradingagents/dataflows/macd_signal.py` | Crossover signal pipeline — entry: `compute_macd_signal()` |
| `tradingagents/dataflows/macd_cache.py` | SQLite param cache (`data/macd_cache.db`) |
| `tradingagents/dataflows/macd_optimizer_algo.py` | Grid-search WFO optimizer — entry: `run_algo_optimizer()` |
| `tradingagents/dataflows/macd_optimizer_llm.py` | LLM-narrowed grid optimizer — entry: `run_llm_optimizer()` |
| `tradingagents/dataflows/wfo_analyzer.py` | WFO statistical significance — entry: `analyze_wfo()` |

---

## Feature Engineering (`_build_features`)

| Feature | Description |
|---------|-------------|
| `macd_hist` | MACD histogram (default 12-26-9) |
| `macd_slope` | 3-bar rate of change on histogram |
| `signal_gap` | MACD line − signal line |
| `vol_ratio` | 30-day avg volume / 60-day avg volume |
| `bb_pct_b` | Bollinger Band %B (price position within 2-std band) |
| `atr14` | ATR-14 (volatility normalizer) |

Labels: binary, `1` if `Close[t + label_horizon] > Close[t]`, else `0`.

---

## Model A — Sliding Window WFO

- **In-sample**: 100 trading days
- **Retrains every**: 30 days (OOS step)
- **Tree**: `max_depth=5`, `min_samples_leaf=10`, `class_weight='balanced'`
- **Signal threshold**: prob > 0.60 → signal = 1
- Captures current market regime / alpha

Each window produces a `SlideResult`:
```python
SlideResult(
    is_start, is_end, oos_start, oos_end,  # bar indices
    train_acc, oos_acc,
    oos_probs,    # prob(class=1) per bar
    oos_signals,  # 1 if prob > 0.60
)
```

---

## Model B — Fixed Crash-Aware Model

- Trained on full 10+ year history
- Shallower tree: `max_depth=3`
- Encodes long-term structural regimes (2008 GFC, 2020 COVID crash)
- Acts as a **veto layer** during regime extremes

---

## Hybrid Signal Logic (`_hybrid_signal`)

```python
def _hybrid_signal(prob_a, prob_b, atr_pct):
    signal = 1 if (prob_a > 0.60 and prob_b > 0.60) else 0
    size_mult = 0.5 if atr_pct > 80 else 1.0
    return signal, size_mult
```

Both models must agree. If ATR is above the 80th percentile (volatility spike), position size is halved.

---

## `run_dt_optimizer()` Return Dict

```python
{
    "symbol": str,
    "curr_date": str,
    "slides": [...],           # list of SlideResult dicts
    "last_signal": 0 | 1,
    "last_prob_a": float,      # Model A confidence
    "last_prob_b": float,      # Model B confidence
    "last_size_mult": 1.0 | 0.5,
    "last_atr_pct": float,
    "avg_oos_acc": float,
    "n_slides": int,
    "feature_names": ["macd_hist", "macd_slope", "signal_gap",
                      "vol_ratio", "bb_pct_b", "atr14"],
    "macd_params": {"fast": 12, "slow": 26, "signal": 9},
    "is_days": 100,
    "oos_days": 30,
    "label_horizon": 10,
    "min_prob_threshold": 0.60,
}
```

---

## Portfolio Layer (`macd_dt_portfolio.py`)

Entry: `run_portfolio_optimizer(symbols, curr_date, capital=100_000)`

**Flow:**
1. Filter by liquidity: 30-day avg dollar volume ≥ $5M
2. Run `run_dt_optimizer()` per symbol
3. Rank survivors by `last_prob_a` (Model A confidence)
4. Cap at 20 positions
5. Size: 5% base per position, 2.5% if `size_mult=0.5`

**Return dict** includes `positions`, `rejected_liquidity`, `rejected_no_signal`, `total_deployed_pct`, `total_deployed_usd`.

---

## MACD Signal Pipeline (`macd_signal.py`)

Entry: `compute_macd_signal(ticker, as_of_date, lookback_days=180)`

5-step pipeline:
1. Load params from `macd_cache.db` (fallback: 12/26/9)
2. Fetch 180-day OHLCV via yfinance
3. Compute MACD (fast EMA − slow EMA, signal = EMA(9))
4. Detect crossover → `"BUY"` | `"SELL"` | `"HOLD"`
5. Return full metadata + 30-bar MACD history

---

## Parameter Cache (`macd_cache.py`)

SQLite: `data/macd_cache.db`, table: `macd_params_cache`

Key columns: `ticker`, `optimal_fast/slow/signal`, `oos_sharpe`, `is_sharpe`, `confidence`, `regime`, `optimizer_provider`, `optimized_at`

API: `get_macd_params(ticker)`, `upsert_macd_params(ticker, ...)`, `list_all_cached_tickers()`, `delete_macd_params(ticker)`

---

## Algorithmic Optimizer (`macd_optimizer_algo.py`)

Entry: `run_algo_optimizer(symbol, curr_date, is_days=180, oos_days=90)`

Grid: fast 6–16, slow 18–34, signal 5–13 → ~713 combos (after `fast < slow` constraint)

Winner selection uses **Plateau Score** = IS Sharpe averaged over param + 26 neighbors (avoids single-row overfit outliers).

Confidence classification:
- `< 30 OOS trades` → LOW
- OOS Sharpe ≥ 0.5 and ratio ≥ 0.4 → HIGH
- OOS Sharpe ≥ 0.2 and ratio ≥ 0.2 → MEDIUM
- else → LOW

---

## LLM-Enhanced Optimizer (`macd_optimizer_llm.py`)

Entry: `run_llm_optimizer(symbol, curr_date, llm_provider=None, llm_model=None)`

**Step 1** — compute 9 regime signals: ATR volatility percentile, trend direction (SMA distance), MACD histogram state, recent 10-day return, etc.

**Step 2** — send signals to LLM (Claude/GPT/Gemini). LLM returns narrowed grid ranges (e.g., `fast_range: [6-10]`, `slow_range: [18-30]`).

**Step 3** — run `run_algo_optimizer()` on narrowed grid (~100-150 combos instead of 713).

Regime → grid heuristic (fallback if LLM unavailable):
- `trending_bull/bear` → longer periods
- `high_volatility` → shorter periods
- `mean_reverting` → standard 12/26 range

Returns all algo optimizer fields plus `regime`, `llm_reasoning`, `llm_grid`, `llm_available`, `regime_signals`, `token_usage`.

---

## WFO Analyzer (`wfo_analyzer.py`)

Entry: `analyze_wfo(is_days, oos_days, is_sharpe, oos_sharpe, oos_positions, bar_frequency) → WFOAnalysis`

Key metrics:
- **WFE** (Walk-Forward Efficiency) = OOS_Sharpe / IS_Sharpe: ≥0.70 = efficient, 0.30–0.70 = degraded, <0.30 = overfit
- **Sample risk**: HIGH (<30 trades), MEDIUM (30–59), LOW (≥60)
- **Confidence score** 0–100: sample size (40pts) + OOS Sharpe (30pts) + WFE (20pts) − penalties (10pts)
- **Confidence label**: Very High ≥80, High 60–79, Moderate 40–59, Low 20–39, Very Low <20

---

## Key Constants

| Constant | Value | Location |
|----------|-------|----------|
| IS_DAYS | 100 | `macd_dt_optimizer.py` |
| OOS_DAYS | 30 | `macd_dt_optimizer.py` |
| LABEL_HORIZON | 10 bars | `macd_dt_optimizer.py` |
| MIN_PROB | 0.60 | `macd_dt_optimizer.py` |
| ATR_VOL_PCT | 80th percentile | `macd_dt_optimizer.py` |
| MAX_TREE_DEPTH (sliding) | 5 | `macd_dt_optimizer.py` |
| MAX_TREE_DEPTH (fixed) | 3 | `macd_dt_optimizer.py` |
| BASE_ALLOC | 5% | `macd_dt_portfolio.py` |
| MAX_POSITIONS | 20 | `macd_dt_portfolio.py` |
| MIN_DOLLAR_VOL | $5M | `macd_dt_portfolio.py` |
| Grid combos | ~713 | `macd_optimizer_algo.py` |
| Transaction cost | 0.1% | `macd_optimizer_algo.py` |

---

## Integration

The DT optimizer is exposed as a LangChain `@tool` in `technical_indicators_tools.py`, returning `"LONG"` or `"NO POSITION"` for agent consumption. The Technical Analyst agent calls this tool alongside RSI and other indicators when building its report.
