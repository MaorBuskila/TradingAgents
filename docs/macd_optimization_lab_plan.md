# MACD Optimization Lab — Implementation Plan
> Mirrors the RSI Lab (`rsi_cache` → `rsi_optimizer_algo` → `rsi_optimizer_llm` → `rsi_signal`)

---

## Overview

The MACD lab applies the exact same Walk-Forward Optimization (WFO) + LLM regime-narrowing
architecture used for RSI, adapted to MACD's 3-parameter space.

**RSI → MACD mapping:**

| RSI concept       | MACD equivalent                          |
|-------------------|------------------------------------------|
| `period`          | `fast_period`, `slow_period`, `signal_period` |
| `overbought`      | signal crossover threshold (histogram)   |
| `oversold`        | signal crossover threshold (histogram)   |
| RSI-14/70/30      | MACD 12/26/9 (industry default baseline) |
| Wilder's smoothing| Standard EMA                             |

---

## Files to Create

```
tradingagents/dataflows/
├── macd_cache.py           ← SQLite cache (mirrors rsi_cache.py)
├── macd_optimizer_algo.py  ← Pure algo WFO (mirrors rsi_optimizer_algo.py)
├── macd_optimizer_llm.py   ← LLM-enhanced optimizer (mirrors rsi_optimizer_llm.py)
└── macd_signal.py          ← Full signal flow (mirrors rsi_signal.py)
```

---

## 1. `macd_cache.py`

**Purpose:** Persistent SQLite cache for optimized MACD params per ticker.

**DB path:** `<project_root>/macd_cache.db`  (separate from `rsi_cache.db` and `portfolio.db`)

**Table schema:**

```sql
CREATE TABLE IF NOT EXISTS macd_params_cache (
    ticker              TEXT PRIMARY KEY,
    optimal_fast        INTEGER NOT NULL,   -- e.g. 12
    optimal_slow        INTEGER NOT NULL,   -- e.g. 26
    optimal_signal      INTEGER NOT NULL,   -- e.g. 9
    oos_sharpe          REAL    NOT NULL,
    is_sharpe           REAL    NOT NULL,
    confidence          TEXT    NOT NULL,   -- HIGH / MEDIUM / LOW
    regime              TEXT,
    training_days       INTEGER,
    test_days           INTEGER,
    optimizer_provider  TEXT,               -- "algo" | "llm" | "heuristic"
    model_used          TEXT,
    optimized_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

**Public API (mirrors rsi_cache.py):**
- `get_macd_params(ticker)` → dict | None
- `upsert_macd_params(ticker, fast, slow, signal, oos_sharpe, is_sharpe, confidence, ...)`
- `list_all_cached_tickers()` → list
- `delete_macd_params(ticker)` → bool

---

## 2. `macd_optimizer_algo.py`

**Purpose:** Pure algorithmic WFO — no LLM. Finds optimal (fast, slow, signal) via grid search + IS/OOS validation.

### MACD Calculation

```python
def _calc_macd(closes, fast, slow, signal):
    ema_fast   = closes.ewm(span=fast,   adjust=False).mean()
    ema_slow   = closes.ewm(span=slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram
```

### Signal (Position) Generation

**Signal line crossover strategy** (primary — most tradeable):
```
histogram goes from negative → positive  →  BUY  (+1, go long)
histogram goes from positive → negative  →  SELL (-1, go short)
hold current position otherwise
```

```python
def _generate_positions(histogram):
    positions = []
    pos = 0
    prev_hist = 0.0
    for hist in histogram:
        if np.isnan(hist):
            positions.append(0); continue
        if prev_hist <= 0 and hist > 0:   # bullish crossover
            pos = 1
        elif prev_hist >= 0 and hist < 0: # bearish crossover
            pos = -1
        prev_hist = hist
        positions.append(pos)
    return pd.Series(positions, index=histogram.index)
```

### Grid Search Parameters

| Parameter      | Range          | Step | Values |
|----------------|----------------|------|--------|
| `fast_period`  | 6 → 16         | 1    | 11     |
| `slow_period`  | 18 → 34        | 2    | 9      |
| `signal_period`| 5 → 13         | 1    | 9      |
| Constraint     | fast < slow    | —    | —      |

Total valid combos: ~11 × 9 × 9 ≈ **891** (minus fast >= slow, ~20%)  → ~713 combos

### Walk-Forward Validation (identical to RSI)

```
|---- IS window (180 trading days) ----|---- OOS window (90 trading days) ----|
  grid search → pick best params           validate on unseen data
```

Scoring: **Sharpe ratio** (identical `_calc_sharpe()` reused from RSI algo)

### Baseline

Default MACD(12, 26, 9) — industry standard, used as comparison.

### Confidence scoring (same logic as RSI)

```
OOS Sharpe >= 0.5 AND ratio >= 0.4  →  HIGH
OOS Sharpe >= 0.2 AND ratio >= 0.2  →  MEDIUM
else                                 →  LOW
```

### Return dict

```python
{
    "optimal_fast":    int,
    "optimal_slow":    int,
    "optimal_signal":  int,
    "is_sharpe":       float,
    "oos_sharpe":      float,
    "confidence":      str,
    "combos_tested":   int,
    "is_days":         int,
    "oos_days":        int,
    "default_is_sharpe":  float,   # MACD 12/26/9 baseline
    "default_oos_sharpe": float,
    "param_sharpes":   list[dict], # {fast, slow, signal, is_sharpe, oos_sharpe}
}
```

---

## 3. `macd_optimizer_llm.py`

**Purpose:** LLM-enhanced optimizer — regime classification narrows grid before WFO runs.

### Regime Signals (pre-computed, MACD-specific)

Mirrors RSI's 4 signals but replaces `rsi14_current` with MACD-specific signals:

| Signal                      | Description                                      |
|-----------------------------|--------------------------------------------------|
| `volatility_percentile`     | ATR-14 vs 252-day history (0–100)                |
| `trend_direction`           | "up" / "down" / "flat" (50-SMA distance)         |
| `trend_strength_pct`        | % distance from 50-day SMA                       |
| `price_above_sma50`         | bool                                             |
| `price_above_sma200`        | bool                                             |
| `macd_histogram_current`    | Current default MACD(12,26,9) histogram value    |
| `histogram_direction`       | "expanding_bullish" / "expanding_bearish" / "contracting" |
| `macd_signal_spread_pct`    | (macd_line - signal_line) / price × 100          |
| `recent_10d_return_pct`     | 10-day price return %                            |

### LLM Prompt — Grid Narrowing

```
You are an expert quant. Narrow the MACD parameter search grid for {symbol}.

Regime signals:
  Volatility Percentile:    {vol_pct}%
  Trend Direction:          {trend_dir}
  Trend Strength (50-SMA):  {trend_pct}%
  MACD Histogram (12/26/9): {histogram_val}
  Histogram Direction:      {hist_dir}
  MACD/Signal Spread:       {spread_pct}%
  Recent 10d Return:        {return_10d}%

MACD optimization logic:
- High volatility  → SHORTER fast/slow periods (faster EMA response to price swings)
- Strong trend     → LONGER periods (avoid premature crossover signals)
- Ranging/choppy   → STANDARD periods, tighten signal_period for sensitivity
- Histogram already large positive → consider wider search in bearish range

Respond ONLY with valid JSON:
{
  "regime": "trending_bull|trending_bear|mean_reverting|high_volatility|choppy",
  "fast_range":   [min, max],    // integers 6-16
  "slow_range":   [min, max],    // integers 18-34
  "signal_range": [min, max],    // integers 5-13
  "reasoning": "2-3 sentences"
}
```

### Heuristic Fallback (no LLM)

```python
if vol > 70:           # high volatility
    fast_range   = [6, 10]
    slow_range   = [18, 24]
    signal_range = [5, 9]
    regime = "high_volatility"

elif abs(trend_pct) > 5:  # strong trend
    fast_range   = [10, 16]
    slow_range   = [24, 34]
    signal_range = [7, 13]
    regime = "trending_bull" or "trending_bear"

else:                  # ranging / mean-reverting
    fast_range   = [8, 14]
    slow_range   = [20, 30]
    signal_range = [6, 11]
    regime = "mean_reverting"
```

### Return dict (extends algo output)

```python
{
    # all fields from algo optimizer, PLUS:
    "regime":          str,
    "llm_reasoning":   str,
    "llm_available":   bool,
    "llm_grid": {
        "fast_range":   [int, int],
        "slow_range":   [int, int],
        "signal_range": [int, int],
    },
    "regime_signals":  dict,
    "debug_logs":      list[str],
    "token_usage":     dict,
}
```

---

## 4. `macd_signal.py`

**Purpose:** Full signal flow — OPTIMIZE → GET PARAMS → CALC MACD → SIGNAL → ACT

### Pipeline (mirrors rsi_signal.py)

```
STEP 1: GET PARAMS   — load (fast, slow, signal) from macd_cache.db
                       fallback: 12 / 26 / 9
STEP 2: FETCH DATA   — yfinance (same cache as optimizer)
STEP 3: CALC MACD    — macd_line, signal_line, histogram
STEP 4: SIGNAL       — crossover detection → BUY / SELL / HOLD
STEP 5: CONTEXT      — return full metadata for UI / agents
```

### Signal Classification

```python
def _classify_signal(histogram_series):
    """Detect crossover on the most recent bar."""
    if len(histogram_series) < 2:
        return "HOLD"
    prev = histogram_series.iloc[-2]
    curr = histogram_series.iloc[-1]
    if prev <= 0 and curr > 0:
        return "BUY"   # bullish crossover
    elif prev >= 0 and curr < 0:
        return "SELL"  # bearish crossover
    return "HOLD"
```

### Return dict

```python
{
    "ticker":             str,
    "as_of_date":         str,
    "action":             "BUY" | "SELL" | "HOLD",
    # Current MACD values
    "macd_line":          float,
    "signal_line":        float,
    "histogram":          float,
    "histogram_direction":"expanding_bullish" | "expanding_bearish" | "contracting",
    # Optimal params used
    "fast_period":        int,
    "slow_period":        int,
    "signal_period":      int,
    # Cache metadata
    "using_cached":       bool,
    "confidence":         str,   # HIGH / MEDIUM / LOW / DEFAULT
    "regime":             str,
    "optimizer_provider": str,
    "optimized_at":       str,
    # Price context
    "latest_close":       float,
    "price_change_pct":   float,
    # History for UI chart (last 30 bars)
    "macd_history": [
        {"date": str, "macd": float, "signal": float, "histogram": float, "close": float}
    ],
    "error":              str | None,
}
```

---

## Integration Points

### Into existing framework

1. **`tradingagents/dataflows/__init__.py`** — export `compute_macd_signal`, `run_algo_optimizer` (macd), `run_llm_optimizer` (macd)

2. **`tradingagents/agents/utils/technical_indicators_tools.py`** — add `get_macd_signal_tool()` alongside RSI tool

3. **API endpoint (optional)** — `api/main.py` add `/macd/signal/{ticker}` and `/macd/optimize/{ticker}`

4. **Frontend** — MACD chart card mirrors RSI card: histogram bar chart + MACD/signal line overlay

---

## Key Differences vs RSI Lab

| Aspect              | RSI Lab                   | MACD Lab                              |
|---------------------|---------------------------|---------------------------------------|
| Parameters          | period, upper, lower      | fast, slow, signal                    |
| Signal logic        | threshold crossing         | line crossover (histogram sign flip)  |
| Signal values       | 0–100 (bounded)           | unbounded (price-relative)            |
| Default baseline    | 14 / 70 / 30              | 12 / 26 / 9                           |
| Grid combos (full)  | ~972                      | ~713 (valid fast < slow)              |
| Chart data          | rsi_series                | macd_line + signal_line + histogram   |
| LLM regime signal   | rsi14_current             | histogram_current + direction + spread|
| Warmup bars needed  | period + 1                | slow_period + signal_period (≈ 39)    |

---

## Implementation Order

1. `macd_cache.py` — DB schema + CRUD (30 min)
2. `macd_optimizer_algo.py` — `_calc_macd`, `_generate_positions`, `run_algo_optimizer` (1 hr)
3. `macd_optimizer_llm.py` — regime signals, LLM prompt, `run_llm_optimizer` (1 hr)
4. `macd_signal.py` — full signal flow + `compute_macd_signal` (45 min)
5. Wire into `__init__.py` + `technical_indicators_tools.py` (15 min)
6. Test with: AAPL, NVDA, BTC-USD — compare default 12/26/9 vs optimized
