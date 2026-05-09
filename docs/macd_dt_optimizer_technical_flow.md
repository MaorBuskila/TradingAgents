# MACD DT Optimizer — Full Technical Flow

> **File:** `tradingagents/dataflows/macd_dt_optimizer.py`  
> **Portfolio wrapper:** `tradingagents/dataflows/macd_dt_portfolio.py`  
> **Entry point:** `run_dt_optimizer(symbol, curr_date, is_days=100, oos_days=30)`

---

## 1. System Overview

The MACD DT Optimizer layers a **Decision Tree (DT) classifier** on top of raw MACD histogram crossover signals. The core insight is that not every MACD crossover is equal — market regime, momentum strength, and volume context determine whether a signal resolves profitably. The DT learns these conditions from history and filters signals before they reach the portfolio.

```
Raw Price Data
     │
     ▼
┌─────────────────────────────────────────────────────┐
│               MACD DT OPTIMIZER                     │
│                                                     │
│  Feature Engineering  →  Model A (Sliding WFO)     │
│                       →  Model B (Fixed 10-yr)      │
│                                 │                   │
│                         Hybrid Signal Fusion        │
│                                 │                   │
│                         Volatility Veto (ATR)       │
│                                 │                   │
│                         Position Size Output        │
└─────────────────────────────────────────────────────┘
     │
     ▼
Portfolio Constructor  →  Ranked Positions  →  Deployment
```

---

## 2. Entry Point

```python
run_dt_optimizer(
    symbol:    str,          # e.g. "AAPL"
    curr_date: str,          # ISO date, e.g. "2024-01-15"
    is_days:   int = 100,    # Model A in-sample window (sliding)
    oos_days:  int = 30      # Model A OOS / re-train cycle
)
```

**Returns:**

| Field | Type | Description |
|---|---|---|
| `last_signal` | str | `"LONG"` / `"NO_SIGNAL"` |
| `prob_a` | float | Model A P(profitable) |
| `prob_b` | float | Model B P(profitable) |
| `size_multiplier` | float | 1.0 or 0.5 (vol veto) |
| `atr_percentile` | float | Current ATR vs 252-day history |
| `avg_oos_accuracy` | float | Mean OOS accuracy across slides |
| `slides` | list[dict] | Per-window WFO results |

---

## 3. Data Acquisition

```python
raw = yf.download(symbol, period="15y", interval="1d")
```

- Pulls up to **15 years** of daily OHLCV via `yf_retry()` (3 retries, exponential backoff)
- Cleaned via `_clean_dataframe()`: parses dates, fills NaN forward, validates non-zero prices
- Model B requires the full 15-year history; Model A uses a rolling window slice

---

## 4. Feature Engineering

Computed once on the full dataset, indexed by bar. Features are lag-safe — all values use data available **at** bar `t`, not after.

```
Features computed per bar:
────────────────────────────────────────────────────────
feat[0]  macd_hist          Histogram value at bar t
                            = MACD_line - Signal_line

feat[1]  macd_slope         3-bar first difference of histogram
                            = hist[t] - hist[t-3]
                            (captures acceleration, not just direction)

feat[2]  signal_gap         Distance between MACD line and signal line
                            = macd_line[t] - signal_line[t]
                            (magnitude of divergence)

feat[3]  vol_ratio          Volume momentum proxy
                            = vol_30d_avg / vol_60d_avg
                            > 1.0 = rising volume participation

feat[4]  bb_pct_b           Bollinger Band %B  (20-day, 2σ)
                            = (close - lower_band) / (upper_band - lower_band)
                            0 = at lower band, 1 = at upper band

feat[5]  atr_14             True Range smoothed over 14 bars
                            (absolute volatility context)
────────────────────────────────────────────────────────
```

**Label construction:**

```
label[t] = 1  if close[t + horizon] > close[t]   # profitable N bars later
           0  otherwise

horizon = 10 bars (forward-looking, ~2 trading weeks)
```

Label is only assigned to bars where a **MACD histogram crossover** occurred (histogram crosses zero), making the dataset sparse and signal-specific.

---

## 5. MACD Calculation

Internal `_calc_macd(closes, fast, slow, signal)` — pure NumPy EMA:

```python
def _ema(series, span):
    k = 2.0 / (span + 1)
    out = np.empty(len(series))
    out[0] = series[0]
    for i in range(1, len(series)):
        out[i] = series[i] * k + out[i-1] * (1 - k)
    return out

macd_line   = _ema(closes, fast) - _ema(closes, slow)
signal_line = _ema(macd_line, signal)
histogram   = macd_line - signal_line
```

Default parameters: `fast=12, slow=26, signal=9` (loaded from `macd_cache.db` if available for the symbol).

---

## 6. Model A — Sliding Walk-Forward Optimizer

Model A is **retrained on a rolling window** to capture the current market regime's alpha.

### 6.1 Window Slicing

```
Total history = N bars

For each slide:
┌─────────────────────────────────────────────────────┐
│        IS Window (is_days=100 bars)        │  OOS   │
│                                            │(30 bars)│
└─────────────────────────────────────────────────────┘
                                     ← step 30 bars →
                                     (OOS size = step)
```

Slides are non-overlapping OOS windows anchored at the end of history:

```python
slides = []
cursor = len(data)
while cursor - (is_days + oos_days) >= 0:
    oos_end   = cursor
    oos_start = cursor - oos_days
    is_end    = oos_start
    is_start  = is_end - is_days
    slides.append((is_start, is_end, oos_start, oos_end))
    cursor -= oos_days
slides.reverse()    # chronological order
```

### 6.2 Per-Slide Training

For each slide:

1. **Filter IS data** to bars where a crossover occurred → `X_train, y_train`
2. **Train `DecisionTreeClassifier`:**
   ```python
   clf = DecisionTreeClassifier(
       max_depth=5,          # prevents memorization
       min_samples_leaf=10,  # enforces generalization
       random_state=42
   )
   clf.fit(X_train, y_train)
   ```
3. **Evaluate on OOS:**
   - Predict probability `P(label=1)` for each OOS crossover bar
   - `signal = "LONG"` if `prob > 0.60`
   - Record OOS accuracy

### 6.3 Final Signal from Model A

The **last slide** in the WFO sequence is trained on the most recent `is_days` bars and evaluated against the most recent `oos_days` bars. The signal it produces for the current bar (`curr_date`) is `prob_a`.

```
prob_a = clf_last_slide.predict_proba([features_today])[0][1]
```

---

## 7. Model B — Fixed Long-History Model

Model B is trained **once** on the full 10-year history of crossover bars.

```python
X_full = features[all_crossover_bars]
y_full = labels[all_crossover_bars]

clf_b = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)
clf_b.fit(X_full, y_full)

prob_b = clf_b.predict_proba([features_today])[0][1]
```

**Why Model B exists:** The 10-year window captures tail regimes (2008 GFC, 2020 COVID crash, 2022 rate shock) that a 100-bar sliding window will never see. Model B prevents Model A from becoming overconfident in benign trending periods.

---

## 8. Hybrid Signal Fusion

```
          prob_a > 0.60 ?
               │
         ┌─────┴─────┐
        YES           NO
         │             │
  prob_b > 0.60 ?    NO_SIGNAL
         │
   ┌─────┴─────┐
  YES           NO
   │             │
 LONG        NO_SIGNAL
```

**Rule:** A `LONG` signal is only emitted when **both** Model A and Model B independently assign probability > 0.60 to the current crossover bar being profitable.

```python
if prob_a > THRESHOLD and prob_b > THRESHOLD:
    last_signal = "LONG"
else:
    last_signal = "NO_SIGNAL"

THRESHOLD = 0.60
```

This dual-gate dramatically reduces false positives at the cost of some true positives — by design (precision over recall).

---

## 9. Volatility Veto (ATR Gate)

Even when both models agree, elevated volatility triggers a position size reduction, not a signal cancellation.

```python
atr_series = compute_atr_14(data)                     # 14-bar True Range EMA
atr_percentile = percentileofscore(                   # current ATR vs 252-day lookback
    atr_series[-252:], atr_series.iloc[-1]
)

if atr_percentile > 80:
    size_multiplier = 0.5     # halve the position
else:
    size_multiplier = 1.0
```

ATR percentile > 80th means volatility is in the top quintile of the past year — the model's edge shrinks because spreads widen, stops are hit more often, and directional moves are noisier.

---

## 10. Walk-Forward Efficiency (WFO) Metrics

Each slide returns:

| Metric | Description |
|---|---|
| `is_accuracy` | Accuracy on in-sample crossover bars |
| `oos_accuracy` | Accuracy on out-of-sample crossover bars |
| `n_oos_signals` | Number of OOS crossovers evaluated |
| `n_oos_longs` | Crossovers where prob > 0.60 |

Aggregate: `avg_oos_accuracy = mean(oos_accuracy across all slides)`

A healthy WFO shows `oos_accuracy` tracking `is_accuracy` within ~10–15 pp. Large divergence signals overfitting in Model A.

---

## 11. Portfolio Construction

**Entry point:** `run_portfolio_optimizer(symbols, curr_date, capital, ...)`

### 11.1 Per-Symbol Pipeline

```
For each symbol in universe:
    ├── Liquidity Gate
    │       30-day avg dollar volume = mean(close * volume, last 30 bars)
    │       if < $5M  →  reject (add to rejected_liquidity)
    │
    ├── Run DT Optimizer
    │       last_signal, prob_a, prob_b, size_multiplier, atr_percentile
    │
    └── Signal Gate
            if last_signal == "LONG"  →  candidate position
            else                      →  reject (add to rejected_no_signal)
```

### 11.2 Position Sizing

```python
base_allocation = 0.05            # 5% of capital per position

allocation = base_allocation * size_multiplier
# size_multiplier = 1.0  →  5.0% per position
# size_multiplier = 0.5  →  2.5% per position (high vol regime)

dollar_amount = capital * allocation
```

### 11.3 Ranking & Cap

Candidate positions are ranked by `prob_a` (Model A confidence, descending). The portfolio takes the **top 20** by rank.

```
Max simultaneous longs = 20
Max theoretical deployment:
    20 × 5.0% = 100% (normal vol)
    20 × 2.5% =  50% (all high-vol)
```

### 11.4 Output Structure

```python
{
  "positions": [
    {
      "symbol":           "AAPL",
      "prob_a":           0.73,
      "prob_b":           0.68,
      "size_multiplier":  1.0,
      "allocation_pct":   5.0,
      "dollar_amount":    50000.0,
      "atr_percentile":   42.1,
      "avg_oos_accuracy": 0.61
    },
    ...
  ],
  "rejected_liquidity": ["SPCE", ...],
  "rejected_no_signal": ["META", ...],
  "rejected_errors":    ["BRKB", ...],
  "total_deployment":   0.45,          # fraction of capital deployed
  "n_positions":        9
}
```

---

## 12. Complete Execution Flow (Sequence Diagram)

```
caller
  │
  ├─► run_dt_optimizer(symbol, curr_date)
  │         │
  │         ├─► yf.download(symbol, period="15y")          [Data layer]
  │         │         └─► yf_retry() + _clean_dataframe()
  │         │
  │         ├─► _calc_macd(closes, fast, slow, signal)     [Indicator]
  │         │         └─► EMA(fast), EMA(slow), EMA(signal)
  │         │
  │         ├─► _build_features(data, macd, histogram)     [Feature Eng]
  │         │         └─► hist, slope, gap, vol_ratio, BB%B, ATR14
  │         │
  │         ├─► _find_crossover_bars(histogram)            [Label]
  │         │         └─► bars where hist crosses zero
  │         │
  │         ├─► _label_bars(crossovers, closes, horizon=10)[Label]
  │         │         └─► forward return > 0 → 1, else → 0
  │         │
  │         ├─► MODEL A: Sliding WFO                       [Training A]
  │         │         │
  │         │         ├─► for each slide:
  │         │         │       split IS / OOS
  │         │         │       DecisionTreeClassifier.fit(X_is, y_is)
  │         │         │       eval OOS accuracy
  │         │         │       store slide metrics
  │         │         │
  │         │         └─► last slide → predict prob_a for today
  │         │
  │         ├─► MODEL B: Fixed 10yr                        [Training B]
  │         │         DecisionTreeClassifier.fit(X_all, y_all)
  │         │         └─► predict prob_b for today
  │         │
  │         ├─► HYBRID FUSION                              [Signal]
  │         │         prob_a > 0.60 AND prob_b > 0.60
  │         │         └─► last_signal = "LONG" | "NO_SIGNAL"
  │         │
  │         └─► ATR VETO                                   [Sizing]
  │                   atr_percentile > 80 → size_multiplier = 0.5
  │                   else               → size_multiplier = 1.0
  │
  └─► run_portfolio_optimizer(symbols, curr_date, capital)
            │
            ├─► for each symbol → run_dt_optimizer()
            ├─► liquidity gate  → filter < $5M avg vol
            ├─► signal gate     → keep LONG only
            ├─► rank by prob_a  → sort descending
            ├─► cap at 20       → top 20
            └─► size positions  → 5% × size_multiplier × capital
```

---

## 13. Key Constants Reference

| Constant | Value | Location | Effect |
|---|---|---|---|
| `is_days` | 100 | `run_dt_optimizer` arg | Model A training window |
| `oos_days` | 30 | `run_dt_optimizer` arg | Model A re-train cadence |
| `FIXED_IS_YEARS` | 10 | hardcoded | Model B history depth |
| `LABEL_HORIZON` | 10 | hardcoded | Forward bars for label |
| `MAX_DEPTH` | 5 | `DecisionTreeClassifier` | Tree depth cap |
| `MIN_SAMPLES_LEAF` | 10 | `DecisionTreeClassifier` | Min leaf size |
| `THRESHOLD` | 0.60 | signal gate | Dual-model probability gate |
| `ATR_VOL_THRESHOLD` | 80 | veto gate | ATR percentile cutoff |
| `SIZE_VETO` | 0.50 | size multiplier | Half-size in high-vol |
| `MAX_POSITIONS` | 20 | portfolio | Max simultaneous longs |
| `BASE_ALLOC` | 0.05 | portfolio | 5% per position |
| `MIN_DOLLAR_VOL` | 5e6 | liquidity gate | $5M 30-day avg |

---

## 14. Design Decisions & Rationale

### Dual-model architecture
A single model trained on recent data risks overfitting to the current regime. A single model trained on 10 years may underweight recent dynamics. The dual-gate forces consensus between "what's working now" (Model A) and "what survives macro regimes" (Model B).

### Decision Tree over other classifiers
Shallow decision trees (`max_depth=5`) are interpretable, fast to retrain on 100-bar windows, and resistant to noise in small samples. Random Forests or gradient boosters would require more data to generalize and would obscure the decision path.

### Feature selection
Features are chosen to capture three independent dimensions:
- **Momentum quality** — histogram value + slope + signal gap
- **Participation** — volume ratio (smart money confirmation)
- **Regime** — BB %B (mean reversion vs breakout context) + ATR (absolute volatility)

### ATR veto as sizing, not filtering
Blocking trades in high-vol entirely would cause the portfolio to go to zero exposure during volatile periods — exactly when some of the best risk/reward entries occur. Halving size preserves participation while reducing dollar risk.

### 0.60 probability threshold
At 0.50 the classifier adds no edge over a coin flip. Empirical testing showed 0.60 as the inflection point where precision materially exceeds base rate without reducing signal count to near zero.

---

## 15. File Map

```
tradingagents/dataflows/
├── macd_dt_optimizer.py      ← Core DT optimizer (this doc)
├── macd_dt_portfolio.py      ← Portfolio wrapper
├── macd_signal.py            ← Real-time signal (uses cached params)
├── macd_optimizer_algo.py    ← Pure WFO grid search (provides params)
├── macd_optimizer_llm.py     ← LLM-enhanced WFO (regime narrowing)
├── macd_cache.py             ← SQLite param cache (macd_cache.db)
├── wfo_analyzer.py           ← WFO statistical significance
└── stockstats_utils.py       ← yf_retry, _clean_dataframe
```

---

*Generated: 2026-04-11 | TradingAgentsMaorFramwork*
