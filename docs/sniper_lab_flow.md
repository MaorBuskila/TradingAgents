# Sniper Lab — Technical Flow

## Overview

The Precision Sniper Lab is a two-layer entry-filtering system that identifies high-conviction BUY/SELL setups. A signal fires only when **both** layers agree:

1. **Classical Walk-Forward Optimization (WFO)** — finds optimal EMA stack parameters
2. **Decision Tree Ensemble** — two XGBoost models vote on today's market regime

---

## Data Flow

```
User fills form (symbol, date, window sizes)
    │
    ▼
POST /api/sniper-optimize
    ├── run_algo_optimizer()      → Classical WFO grid search
    ├── run_sniper_dt_optimizer() → DT ensemble training
    ├── RSI WFO (if not cached)
    └── MACD WFO (if not cached)
         │
         ▼
    upsert_sniper_params() → sniper_cache.db (SQLite)
         │
         ▼
GET /api/sniper-signal/{symbol}?date={date}
    ├── get_sniper_params()         ← loads cached params
    ├── fetch_price_data()          ← yfinance (15-year history, CSV-cached)
    ├── build_sniper_feature_frame() ← 18+ indicator columns
    ├── compute_sniper_score()      ← bull/bear composite 0–10
    ├── signal logic (cross + score gate)
    ├── compute_stop()              ← structure + ATR hybrid SL
    └── compute_tps()               ← 1R / 2R / 3R TP ladder
         │
         ▼
Frontend renders: signal card, WFO card, DT card, RSI card, MACD card
```

---

## Layer 1 — Classical WFO

**File:** `tradingagents/quant_ml/optimizers/sniper_optimizer_algo.py`

Grid-searches ~800–900 EMA parameter combinations:

| Parameter | Search Space |
|-----------|-------------|
| EMA fast | 8, 9, 12 |
| EMA slow | 18, 21, 26, 34 (must be > fast) |
| EMA trend | 50, 55 |
| Min score threshold | 5.0, 6.0, 7.0, 8.0 |
| SL multiplier | 1.0, 1.5, 2.0, 2.5 |
| Volume multiplier | 1.0, 1.2, 1.5 |

**Scoring:** R-multiple Sharpe on simulated non-overlapping long trades over the in-sample window. Out-of-sample validation determines `HIGH / MEDIUM / LOW` confidence.

---

## Layer 2 — DT Ensemble

**File:** `tradingagents/quant_ml/optimizers/sniper_dt_optimizer.py`

Two XGBoost models trained on 19 features:

| Feature Group | Features |
|---------------|---------|
| EMA | `ema_fs_ratio`, `close_ema_f_dist`, `ema_trend_dist` |
| Momentum | `rsi_13`, `rsi_slope5`, `rsi_opt` (WFO-optimized period) |
| MACD | `macd_hist`, `macd_gap`, `macd_hist_opt` (WFO-optimized params) |
| Volume / Volatility | `vol_ratio`, `vol_regime_ratio`, `atr_norm` |
| Structure | `vwap_dist`, `bb_pct_b`, `adx_14`, `di_spread`, `htf_bias_f` |
| Composite | `bull_score`, `bear_score` (0–10 confluence scores) |

**Model A — Sliding WFO:** Trains on `IS_bars`, tests on `OOS_step`, slides forward each fold. Adapts to recent regime.

**Model B — Crash-aware reference:** Trains on the first ~1500 bars, holds its scaler fixed throughout the OOS phase. Detects distributional drift.

**Labels:** Triple-barrier:
- `+1` if any TP (1R, 2R, 3R) is hit first
- `-1` if SL is hit first
- `0` if neither hits within 10 bars

A LONG signal requires **both** `prob_A > threshold_A` **and** `prob_B > threshold_B`, plus a grade veto: `bull_score ≥ 5.0` (grade B+).

---

## Composite Scoring Engine

**File:** `tradingagents/quant_ml/composite/sniper_score.py`

10 weighted factors (total weight = 10.0) evaluated per bar:

| Factor | Weight | Condition |
|--------|--------|-----------|
| Fast EMA > Slow EMA | 1.0 | Uptrend |
| Close > Trend EMA | 1.0 | Above trend |
| RSI in 50–75 zone | 1.0 | Momentum without overbought |
| MACD histogram > 0 | 1.0 | Positive momentum |
| MACD line > signal | 1.0 | Bullish crossover |
| Close > VWAP | 1.0 | Institutional bias |
| Volume burst | 1.0 | Volume > 20-SMA × vol_mult |
| ADX > 20 AND +DI > -DI | 1.0 | Trending and directional |
| HTF EMA bias | 1.5 | Higher-timeframe fast > slow |
| Close > Fast EMA | 0.5 | Near-term momentum |

**Grade mapping:**

| Grade | Score |
|-------|-------|
| A+ | ≥ 8.0 |
| A | ≥ 6.5 |
| B | ≥ 5.0 |
| C | < 5.0 |

---

## Entry Signal Logic

**File:** `tradingagents/quant_ml/signals/sniper_signal.py:95–99`

```python
action = "HOLD"
if cross_up and bull_score >= min_score:
    action = "BUY"
elif cross_dn and bear_score >= min_score:
    action = "SELL"
```

Both conditions must be true:
- An EMA fast/slow crossover (`cross_up` or `cross_dn`)
- Composite score at or above the WFO-optimized minimum threshold

---

## Risk Management

### Stop-Loss
**File:** `tradingagents/quant_ml/risk/structure_sl.py`

For a LONG entry:
1. Find swing low over 10 bars before entry
2. Structure stop = `swing_low − (0.2 × ATR)`
3. ATR stop = `entry − (atr_mult × ATR)`
4. Use the tighter of the two, with a minimum distance of `0.5 × ATR` from entry

### Take-Profit Ladder
**File:** `tradingagents/quant_ml/risk/tp_ladder.py`

```
R = |entry − stop_loss|
TP1 = entry + 1R
TP2 = entry + 2R
TP3 = entry + 3R
```

### Volatility Regime
**File:** `tradingagents/quant_ml/indicators/vol_regime.py`

```
ratio = ATR(14) / SMA(ATR(14), 42)
HIGH   → ratio > 1.3   (expanding volatility)
NORMAL → 0.7 ≤ ratio ≤ 1.3
LOW    → ratio < 0.7   (compression, potential breakout setup)
```

---

## Caching Layer

**File:** `tradingagents/dataflows/sniper_cache.py` → SQLite `sniper_cache.db`

Stored per ticker:

| Field | Description |
|-------|-------------|
| `optimal_ema_fast/slow/trend` | Best EMA stack from WFO |
| `optimal_min_score` | Score gate threshold |
| `optimal_sl_mult`, `optimal_vol_mult` | Risk multipliers |
| `classical_is_sharpe`, `classical_oos_sharpe` | WFO performance |
| `confidence` | HIGH / MEDIUM / LOW |
| `threshold_a`, `threshold_b` | DT model probability thresholds |
| `dt_oos_sharpe`, `dt_oos_hit_rate` | DT performance |
| `optimized_at` | Timestamp |

Signal computation loads from cache; re-running the optimizer overwrites it.

---

## Frontend UI

**File:** `frontend/src/pages/SniperLab.tsx`

The page renders six cards after a successful optimize run:

| Card | Contents |
|------|----------|
| **Classical WFO** | EMA stack, score threshold, multipliers, IS/OOS Sharpe, confidence |
| **DT Ensemble** | OOS Sharpe, hit rate, model thresholds, latest probs, grade veto |
| **RSI WFO** | Optimal RSI period & thresholds, IS/OOS Sharpe |
| **MACD WFO** | Optimal fast/slow/signal periods, IS/OOS Sharpe |
| **Latest Signal** | BUY/SELL/HOLD badge, bull/bear grades, price, SL, TP1/TP2/TP3, vol regime |
| **Track Signal** | Button to push signal into TP Tracker (`POST /tp-tracker/from-signal`) |

Form state and results are persisted to `sessionStorage` (key `tradingagents_sniperlab_v1`) so a page reload restores the last run.

---

## Known Gap

`POST /api/sniper-optimize` is called by the frontend but is **not yet implemented** in `api/main.py`. Only `GET /api/sniper-signal/{ticker}` exists. The optimize button returns a 404 until the endpoint is added.
