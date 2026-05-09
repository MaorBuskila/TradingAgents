# RSI DT Optimizer — Beginner's Guide

A line-by-line explanation of [test_rsi_dt_optimizer_wfo.py](test_rsi_dt_optimizer_wfo.py)
and the pipeline it tests at
[rsi_dt_optimizer.py](../tradingagents/quant_ml/optimizers/rsi_dt_optimizer.py).

If you are new to machine learning, quantitative trading, or walk-forward
validation, read in order: **§1 → §3** for the idea, **§4–§6** for how the
optimizer behaves, **§7** for what the tests guarantee, **§11** for Python
keywords in the test file.

## Contents

| § | Topic |
|---|--------|
| [1](#1-the-big-picture) | Big picture |
| [2](#2-vocabulary-you-must-know) | Vocabulary (OHLC, RSI, WFO terms, …) |
| [3](#3-triple-barrier-labeling-the-y) | Triple barrier labels (the `y`) |
| [4](#4-walk-forward-optimization-wfo) | Walk-forward optimization |
| [5](#5-dynamic-thresholds-why-not-05) | Dynamic thresholds |
| [6](#6-the-signal-gate) | Live signal gate |
| [7](#7-now-read-the-test) | What each test asserts |
| [8](#8-pytest-flags-you-used) | Pytest flags |
| [9](#9-reading-the-output) | Reading JSON output |
| [10](#10-running-it-yourself) | Commands |
| [11](#11-line-by-line-the-test-file-itself-keywords-in-order) | Test file, keyword by keyword |
| [12](#12-further-reading) | Further reading |

---

## 1. The big picture

Each day the pipeline asks one *classification* question (the exact rule is
in [§3](#3-triple-barrier-labeling-the-y)):

> "Does today's chart setup resemble past setups where, within the **next 10
> trading days**, price **touched** a line **2×ATR above** today's close
> **before** it **touched** a line **1×ATR below** (using today's ATR and
> each day's **high/low**)?"

Informally people shorten that to “went up 2×ATR before down 1×ATR,” but the
implementation is **barrier touches**, not “close finished 2 ATR higher.”

If the models assign high probability to class **1**, and other gates pass
([§6](#6-the-signal-gate)) → `last_signal` can be **1** (long). Otherwise stay flat.

Four features feed the model:

| Feature | What it measures |
|---|---|
| `rsi_14` | Momentum (0 = crashing, 100 = parabolic) |
| `ema_ratio_50_200` | Short-term trend strength vs. long-term trend |
| `macd_hist` | Acceleration of momentum (slope of MACD) |
| `atr_norm_14` | Volatility as a % of price |

The model is an **XGBoost classifier** — a gradient-boosted decision tree
ensemble. It outputs a probability in [0, 1].

---

## 2. Vocabulary you must know

### 2.1 OHLC data
`Open / High / Low / Close` — the four prices that describe one bar (here, one day) of trading. `yfinance` gives us these.

### 2.2 RSI (Relative Strength Index)
Ratio of recent gains to recent losses, smoothed over 14 days.
- RSI > 70 → "overbought" (usually reverts down)
- RSI < 30 → "oversold" (usually bounces up)

### 2.3 EMA (Exponential Moving Average)
A moving average where recent prices matter more than old ones.
`ema_ratio_50_200 = ema(50-day) / ema(200-day)`. Above 1.0 means short-term trend is stronger than long-term.

### 2.4 MACD (Moving Average Convergence Divergence)
`macd_line = ema(12) - ema(26)`, smoothed by another EMA.
`macd_hist = macd_line - signal_line`. Rising histogram = accelerating up-trend.

### 2.5 ATR (Average True Range)
The typical daily price swing, smoothed over 14 days. Used here to **scale barriers to volatility** — a +2×ATR move means the same relative magnitude for a calm stock and a wild one.

### 2.6 Classifier / Probability
A classifier predicts **which class** an input belongs to. `XGBClassifier.predict_proba(X)[:, 1]` returns the probability of class **1** (the "upper barrier hit first" case).

### 2.7 Features / Labels
- **Features (X):** what the model looks at (RSI, MACD, etc.)
- **Label (y):** the answer we want it to predict (0 or 1)

### 2.8 Training vs. Test set
- **Training:** data the model learns from
- **Test (out-of-sample, OOS):** data the model has **never seen** — used to measure real performance

If you evaluate on data the model already saw, you get fake-good results. This is the #1 rule of ML.

### 2.9 Standard Scaler
Rescales each feature to mean=0, std=1. XGBoost doesn't strictly need it, but the scaler is useful for diagnostics and keeps the pipeline consistent.
**Critical:** the scaler must be fit on **training data only**, then *applied* to test data. Fitting on everything causes **leakage**.

### 2.10 Leakage
When information from the future (or from the test set) sneaks into training. Leakage produces great-looking backtests that die in production. The two leaks we fixed:
1. **Scaler fit outside the fold** — old code fit a single scaler on all data.
2. **FinBERT sentiment ffilled backward** — old code applied *today's* news score to every historical row.

---

## 3. Triple Barrier labeling (the "y")

Method associated with [López de Prado](https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos/dp/1119482089). Instead of “did the stock go up tomorrow?” (noisy), we use a **path-dependent** rule over the next `horizon` bars.

### 3.1 Exact rule (matches the code)

Fix a day `t`. Let `ATR_t` be volatility at `t` (14-day ATR in this project).

- **Upper barrier:** `tp = close[t] + tp_mult * ATR_t` (default `tp_mult = 2`)
- **Lower barrier:** `sl = close[t] - sl_mult * ATR_t` (default `sl_mult = 1`)

Scan days `t+1, t+2, …` in order:

| What happens first | Label |
|--------------------|--------|
| Some day’s **high** ≥ `tp` | **1** |
| Some day’s **low** ≤ `sl` | **0** |
| Neither within `horizon` bars | **0** (timeout) |

So “before” means **first hit in calendar order** among future bars. The
**last `horizon` rows** of the label series are `NaN`: you cannot know the
future from the end of the sample ([test](test_rsi_dt_optimizer_wfo.py)).

### 3.2 Why this is not the same as “+2 ATR on the close”

Barriers are checked against **intraday extremes** (`High`, `Low`), not only
the close. A stock can wick through `tp` and still close red that day.

### 3.3 Why it is called “triple” barrier

Three ways the episode ends: hit **upper**, hit **lower**, or hit **time**
(horizon) without either. Asymmetric multipliers (2 vs 1) mean the upper line
is farther in price units than the lower line, so **random** paths tend to
hit the lower side first more often; a useful model tilts the odds toward
label **1** when features align.

```python
# rsi_dt_optimizer.py (conceptually)
for t in range(n - horizon):
    tp = c[t] + tp_mult * a[t]
    sl = c[t] - sl_mult * a[t]
    for k in range(1, horizon + 1):
        if h[t + k] >= tp:
            label = 1
            break
        if lo[t + k] <= sl:
            label = 0
            break
```

---

## 4. Walk-Forward Optimization (WFO)

The naïve approach: train once on years 1–5, test on year 6. But markets change — a model trained in 2018 may be useless in 2024.

WFO instead **slides** the training window forward through time:

```
fold 1:  [—— train 1000 days ——][test 20]
fold 2:          [—— train 1000 days ——][test 20]
fold 3:                 [—— train 1000 days ——][test 20]
...
fold 39:                                          [—— train ——][test 20]
```

Each fold:
1. Trains a **fresh** model on the trailing 1000 days.
2. Predicts on the next 20 days (which the model has never seen).
3. Collects those 20 predictions into `oos_preds_a`.

After all folds, `oos_preds_a` is a long list of **honest** out-of-sample predictions — one per test day, produced by a model that didn't see that day during training. That's what gives us a trustworthy Sharpe ratio and a realistic threshold.

See [rsi_dt_optimizer.py:291-330](../tradingagents/quant_ml/optimizers/rsi_dt_optimizer.py:291).

### 4.1 Why two models?

- **Model A (sliding WFO)** — retrained every 20 days on the last 1000. Adapts to recent regime.
- **Model B (fixed crash-aware)** — trained *once* on the oldest 1500 bars, which include stress periods (Q4 2018 selloff, March 2020 COVID crash, 2022 bear). It provides a "what would a crash-trained model say?" second opinion.

We require **both** to agree before going long. Model A's recent optimism is overruled if Model B (which has seen carnage) disagrees.

### 4.2 `scale_pos_weight`

The Triple Barrier labels are often imbalanced — e.g. 70% zeros, 30% ones. Naïve training makes the model lazy ("always predict 0 → 70% accuracy!"). `scale_pos_weight = neg/pos` tells XGBoost to weight the minority class more heavily during training.

```python
# rsi_dt_optimizer.py:195-198
def _spw(y):
    pos = y.sum()
    neg = len(y) - pos
    return max(neg / max(pos, 1.0), 1.0)
```

---

## 5. Dynamic thresholds (why not 0.5?)

A naïve threshold would be: "signal LONG when prob > 0.5." But:

- Models systematically under- or over-confident depending on feature distribution.
- What's a "high" probability differs per stock.
- Regime changes shift the prediction distribution.

Instead, we use the **rolling 85th percentile** of the last ~250 OOS
predictions (`ROLLING_THRESHOLD_WINDOW`, `THRESHOLD_PERCENTILE` in
[rsi_dt_optimizer.py](../tradingagents/quant_ml/optimizers/rsi_dt_optimizer.py)):

```python
def _rolling_quantile(preds: list[float], window: int, q: float) -> float:
    if not preds:
        return 0.55
    tail = preds[-window:] if len(preds) > window else preds
    return float(np.quantile(tail, q))
```

Meaning: *fire only when today’s probability is in the **top ~15%** of
recent OOS probabilities for that model slice.* That adapts per ticker without
hard-coding `0.5`.

Illustration (one past run): thresholds around **0.63** while raw probs were
**~0.2–0.4** — crossing the threshold is intentionally **rare**; your numbers
will differ by symbol and date.

---

## 6. The signal gate

`last_signal = 1` only if **all** of the following are true (one line each
in code; `model_agree` bundles two numeric checks on the two models):

```python
# rsi_dt_optimizer.py
model_agree   = prob_a > threshold_a and prob_b > threshold_b
sentiment_ok  = sentiment_today > 0.0
macro_up      = price_today > ema200_today
last_signal   = int(model_agree and sentiment_ok and macro_up)
```

| Gate | Role |
|---|---|
| `model_agree` | Both ensemble models must exceed their dynamic thresholds. |
| `sentiment_ok` | FinBERT on **today’s** headlines must not be net bearish (veto). |
| `macro_up` | Price above 200-day EMA (avoid structural downtrend entries). |

**Sentiment is a veto, not a training feature** — no historical FinBERT
column in `X` ([§2.10](#210-leakage)); that avoids “paste today’s news into
every past row” leakage. Headlines only affect **today’s** go/no-go.

---

## 7. Now read the test

Short map of what each test protects. **Python / pytest keywords** for the
same file live in [§11](#11-line-by-line-the-test-file-itself-keywords-in-order).

### 7.1 `_synthetic_ohlc` — test fixture

```python
def _synthetic_ohlc(n=2200, seed=7):
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0005, 0.015, size=n)  # mild drift
    returns[n // 3 : 2 * n // 3] -= 0.002        # bear patch
    close = 100.0 * np.exp(np.cumsum(returns))
    ...
```

Creates **fake** price data — reproducible (`seed=7`), with a built-in
"bear patch" so the tests exercise both up- and down-regimes without
relying on the network. **Unit tests should never depend on the internet.**

- `rng.normal(0.0005, 0.015, size=n)` — daily returns from a normal distribution (tiny positive drift, 1.5% daily vol).
- `np.exp(np.cumsum(returns))` — turn log-returns into a price path.
- `pd.date_range("2018-01-02", periods=n, freq="B")` — business-day index (skips weekends).

### 7.2 `test_triple_barrier_produces_binary_labels`

```python
labels = _triple_barrier_labels(close=..., atr=..., horizon=10, tp_mult=2.0, sl_mult=1.0)
assert set(valid.unique()).issubset({0, 1})
assert labels.iloc[-10:].isna().all()
```

Two invariants:
1. Every non-missing label is exactly **0 or 1** (no floats, no garbage).
2. The **last 10 rows** must be `NaN` — we can't label bar `t` until we've seen bars `t+1..t+10`, and those don't exist yet at the end. If they *weren't* NaN, we'd have **look-ahead bias**.

### 7.3 `test_wfo_runs_many_folds_not_single_shot`

This is the regression test for the real bug the rewrite fixed:

```python
out = _run_wfo(feat_df, y, is_days=1000, oos_days=20)
assert len(out["slides"]) >= 20, "WFO must produce many folds, not one"
assert len(out["oos_preds_a"]) >= 400
assert float(out["oos_preds_a"].min()) >= 0.0
assert float(out["oos_preds_a"].max()) <= 1.0
```

- `len(slides) >= 20` — proves the loop actually iterates. The old code trained **once**; this guards against regressing to that.
- `len(oos_preds_a) == len(oos_preds_b)` — both models score the same OOS slices (same number of predictions).
- `min/max in [0, 1]` — sanity check that they're probabilities.

### 7.4 `test_features_exclude_sentiment`

```python
assert "sentiment" not in feat.columns
assert "sentiment" not in ML_COLS
```

Guards the leakage fix. If someone accidentally adds sentiment back to the feature matrix, this test fails.

### 7.5 `test_full_optimizer_end_to_end` (marked `@pytest.mark.integration`)

Calls the real optimizer against the real `yfinance` API. Marker lets CI skip it when running `-m "not integration"`.

```python
delete_rsi_dt_params(symbol)                    # force a fresh WFO
out = run_rsi_dt_optimizer(symbol, curr_date, is_days=1000, oos_days=20)
```

The `delete_rsi_dt_params` call ensures we don't short-circuit through the 7-day cache and miss exercising the WFO loop.

**Contract assertions:**

```python
required = {"symbol", "curr_date", "last_signal", "last_prob_a", ...}
missing = required - set(out.keys())
assert not missing
```

These keys must stay in sync with whatever the **API** and **portfolio**
layers deserialize (search the repo for `RsiDt` / `rsi_dt` response types).
If the optimizer drops a field, this test fails before production.

**Cache round-trip:**

```python
cached = get_rsi_dt_params(symbol)
assert cached["threshold_a"] == pytest.approx(out["threshold_a"], abs=1e-4)
```

Proves the SQLite upsert actually happened and that the stored value round-trips to within floating-point tolerance.

---

## 8. Pytest flags you used

| Flag | What it does |
|---|---|
| `-v` | Verbose — shows each test name and pass/fail |
| `-s` | **Don't capture stdout** — `print()` calls actually appear on your terminal (without this, you see nothing but pass/fail) |
| `-x` | Stop at the first failure |
| `-k "partial_name"` | Run only tests whose name matches |
| `-m "not integration"` | Skip tests marked `@pytest.mark.integration` |
| `--tb=short` | Compact traceback on failure |
| `--log-cli-level=INFO` | Stream `logging.INFO` to terminal during tests (does **not** show `print()` — use `-s` for that) |

---

## 9. Reading the output

**Illustrative** JSON (one historical run shape). Numbers change with ticker,
date, and market; treat field *names* and *ranges* as the stable lesson.

```json
{
  "last_signal": 0,
  "last_prob_a": 0.1899,
  "last_prob_b": 0.176,
  "threshold_a": 0.6285,
  "threshold_b": 0.6356,
  "oos_sharpe": 1.294,
  "oos_hit_rate": 0.432,
  "oos_trade_count": 95,
  "confidence": "MEDIUM",
  "n_slides": 39,
  "last_rsi": 77.31,
  "last_atr_pct": 83.92,
  "last_size_mult": 0.5,
  "cache_hit": false
}
```

How to read this example:

- **`last_signal`** — `0` means “no long”; here probs sit **below** both thresholds, so `model_agree` is false.
- **`last_prob_a` / `last_prob_b`** — class-1 probability from each model; compare to **`threshold_a` / `threshold_b`**.
- **`oos_sharpe`** — annualized Sharpe on gated OOS entries (see `_oos_backtest_stats` in the optimizer); rough quality of the **historical** rule, not a forecast.
- **`oos_hit_rate`** — among OOS bars where the model “would trade,” fraction of label **1**; can be below 0.5 yet still OK with **2:1** barrier asymmetry ([§3](#3-triple-barrier-labeling-the-y)).
- **`n_slides`** — number of WFO folds executed when `cache_hit` is false.
- **`confidence`** — `HIGH` / `MEDIUM` / `LOW` from OOS Sharpe + model agreement ([`_confidence_tier`](../tradingagents/quant_ml/optimizers/rsi_dt_optimizer.py)).
- **`last_rsi`** — RSI level on the as-of bar (context only; signal still needs gates).
- **`last_atr_pct`** — where today’s normalized ATR sits in the historical distribution; high → **`last_size_mult`** may be `0.5`.
- **`cache_hit`** — `false` = thresholds recomputed / WFO ran; a **fresh** cache within TTL can yield `true` and fewer recomputed fields.

---

## 10. Running it yourself

```bash
# Fast path — unit tests only, no network
.venv/bin/python -m pytest tests/test_rsi_dt_optimizer_wfo.py -v -m "not integration"

# Full suite including the integration test (downloads from yfinance)
.venv/bin/python -m pytest tests/test_rsi_dt_optimizer_wfo.py -v

# Inspect the optimizer's full dict output
.venv/bin/python -m pytest \
  tests/test_rsi_dt_optimizer_wfo.py::test_full_optimizer_end_to_end \
  -v -s
```

To inspect the cached row after a run:

```bash
sqlite3 data/rsi_cache.db \
  "SELECT * FROM rsi_dt_params_cache WHERE ticker='FORM';"
```

---

## 11. Line-by-line: the test file itself (keywords in order)

This section follows [test_rsi_dt_optimizer_wfo.py](test_rsi_dt_optimizer_wfo.py) from top to bottom. If a word looks unfamiliar, find it here.

### 11.1 Module docstring (lines 1–34)

- **Triple docstring `"""..."""`:** A string at the top of a file is a *module docstring*. Tools and humans read it; Python keeps it in `__doc__`. It does not run code.
- **`pytest` / `-m "not integration"`:** Pytest can *mark* tests. `@pytest.mark.integration` tags slow or network tests. `-m "not integration"` means “run tests that do **not** have that marker.”
- **`-s`:** “Do not capture stdout” — lets `print()` show during the test run.
- **`-x`:** Stop the whole run on the first failure (useful while debugging).
- **`--tb=short`:** Shorter tracebacks when something fails.
- **`-c "..."`:** Run a one-liner *command* as Python code (here: import optimizer, call it, `json.dumps`).

### 11.2 Imports (lines 36–54)

- **`from __future__ import annotations`:** Makes type hints (like `def f(x: list[str])`) lazily evaluated so forward references work cleanly on older Python semantics. Safe boilerplate in modern codebases.
- **`import json`:** Standard library for reading/writing JSON text. Here: pretty-print dicts with `json.dumps(..., indent=2, default=str)` — `default=str` converts non-JSON types (e.g. dates) to strings instead of crashing.
- **`import numpy as np`:** *Numerical* arrays and fast math. `np.random`, `np.exp`, `np.cumsum`, etc.
- **`import pandas as pd`:** *Tabular* data as `DataFrame` (named columns, index) and `Series` (one column). Finance code uses pandas heavily.
- **`import pytest`:** The test framework. Gives `pytest.mark`, `pytest.approx`, and runs functions named `test_*`.
- **`from tradingagents... import (...)`:** Imports *private* helpers (names starting with `_`) from production code. Tests are allowed to touch internals to lock behavior; app code usually should not import private symbols from elsewhere.
  - **`ML_COLS`:** List of column names used as model inputs (`rsi_14`, `ema_ratio_50_200`, `macd_hist`, `atr_norm_14`).
  - **`_build_features`:** Builds indicators + helper columns from OHLC.
  - **`_run_wfo`:** Walk-forward loop; returns dict with `slides`, `oos_preds_a`, etc.
  - **`_triple_barrier_labels`:** Builds supervised labels `0/1/NaN`.
  - **`run_rsi_dt_optimizer`:** Public API: download data, features, labels, WFO, thresholds, signal, cache.
- **`delete_rsi_dt_params` / `get_rsi_dt_params`:** SQLite cache helpers for RSI-DT parameters.

### 11.3 Helper `_synthetic_ohlc` (lines 57–66)

- **`def _synthetic_ohlc(n: int = 2200, seed: int = 7)`:** A *fixture* factory. Underscore prefix = “internal to this test module.” Default args make calls short.
- **`rng = np.random.default_rng(seed)`:** A *reproducible* random number generator. Same `seed` → same “random” series every run (tests must be deterministic).
- **`rng.normal(mean, std, size=n)`:** Draw `n` samples from a bell curve. Here: tiny positive mean (`0.0005`) = slight upward drift; `0.015` ≈ daily volatility scale.
- **`returns[n // 3 : 2 * n // 3] -= 0.002`:** Integer division `//`; slice from one-third to two-thirds of the array; subtract a fixed amount to simulate a *weaker* or negative period (“bear patch”).
- **`close = 100.0 * np.exp(np.cumsum(returns))`:** Treat `returns` as log-returns; cumulative sum then `exp` converts to a price level starting near 100.
- **`high = close * (1 + abs(...))` / `low = close * (1 - abs(...))`:** Ensure High ≥ Close-ish and Low ≤ Close-ish with small random wicks (`abs` drops sign).
- **`pd.date_range("2018-01-02", periods=n, freq="B")`:** Business-day (`B`) datetime index — skips weekends; length `n`.
- **`pd.DataFrame({...}, index=idx)`:** Table with columns `Close`, `High`, `Low` and the business-day index.

### 11.4 `test_triple_barrier_produces_binary_labels` (lines 73–84)

**Step order:**

1. **`df = _synthetic_ohlc(500)`** — small synthetic history.
2. **`feat = _build_features(df, rsi_period=14).iloc[210:]`**
   - **`rsi_period=14`** — keyword argument; classic RSI window.
   - **`.iloc[210:]`** — *integer location* slice: drop the first 210 rows. Warm-up: indicators like RSI/EMA need history before values stabilize (NaNs at the start).
3. **`labels = _triple_barrier_labels(...)`** — pass aligned `Series` columns by name.
   - **`horizon=10`** — look ahead up to 10 bars for barrier hits.
   - **`tp_mult=2.0` / `sl_mult=1.0`** — take-profit / stop-loss distances in multiples of ATR at entry bar.
4. **`valid = labels.dropna()`** — remove `NaN` (unlabeled tail + any skipped bars).
5. **`assert len(valid) > 0`** — must have at least one usable label.
6. **`assert set(valid.unique()).issubset({0, 1})`**
   - **`Series.unique()`** — distinct values present.
   - **`set(...)`** — unordered collection.
   - **`issubset({0, 1})`** — every label is 0 or 1 (binary classification).
7. **`assert labels.iloc[-10:].isna().all()`**
   - **`.iloc[-10:]`** — last 10 rows.
   - **`.isna()`** — boolean mask where value is missing.
   - **`.all()`** — every boolean is True — *all* of the last `horizon` rows must be unlabeled (no future peeking).

### 11.5 `test_wfo_runs_many_folds_not_single_shot` (lines 87–104)

**Step order:**

1. Longer synthetic series: **`n=2100`** so many WFO folds exist.
2. **`feat_full = _build_features(...).iloc[210:]`** — full feature matrix after warm-up.
3. **`labels = _triple_barrier_labels(...)`** — same parameters as other tests for consistency.
4. **`mask = labels.notna() & feat_full[ML_COLS].notna().all(axis=1)`**
   - **`labels.notna()`** — True where a label exists.
   - **`feat_full[ML_COLS]`** — sub-table of only model input columns.
   - **`.notna().all(axis=1)`** — row-wise: True only if *every* feature in that row is non-NaN.
   - **`&`** — element-wise boolean AND for aligned indexes.
5. **`feat_df = feat_full[mask].copy()`** — keep only clean rows; **`.copy()`** avoids accidentally mutating a slice of the parent frame.
6. **`y = labels[mask].astype(int)`** — labels aligned to the same rows; cast to integer type for sklearn/XGBoost.
7. **`out = _run_wfo(feat_df, y, is_days=1000, oos_days=20)`**
   - **`is_days`** — in-sample training window length (bars).
   - **`oos_days`** — out-of-sample test slice length per fold; also the slide *step*.
8. **`assert len(out["slides"]) >= 20`** — `slides` is one record per fold; many folds ⇒ WFO is not degenerate.
9. **`assert len(out["oos_preds_a"]) == len(out["oos_preds_b"])`** — both models score identical OOS index ranges.
10. **`assert len(out["oos_preds_a"]) >= 400`** — many OOS bar predictions accumulated across folds (magnitude regression guard).
11. **`float(out["oos_preds_a"].min())`** — `min`/`max` on numpy arrays; cast to Python `float` for the assertion message path; probabilities must stay in **[0, 1]**.

### 11.6 `test_features_exclude_sentiment` (lines 107–112)

- **Docstring under `def`:** Explains *why* the test exists (leakage / FinBERT).
- **`assert "sentiment" not in feat.columns`** — column name must not appear in engineered features.
- **`assert "sentiment" not in ML_COLS`** — the canonical feature list must not silently re-include sentiment.

### 11.7 `test_full_optimizer_end_to_end` (lines 119–158)

- **`@pytest.mark.integration`:** *Decorator* — wraps the function so pytest attaches metadata. Paired with `-m "not integration"` to skip.
- **`symbol = "FORM"`** — ticker string passed to yfinance / cache.
- **`curr_date = "2026-04-18"`** — “as of” date; downloader uses it to bound history.
- **`delete_rsi_dt_params(symbol)`** — clears SQLite row so the run cannot reuse thresholds (`cache_hit` path) without recomputing WFO.
- **`out = run_rsi_dt_optimizer(...)`** — calls the full production pipeline.
- **`summary = {k: v for k, v in out.items() if k != "slides"}`** — dict comprehension; drops bulky per-fold list for printing.
- **`summary["n_slides_recorded"] = len(out.get("slides", []))`** — `dict.get` with default if key missing (defensive); adds a count field for humans.
- **`print(...)`** — only visible with `pytest -s` (or similar).
- **`required = {"symbol", "curr_date", ...}`** — a `set` literal of required string keys (API contract).
- **`missing = required - set(out.keys())`** — set difference: keys we require but `out` does not provide.
- **`assert not missing, f"Missing keys: {missing}"`** — empty set is falsy; if non-empty, assertion fails with helpful message.
- **`assert out["last_signal"] in (0, 1)`** — integer binary signal.
- **`assert 0.0 <= out["last_prob_a"] <= 1.0`** — chained comparison; probability bounds.
- **`assert out["n_slides"] >= 20`** — end-to-end WFO must have run many folds on ~8y of daily data.
- **`assert out["confidence"] in ("HIGH", "MEDIUM", "LOW")`** — categorical confidence tier from optimizer logic.
- **`assert set(out["feature_names"]) == set(ML_COLS)`** — API-reported feature names match the training column list (no drift).
- **`cached = get_rsi_dt_params(symbol)`** — read back what was stored.
- **`assert cached is not None`** — cache row exists after fresh optimization.
- **`pytest.approx(out["threshold_a"], abs=1e-4)`** — floating-point *approximate* equality within absolute tolerance `1e-4` (SQLite/rounding noise).

---

## 12. Further reading

- **Triple Barrier method:** Marcos López de Prado, *Advances in Financial Machine Learning* (2018), Ch. 3.
- **Walk-forward analysis:** Robert Pardo, *The Evaluation and Optimization of Trading Strategies* (2008).
- **XGBoost:** [xgboost.readthedocs.io](https://xgboost.readthedocs.io/) — start with the [Introduction to Boosted Trees](https://xgboost.readthedocs.io/en/stable/tutorials/model.html) page.
- **Sharpe ratio:** [investopedia.com/terms/s/sharperatio.asp](https://www.investopedia.com/terms/s/sharperatio.asp)
- **Data leakage in finance:** López de Prado Ch. 7 (fractionally differentiated features) — explains why "use train/test split" is harder than it sounds for time-series.
