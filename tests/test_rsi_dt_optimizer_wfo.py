"""
Integration tests for the RSI DT optimizer (WFO + Triple Barrier + two-model ensemble).

What is tested
--------------
  - Triple Barrier labeling produces deterministic binary {0, 1} labels
  - WFO loop runs many overlapping folds, not a single train/test split
  - FinBERT sentiment is NOT a training feature (leakage guard)
  - run_rsi_dt_optimizer() returns every key the API and portfolio layer require,
    including the renamed rsi_params_nondefault (was wfo_params_used)
  - Cache hit reuses stored thresholds; cache miss triggers a fresh WFO run and
    writes avg_oos_acc to the DB so subsequent cache hits return the real value

Run commands (from project root)
---------------------------------
  # Fast unit tests only — no network, no model load (~3 s)
  .venv/bin/python -m pytest tests/test_rsi_dt_optimizer_wfo.py -v -m "not integration"

  # Full suite including the live yfinance + FinBERT integration test
  .venv/bin/python -m pytest tests/test_rsi_dt_optimizer_wfo.py -v -s --tb=short

  # Single integration test with Rich pretty-print output visible
  .venv/bin/python -m pytest tests/test_rsi_dt_optimizer_wfo.py::test_full_optimizer_end_to_end -v -s --tb=short

  # Stop at first failure
  .venv/bin/python -m pytest tests/test_rsi_dt_optimizer_wfo.py -x --tb=short

  # Run the optimizer and render the Rich report directly (no pytest)
  .venv/bin/python -c "
  from tradingagents.quant_ml.optimizers.rsi_dt_optimizer import run_rsi_dt_optimizer
  from tradingagents.quant_ml.optimizers.rsi_dt_report import pretty_print_rsi_dt
  import logging; logging.basicConfig(level=logging.WARNING)
  pretty_print_rsi_dt(run_rsi_dt_optimizer('NVDA', '2026-04-17'))
  "

Key pytest flags
----------------
  -v            verbose: show each test name and PASSED/FAILED
  -s            don't capture stdout — required to see the Rich report in the terminal
  -x            stop after the first failure
  --tb=short    compact traceback: failing line + assertion message only,
                no surrounding frames (avoids 40-line XGBoost/sklearn noise)
  --tb=long     full traceback with local variables (default; useful for deep bugs)
  -m "not integration"  skip tests marked @pytest.mark.integration (network-free)
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from tradingagents.quant_ml.optimizers.rsi_dt_optimizer import (
    ML_COLS,
    _build_features,
    _run_wfo,
    _triple_barrier_labels,
    run_rsi_dt_optimizer,
)
from tradingagents.quant_ml.optimizers.rsi_dt_report import pretty_print_rsi_dt
from tradingagents.dataflows.rsi_dt_cache import (
    get_rsi_dt_params,
)
from tradingagents.dataflows.rsi_cache import get_rsi_params

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

TEST_SYMBOL = "FORM"
TEST_DATE = "2026-04-18"

# Fallback values used only when no DB cache exists and network is unavailable.
_DEFAULT_RSI_PERIOD = 14
_DEFAULT_TP_MULT = 2.0
_DEFAULT_SL_MULT = 1.0
_DEFAULT_LABEL_HORIZON = 10
_DEFAULT_IS_DAYS = 1000
_DEFAULT_OOS_DAYS = 20


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_ohlc(n: int = 2200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    # Mild upward drift with regime shifts
    returns = rng.normal(0.0005, 0.015, size=n)
    returns[n // 3: 2 * n // 3] -= 0.002  # bear patch
    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(rng.normal(0, 0.006, size=n)))
    low = close * (1 - np.abs(rng.normal(0, 0.006, size=n)))
    idx = pd.date_range("2018-01-02", periods=n, freq="B")
    return pd.DataFrame({"Close": close, "High": high, "Low": low}, index=idx)


def _load_cached_params(symbol: str) -> dict:
    """
    Load optimized WFO and RSI params from the DB caches.

    Reads rsi_dt_params_cache for (tp_mult, sl_mult, label_horizon, training_days,
    step_days) and rsi_params_cache for the optimized RSI period.
    Falls back to module-level defaults when no cache entry exists so that unit
    tests remain runnable offline without network access.
    """
    dt_cached = get_rsi_dt_params(symbol)
    rsi_cached = get_rsi_params(symbol)
    return {
        # RSI WFO-optimized params — feed into DT optimizer so it runs on real signal params
        "rsi_period": int(rsi_cached["optimal_period"]) if rsi_cached else _DEFAULT_RSI_PERIOD,
        "rsi_upper":  float(rsi_cached["optimal_upper"])  if rsi_cached else 70.0,
        "rsi_lower":  float(rsi_cached["optimal_lower"])  if rsi_cached else 30.0,
        # DT WFO params
        "tp_mult":       float(dt_cached["tp_mult"])       if dt_cached and dt_cached.get("tp_mult")       is not None else _DEFAULT_TP_MULT,
        "sl_mult":       float(dt_cached["sl_mult"])       if dt_cached and dt_cached.get("sl_mult")       is not None else _DEFAULT_SL_MULT,
        "label_horizon": int(dt_cached["label_horizon"])   if dt_cached and dt_cached.get("label_horizon") is not None else _DEFAULT_LABEL_HORIZON,
        "is_days":       int(dt_cached["training_days"])   if dt_cached and dt_cached.get("training_days") is not None else _DEFAULT_IS_DAYS,
        "oos_days":      int(dt_cached["step_days"])       if dt_cached and dt_cached.get("step_days")     is not None else _DEFAULT_OOS_DAYS,
        "dt_cache_hit": dt_cached is not None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests (no network)
# ─────────────────────────────────────────────────────────────────────────────

def test_triple_barrier_produces_binary_labels():
    p = _load_cached_params(TEST_SYMBOL)
    df = _synthetic_ohlc(500)
    feat = _build_features(df, rsi_period=p["rsi_period"]).iloc[210:]
    labels = _triple_barrier_labels(
        close=feat["price"], high=feat["high"], low=feat["low"],
        atr=feat["atr_14"], horizon=p["label_horizon"],
        tp_mult=p["tp_mult"], sl_mult=p["sl_mult"],
    )
    valid = labels.dropna()
    assert len(valid) > 0
    assert set(valid.unique()).issubset({0, 1})
    # Last `label_horizon` rows should be unlabelable
    assert labels.iloc[-p["label_horizon"]:].isna().all()


def test_wfo_runs_many_folds_not_single_shot():
    p = _load_cached_params(TEST_SYMBOL)
    df = _synthetic_ohlc(2100)
    feat_full = _build_features(df, rsi_period=p["rsi_period"]).iloc[210:]
    labels = _triple_barrier_labels(
        close=feat_full["price"], high=feat_full["high"], low=feat_full["low"],
        atr=feat_full["atr_14"], horizon=p["label_horizon"],
        tp_mult=p["tp_mult"], sl_mult=p["sl_mult"],
    )
    mask = labels.notna() & feat_full[ML_COLS].notna().all(axis=1)
    feat_df = feat_full[mask].copy()
    y = labels[mask].astype(int)

    out = _run_wfo(feat_df, y, is_days=p["is_days"], oos_days=p["oos_days"])
    assert len(out["slides"]) >= 20, "WFO must produce many folds, not one"
    assert len(out["oos_preds_a"]) == len(out["oos_preds_b"])
    assert len(out["oos_preds_a"]) >= 400
    # Sanity: probabilities in [0, 1]
    assert float(out["oos_preds_a"].min()) >= 0.0
    assert float(out["oos_preds_a"].max()) <= 1.0


def test_features_exclude_sentiment():
    """Sentiment must NOT be in the training feature set (leakage fix)."""
    p = _load_cached_params(TEST_SYMBOL)
    df = _synthetic_ohlc(300)
    feat = _build_features(df, rsi_period=p["rsi_period"])
    assert "sentiment" not in feat.columns
    assert "sentiment" not in ML_COLS


def test_params_sourced_from_db_or_defaults():
    """Verify _load_cached_params always returns typed, in-range values."""
    p = _load_cached_params(TEST_SYMBOL)
    assert isinstance(p["rsi_period"], int) and 2 <= p["rsi_period"] <= 50
    assert isinstance(p["tp_mult"], float) and p["tp_mult"] > 0
    assert isinstance(p["sl_mult"], float) and p["sl_mult"] > 0
    assert isinstance(p["label_horizon"], int) and p["label_horizon"] > 0
    assert isinstance(p["is_days"], int) and p["is_days"] >= 500
    assert isinstance(p["oos_days"], int) and p["oos_days"] >= 5


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end integration (hits yfinance — network required)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.timeout(600)
def test_full_optimizer_end_to_end():
    symbol = TEST_SYMBOL
    curr_date = TEST_DATE

    # Load WFO-optimized RSI params (period/upper/lower) from rsi_params_cache and
    # DT WFO params (thresholds, window sizes) from rsi_dt_params_cache.
    # Falls back to defaults offline so unit tests stay network-free.
    p = _load_cached_params(symbol)
    cached_dt = get_rsi_dt_params(symbol)

    if cached_dt is None:
        # No DT cache — run fresh WFO with the RSI-WFO-optimized signal params.
        out = run_rsi_dt_optimizer(
            symbol, curr_date,
            is_days=_DEFAULT_IS_DAYS,
            oos_days=_DEFAULT_OOS_DAYS,
            rsi_period=p["rsi_period"],
            rsi_upper=p["rsi_upper"],
            rsi_lower=p["rsi_lower"],
        )
        expected_cache_hit = False
    else:
        # DT cache exists — pass all optimized params back so the optimizer
        # exercises the full cache-hit path with realistic signal configuration.
        out = run_rsi_dt_optimizer(
            symbol, curr_date,
            is_days=p["is_days"],
            oos_days=p["oos_days"],
            tp_mult=p["tp_mult"],
            sl_mult=p["sl_mult"],
            label_horizon=p["label_horizon"],
            rsi_period=p["rsi_period"],
            rsi_upper=p["rsi_upper"],
            rsi_lower=p["rsi_lower"],
        )
        expected_cache_hit = True

    # Pretty-print the full result for human inspection (visible with pytest -s).
    pretty_print_rsi_dt(out)

    # API-contract keys the portfolio + response layer consume
    required = {
        "symbol", "curr_date", "last_signal", "last_prob_a", "last_prob_b",
        "last_size_mult", "last_atr_pct", "last_rsi", "avg_oos_acc", "n_slides",
        "feature_names", "rsi_params", "rsi_params_nondefault",
        "is_days", "oos_days", "label_horizon",
        "min_prob_threshold", "slides",
        "threshold_a", "threshold_b", "oos_sharpe", "confidence", "cache_hit",
    }
    missing = required - set(out.keys())
    assert not missing, f"Missing keys: {missing}"

    assert out["last_signal"] in (0, 1)
    assert 0.0 <= out["last_prob_a"] <= 1.0
    assert 0.0 <= out["last_prob_b"] <= 1.0
    assert 0.0 <= out["threshold_a"] <= 1.0
    assert 0.0 <= out["threshold_b"] <= 1.0
    assert out["confidence"] in ("HIGH", "MEDIUM", "LOW")
    assert set(out["feature_names"]) == set(ML_COLS)
    assert out["cache_hit"] == expected_cache_hit
    # rsi_params_nondefault must reflect whether we passed non-default RSI signal params
    expected_nondefault = (p["rsi_period"] != 14 or p["rsi_upper"] != 70.0 or p["rsi_lower"] != 30.0)
    assert out["rsi_params_nondefault"] == expected_nondefault

    if not expected_cache_hit:
        # Fresh WFO run must produce many folds and populate the DB.
        assert out["n_slides"] >= 20, "should run many WFO folds over 2000 days"
        cached_after = get_rsi_dt_params(symbol)
        assert cached_after is not None
        assert cached_after["threshold_a"] == pytest.approx(out["threshold_a"], abs=1e-4)
    else:
        # Cache-hit path: output params must match what was stored in the DB.
        assert out["is_days"] == cached_dt["training_days"]
        assert out["oos_days"] == cached_dt["step_days"]
        assert out["threshold_a"] == pytest.approx(cached_dt["threshold_a"], abs=1e-4)
        assert out["threshold_b"] == pytest.approx(cached_dt["threshold_b"], abs=1e-4)
