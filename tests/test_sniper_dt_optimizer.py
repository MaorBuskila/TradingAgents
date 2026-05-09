"""Sniper DT optimizer — triple-barrier labels + WFO structure (fast tests)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.quant_ml.optimizers.rsi_dt_optimizer import _triple_barrier_labels
from tradingagents.quant_ml.optimizers.sniper_dt_optimizer import ML_COLS, _build_ml_features


def test_triple_barrier_binary():
    n = 400
    rng = np.random.default_rng(3)
    r = rng.normal(0.0003, 0.012, size=n)
    close = 100 * np.exp(np.cumsum(r))
    high = close * (1 + np.abs(rng.normal(0, 0.005, size=n)))
    low = close * (1 - np.abs(rng.normal(0, 0.005, size=n)))
    raw = pd.DataFrame(
        {
            "Date": pd.date_range("2019-01-02", periods=n, freq="B"),
            "Close": close,
            "High": high,
            "Low": low,
            "Open": close,
            "Volume": np.full(n, 1_000_000),
        }
    )
    feat = _build_ml_features(raw).iloc[210:]
    atr = (feat["atr_norm"] * feat["Close"]).squeeze()
    lab = _triple_barrier_labels(
        feat["Close"].squeeze(),
        feat["High"].squeeze(),
        feat["Low"].squeeze(),
        atr,
        horizon=10,
        tp_mult=1.5,
        sl_mult=1.0,
    )
    v = lab.dropna()
    assert len(v) > 0
    assert set(np.unique(v.to_numpy())).issubset({0.0, 1.0})


def test_ml_cols_present():
    n = 600
    rng = np.random.default_rng(1)
    r = rng.normal(0.0, 0.01, size=n)
    close = 50 * np.exp(np.cumsum(r))
    high = close * 1.002
    low = close * 0.998
    raw = pd.DataFrame(
        {
            "Date": pd.date_range("2018-01-02", periods=n, freq="B"),
            "Open": close,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, size=n),
        }
    )
    feat = _build_ml_features(raw)
    for c in ML_COLS:
        assert c in feat.columns


@pytest.mark.integration
@pytest.mark.timeout(600)
def test_sniper_dt_end_to_end_form():
    """Live SPY run — requires network + XGBoost."""
    pytest.importorskip("xgboost")
    from tradingagents.quant_ml.optimizers.sniper_dt_optimizer import run_sniper_dt_optimizer

    from tradingagents.dataflows.config import set_config
    from tradingagents.default_config import DEFAULT_CONFIG

    set_config(DEFAULT_CONFIG.copy())
    out = run_sniper_dt_optimizer("SPY", "2024-06-15", force_reoptimize=True, is_days=600, oos_days=15)
    assert out["symbol"] == "SPY"
    assert "last_prob_a" in out
    assert len(ML_COLS) == len(out["feature_names"])
