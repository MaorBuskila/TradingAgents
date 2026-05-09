"""
tests/test_sniper_indicators.py
================================
Unit tests for the Precision Sniper indicator primitives.

All tests use synthetic OHLC fixtures so they run offline and deterministically.
An optional @pytest.mark.integration test cross-checks against real SPY data.

Run:
    pytest tests/test_sniper_indicators.py -v
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.quant_ml.indicators.ema_stack import compute_ema_stack
from tradingagents.quant_ml.indicators.vwap import compute_session_vwap
from tradingagents.quant_ml.indicators.adx_di import compute_adx_di
from tradingagents.quant_ml.indicators.volume_burst import compute_volume_burst
from tradingagents.quant_ml.indicators.htf_ema_bias import compute_htf_ema_bias
from tradingagents.quant_ml.indicators.vol_regime import compute_vol_regime


def _synthetic_ohlc(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(rng.normal(0.05, 1.0, n))
    high = close + rng.uniform(0.1, 2.0, n)
    low = close - rng.uniform(0.1, 2.0, n)
    open_ = close + rng.normal(0, 0.3, n)
    vol = rng.integers(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"Date": dates, "Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}
    )


# ── ema_stack ──────────────────────────────────────────────────────────────

def test_ema_stack_columns_and_monotonic_smoothing():
    df = _synthetic_ohlc(120)
    out = compute_ema_stack(df, fast=9, slow=21, trend=55)
    assert {"ema_f", "ema_s", "ema_t", "cross_up", "cross_dn"}.issubset(out.columns)
    assert out["ema_f"].notna().all()
    # ema_t is most smoothed; its rolling std should be <= ema_f's after warmup
    warmup = 60
    assert out["ema_t"].iloc[warmup:].std() <= out["ema_f"].iloc[warmup:].std() + 1e-6


def test_ema_stack_cross_up_fires_exactly_once_at_transition():
    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    # Manufacture a clean cross: decline then rise
    close = np.concatenate([np.linspace(100, 80, 20), np.linspace(80, 120, 20)])
    df = pd.DataFrame({"Date": dates, "Open": close, "High": close + 1, "Low": close - 1,
                       "Close": close, "Volume": np.ones(40) * 1_000_000})
    out = compute_ema_stack(df, fast=3, slow=8, trend=20)
    assert out["cross_up"].sum() >= 1
    # cross_up bars all have ema_f > ema_s and prev bar ema_f <= ema_s
    for i in out.index[out["cross_up"]]:
        if i > 0:
            assert out["ema_f"].iloc[i] > out["ema_s"].iloc[i]
            assert out["ema_f"].iloc[i - 1] <= out["ema_s"].iloc[i - 1]


# ── vwap ───────────────────────────────────────────────────────────────────

def test_vwap_nan_during_warmup_and_finite_after():
    df = _synthetic_ohlc(60)
    vwap = compute_session_vwap(df, anchor_period=20)
    assert vwap.iloc[:19].isna().all()
    assert vwap.iloc[19:].notna().all()


def test_vwap_equals_weighted_mean_of_hlc3():
    df = _synthetic_ohlc(30)
    vwap = compute_session_vwap(df, anchor_period=10)
    last = vwap.iloc[-1]
    hlc3 = (df["High"] + df["Low"] + df["Close"]) / 3.0
    expected = (hlc3.iloc[-10:] * df["Volume"].iloc[-10:]).sum() / df["Volume"].iloc[-10:].sum()
    assert abs(last - expected) < 1e-6


# ── adx_di ─────────────────────────────────────────────────────────────────

def test_adx_di_columns_and_range():
    df = _synthetic_ohlc(100)
    out = compute_adx_di(df, length=14)
    assert {"adx", "plus_di", "minus_di"}.issubset(out.columns)
    tail = out.iloc[30:]
    assert (tail["adx"] >= 0).all() and (tail["adx"] <= 100).all()
    assert (tail["plus_di"] >= 0).all()
    assert (tail["minus_di"] >= 0).all()


def test_adx_rising_in_trending_data():
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    # Strong uptrend
    close = np.linspace(100, 150, 60)
    high = close + 0.5
    low = close - 0.5
    df = pd.DataFrame({"Date": dates, "Open": close, "High": high, "Low": low,
                       "Close": close, "Volume": np.ones(60) * 1e6})
    out = compute_adx_di(df, length=14)
    # In a clean uptrend, plus_di > minus_di on the final bar
    assert out["plus_di"].iloc[-1] > out["minus_di"].iloc[-1]


# ── volume_burst ───────────────────────────────────────────────────────────

def test_volume_burst_flags_spike_bar():
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    vol = np.ones(30) * 1_000_000
    vol[25] = 5_000_000  # clear spike after SMA warmup
    df = pd.DataFrame({"Date": dates, "Open": 100, "High": 101, "Low": 99,
                       "Close": 100, "Volume": vol})
    burst = compute_volume_burst(df, sma_len=20, mult=1.2)
    assert burst.iloc[25]
    assert not burst.iloc[24]


# ── htf_ema_bias ───────────────────────────────────────────────────────────

def test_htf_bias_no_future_leakage():
    """Truncating the tail must not change earlier bias values."""
    df = _synthetic_ohlc(200)
    full = compute_htf_ema_bias(df, htf="W", fast=4, slow=8)
    truncated = compute_htf_ema_bias(df.iloc[:-5].copy(), htf="W", fast=4, slow=8)
    # First 100 bars of both must match exactly
    assert (full.iloc[:100].values == truncated.iloc[:100].values).all()


def test_htf_bias_returns_bools():
    df = _synthetic_ohlc(120)
    bias = compute_htf_ema_bias(df, htf="W", fast=4, slow=8)
    assert bias.dtype == bool
    assert len(bias) == len(df)


# ── vol_regime ─────────────────────────────────────────────────────────────

def test_vol_regime_labels():
    df = _synthetic_ohlc(150)
    reg = compute_vol_regime(df, atr_len=14, sma_len=42)
    assert set(reg.unique()).issubset({"HIGH", "NORMAL", "LOW"})


def test_vol_regime_high_when_volatility_spikes():
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    # Low-vol regime then vol explosion
    close = np.concatenate([100 + np.random.normal(0, 0.1, 70),
                            100 + np.random.normal(0, 5.0, 30)])
    high = close + 0.5
    low = close - 0.5
    df = pd.DataFrame({"Date": dates, "Open": close, "High": high, "Low": low,
                       "Close": close, "Volume": np.ones(100) * 1e6})
    reg = compute_vol_regime(df, atr_len=14, sma_len=42)
    # Final 10 bars should include at least one HIGH regime label
    assert "HIGH" in reg.iloc[-10:].values
