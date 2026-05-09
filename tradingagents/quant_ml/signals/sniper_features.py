"""
sniper_features.py
------------------
Build the Precision Sniper feature DataFrame from OHLCV (daily).

Used by integration tests, classical WFO optimizer, DT optimizer, and live signals.
"""

from __future__ import annotations

import pandas as pd

from tradingagents.quant_ml.composite.sniper_score import compute_sniper_score_series
from tradingagents.quant_ml.indicators.adx_di import compute_adx_di
from tradingagents.quant_ml.indicators.ema_stack import compute_ema_stack
from tradingagents.quant_ml.indicators.htf_ema_bias import compute_htf_ema_bias
from tradingagents.quant_ml.indicators.volume_burst import compute_volume_burst
from tradingagents.quant_ml.indicators.vwap import compute_session_vwap
from tradingagents.quant_ml.optimizers.macd_optimizer_algo import _calc_macd
from tradingagents.quant_ml.optimizers.rsi_optimizer_algo import _calc_rsi


def build_sniper_base_indicators(raw: pd.DataFrame) -> pd.DataFrame:
    """OHLCV + RSI/MACD/VWAP/ADX/HTF (no EMA stack, no volume burst, no scores)."""
    df = raw.sort_values("Date").reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df["rsi"] = _calc_rsi(df["Close"].astype(float), 13)
    m_line, m_sig, m_hist = _calc_macd(df["Close"].astype(float), 12, 26, 9)
    df["macd_line"] = m_line
    df["macd_signal"] = m_sig
    df["macd_hist"] = m_hist
    df["vwap"] = compute_session_vwap(df, anchor_period=20)
    adx_part = compute_adx_di(df, length=14)
    df["adx"] = adx_part["adx"]
    df["plus_di"] = adx_part["plus_di"]
    df["minus_di"] = adx_part["minus_di"]
    df["htf_bias"] = compute_htf_ema_bias(df, htf="W", fast=9, slow=21).values
    return df


def build_sniper_feature_frame(
    raw: pd.DataFrame,
    *,
    fast: int = 9,
    slow: int = 21,
    trend: int = 55,
    vol_mult: float = 1.2,
) -> pd.DataFrame:
    """Return OHLCV plus Sniper indicators and bull/bear scores.

    Args:
        raw: columns Date, Open, High, Low, Close, Volume (Date parseable).
        fast/slow/trend: EMA stack periods.
        vol_mult: volume burst threshold multiplier.
    """
    base = build_sniper_base_indicators(raw)
    df = compute_ema_stack(base, fast=fast, slow=slow, trend=trend)
    df["volume_burst"] = compute_volume_burst(df, sma_len=20, mult=vol_mult)
    scores = compute_sniper_score_series(df)
    return pd.concat([df, scores], axis=1)


def finalize_sniper_frame(
    base: pd.DataFrame,
    *,
    fast: int,
    slow: int,
    trend: int,
    vol_mult: float,
) -> pd.DataFrame:
    """Cheaper than full reload: reuse `build_sniper_base_indicators` output."""
    df = compute_ema_stack(base, fast=fast, slow=slow, trend=trend)
    df["volume_burst"] = compute_volume_burst(df, sma_len=20, mult=vol_mult)
    scores = compute_sniper_score_series(df)
    return pd.concat([df, scores], axis=1)
