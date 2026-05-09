"""
vol_regime.py
-------------
Volatility regime classifier.

regime = ATR(atr_len) / SMA(ATR(atr_len), sma_len)
  HIGH    : regime > high_th  (default 1.3)   — consider wider stops
  NORMAL  : otherwise
  LOW     : regime < low_th   (default 0.7)   — compression, watch for breakout
"""

from __future__ import annotations

import pandas as pd


def _wilder_atr(df: pd.DataFrame, length: int) -> pd.Series:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    prev_close = df["Close"].astype(float).shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / length, adjust=False).mean()


def compute_vol_regime(
    df: pd.DataFrame,
    atr_len: int = 14,
    sma_len: int = 42,
    high_th: float = 1.3,
    low_th: float = 0.7,
) -> pd.Series:
    """Return a Series of regime labels: "HIGH" | "NORMAL" | "LOW".

    Args:
        df: must contain High, Low, Close.
        atr_len: ATR Wilder length (default 14).
        sma_len: SMA window over ATR (default 42).
        high_th/low_th: regime thresholds (default 1.3 / 0.7).
    """
    atr = _wilder_atr(df, atr_len)
    atr_sma = atr.rolling(window=sma_len, min_periods=sma_len).mean()
    ratio = atr / atr_sma.replace(0, 1e-10)

    labels = pd.Series(["NORMAL"] * len(df), index=df.index, name="vol_regime")
    labels[ratio > high_th] = "HIGH"
    labels[ratio < low_th] = "LOW"
    labels[ratio.isna()] = "NORMAL"
    return labels


def compute_vol_regime_ratio(df: pd.DataFrame, atr_len: int = 14, sma_len: int = 42) -> pd.Series:
    """Return the raw ATR/SMA(ATR) ratio. Useful for DT features."""
    atr = _wilder_atr(df, atr_len)
    atr_sma = atr.rolling(window=sma_len, min_periods=sma_len).mean()
    ratio = (atr / atr_sma.replace(0, 1e-10)).fillna(1.0)
    ratio.name = "vol_regime_ratio"
    return ratio
