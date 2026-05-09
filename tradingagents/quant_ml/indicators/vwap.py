"""
vwap.py
-------
Session-anchored VWAP using hlc3 and volume.

On daily bars, each bar IS a full session, so session-VWAP collapses to hlc3 —
useless for confluence. We instead compute a rolling/anchored VWAP where the
anchor resets every N bars (N=1 for intraday would be session; on daily we use
N = anchor_period, default 20 trading days, i.e. a rolling monthly VWAP).

The original TradingView script uses hlc3 * volume / cumulative volume reset
per session; on daily the analog is a rolling volume-weighted mean over a
window long enough to smooth single-day noise but short enough to still reflect
"institutional fair value" for the current trend.
"""

from __future__ import annotations

import pandas as pd


def compute_session_vwap(df: pd.DataFrame, anchor_period: int = 20) -> pd.Series:
    """Anchored VWAP on daily bars — rolling hlc3 * volume / sum(volume).

    Args:
        df: must contain High, Low, Close, Volume columns.
        anchor_period: lookback window in bars. 20 ≈ 1 trading month.

    Returns:
        Series aligned with df.index. NaN for the first (anchor_period - 1) bars.
    """
    if anchor_period < 1:
        raise ValueError("anchor_period must be >= 1")

    hlc3 = (df["High"] + df["Low"] + df["Close"]) / 3.0
    vol = df["Volume"].astype(float).replace(0, 1e-10)
    pv = hlc3 * vol

    cum_pv = pv.rolling(window=anchor_period, min_periods=anchor_period).sum()
    cum_v = vol.rolling(window=anchor_period, min_periods=anchor_period).sum()

    vwap = cum_pv / cum_v
    vwap.name = "vwap"
    return vwap
