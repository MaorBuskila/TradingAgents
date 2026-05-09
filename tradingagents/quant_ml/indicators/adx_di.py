"""
adx_di.py
---------
ADX + Directional Indicators (+DI, -DI) with Wilder's smoothing.

The Precision Sniper uses ADX > 20 together with +DI > -DI (bullish) or
-DI > +DI (bearish) as its trend-strength confluence factor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_adx_di(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """Compute ADX, +DI, -DI with Wilder smoothing (alpha = 1/length).

    Args:
        df: must contain High, Low, Close columns.
        length: Wilder smoothing period (default 14).

    Returns:
        DataFrame with columns: adx, plus_di, minus_di  (aligned with df.index).
    """
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    alpha = 1.0 / length
    atr = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_dm_smooth = pd.Series(plus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean()
    minus_dm_smooth = pd.Series(minus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean()

    atr_safe = atr.replace(0, 1e-10)
    plus_di = 100.0 * (plus_dm_smooth / atr_safe)
    minus_di = 100.0 * (minus_dm_smooth / atr_safe)

    di_sum = (plus_di + minus_di).replace(0, 1e-10)
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum
    adx = dx.ewm(alpha=alpha, adjust=False).mean()

    out = pd.DataFrame(
        {"adx": adx, "plus_di": plus_di, "minus_di": minus_di},
        index=df.index,
    )
    return out
