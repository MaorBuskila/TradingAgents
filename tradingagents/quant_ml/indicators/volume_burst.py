"""
volume_burst.py
---------------
Participation filter: True when current bar's volume exceeds
SMA(Volume, sma_len) * mult.

Default mult = 1.2 matches the Precision Sniper TradingView script.
"""

from __future__ import annotations

import pandas as pd


def compute_volume_burst(
    df: pd.DataFrame,
    sma_len: int = 20,
    mult: float = 1.2,
) -> pd.Series:
    """Return a boolean Series: True where Volume > SMA(Volume, sma_len) * mult.

    Args:
        df: must contain Volume column.
        sma_len: SMA window (default 20).
        mult: multiplier threshold (default 1.2).

    Returns:
        Series[bool] aligned with df.index. NaN SMA warmup bars return False.
    """
    vol = df["Volume"].astype(float)
    sma = vol.rolling(window=sma_len, min_periods=sma_len).mean()
    burst = (vol > sma * mult).fillna(False)
    burst.name = "volume_burst"
    return burst
