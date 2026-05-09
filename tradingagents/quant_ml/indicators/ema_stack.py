"""
ema_stack.py
------------
3-EMA trend structure for the Precision Sniper composite engine.

fast  = short-term momentum   (default 9)
slow  = medium-term trend     (default 21)
trend = macro filter          (default 55)

Crossover semantics match the Pine script: cross_up fires only on the bar
where fast crosses *above* slow (prev bar was fast <= slow, this bar is fast > slow).
"""

from __future__ import annotations

import pandas as pd


def compute_ema_stack(
    df: pd.DataFrame,
    fast: int = 9,
    slow: int = 21,
    trend: int = 55,
) -> pd.DataFrame:
    """Add ema_f/ema_s/ema_t plus cross_up/cross_dn flags to a copy of df.

    Args:
        df: must contain a "Close" column.
        fast/slow/trend: EMA periods.

    Returns:
        DataFrame with original columns plus:
            ema_f, ema_s, ema_t  — EMA series
            cross_up             — bool, fast crosses above slow on this bar
            cross_dn             — bool, fast crosses below slow on this bar
    """
    out = df.copy()
    close = out["Close"].astype(float)

    out["ema_f"] = close.ewm(span=fast, adjust=False).mean()
    out["ema_s"] = close.ewm(span=slow, adjust=False).mean()
    out["ema_t"] = close.ewm(span=trend, adjust=False).mean()

    fast_above = out["ema_f"] > out["ema_s"]
    fast_above_prev = fast_above.shift(1, fill_value=False)
    out["cross_up"] = fast_above & ~fast_above_prev
    out["cross_dn"] = ~fast_above & fast_above_prev

    return out
