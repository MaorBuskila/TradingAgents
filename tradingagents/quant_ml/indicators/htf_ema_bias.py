"""
htf_ema_bias.py
---------------
Higher-timeframe EMA bias with no look-ahead.

The Pine script uses request.security(htf, "close", lookahead_on)[1] to read
the CONFIRMED previous HTF bar — never the currently-forming one. We replicate
that by resampling to the higher timeframe, computing EMAs on that series,
forward-filling back to daily, then shifting one HTF-period so each daily bar
only sees the previous *completed* HTF bar.
"""

from __future__ import annotations

import pandas as pd


def compute_htf_ema_bias(
    daily_df: pd.DataFrame,
    htf: str = "W",
    fast: int = 9,
    slow: int = 21,
) -> pd.Series:
    """Return a boolean Series: True where HTF EMA fast > HTF EMA slow.

    Args:
        daily_df: must contain a Date column (datetime-parseable) and Close.
        htf:      pandas resample rule (e.g. "W" for weekly, "M" for monthly).
        fast/slow: EMA periods applied on the resampled series.

    Returns:
        Series[bool] aligned to daily_df.index. No future leakage: each bar
        reflects the *previous* confirmed HTF bar's EMA relationship.
    """
    df = daily_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    htf_close = df.set_index("Date")["Close"].resample(htf).last().dropna()
    if len(htf_close) < slow + 1:
        return pd.Series([False] * len(df), index=daily_df.index, name="htf_bias")

    htf_ema_fast = htf_close.ewm(span=fast, adjust=False).mean()
    htf_ema_slow = htf_close.ewm(span=slow, adjust=False).mean()
    htf_bias = (htf_ema_fast > htf_ema_slow).shift(1)
    htf_bias = htf_bias.infer_objects(copy=False).fillna(False).astype(bool)

    mapped = htf_bias.reindex(df["Date"], method="ffill")
    mapped = mapped.infer_objects(copy=False).fillna(False).astype(bool).reset_index(drop=True)
    mapped.index = daily_df.index
    mapped.name = "htf_bias"
    return mapped.astype(bool)
