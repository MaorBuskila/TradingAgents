"""
structure_sl.py
---------------
Structure-based stop loss with ATR fallback.

For LONGS:
  swing_low = min(Low over `lookback` bars ending at entry bar)
  structure_stop = swing_low - atr_pad * atr
  atr_stop = entry - atr * atr_mult
  SL = max(structure_stop, atr_stop)  # TIGHTER of the two
  Floor: entry - SL >= atr_floor * atr  (min distance from entry)

For SHORTS: mirror (swing_high, add atr_pad, min entry+atr_mult*atr).
"""

from __future__ import annotations

import pandas as pd


def _wilder_atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
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


def compute_stop(
    side: str,
    entry: float,
    df: pd.DataFrame,
    entry_idx: int | None = None,
    lookback: int = 10,
    atr_mult: float = 1.5,
    atr_pad: float = 0.2,
    atr_floor: float = 0.5,
    atr_len: int = 14,
) -> float:
    """Compute structure+ATR hybrid stop.

    Args:
        side:     "long" or "short".
        entry:    entry price.
        df:       OHLC DataFrame up to and including the entry bar.
        entry_idx: iloc index of the entry bar in df. Defaults to last row.
        lookback: swing lookback window in bars.
        atr_mult: ATR multiplier for the ATR-only stop.
        atr_pad:  ATR padding added below swing-low / above swing-high.
        atr_floor: minimum SL distance from entry as multiple of ATR.
        atr_len:  Wilder ATR length.

    Returns:
        Stop-loss price (float).
    """
    if side not in ("long", "short"):
        raise ValueError("side must be 'long' or 'short'")
    if entry_idx is None:
        entry_idx = len(df) - 1
    if entry_idx < 1:
        raise ValueError("Need at least 2 bars of history to compute ATR/structure.")

    atr_series = _wilder_atr(df, atr_len)
    atr_val = float(atr_series.iloc[entry_idx])
    if not (atr_val > 0):
        atr_val = float(df["Close"].iloc[entry_idx]) * 0.01  # 1% fallback

    start = max(0, entry_idx - lookback + 1)
    window = df.iloc[start : entry_idx + 1]

    if side == "long":
        swing = float(window["Low"].min())
        structure_stop = swing - atr_pad * atr_val
        atr_stop = entry - atr_mult * atr_val
        sl = max(structure_stop, atr_stop)  # tighter = higher for longs
        floor_sl = entry - atr_floor * atr_val
        sl = min(sl, floor_sl)  # enforce minimum distance
        sl = min(sl, entry - 1e-8)  # SL must be below entry
    else:
        swing = float(window["High"].max())
        structure_stop = swing + atr_pad * atr_val
        atr_stop = entry + atr_mult * atr_val
        sl = min(structure_stop, atr_stop)  # tighter = lower for shorts
        floor_sl = entry + atr_floor * atr_val
        sl = max(sl, floor_sl)
        sl = max(sl, entry + 1e-8)

    return float(sl)
