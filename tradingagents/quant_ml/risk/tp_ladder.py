"""
tp_ladder.py
------------
Three take-profit targets at configurable R:R multiples from entry.

risk = |entry - sl|
TP_i = entry + sign * risk * ratio_i

For longs, sign=+1; for shorts, sign=-1.
"""

from __future__ import annotations

from typing import Tuple


def compute_tps(
    side: str,
    entry: float,
    sl: float,
    ratios: Tuple[float, float, float] = (1.0, 2.0, 3.0),
) -> Tuple[float, float, float]:
    """Return (tp1, tp2, tp3) given side, entry, SL and R:R ratios."""
    if side not in ("long", "short"):
        raise ValueError("side must be 'long' or 'short'")
    risk = abs(entry - sl)
    if risk <= 0:
        raise ValueError(f"Invalid entry/SL: risk={risk}")
    sign = 1.0 if side == "long" else -1.0
    tps = tuple(float(entry + sign * risk * r) for r in ratios)
    return tps  # type: ignore[return-value]
