"""
sniper_sim.py
-------------
Long-only first-touch R simulation for Precision Sniper (matches integration test).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tradingagents.quant_ml.risk.structure_sl import compute_stop
from tradingagents.quant_ml.risk.tp_ladder import compute_tps


def simulate_long_trade_r(
    df: pd.DataFrame,
    entry_i: int,
    *,
    sl_mult: float = 1.5,
) -> tuple[float, bool, int]:
    """Entry at bar `entry_i` open. Returns (R multiple, tp1_hit, exit_bar_idx)."""
    if entry_i >= len(df):
        return 0.0, False, max(0, entry_i - 1)
    entry = float(df["Open"].iloc[entry_i])
    hist = df.iloc[: entry_i + 1]
    sl = compute_stop("long", entry, hist, entry_idx=entry_i, atr_mult=sl_mult)
    tp1, tp2, tp3 = compute_tps("long", entry, sl)
    tp1_hit = False
    for j in range(entry_i + 1, len(df)):
        low = float(df["Low"].iloc[j])
        high = float(df["High"].iloc[j])
        if low <= sl and high >= tp1:
            return -1.0, tp1_hit, j
        if low <= sl:
            return -1.0, tp1_hit, j
        if high >= tp3:
            return 3.0, True, j
        if high >= tp2:
            return 2.0, True, j
        if high >= tp1:
            return 1.0, True, j
    return 0.0, tp1_hit, len(df) - 1


def run_long_backtest_rs(
    df: pd.DataFrame,
    *,
    start_i: int,
    end_i: int,
    min_score: float,
    sl_mult: float,
) -> list[float]:
    """Collect R-multiples for non-overlapping long trades in [start_i, end_i) on signal bars."""
    rs: list[float] = []
    next_allowed = -1
    for i in range(start_i, min(end_i, len(df) - 1)):
        if i < next_allowed:
            continue
        if not bool(df["cross_up"].iloc[i]):
            continue
        if float(df["bull_score"].iloc[i]) < min_score:
            continue
        entry_i = i + 1
        if entry_i >= len(df):
            break
        r_mult, _, exit_j = simulate_long_trade_r(df, entry_i, sl_mult=sl_mult)
        rs.append(float(r_mult))
        next_allowed = exit_j + 1
    return rs


def sharpe_r(rs: list[float]) -> float:
    """Heuristic Sharpe on R-multiples (not annualized)."""
    if len(rs) < 2:
        return 0.0
    a = np.array(rs, dtype=float)
    s = float(a.std(ddof=0))
    if s < 1e-9:
        return 0.0
    return float(a.mean() / s * np.sqrt(min(len(a), 252)))


def confidence_tier(is_sharpe: float, oos_sharpe: float) -> str:
    if oos_sharpe > 0.5 and is_sharpe > 0.3:
        return "HIGH"
    if oos_sharpe > 0.2:
        return "MEDIUM"
    return "LOW"
