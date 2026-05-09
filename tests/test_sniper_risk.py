"""
tests/test_sniper_risk.py
=========================
Unit tests for the Precision Sniper risk module: structure SL, TP ladder,
progressive trailing state machine, and direction lock.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.quant_ml.risk.structure_sl import compute_stop
from tradingagents.quant_ml.risk.tp_ladder import compute_tps
from tradingagents.quant_ml.risk.trailing import ProgressiveTrail
from tradingagents.quant_ml.risk.direction_lock import DirectionLock


# ── structure_sl ───────────────────────────────────────────────────────────

def _constant_atr_df(n: int = 30, close: float = 100.0, atr_width: float = 1.0) -> pd.DataFrame:
    """OHLC with ~constant ATR so Wilder's smoothing converges to `atr_width`."""
    highs = np.full(n, close + atr_width / 2)
    lows = np.full(n, close - atr_width / 2)
    closes = np.full(n, float(close))
    return pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=n, freq="B"),
            "Open": closes,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": np.ones(n) * 1e6,
        }
    )


def test_structure_sl_long_below_entry_and_respects_swing_low():
    df = _constant_atr_df(n=30, close=100.0, atr_width=1.0)
    # Inject a known swing low
    df.loc[25, "Low"] = 97.5
    sl = compute_stop(
        side="long",
        entry=100.0,
        df=df,
        lookback=10,
        atr_mult=1.5,
        atr_pad=0.2,
        atr_floor=0.5,
    )
    assert sl < 100.0
    # Should sit near swing_low - 0.2*atr ≈ 97.3 (tighter than entry - 1.5*atr = 98.5)
    # structure_stop = max(97.3, 98.5) = 98.5, floor=entry-0.5=99.5 → min(98.5, 99.5) = 98.5
    assert sl <= 99.5 + 1e-6


def test_structure_sl_short_above_entry():
    df = _constant_atr_df(n=30, close=100.0, atr_width=1.0)
    df.loc[25, "High"] = 102.5
    sl = compute_stop(side="short", entry=100.0, df=df)
    assert sl > 100.0


# ── tp_ladder ──────────────────────────────────────────────────────────────

def test_tp_ladder_long():
    tp1, tp2, tp3 = compute_tps("long", entry=100.0, sl=95.0)
    assert (tp1, tp2, tp3) == (105.0, 110.0, 115.0)


def test_tp_ladder_short():
    tp1, tp2, tp3 = compute_tps("short", entry=100.0, sl=105.0)
    assert (tp1, tp2, tp3) == (95.0, 90.0, 85.0)


def test_tp_ladder_custom_ratios():
    tp1, tp2, tp3 = compute_tps("long", entry=100.0, sl=98.0, ratios=(0.5, 1.5, 4.0))
    assert tp1 == 101.0 and tp2 == 103.0 and tp3 == 108.0


def test_tp_ladder_invalid_risk():
    with pytest.raises(ValueError):
        compute_tps("long", entry=100.0, sl=100.0)


# ── ProgressiveTrail ───────────────────────────────────────────────────────

def test_trail_stops_out_on_first_bar_below_sl():
    trail = ProgressiveTrail(side="long", entry=100, sl=95, tp1=105, tp2=110, tp3=115)
    ev = trail.on_bar(high=99, low=94.9)
    assert ev == "SL_HIT"
    assert trail.closed
    assert trail.r_multiple() == -1.0


def test_trail_hits_tp1_then_sl_moves_to_entry():
    trail = ProgressiveTrail(side="long", entry=100, sl=95, tp1=105, tp2=110, tp3=115)
    ev = trail.on_bar(high=106, low=99)  # reach TP1
    assert ev == "TP1"
    assert trail.tp1_hit and not trail.closed
    assert trail.sl == 100.0  # moved to entry
    # Next bar pulls back — should stop out at breakeven, not below
    ev2 = trail.on_bar(high=101, low=99.5)
    assert ev2 == "SL_HIT"
    assert trail.r_multiple() == 1.0


def test_trail_full_ladder_to_tp3():
    trail = ProgressiveTrail(side="long", entry=100, sl=95, tp1=105, tp2=110, tp3=115)
    trail.on_bar(high=106, low=99)      # TP1
    trail.on_bar(high=111, low=104)     # TP2
    ev = trail.on_bar(high=116, low=109)  # TP3 → close
    assert ev == "TP3"
    assert trail.closed
    assert trail.r_multiple() == 3.0


def test_trail_short_path():
    trail = ProgressiveTrail(side="short", entry=100, sl=105, tp1=95, tp2=90, tp3=85)
    ev = trail.on_bar(high=101, low=94)  # TP1
    assert ev == "TP1"
    assert trail.sl == 100.0  # breakeven
    # Pullback up to 100.1 — stops out at breakeven
    ev2 = trail.on_bar(high=100.5, low=99)
    assert ev2 == "SL_HIT"
    assert trail.r_multiple() == 1.0


def test_trail_no_action_on_boring_bar():
    trail = ProgressiveTrail(side="long", entry=100, sl=95, tp1=105, tp2=110, tp3=115)
    ev = trail.on_bar(high=102, low=98)
    assert ev is None
    assert not trail.closed


# ── DirectionLock ──────────────────────────────────────────────────────────

def test_direction_lock_first_trade_any_side_allowed():
    lock = DirectionLock()
    assert lock.can_enter("long")
    assert lock.can_enter("short")


def test_direction_lock_blocks_reentry_while_open():
    lock = DirectionLock()
    lock.on_enter("long")
    assert not lock.can_enter("long")
    assert not lock.can_enter("short")
    lock.on_exit()
    # now can enter — but only the opposite side
    assert not lock.can_enter("long")
    assert lock.can_enter("short")


def test_direction_lock_alternation_after_exit():
    lock = DirectionLock()
    lock.on_enter("long")
    lock.on_exit()
    assert lock.can_enter("short") and not lock.can_enter("long")
    lock.on_enter("short")
    lock.on_exit()
    assert lock.can_enter("long") and not lock.can_enter("short")


def test_direction_lock_double_enter_raises():
    lock = DirectionLock()
    lock.on_enter("long")
    with pytest.raises(RuntimeError):
        lock.on_enter("short")
