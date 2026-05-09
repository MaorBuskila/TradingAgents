"""
tests/test_sniper_composite.py
==============================
Unit tests for the 10-factor composite scoring engine and A+/A/B/C grade map.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.quant_ml.composite.sniper_score import (
    DEFAULT_WEIGHTS,
    SniperBarInputs,
    compute_sniper_score,
    compute_sniper_score_series,
    grade_from_score,
)


def _all_bull_bar() -> SniperBarInputs:
    """Bar where every bull factor is True."""
    return SniperBarInputs(
        close=105.0,
        ema_f=104.0,
        ema_s=102.0,
        ema_t=100.0,
        rsi=60.0,           # in zone (50, 75)
        macd_hist=0.5,
        macd_line=1.0,
        macd_signal=0.5,
        vwap=103.0,
        volume_burst=True,
        adx=30.0,
        plus_di=25.0,
        minus_di=10.0,
        htf_bias_bull=True,
    )


def _all_bear_bar() -> SniperBarInputs:
    """Bar where every bear factor is True."""
    return SniperBarInputs(
        close=95.0,
        ema_f=96.0,
        ema_s=98.0,
        ema_t=100.0,
        rsi=40.0,           # in bear zone (25, 50)
        macd_hist=-0.5,
        macd_line=-1.0,
        macd_signal=-0.5,
        vwap=97.0,
        volume_burst=True,
        adx=30.0,
        plus_di=10.0,
        minus_di=25.0,
        htf_bias_bull=False,
    )


# ── Grade map ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "score,expected",
    [
        (10.0, "A+"),
        (8.0, "A+"),
        (7.99, "A"),
        (6.5, "A"),
        (6.49, "B"),
        (5.0, "B"),
        (4.99, "C"),
        (0.0, "C"),
    ],
)
def test_grade_thresholds(score, expected):
    assert grade_from_score(score) == expected


# ── Single-bar scoring ─────────────────────────────────────────────────────

def test_all_bull_factors_yield_max_score_and_Aplus():
    out = compute_sniper_score(_all_bull_bar())
    assert pytest.approx(out["bull_score"], abs=1e-6) == 10.0
    assert out["bull_grade"] == "A+"
    # bear side shouldn't be zero-checked — volume_burst is side-independent
    assert out["bear_score"] < 10.0


def test_all_bear_factors_yield_max_bear_score():
    out = compute_sniper_score(_all_bear_bar())
    assert pytest.approx(out["bear_score"], abs=1e-6) == 10.0
    assert out["bear_grade"] == "A+"


def test_flipping_htf_drops_1_5_points():
    bar = _all_bull_bar()
    full = compute_sniper_score(bar)["bull_score"]
    bar.htf_bias_bull = False
    flipped = compute_sniper_score(bar)["bull_score"]
    assert pytest.approx(full - flipped, abs=1e-6) == 1.5


def test_flipping_close_vs_ema_f_drops_half_point():
    bar = _all_bull_bar()
    full = compute_sniper_score(bar)["bull_score"]
    bar.close = bar.ema_f - 0.01  # below fast EMA
    # but keep close > ema_t (trend) — otherwise two factors flip
    assert bar.close > bar.ema_t
    flipped = compute_sniper_score(bar)["bull_score"]
    assert pytest.approx(full - flipped, abs=1e-6) == 0.5


def test_weights_sum_to_ten():
    assert abs(sum(DEFAULT_WEIGHTS.values()) - 10.0) < 1e-9


# ── Vectorized scoring ─────────────────────────────────────────────────────

def test_vectorized_matches_per_bar():
    """Build a tiny 3-row DataFrame and verify row-wise equivalence."""
    rows = []
    bars = [_all_bull_bar(), _all_bear_bar(), _all_bull_bar()]
    for b in bars:
        rows.append(
            {
                "Close": b.close,
                "ema_f": b.ema_f,
                "ema_s": b.ema_s,
                "ema_t": b.ema_t,
                "rsi": b.rsi,
                "macd_hist": b.macd_hist,
                "macd_line": b.macd_line,
                "macd_signal": b.macd_signal,
                "vwap": b.vwap,
                "volume_burst": b.volume_burst,
                "adx": b.adx,
                "plus_di": b.plus_di,
                "minus_di": b.minus_di,
                "htf_bias": b.htf_bias_bull,
            }
        )
    df = pd.DataFrame(rows)
    series = compute_sniper_score_series(df)
    assert len(series) == 3
    assert series["bull_grade"].iloc[0] == "A+"
    assert series["bear_grade"].iloc[1] == "A+"
    assert series["bull_grade"].iloc[2] == "A+"
