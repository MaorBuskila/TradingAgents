"""
ema_optimizer_algo.py
=====================
Walk-forward grid search over EMA stack parameters (fast / slow / trend).

Follows the same structure as rsi_optimizer_algo.py and macd_optimizer_algo.py:
  1. Load OHLCV price data
  2. Split into IS (in-sample) and OOS (out-of-sample) windows
  3. Grid search: EMA fast × slow × trend → score each combo with annualized Sharpe
  4. Report best IS params, validate on OOS, return confidence tier

Signal logic:
  LONG  when close > ema_fast  AND  ema_fast > ema_slow   (uptrend aligned)
  FLAT  otherwise
"""

from __future__ import annotations

import itertools
import logging
from typing import Annotated, Any

import numpy as np
import pandas as pd

from tradingagents.dataflows.ema_cache import upsert_ema_params
from tradingagents.quant_ml.indicators.stockstats_utils import yf_retry
from tradingagents.quant_ml.optimizers.rsi_optimizer_algo import (
    _calc_sharpe,
    _load_price_data,
)
from tradingagents.quant_ml.walk_forward.wfo_analyzer import analyze_wfo

log = logging.getLogger("ema_optimizer_algo")

# ─────────────────────────────────────────────────────────────────────────────
# Grid constants
# ─────────────────────────────────────────────────────────────────────────────
FAST_GRID  = [8, 9, 12]
SLOW_GRID  = [18, 21, 26, 34]   # constraint: slow > fast enforced below
TREND_GRID = [50, 55]

# Default EMA stack for baseline comparison
DEFAULT_FAST  = 9
DEFAULT_SLOW  = 21
DEFAULT_TREND = 55


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _calc_ema(closes: pd.Series, span: int) -> pd.Series:
    return closes.ewm(span=span, adjust=False).mean()


def _ema_positions(closes: pd.Series, fast: int, slow: int) -> pd.Series:
    """Long when close > ema_fast AND ema_fast > ema_slow, else flat."""
    ema_f = _calc_ema(closes, fast)
    ema_s = _calc_ema(closes, slow)
    long_signal = (closes > ema_f) & (ema_f > ema_s)
    # Vectorized ffill: hold position until condition flips
    positions = long_signal.astype(int)
    return positions


def _confidence_tier(is_sharpe: float, oos_sharpe: float) -> str:
    """Same thresholds as rsi_optimizer_algo."""
    if is_sharpe > 0 and oos_sharpe > 0:
        ratio = oos_sharpe / is_sharpe
        if oos_sharpe >= 0.5 and ratio >= 0.4:
            return "HIGH"
        if oos_sharpe >= 0.2 and ratio >= 0.2:
            return "MEDIUM"
        return "LOW"
    if oos_sharpe > 0.3:
        return "MEDIUM"
    return "LOW"


# ─────────────────────────────────────────────────────────────────────────────
# Main optimizer
# ─────────────────────────────────────────────────────────────────────────────

def run_algo_optimizer(
    symbol: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "YYYY-MM-DD"],
    is_days: Annotated[int, "in-sample window (trading days)"] = 180,
    oos_days: Annotated[int, "out-of-sample window (trading days)"] = 90,
) -> dict[str, Any]:
    """
    Grid-search EMA stack params; maximize IS Sharpe, validate on OOS.

    Grid:
      fast  ∈ [8, 9, 12]
      slow  ∈ [18, 21, 26, 34]  (slow > fast enforced)
      trend ∈ [50, 55]           (not used in signal, used in sniper features)
      Total valid combos: up to 3 × 4 × 2 = 24 (minus fast>=slow pairs)

    Returns dict with:
      optimal_fast, optimal_slow, optimal_trend,
      is_sharpe, oos_sharpe, confidence, combos_tested,
      default_is_sharpe, default_oos_sharpe, wfo_analysis
    """
    data = _load_price_data(symbol)
    curr_dt = pd.to_datetime(curr_date)
    data = data[data["Date"] <= curr_dt].copy()
    closes = data["Close"].reset_index(drop=True)
    total_days = len(closes)

    min_required = is_days + oos_days + max(TREND_GRID)
    if total_days < min_required:
        raise ValueError(
            f"Not enough data for {symbol}: {total_days} bars, need {min_required}"
        )

    oos_start = total_days - oos_days
    is_start  = max(oos_start - is_days, 0)

    # ── Grid search ──────────────────────────────────────────────────────────
    best_is  = -np.inf
    best_params: dict[str, Any] = {
        "optimal_fast": DEFAULT_FAST,
        "optimal_slow": DEFAULT_SLOW,
        "optimal_trend": DEFAULT_TREND,
    }
    combos_tested = 0
    best_oos = 0.0
    oos_positions_best: pd.Series | None = None

    for fast, slow in itertools.product(FAST_GRID, SLOW_GRID):
        if slow <= fast:
            continue
        positions = _ema_positions(closes, fast, slow)
        is_sh  = _calc_sharpe(closes.iloc[is_start:oos_start], positions.iloc[is_start:oos_start])
        oos_sh = _calc_sharpe(closes.iloc[oos_start:], positions.iloc[oos_start:])
        combos_tested += 1
        if is_sh > best_is:
            best_is = is_sh
            best_oos = oos_sh
            oos_positions_best = positions.iloc[oos_start:].copy()
            for trend in TREND_GRID:
                # trend doesn't affect signal Sharpe; pick the one with better IS
                # (they produce identical Sharpe — just record best trend separately)
                pass
            best_params = {
                "optimal_fast":  fast,
                "optimal_slow":  slow,
                "optimal_trend": DEFAULT_TREND,  # fixed until trend affects signal
            }

    # ── Best trend: pick whichever trend produces highest IS Sharpe ──────────
    # (trend is used in sniper features but not in the simple crossover signal;
    #  we still let users know which trend EMA was best in the feature frame)
    best_trend_sh = -np.inf
    best_trend = DEFAULT_TREND
    best_fast  = best_params["optimal_fast"]
    best_slow  = best_params["optimal_slow"]
    for trend in TREND_GRID:
        # Score: how often is close > trend EMA on IS window (alignment bonus)
        ema_t = _calc_ema(closes, trend)
        above_trend = (closes.iloc[is_start:oos_start] > ema_t.iloc[is_start:oos_start]).mean()
        if above_trend > best_trend_sh:
            best_trend_sh = above_trend
            best_trend = trend
    best_params["optimal_trend"] = best_trend

    # ── Baseline: default EMA(9/21) ──────────────────────────────────────────
    def_pos = _ema_positions(closes, DEFAULT_FAST, DEFAULT_SLOW)
    default_is_sharpe  = _calc_sharpe(closes.iloc[is_start:oos_start], def_pos.iloc[is_start:oos_start])
    default_oos_sharpe = _calc_sharpe(closes.iloc[oos_start:], def_pos.iloc[oos_start:])

    # ── Confidence ───────────────────────────────────────────────────────────
    confidence = _confidence_tier(best_is, best_oos)

    # ── WFO analysis (same as MACD algo) ─────────────────────────────────────
    oos_trade_count = int((oos_positions_best > 0).sum()) if oos_positions_best is not None else 0
    wfo = analyze_wfo(
        is_days=is_days,
        oos_days=oos_days,
        is_sharpe=float(best_is),
        oos_sharpe=float(best_oos),
        oos_trade_count=oos_trade_count,
        bar_frequency="daily",
    )

    # ── Cache result ──────────────────────────────────────────────────────────
    upsert_ema_params(
        ticker=symbol,
        optimal_fast=int(best_params["optimal_fast"]),
        optimal_slow=int(best_params["optimal_slow"]),
        optimal_trend=int(best_params["optimal_trend"]),
        is_sharpe=round(float(best_is), 4),
        oos_sharpe=round(float(best_oos), 4),
        confidence=confidence,
        combos_tested=combos_tested,
        training_days=is_days,
        test_days=oos_days,
    )

    log.info(
        "[ema_algo] %s EMA(%d/%d/%d) IS=%.3f OOS=%.3f [%s] combos=%d",
        symbol,
        best_params["optimal_fast"], best_params["optimal_slow"], best_params["optimal_trend"],
        best_is, best_oos, confidence, combos_tested,
    )

    return {
        **best_params,
        "is_sharpe":          round(float(best_is), 4),
        "oos_sharpe":         round(float(best_oos), 4),
        "confidence":         confidence,
        "combos_tested":      combos_tested,
        "is_days":            is_days,
        "oos_days":           oos_days,
        "default_is_sharpe":  round(float(default_is_sharpe), 4),
        "default_oos_sharpe": round(float(default_oos_sharpe), 4),
        "wfo_analysis":       vars(wfo) if wfo else {},
    }
