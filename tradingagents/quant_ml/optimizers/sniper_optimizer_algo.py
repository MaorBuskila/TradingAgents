"""
sniper_optimizer_algo.py
==========================
Walk-forward grid search over Precision Sniper classical parameters.
Scores with Sharpe on simulated R-multiples (non-overlapping long trades).
"""

from __future__ import annotations

import itertools
import logging
from typing import Annotated, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from tradingagents.quant_ml.optimizers.rsi_optimizer_algo import _load_price_data
from tradingagents.quant_ml.signals.sniper_features import (
    build_sniper_base_indicators,
    finalize_sniper_frame,
)
from tradingagents.quant_ml.walk_forward.sniper_sim import (
    confidence_tier,
    run_long_backtest_rs,
    sharpe_r,
)

log = logging.getLogger("sniper_optimizer_algo")

# Default search space (~800–900 valid combos after slow>fast).
FAST_GRID = [8, 9, 12]
SLOW_GRID = [18, 21, 26, 34]
TREND_GRID = [50, 55]
MIN_SCORE_GRID = [5.0, 6.0, 7.0, 8.0]
SL_MULT_GRID = [1.0, 1.5, 2.0, 2.5]
VOL_MULT_GRID = [1.0, 1.2, 1.5]


def _valid_ema_pairs() -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for f, s in itertools.product(FAST_GRID, SLOW_GRID):
        if s > f:
            pairs.append((f, s))
    return pairs


def _ema_fast_sharpes(best_by_key: dict) -> List[dict]:
    """Best OOS Sharpe observed per fast EMA (for charts)."""
    by_fast: dict[int, float] = {}
    for k, v in best_by_key.items():
        if not isinstance(k, tuple) or len(k) < 6:
            continue
        f = int(k[0])
        oos = float(v.get("oos_sharpe", 0.0))
        by_fast[f] = max(by_fast.get(f, -1e9), oos)
    return [{"ema_fast": f, "oos_sharpe": round(s, 4)} for f, s in sorted(by_fast.items())]


def run_algo_optimizer(
    symbol: Annotated[str, "ticker"],
    curr_date: Annotated[str, "YYYY-MM-DD"],
    is_days: Annotated[int, "in-sample bars"] = 180,
    oos_days: Annotated[int, "out-of-sample bars"] = 90,
    *,
    fast_grid: Optional[Sequence[int]] = None,
    slow_grid: Optional[Sequence[int]] = None,
    trend_grid: Optional[Sequence[int]] = None,
    min_score_grid: Optional[Sequence[float]] = None,
    sl_mult_grid: Optional[Sequence[float]] = None,
    vol_mult_grid: Optional[Sequence[float]] = None,
) -> dict[str, Any]:
    """Grid-search Sniper params; maximize IS Sharpe on R-multiples, report OOS."""
    data = _load_price_data(symbol)
    curr_dt = pd.to_datetime(curr_date)
    data = data[data["Date"] <= curr_dt].copy()
    closes = data["Close"].reset_index(drop=True)
    total_days = len(closes)
    min_required = is_days + oos_days + 60
    if total_days < min_required:
        raise ValueError(
            f"Not enough data: {total_days} rows, need at least {min_required}"
        )

    oos_start = total_days - oos_days
    is_start = oos_start - is_days
    if is_start < 0:
        is_start = 0

    fg = list(fast_grid) if fast_grid is not None else FAST_GRID
    sg = list(slow_grid) if slow_grid is not None else SLOW_GRID
    tg = list(trend_grid) if trend_grid is not None else TREND_GRID
    mg = list(min_score_grid) if min_score_grid is not None else MIN_SCORE_GRID
    slg = list(sl_mult_grid) if sl_mult_grid is not None else SL_MULT_GRID
    vg = list(vol_mult_grid) if vol_mult_grid is not None else VOL_MULT_GRID

    base = build_sniper_base_indicators(data)

    best_is = -np.inf
    best_params: dict[str, Any] = {}
    best_by_key: dict = {}
    combos_tested = 0
    debug_logs: List[str] = []

    ema_pairs = [(f, s) for f, s in itertools.product(fg, sg) if s > f]
    if not ema_pairs:
        raise ValueError("No valid EMA pairs (need slow > fast)")

    def _eval_frame(full: pd.DataFrame, ms: float, sl_m: float, lo: int, hi: int) -> float:
        rs = run_long_backtest_rs(
            full, start_i=lo, end_i=hi, min_score=ms, sl_mult=sl_m,
        )
        return sharpe_r(rs)

    default_frame = finalize_sniper_frame(base, fast=9, slow=21, trend=55, vol_mult=1.2)
    default_is = _eval_frame(default_frame, 5.0, 1.5, is_start, oos_start)
    default_oos = _eval_frame(default_frame, 5.0, 1.5, oos_start, total_days)

    for fast, slow in ema_pairs:
        for trend in tg:
            for vol_m in vg:
                full = finalize_sniper_frame(
                    base, fast=fast, slow=slow, trend=trend, vol_mult=vol_m
                )
                for ms in mg:
                    for sl_m in slg:
                        combos_tested += 1
                        is_sh = _eval_frame(full, ms, sl_m, is_start, oos_start)
                        oos_sh = _eval_frame(full, ms, sl_m, oos_start, total_days)
                        key = (fast, slow, trend, vol_m, ms, sl_m)
                        best_by_key[key] = {"is_sharpe": is_sh, "oos_sharpe": oos_sh}
                        if is_sh > best_is:
                            best_is = is_sh
                            best_params = {
                                "optimal_ema_fast": fast,
                                "optimal_ema_slow": slow,
                                "optimal_ema_trend": trend,
                                "optimal_min_score": ms,
                                "optimal_sl_mult": sl_m,
                                "optimal_vol_mult": vol_m,
                                "is_sharpe": round(is_sh, 4),
                                "oos_sharpe": round(oos_sh, 4),
                            }

    assert best_params
    conf = confidence_tier(float(best_params["is_sharpe"]), float(best_params["oos_sharpe"]))
    best_params["confidence"] = conf
    heatmap: List[dict[str, Any]] = []
    for ms in mg:
        for sl_m in slg:
            sub = [
                v["oos_sharpe"]
                for k, v in best_by_key.items()
                if len(k) >= 6 and float(k[4]) == float(ms) and float(k[5]) == float(sl_m)
            ]
            heatmap.append({
                "min_score": ms,
                "sl_mult": sl_m,
                "best_oos_sharpe": round(max(sub) if sub else 0.0, 4),
            })

    out = {
        **best_params,
        "combos_tested": combos_tested,
        "default_is_sharpe": round(float(default_is), 4),
        "default_oos_sharpe": round(float(default_oos), 4),
        "ema_fast_sharpes": _ema_fast_sharpes(best_by_key),
        "min_score_sl_heatmap": heatmap,
        "debug_logs": debug_logs,
        "is_days": is_days,
        "oos_days": oos_days,
    }
    log.info("[sniper_algo] %s best OOS %.3f (combos=%d)", symbol, best_params["oos_sharpe"], combos_tested)
    return out
