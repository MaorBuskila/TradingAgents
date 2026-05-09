"""
sniper_score.py
---------------
10-factor weighted confluence engine for the Precision Sniper strategy.

For each bar, compute BOTH a bullish score and a bearish score independently
(a bar could pass some bull factors and some bear factors; only the dominant
side typically triggers a signal). Each factor contributes its weight if its
condition is True. Max possible score = 10.0.

Grade thresholds match the TradingView script:
    A+  score >= 8.0
    A   score >= 6.5
    B   score >= 5.0
    C   score <  5.0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import pandas as pd


DEFAULT_WEIGHTS: Dict[str, float] = {
    "ema_f_vs_s": 1.0,
    "close_vs_trend": 1.0,
    "rsi_in_zone": 1.0,
    "macd_hist": 1.0,
    "macd_line_vs_signal": 1.0,
    "close_vs_vwap": 1.0,
    "volume_burst": 1.0,
    "adx_di_aligned": 1.0,
    "htf_bias": 1.5,
    "close_vs_ema_f": 0.5,
}

assert abs(sum(DEFAULT_WEIGHTS.values()) - 10.0) < 1e-9


def grade_from_score(score: float) -> str:
    """Map a composite score (0..10) to a letter grade."""
    if score >= 8.0:
        return "A+"
    if score >= 6.5:
        return "A"
    if score >= 5.0:
        return "B"
    return "C"


@dataclass
class SniperBarInputs:
    """Everything needed to score a single bar.

    RSI uses the Sniper defaults (50..75 bull, 25..50 bear). If you want to
    plug your already-optimized RSI thresholds in, pass them via
    rsi_upper / rsi_lower.
    """
    close: float
    ema_f: float
    ema_s: float
    ema_t: float
    rsi: float
    macd_hist: float
    macd_line: float
    macd_signal: float
    vwap: float
    volume_burst: bool
    adx: float
    plus_di: float
    minus_di: float
    htf_bias_bull: bool  # HTF EMA fast > slow
    rsi_upper: float = 75.0
    rsi_lower: float = 25.0
    rsi_mid: float = 50.0
    adx_th: float = 20.0


def _bull_factors(b: SniperBarInputs) -> Dict[str, bool]:
    return {
        "ema_f_vs_s": b.ema_f > b.ema_s,
        "close_vs_trend": b.close > b.ema_t,
        "rsi_in_zone": (b.rsi > b.rsi_mid) and (b.rsi < b.rsi_upper),
        "macd_hist": b.macd_hist > 0,
        "macd_line_vs_signal": b.macd_line > b.macd_signal,
        "close_vs_vwap": b.close > b.vwap,
        "volume_burst": bool(b.volume_burst),
        "adx_di_aligned": (b.adx > b.adx_th) and (b.plus_di > b.minus_di),
        "htf_bias": bool(b.htf_bias_bull),
        "close_vs_ema_f": b.close > b.ema_f,
    }


def _bear_factors(b: SniperBarInputs) -> Dict[str, bool]:
    return {
        "ema_f_vs_s": b.ema_f < b.ema_s,
        "close_vs_trend": b.close < b.ema_t,
        "rsi_in_zone": (b.rsi < b.rsi_mid) and (b.rsi > b.rsi_lower),
        "macd_hist": b.macd_hist < 0,
        "macd_line_vs_signal": b.macd_line < b.macd_signal,
        "close_vs_vwap": b.close < b.vwap,
        "volume_burst": bool(b.volume_burst),
        "adx_di_aligned": (b.adx > b.adx_th) and (b.minus_di > b.plus_di),
        "htf_bias": not b.htf_bias_bull,
        "close_vs_ema_f": b.close < b.ema_f,
    }


def compute_sniper_score(
    bar: SniperBarInputs,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, object]:
    """Compute bull/bear composite scores + grades + per-factor contributions.

    Returns:
        {
            bull_score, bear_score       — float in [0, sum(weights)]
            bull_grade,  bear_grade      — "A+" | "A" | "B" | "C"
            bull_factors, bear_factors   — dict[str, bool]
            weights                      — dict[str, float] used
        }
    """
    w = weights or DEFAULT_WEIGHTS
    bulls = _bull_factors(bar)
    bears = _bear_factors(bar)

    bull_score = sum(w[k] for k, v in bulls.items() if v)
    bear_score = sum(w[k] for k, v in bears.items() if v)

    return {
        "bull_score": round(bull_score, 4),
        "bear_score": round(bear_score, 4),
        "bull_grade": grade_from_score(bull_score),
        "bear_grade": grade_from_score(bear_score),
        "bull_factors": bulls,
        "bear_factors": bears,
        "weights": dict(w),
    }


def compute_sniper_score_series(
    df: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    rsi_upper: float = 75.0,
    rsi_lower: float = 25.0,
    adx_th: float = 20.0,
) -> pd.DataFrame:
    """Vectorized scoring across a full feature DataFrame.

    Required columns: Close, ema_f, ema_s, ema_t, rsi, macd_hist, macd_line,
    macd_signal, vwap, volume_burst, adx, plus_di, minus_di, htf_bias.

    Returns:
        DataFrame with: bull_score, bear_score, bull_grade, bear_grade
        (aligned with df.index). Grades are computed per-bar via grade_from_score.
    """
    w = weights or DEFAULT_WEIGHTS
    rsi_mid = 50.0

    bull_factors_raw = {
        "ema_f_vs_s": df["ema_f"] > df["ema_s"],
        "close_vs_trend": df["Close"] > df["ema_t"],
        "rsi_in_zone": (df["rsi"] > rsi_mid) & (df["rsi"] < rsi_upper),
        "macd_hist": df["macd_hist"] > 0,
        "macd_line_vs_signal": df["macd_line"] > df["macd_signal"],
        "close_vs_vwap": df["Close"] > df["vwap"],
        "volume_burst": df["volume_burst"].astype(bool),
        "adx_di_aligned": (df["adx"] > adx_th) & (df["plus_di"] > df["minus_di"]),
        "htf_bias": df["htf_bias"].astype(bool),
        "close_vs_ema_f": df["Close"] > df["ema_f"],
    }
    bear_factors_raw = {
        "ema_f_vs_s": df["ema_f"] < df["ema_s"],
        "close_vs_trend": df["Close"] < df["ema_t"],
        "rsi_in_zone": (df["rsi"] < rsi_mid) & (df["rsi"] > rsi_lower),
        "macd_hist": df["macd_hist"] < 0,
        "macd_line_vs_signal": df["macd_line"] < df["macd_signal"],
        "close_vs_vwap": df["Close"] < df["vwap"],
        "volume_burst": df["volume_burst"].astype(bool),
        "adx_di_aligned": (df["adx"] > adx_th) & (df["minus_di"] > df["plus_di"]),
        "htf_bias": ~df["htf_bias"].astype(bool),
        "close_vs_ema_f": df["Close"] < df["ema_f"],
    }

    bull_score = sum(w[k] * flag.astype(float).fillna(0.0) for k, flag in bull_factors_raw.items())
    bear_score = sum(w[k] * flag.astype(float).fillna(0.0) for k, flag in bear_factors_raw.items())

    out = pd.DataFrame(
        {
            "bull_score": bull_score.round(4),
            "bear_score": bear_score.round(4),
        },
        index=df.index,
    )
    out["bull_grade"] = out["bull_score"].apply(grade_from_score)
    out["bear_grade"] = out["bear_score"].apply(grade_from_score)
    return out
