"""
wfo_analyzer.py
===============
Walk-Forward Optimization (WFO) Statistical Significance Analyzer.

For weekly MACD strategies, OOS windows produce only ~52 bars/year.
A histogram crossover signal fires ~4–8 times per year on weekly data,
so 6 months of OOS = ~3–5 trades — statistically meaningless.

This module:
  1. Counts OOS trades from position array
  2. Classifies sample-size risk (< 30 trades = High Risk)
  3. Computes Walk-Forward Efficiency  = OOS_Sharpe / IS_Sharpe
  4. Returns a 0–100 Confidence Score + risk label
"""

from __future__ import annotations
import logging
import math
from dataclasses import dataclass, asdict
from typing import Sequence

log = logging.getLogger("wfo_analyzer")


# ─────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────

MIN_TRADES_WEEKLY   = 30   # below this → High Risk on weekly timeframe
MIN_TRADES_DAILY    = 30   # same threshold for daily (standard stat sig)
WFE_DEGRADATION_CAP = 2.0  # WFE > 2.0 is suspicious (overfit IS)

RISK_LABELS = {
    "HIGH":   "High Risk — Insufficient Sample Size",
    "MEDIUM": "Medium Risk — Marginal Sample Size",
    "LOW":    "Low Risk — Adequate Sample",
}


# ─────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────

@dataclass
class WFOAnalysis:
    # inputs (echoed back)
    is_days:        int
    oos_days:       int
    is_sharpe:      float
    oos_sharpe:     float
    oos_trade_count: int
    bar_frequency:  str          # "weekly" | "daily"

    # derived
    wfe:                 float   # Walk-Forward Efficiency = OOS / IS
    wfe_label:           str     # "efficient" | "degraded" | "suspicious"
    sample_risk:         str     # "HIGH" | "MEDIUM" | "LOW"
    sample_risk_label:   str     # human-readable
    confidence_score:    int     # 0–100
    confidence_label:    str     # "Very Low" | "Low" | "Moderate" | "High" | "Very High"
    assessment:          str     # 2-3 sentence narrative

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────
# Core scoring
# ─────────────────────────────────────────────────────────

def _count_trades(positions: Sequence[int]) -> int:
    """Count round-trip trade entries from a position array (+1/0/−1)."""
    trades = 0
    prev = 0
    for p in positions:
        if p != 0 and p != prev:
            trades += 1
        prev = p
    log.debug("[trades] position array len=%d  trade count=%d", len(positions), trades)
    return trades


def _wfe(oos_sharpe: float, is_sharpe: float) -> tuple[float, str]:
    """
    Walk-Forward Efficiency.
    >  0.7  → efficient  (OOS retains ≥70 % of IS performance)
    0.3–0.7 → degraded   (expected decay, borderline acceptable)
    < 0.3   → poor       (IS overfit likely)
    > WFE_DEGRADATION_CAP → suspicious (IS Sharpe was too low to trust ratio)
    """
    if is_sharpe <= 0:
        log.debug("[wfe] IS sharpe=%.4f ≤ 0 → invalid_is", is_sharpe)
        return 0.0, "invalid_is"
    ratio = oos_sharpe / is_sharpe
    if ratio > WFE_DEGRADATION_CAP:
        label = "suspicious_high"
    elif ratio >= 0.70:
        label = "efficient"
    elif ratio >= 0.30:
        label = "degraded"
    else:
        label = "poor"
    log.debug("[wfe] IS=%.4f  OOS=%.4f  ratio=%.4f  label=%s", is_sharpe, oos_sharpe, ratio, label)
    return round(ratio, 4), label


def _sample_risk(trade_count: int, bar_frequency: str) -> tuple[str, str]:
    min_t = MIN_TRADES_WEEKLY if bar_frequency == "weekly" else MIN_TRADES_DAILY
    if trade_count < min_t:
        risk, label = "HIGH", RISK_LABELS["HIGH"]
    elif trade_count < min_t * 2:          # 30–59 → medium
        risk, label = "MEDIUM", RISK_LABELS["MEDIUM"]
    else:
        risk, label = "LOW", RISK_LABELS["LOW"]
    log.debug("[sample_risk] trades=%d  min_threshold=%d  risk=%s", trade_count, min_t, risk)
    return risk, label


def _confidence_score(
    trade_count: int,
    oos_sharpe: float,
    wfe: float,
    wfe_label: str,
    sample_risk: str,
) -> int:
    """
    Composite 0–100 score.

    Components:
      A) Sample size    (0–40 pts)  — most weight; sparse data kills everything
      B) OOS Sharpe     (0–30 pts)  — raw performance
      C) WFE            (0–20 pts)  — IS→OOS stability
      D) Penalty bucket (0–10 pts)  — suspicious patterns
    """
    # A: Sample size component
    if trade_count >= 100:
        a = 40
    elif trade_count >= 60:
        a = 30
    elif trade_count >= 30:
        a = 20
    elif trade_count >= 15:
        a = 10
    elif trade_count >= 5:
        a = 5
    else:
        a = 0

    # B: OOS Sharpe component
    if oos_sharpe >= 1.5:
        b = 30
    elif oos_sharpe >= 1.0:
        b = 22
    elif oos_sharpe >= 0.5:
        b = 14
    elif oos_sharpe >= 0.0:
        b = 6
    else:
        b = 0   # negative Sharpe

    # C: WFE component
    wfe_pts = {
        "efficient":       20,
        "degraded":        10,
        "poor":             3,
        "suspicious_high":  5,   # might still work, but not trustworthy
        "invalid_is":       0,
    }
    c = wfe_pts.get(wfe_label, 0)

    # D: Penalty
    d = 10  # start full, subtract
    if sample_risk == "HIGH":
        d -= 8
    elif sample_risk == "MEDIUM":
        d -= 3
    if wfe_label in ("suspicious_high", "invalid_is"):
        d -= 5
    d = max(d, 0)

    raw = a + b + c + d
    score = min(100, max(0, raw))
    log.debug("[confidence_score] A(sample)=%d  B(oos_sharpe)=%d  C(wfe)=%d  D(penalty)=%d  raw=%d  final=%d",
              a, b, c, d, raw, score)
    return score


def _confidence_label(score: int) -> str:
    if score >= 80:
        return "Very High"
    if score >= 60:
        return "High"
    if score >= 40:
        return "Moderate"
    if score >= 20:
        return "Low"
    return "Very Low"


def _narrative(
    trade_count: int,
    wfe: float,
    wfe_label: str,
    sample_risk: str,
    oos_sharpe: float,
    bar_frequency: str,
    confidence_score: int,
) -> str:
    freq_note = (
        "Weekly bars produce ~52 candles/year; a histogram crossover fires ~4–8×/year, "
        "so OOS windows under 2 years rarely exceed 30 trades."
        if bar_frequency == "weekly"
        else "Daily bars still require at least 30 OOS trades for stat significance."
    )

    sample_clause = (
        f"OOS trade count is {trade_count} — {'BELOW' if sample_risk == 'HIGH' else 'at the lower bound of'} "
        f"the 30-trade minimum for statistical significance. {freq_note}"
        if sample_risk in ("HIGH", "MEDIUM")
        else f"OOS trade count is {trade_count}, meeting the 30-trade significance threshold."
    )

    wfe_clause = (
        f"Walk-Forward Efficiency is {wfe:.2f} ({wfe_label.replace('_', ' ')}): "
        + {
            "efficient":       "OOS retained ≥70% of IS Sharpe — robust carry-through.",
            "degraded":        "OOS captured 30–70% of IS Sharpe — typical for live decay, borderline acceptable.",
            "poor":            "OOS retained <30% of IS Sharpe — strong IS overfitting signal.",
            "suspicious_high": "WFE >2.0 suggests IS Sharpe was artificially low (e.g. too few IS trades).",
            "invalid_is":      "IS Sharpe ≤ 0; WFE ratio is undefined — IS optimization may have failed.",
        }.get(wfe_label, "")
    )

    risk_clause = (
        f"Overall confidence score is {confidence_score}/100. "
        + ("DO NOT trade this configuration live." if confidence_score < 30 else
           "Paper-trade only until confirmed." if confidence_score < 55 else
           "Proceed with reduced position sizing pending further OOS data." if confidence_score < 75 else
           "Configuration shows statistical robustness.")
    )

    return f"{sample_clause} {wfe_clause} {risk_clause}"


# ─────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────

def analyze_wfo(
    *,
    is_days: int,
    oos_days: int,
    is_sharpe: float,
    oos_sharpe: float,
    oos_positions: Sequence[int] | None = None,
    oos_trade_count: int | None = None,
    bar_frequency: str = "weekly",
) -> WFOAnalysis:
    """
    Analyze Walk-Forward Optimization results.

    Args:
        is_days:          calendar days in the In-Sample window
        oos_days:         calendar days in the Out-of-Sample window
        is_sharpe:        annualized Sharpe from IS optimization
        oos_sharpe:       annualized Sharpe from OOS backtest
        oos_positions:    raw +1/0/-1 position array (used to count trades)
        oos_trade_count:  if positions not available, pass count directly
        bar_frequency:    "weekly" or "daily"

    Returns:
        WFOAnalysis dataclass with all derived fields
    """
    log.debug("[wfo] ── analyze_wfo START ──────────────────────────────────────")
    log.debug("[wfo] is_days=%d  oos_days=%d  is_sharpe=%.4f  oos_sharpe=%.4f  freq=%s",
              is_days, oos_days, is_sharpe, oos_sharpe, bar_frequency)

    if oos_positions is not None:
        count = _count_trades(oos_positions)
    elif oos_trade_count is not None:
        count = oos_trade_count
        log.debug("[wfo] trade count provided directly: %d", count)
    else:
        raise ValueError("Provide either oos_positions or oos_trade_count")

    wfe, wfe_label         = _wfe(oos_sharpe, is_sharpe)
    s_risk, s_risk_label   = _sample_risk(count, bar_frequency)
    score                  = _confidence_score(count, oos_sharpe, wfe, wfe_label, s_risk)
    c_label                = _confidence_label(score)
    assessment             = _narrative(count, wfe, wfe_label, s_risk, oos_sharpe, bar_frequency, score)

    log.debug("[wfo] result — wfe=%.4f (%s)  sample_risk=%s  score=%d  label=%s",
              wfe, wfe_label, s_risk, score, c_label)
    log.debug("[wfo] ── analyze_wfo END ────────────────────────────────────────")

    return WFOAnalysis(
        is_days=is_days,
        oos_days=oos_days,
        is_sharpe=round(is_sharpe, 4),
        oos_sharpe=round(oos_sharpe, 4),
        oos_trade_count=count,
        bar_frequency=bar_frequency,
        wfe=wfe,
        wfe_label=wfe_label,
        sample_risk=s_risk,
        sample_risk_label=s_risk_label,
        confidence_score=score,
        confidence_label=c_label,
        assessment=assessment,
    )


def analyze_wfo_from_optimizer_result(result: dict, bar_frequency: str = "daily") -> dict:
    """
    Convenience wrapper: takes the dict returned by run_algo_optimizer / run_llm_optimizer
    and augments it with WFO analysis fields.

    Expected keys in result:
        is_sharpe, oos_sharpe, is_days, oos_days, oos_positions (optional)
    """
    analysis = analyze_wfo(
        is_days=result.get("is_days", 180),
        oos_days=result.get("oos_days", 90),
        is_sharpe=result.get("is_sharpe", 0.0),
        oos_sharpe=result.get("oos_sharpe", 0.0),
        oos_positions=result.get("oos_positions"),
        oos_trade_count=result.get("oos_trade_count"),
        bar_frequency=bar_frequency,
    )
    return {**result, "wfo_analysis": analysis.to_dict()}
