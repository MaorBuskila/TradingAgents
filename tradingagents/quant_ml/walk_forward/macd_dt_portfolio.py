"""
macd_dt_portfolio.py
====================
Portfolio construction layer for the DT-filtered MACD optimizer.

Rules
-----
  - Liquidity gate : skip symbols where 30-day avg dollar-volume < min_dollar_vol
  - 5% Rule        : allocate 5% of capital per "High Confidence" DT signal
  - Max Exposure   : cap at 20 simultaneous positions  (max 100% deployed)
  - Size Multiplier: halved to 2.5% when DT model detects high-volatility regime

Entry point
-----------
    from tradingagents.quant_ml.walk_forward.macd_dt_portfolio import run_portfolio_optimizer
    result = run_portfolio_optimizer(
        symbols=["AAPL", "MSFT", ...],
        curr_date="2024-01-01",
        capital=100_000,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, asdict
from typing import Annotated, Sequence

import numpy as np
import pandas as pd
import yfinance as yf

from tradingagents.quant_ml.indicators.stockstats_utils import yf_retry, _clean_dataframe
from tradingagents.quant_ml.optimizers.macd_dt_optimizer import run_dt_optimizer

log = logging.getLogger("macd_dt_portfolio")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MAX_POSITIONS    = 20          # maximum simultaneous longs
BASE_ALLOC       = 0.05        # 5% of capital per position (full size)
MIN_DOLLAR_VOL   = 5_000_000   # $5M 30-day avg dollar volume gate
VOL_HALF_MULT    = 0.5         # halved allocation in high-vol regime


# ─────────────────────────────────────────────────────────────────────────────
# Liquidity filter — single method
# ─────────────────────────────────────────────────────────────────────────────

def _passes_liquidity(symbol: str, curr_date: str, min_dv: float = MIN_DOLLAR_VOL) -> bool:
    """
    Return True if 30-day average dollar volume >= min_dv.
    Downloads only 90 days of data to keep it fast.
    """
    try:
        end   = pd.Timestamp(curr_date)
        start = end - pd.DateOffset(days=90)
        df = yf_retry(lambda: yf.download(
            symbol,
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            multi_level_index=False,
            progress=False,
            auto_adjust=True,
        ))
        if df is None or len(df) < 20:
            return False
        df = _clean_dataframe(df.reset_index())
        df = df.sort_values("Date").tail(30)
        avg_dv = float((df["Close"] * df["Volume"]).mean())
        ok = avg_dv >= min_dv
        log.debug("[liquidity] %s  avg_dv=$%.0f  threshold=$%.0f  pass=%s",
                  symbol, avg_dv, min_dv, ok)
        return ok
    except Exception as exc:
        log.warning("[liquidity] %s failed: %s", symbol, exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Position Sizing
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Position:
    symbol:        str
    signal:        int          # 1 = Long
    prob_a:        float        # Model A confidence
    prob_b:        float        # Model B confidence
    atr_pct:       float        # volatility percentile
    size_mult:     float        # 1.0 or 0.5
    alloc_pct:     float        # final % of capital
    alloc_dollars: float        # absolute dollar allocation
    avg_oos_acc:   float        # sliding WFO OOS accuracy

    def to_dict(self) -> dict:
        return asdict(self)


def _size_position(capital: float, size_mult: float) -> tuple[float, float]:
    """Returns (alloc_pct, alloc_dollars)."""
    pct = BASE_ALLOC * size_mult
    return round(pct, 4), round(capital * pct, 2)


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio Optimizer — single entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_portfolio_optimizer(
    symbols:   Annotated[Sequence[str], "List of ticker symbols to evaluate"],
    curr_date: Annotated[str, "Current date YYYY-MM-DD"],
    capital:   Annotated[float, "Total portfolio capital in USD"] = 100_000.0,
    min_dollar_vol: Annotated[float, "Min 30-day avg dollar volume for liquidity gate"] = MIN_DOLLAR_VOL,
    max_positions:  Annotated[int,   "Max simultaneous long positions"] = MAX_POSITIONS,
    is_days:   Annotated[int, "Sliding IS window in bars"] = 100,
    oos_days:  Annotated[int, "OOS / re-train cycle in bars"] = 30,
) -> dict:
    """
    Runs the DT-filtered MACD optimizer across a universe of symbols and
    constructs a portfolio obeying the 5%-rule and MAX_POSITIONS cap.

    Returns dict with:
        positions        — list of Position dicts (long signals only, ranked by prob_a)
        rejected_liquidity — symbols that failed the dollar-volume gate
        rejected_no_signal — symbols that passed liquidity but had no DT signal
        rejected_errors    — symbols that errored during optimization
        total_deployed_pct — sum of alloc_pct across accepted positions
        total_deployed_usd — sum of alloc_dollars
        capital            — input capital
        curr_date          — input date
        n_evaluated        — total symbols attempted
    """
    log.debug("[portfolio] START  n_symbols=%d  capital=$%.0f  date=%s",
              len(symbols), capital, curr_date)

    rejected_liquidity: list[str] = []
    rejected_no_signal: list[str] = []
    rejected_errors:    list[str] = []
    candidates:         list[Position] = []

    for sym in symbols:
        # ── Liquidity gate ───────────────────────────────────────────────────
        if not _passes_liquidity(sym, curr_date, min_dollar_vol):
            log.debug("[portfolio] %s → rejected (liquidity)", sym)
            rejected_liquidity.append(sym)
            continue

        # ── DT optimizer ─────────────────────────────────────────────────────
        try:
            res = run_dt_optimizer(sym, curr_date, is_days=is_days, oos_days=oos_days)
        except Exception as exc:
            log.warning("[portfolio] %s optimizer error: %s", sym, exc)
            rejected_errors.append(sym)
            continue

        if res["last_signal"] == 0:
            log.debug("[portfolio] %s → no signal (prob_a=%.3f  prob_b=%.3f)",
                      sym, res["last_prob_a"], res["last_prob_b"])
            rejected_no_signal.append(sym)
            continue

        alloc_pct, alloc_usd = _size_position(capital, res["last_size_mult"])
        candidates.append(Position(
            symbol=sym,
            signal=res["last_signal"],
            prob_a=res["last_prob_a"],
            prob_b=res["last_prob_b"],
            atr_pct=res["last_atr_pct"],
            size_mult=res["last_size_mult"],
            alloc_pct=alloc_pct,
            alloc_dollars=alloc_usd,
            avg_oos_acc=res["avg_oos_acc"],
        ))

    # ── Rank by Model A confidence, cap at max_positions ────────────────────
    candidates.sort(key=lambda p: p.prob_a, reverse=True)
    selected = candidates[:max_positions]

    total_pct = round(sum(p.alloc_pct for p in selected), 4)
    total_usd = round(sum(p.alloc_dollars for p in selected), 2)

    log.debug(
        "[portfolio] DONE  selected=%d  deployed_pct=%.1f%%  deployed_usd=$%.0f",
        len(selected), total_pct * 100, total_usd,
    )

    return {
        "positions":              [p.to_dict() for p in selected],
        "rejected_liquidity":     rejected_liquidity,
        "rejected_no_signal":     rejected_no_signal,
        "rejected_errors":        rejected_errors,
        "total_deployed_pct":     total_pct,
        "total_deployed_usd":     total_usd,
        "capital":                capital,
        "curr_date":              curr_date,
        "n_evaluated":            len(symbols),
        "n_selected":             len(selected),
        "base_alloc_pct":         BASE_ALLOC,
        "max_positions":          max_positions,
        "min_dollar_vol":         min_dollar_vol,
    }
