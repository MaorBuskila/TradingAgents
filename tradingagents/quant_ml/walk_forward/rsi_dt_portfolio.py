"""
rsi_dt_portfolio.py
===================
Portfolio construction layer for the DT-filtered RSI optimizer.

Rules
-----
  - Liquidity gate : skip symbols where 30-day avg dollar-volume < min_dollar_vol
  - 5% Rule        : allocate 5% of capital per "High Confidence" DT signal
  - Max Exposure   : cap at 20 simultaneous positions (max 100% deployed)
  - Size Multiplier: halved to 2.5% when DT model detects high-volatility regime

Entry point
-----------
    from tradingagents.quant_ml.walk_forward.rsi_dt_portfolio import run_rsi_portfolio_optimizer
    result = run_rsi_portfolio_optimizer(
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
from tradingagents.quant_ml.optimizers.rsi_dt_optimizer import run_rsi_dt_optimizer

log = logging.getLogger("rsi_dt_portfolio")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MAX_POSITIONS  = 20
BASE_ALLOC     = 0.05        # 5% of capital per position (full size)
MIN_DOLLAR_VOL = 5_000_000   # $5M 30-day avg dollar volume gate
VOL_HALF_MULT  = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Liquidity filter
# ─────────────────────────────────────────────────────────────────────────────

def _passes_liquidity(symbol: str, curr_date: str, min_dv: float = MIN_DOLLAR_VOL) -> bool:
    """Return True if 30-day average dollar volume >= min_dv."""
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
        if df is None or df.empty:
            return False
        df = _clean_dataframe(df)
        if "Close" not in df.columns or "Volume" not in df.columns:
            return False
        dv30 = (df["Close"] * df["Volume"]).rolling(30, min_periods=5).mean().iloc[-1]
        return float(dv30) >= min_dv
    except Exception as exc:
        log.warning("[rsi_dt_portfolio] liquidity check failed for %s: %s", symbol, exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Position dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RsiDtPosition:
    symbol:        str
    signal:        int
    prob_a:        float
    prob_b:        float
    atr_pct:       float
    last_rsi:      float
    size_mult:     float
    alloc_pct:     float
    alloc_dollars: float
    avg_oos_acc:   float

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_rsi_portfolio_optimizer(
    symbols:        Annotated[Sequence[str], "List of ticker symbols to evaluate"],
    curr_date:      Annotated[str,   "Current date YYYY-MM-DD"],
    capital:        Annotated[float, "Total portfolio capital USD"] = 100_000.0,
    min_dollar_vol: Annotated[float, "Min 30-day avg dollar volume"] = MIN_DOLLAR_VOL,
    max_positions:  Annotated[int,   "Max simultaneous long positions"] = MAX_POSITIONS,
    is_days:        Annotated[int,   "Sliding IS window in bars"] = 100,
    oos_days:       Annotated[int,   "OOS / re-train cycle in bars"] = 30,
    rsi_period:     Annotated[int,   "RSI period (WFO-optimized or default 14)"] = 14,
    rsi_upper:      Annotated[float, "RSI overbought threshold (WFO or default 70)"] = 70.0,
    rsi_lower:      Annotated[float, "RSI oversold threshold (WFO or default 30)"] = 30.0,
) -> dict:
    """
    Run RSI DT optimizer on every symbol and build a portfolio.

    Returns dict with:
        positions            — list of selected positions
        rejected_liquidity   — symbols failing the liquidity gate
        rejected_no_signal   — symbols where DT signal = 0
        rejected_errors      — symbols that errored during optimization
        total_deployed_pct   — fraction of capital deployed (0.0–1.0)
        total_deployed_usd   — dollar amount deployed
        capital, curr_date, n_evaluated, n_selected,
        base_alloc_pct, max_positions, min_dollar_vol
    """
    rejected_liquidity: list[str] = []
    rejected_no_signal: list[str] = []
    rejected_errors:    list[str] = []
    candidates:         list[RsiDtPosition] = []

    for sym in symbols:
        # ── liquidity gate ────────────────────────────────────────────────────
        if not _passes_liquidity(sym, curr_date, min_dollar_vol):
            log.info("[rsi_dt_portfolio] %s: REJECTED (liquidity)", sym)
            rejected_liquidity.append(sym)
            continue

        # ── run DT optimizer ─────────────────────────────────────────────────
        try:
            raw = run_rsi_dt_optimizer(
                symbol=sym,
                curr_date=curr_date,
                is_days=is_days,
                oos_days=oos_days,
                rsi_period=rsi_period,
                rsi_upper=rsi_upper,
                rsi_lower=rsi_lower,
            )
        except Exception as exc:
            log.warning("[rsi_dt_portfolio] %s: ERROR — %s", sym, exc)
            rejected_errors.append(sym)
            continue

        if raw["last_signal"] == 0:
            log.info("[rsi_dt_portfolio] %s: no signal (prob_a=%.3f, prob_b=%.3f)",
                     sym, raw["last_prob_a"], raw["last_prob_b"])
            rejected_no_signal.append(sym)
            continue

        alloc_pct     = BASE_ALLOC * raw["last_size_mult"]
        alloc_dollars = alloc_pct * capital

        candidates.append(RsiDtPosition(
            symbol=sym,
            signal=raw["last_signal"],
            prob_a=raw["last_prob_a"],
            prob_b=raw["last_prob_b"],
            atr_pct=raw["last_atr_pct"],
            last_rsi=raw.get("last_rsi", 50.0),
            size_mult=raw["last_size_mult"],
            alloc_pct=alloc_pct,
            alloc_dollars=alloc_dollars,
            avg_oos_acc=raw["avg_oos_acc"],
        ))

    # ── cap at max_positions (rank by prob_a descending) ─────────────────────
    candidates.sort(key=lambda p: p.prob_a, reverse=True)
    selected = candidates[:max_positions]

    total_deployed_usd = sum(p.alloc_dollars for p in selected)
    total_deployed_pct = total_deployed_usd / capital if capital > 0 else 0.0

    return {
        "positions":           [p.to_dict() for p in selected],
        "rejected_liquidity":  rejected_liquidity,
        "rejected_no_signal":  rejected_no_signal,
        "rejected_errors":     rejected_errors,
        "total_deployed_pct":  round(total_deployed_pct, 4),
        "total_deployed_usd":  round(total_deployed_usd, 2),
        "capital":             capital,
        "curr_date":           curr_date,
        "n_evaluated":         len(symbols),
        "n_selected":          len(selected),
        "base_alloc_pct":      BASE_ALLOC,
        "max_positions":       max_positions,
        "min_dollar_vol":      min_dollar_vol,
    }
