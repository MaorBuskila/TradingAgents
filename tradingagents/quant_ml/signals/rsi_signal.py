"""
rsi_signal.py
-------------
Full RSI signal flow:  OPTIMIZE → GET PARAMS → CALCULATE RSI → SIGNAL → ACT

Public entry point:
    compute_rsi_signal(ticker, as_of_date=None) -> dict

Steps:
    1. GET PARAMS   — load optimal (period, overbought, oversold) from rsi_cache.db.
                      Falls back to RSI-14 / 70 / 30 if the ticker was never optimized.
    2. FETCH DATA   — pull recent price history via yfinance (same cache as optimizer).
    3. CALC RSI     — Wilder's smoothing method (identical to the optimizer).
    4. SIGNAL       — apply adaptive thresholds → BUY / SELL / HOLD.
    5. CONTEXT      — return full metadata so the UI / agents can explain the decision.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from tradingagents.dataflows.rsi_cache import get_rsi_params
from tradingagents.quant_ml.indicators.stockstats_utils import yf_retry, _clean_dataframe
from tradingagents.dataflows.config import get_config

logger = logging.getLogger(__name__)

# ── Defaults used when no optimization params exist for a ticker ───────────────
_DEFAULT_PERIOD = 14
_DEFAULT_UPPER  = 70.0
_DEFAULT_LOWER  = 30.0

# Minimum bars needed before the RSI value is considered reliable
_MIN_BARS = 50


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _calc_rsi(closes: pd.Series, period: int) -> pd.Series:
    """Wilder's RSI — identical to the optimizer so signals are consistent."""
    delta    = closes.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, 1e-10)
    return 100.0 - (100.0 / (1.0 + rs))


def _load_recent_prices(symbol: str, lookback_days: int = 120) -> pd.DataFrame:
    """
    Fetch the last `lookback_days` calendar days of OHLCV data.
    Uses the same yfinance cache as the optimizer when available; otherwise
    does a fresh download for the requested window only.
    """
    config   = get_config()
    today    = pd.Timestamp.today()
    start    = today - pd.DateOffset(years=15)
    end      = today

    start_str = start.strftime("%Y-%m-%d")
    end_str   = end.strftime("%Y-%m-%d")

    data_file = os.path.join(
        config["data_cache_dir"],
        f"{symbol}-YFin-data-{start_str}-{end_str}.csv",
    )

    if os.path.exists(data_file):
        data = pd.read_csv(data_file, on_bad_lines="skip")
    else:
        os.makedirs(config["data_cache_dir"], exist_ok=True)
        data = yf_retry(lambda: yf.download(
            symbol,
            start=start_str,
            end=end_str,
            multi_level_index=False,
            progress=False,
            auto_adjust=True,
        ))
        data = data.reset_index()
        data.to_csv(data_file, index=False)

    data = _clean_dataframe(data)
    data = data.sort_values("Date").reset_index(drop=True)

    # Keep only the last lookback_days calendar-day window
    cutoff = today - pd.DateOffset(days=lookback_days)
    data   = data[data["Date"] >= cutoff].reset_index(drop=True)
    return data


def _classify_signal(rsi_val: float, upper: float, lower: float) -> str:
    """Map a single RSI value to BUY / SELL / HOLD."""
    if rsi_val < lower:
        return "BUY"
    elif rsi_val > upper:
        return "SELL"
    return "HOLD"


def _rsi_zone(rsi_val: float, upper: float, lower: float) -> str:
    """Human-readable zone label."""
    if rsi_val < lower:
        return "OVERSOLD"
    elif rsi_val > upper:
        return "OVERBOUGHT"
    return "NEUTRAL"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_rsi_signal(
    ticker: str,
    as_of_date: Optional[str] = None,
    lookback_days: int = 120,
) -> Dict[str, Any]:
    """
    Run the full OPTIMIZE → GET PARAMS → CALC RSI → SIGNAL → ACT pipeline
    for a single ticker.

    Args:
        ticker:        Stock symbol, e.g. "AAPL".
        as_of_date:    Treat this as "today" (YYYY-MM-DD). Defaults to today.
        lookback_days: How many calendar days of price history to load.
                       Must be enough to warm up the RSI (>=50 trading days).

    Returns a dict with keys:
        ticker, as_of_date,
        action            — "BUY" | "SELL" | "HOLD"
        current_rsi       — float
        rsi_zone          — "OVERSOLD" | "OVERBOUGHT" | "NEUTRAL"
        period            — int   (optimal RSI period used)
        overbought        — float (optimal upper threshold)
        oversold          — float (optimal lower threshold)
        using_cached      — bool  (False = fell back to defaults)
        confidence        — str   (from optimizer: "HIGH"/"MEDIUM"/"LOW" or "DEFAULT")
        regime            — str   (market regime from LLM run, may be empty)
        optimizer_provider— str
        optimized_at      — str
        latest_close      — float
        price_change_pct  — float (1-day % change)
        rsi_series        — list[dict] of {date, rsi, close} for the last 30 bars
        error             — str | None
    """
    symbol = ticker.upper().strip()
    today  = as_of_date or datetime.now().strftime("%Y-%m-%d")

    # ── STEP 1: GET PARAMS ────────────────────────────────────────────────
    cached = get_rsi_params(symbol)
    if cached:
        period     = int(cached["optimal_period"])
        overbought = float(cached["optimal_upper"])
        oversold   = float(cached["optimal_lower"])
        confidence = cached.get("confidence", "UNKNOWN")
        regime     = cached.get("regime") or ""
        opt_prov   = cached.get("optimizer_provider") or "algo"
        opt_at     = cached.get("optimized_at") or ""
        using_cached = True
    else:
        period     = _DEFAULT_PERIOD
        overbought = _DEFAULT_UPPER
        oversold   = _DEFAULT_LOWER
        confidence = "DEFAULT"
        regime     = ""
        opt_prov   = "none"
        opt_at     = ""
        using_cached = False
        logger.warning(
            "No cached RSI params for %s — using defaults RSI-%d / %g / %g",
            symbol, period, overbought, oversold,
        )

    # ── STEP 2: FETCH PRICE DATA ──────────────────────────────────────────
    try:
        data = _load_recent_prices(symbol, lookback_days=lookback_days)
    except Exception as exc:
        return _error_response(symbol, today, str(exc))

    if len(data) < _MIN_BARS:
        return _error_response(
            symbol, today,
            f"Only {len(data)} price bars available; need at least {_MIN_BARS}."
        )

    # Filter to as_of_date
    as_of_dt = pd.to_datetime(today)
    data = data[data["Date"] <= as_of_dt].reset_index(drop=True)

    if len(data) < _MIN_BARS:
        return _error_response(
            symbol, today,
            f"Only {len(data)} bars up to {today}; need at least {_MIN_BARS}."
        )

    closes = data["Close"]

    # ── STEP 3: CALCULATE RSI ─────────────────────────────────────────────
    rsi_series = _calc_rsi(closes, period)

    current_rsi = float(rsi_series.iloc[-1])
    if np.isnan(current_rsi):
        return _error_response(symbol, today, "RSI calculation returned NaN.")

    # ── STEP 4: SIGNAL ────────────────────────────────────────────────────
    action   = _classify_signal(current_rsi, overbought, oversold)
    rsi_zone = _rsi_zone(current_rsi, overbought, oversold)

    # ── STEP 5: BUILD CONTEXT ─────────────────────────────────────────────
    latest_close = float(closes.iloc[-1])

    # 1-day price change %
    price_change_pct = 0.0
    if len(closes) >= 2:
        prev_close = float(closes.iloc[-2])
        if prev_close > 0:
            price_change_pct = round((latest_close - prev_close) / prev_close * 100, 3)

    # Last 30 bars of {date, rsi, close} for a mini-chart in the UI
    tail_n = min(30, len(data))
    tail_data  = data.tail(tail_n).reset_index(drop=True)
    tail_rsi   = rsi_series.tail(tail_n).reset_index(drop=True)

    rsi_history = []
    for i in range(len(tail_data)):
        dt  = tail_data["Date"].iloc[i]
        dt_str = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)[:10]
        rsi_val = float(tail_rsi.iloc[i]) if not np.isnan(tail_rsi.iloc[i]) else None
        rsi_history.append({
            "date":  dt_str,
            "rsi":   round(rsi_val, 2) if rsi_val is not None else None,
            "close": round(float(tail_data["Close"].iloc[i]), 2),
        })

    return {
        "ticker":             symbol,
        "as_of_date":         today,
        "action":             action,       # "BUY" | "SELL" | "HOLD"
        "current_rsi":        round(current_rsi, 2),
        "rsi_zone":           rsi_zone,
        "period":             period,
        "overbought":         overbought,
        "oversold":           oversold,
        "using_cached":       using_cached,
        "confidence":         confidence,
        "regime":             regime,
        "optimizer_provider": opt_prov,
        "optimized_at":       opt_at,
        "latest_close":       round(latest_close, 2),
        "price_change_pct":   price_change_pct,
        "rsi_history":        rsi_history,
        "error":              None,
    }


def compute_rsi_signals_batch(tickers: list[str]) -> list[Dict[str, Any]]:
    """Run compute_rsi_signal for a list of tickers. Errors per-ticker don't abort others."""
    results = []
    for t in tickers:
        results.append(compute_rsi_signal(t))
    return results


def _error_response(symbol: str, today: str, msg: str) -> Dict[str, Any]:
    logger.error("RSI signal error for %s: %s", symbol, msg)
    return {
        "ticker":             symbol,
        "as_of_date":         today,
        "action":             None,
        "current_rsi":        None,
        "rsi_zone":           None,
        "period":             None,
        "overbought":         None,
        "oversold":           None,
        "using_cached":       False,
        "confidence":         None,
        "regime":             None,
        "optimizer_provider": None,
        "optimized_at":       None,
        "latest_close":       None,
        "price_change_pct":   None,
        "rsi_history":        [],
        "error":              msg,
    }
