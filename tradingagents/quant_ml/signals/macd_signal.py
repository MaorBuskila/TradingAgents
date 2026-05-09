"""
macd_signal.py
--------------
Full MACD signal flow:  OPTIMIZE → GET PARAMS → CALCULATE MACD → SIGNAL → ACT

Public entry point:
    compute_macd_signal(ticker, as_of_date=None) -> dict

Steps:
    1. GET PARAMS   — load optimal (fast, slow, signal) from macd_cache.db.
                      Falls back to MACD(12, 26, 9) if the ticker was never optimized.
    2. FETCH DATA   — pull recent price history via yfinance (same cache as optimizer).
    3. CALC MACD    — standard EMA-based MACD + signal line + histogram.
    4. SIGNAL       — crossover detection on histogram sign flip → BUY / SELL / HOLD.
    5. CONTEXT      — return full metadata so the UI / agents can explain the decision.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from tradingagents.dataflows.macd_cache import get_macd_params
from tradingagents.quant_ml.indicators.stockstats_utils import yf_retry, _clean_dataframe
from tradingagents.dataflows.config import get_config

logger = logging.getLogger(__name__)

# ── Defaults used when no optimization params exist for a ticker ───────────────
_DEFAULT_FAST   = 12
_DEFAULT_SLOW   = 26
_DEFAULT_SIGNAL = 9

# Minimum bars needed before the MACD value is considered reliable
# slow_period (34 max) + signal_period (13 max) + warmup buffer
_MIN_BARS = 60


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _calc_macd(
    closes: pd.Series,
    fast: int,
    slow: int,
    signal: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Standard MACD via EMA.
    Returns (macd_line, signal_line, histogram).
    """
    ema_fast    = closes.ewm(span=fast,   adjust=False).mean()
    ema_slow    = closes.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


def _load_recent_prices(symbol: str, lookback_days: int = 180) -> pd.DataFrame:
    """
    Fetch the last `lookback_days` calendar days of OHLCV data.
    Uses the same yfinance cache as the optimizer.
    """
    config    = get_config()
    today     = pd.Timestamp.today()
    start     = today - pd.DateOffset(years=15)
    end       = today
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

    cutoff = today - pd.DateOffset(days=lookback_days)
    data   = data[data["Date"] >= cutoff].reset_index(drop=True)
    return data


def _classify_signal(histogram: pd.Series) -> str:
    """
    Detect a MACD signal crossover on the last two bars.

    BUY:  histogram flips from ≤ 0 to > 0  (bullish crossover)
    SELL: histogram flips from ≥ 0 to < 0  (bearish crossover)
    HOLD: no crossover
    """
    if len(histogram) < 2:
        return "HOLD"
    prev = float(histogram.iloc[-2])
    curr = float(histogram.iloc[-1])
    if np.isnan(prev) or np.isnan(curr):
        return "HOLD"
    if prev <= 0 and curr > 0:
        return "BUY"
    elif prev >= 0 and curr < 0:
        return "SELL"
    return "HOLD"


def _histogram_direction(histogram: pd.Series) -> str:
    """Describe whether histogram momentum is growing or shrinking."""
    if len(histogram) < 2:
        return "contracting"
    curr = float(histogram.iloc[-1])
    prev = float(histogram.iloc[-2])
    if np.isnan(curr) or np.isnan(prev):
        return "contracting"
    if curr > 0:
        return "expanding_bullish" if curr > prev else "contracting"
    elif curr < 0:
        return "expanding_bearish" if curr < prev else "contracting"
    return "contracting"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_macd_signal(
    ticker: str,
    as_of_date: Optional[str] = None,
    lookback_days: int = 180,
) -> Dict[str, Any]:
    """
    Run the full OPTIMIZE → GET PARAMS → CALC MACD → SIGNAL → ACT pipeline
    for a single ticker.

    Args:
        ticker:        Stock symbol, e.g. "AAPL".
        as_of_date:    Treat this as "today" (YYYY-MM-DD). Defaults to today.
        lookback_days: How many calendar days of price history to load.
                       Must be enough to warm up the MACD (≥ 60 trading days).

    Returns a dict with keys:
        ticker, as_of_date,
        action            — "BUY" | "SELL" | "HOLD"
        macd_line         — float (current)
        signal_line       — float (current)
        histogram         — float (current)
        histogram_direction — "expanding_bullish" | "expanding_bearish" | "contracting"
        fast_period       — int
        slow_period       — int
        signal_period     — int
        using_cached      — bool  (False = fell back to defaults)
        confidence        — str   ("HIGH"/"MEDIUM"/"LOW" or "DEFAULT")
        regime            — str   (from optimizer, may be empty)
        optimizer_provider— str
        optimized_at      — str
        latest_close      — float
        price_change_pct  — float (1-day % change)
        macd_history      — list[dict] of {date, macd, signal, histogram, close} last 30 bars
        error             — str | None
    """
    symbol = ticker.upper().strip()
    today  = as_of_date or datetime.now().strftime("%Y-%m-%d")

    # ── STEP 1: GET PARAMS ────────────────────────────────────────────────
    cached = get_macd_params(symbol)
    if cached:
        fast_period   = int(cached["optimal_fast"])
        slow_period   = int(cached["optimal_slow"])
        signal_period = int(cached["optimal_signal"])
        confidence    = cached.get("confidence", "UNKNOWN")
        regime        = cached.get("regime") or ""
        opt_prov      = cached.get("optimizer_provider") or "algo"
        opt_at        = cached.get("optimized_at") or ""
        using_cached  = True
    else:
        fast_period   = _DEFAULT_FAST
        slow_period   = _DEFAULT_SLOW
        signal_period = _DEFAULT_SIGNAL
        confidence    = "DEFAULT"
        regime        = ""
        opt_prov      = "none"
        opt_at        = ""
        using_cached  = False
        logger.warning(
            "No cached MACD params for %s — using defaults MACD(%d,%d,%d)",
            symbol, fast_period, slow_period, signal_period,
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

    as_of_dt = pd.to_datetime(today)
    data     = data[data["Date"] <= as_of_dt].reset_index(drop=True)

    if len(data) < _MIN_BARS:
        return _error_response(
            symbol, today,
            f"Only {len(data)} bars up to {today}; need at least {_MIN_BARS}."
        )

    closes = data["Close"]

    # ── STEP 3: CALCULATE MACD ────────────────────────────────────────────
    macd_s, signal_s, hist_s = _calc_macd(closes, fast_period, slow_period, signal_period)

    current_macd   = float(macd_s.iloc[-1])
    current_signal = float(signal_s.iloc[-1])
    current_hist   = float(hist_s.iloc[-1])

    if np.isnan(current_macd) or np.isnan(current_hist):
        return _error_response(symbol, today, "MACD calculation returned NaN.")

    # ── STEP 4: SIGNAL ────────────────────────────────────────────────────
    action    = _classify_signal(hist_s)
    hist_dir  = _histogram_direction(hist_s)

    # ── STEP 5: BUILD CONTEXT ─────────────────────────────────────────────
    latest_close    = float(closes.iloc[-1])
    price_change_pct = 0.0
    if len(closes) >= 2:
        prev_close = float(closes.iloc[-2])
        if prev_close > 0:
            price_change_pct = round((latest_close - prev_close) / prev_close * 100, 3)

    # Last 30 bars of {date, macd, signal, histogram, close} for UI chart
    tail_n    = min(30, len(data))
    tail_data = data.tail(tail_n).reset_index(drop=True)
    tail_macd = macd_s.tail(tail_n).reset_index(drop=True)
    tail_sig  = signal_s.tail(tail_n).reset_index(drop=True)
    tail_hist = hist_s.tail(tail_n).reset_index(drop=True)

    macd_history = []
    for i in range(len(tail_data)):
        dt = tail_data["Date"].iloc[i]
        dt_str = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)[:10]

        def _safe(val):
            v = float(val)
            return round(v, 4) if not np.isnan(v) else None

        macd_history.append({
            "date":      dt_str,
            "macd":      _safe(tail_macd.iloc[i]),
            "signal":    _safe(tail_sig.iloc[i]),
            "histogram": _safe(tail_hist.iloc[i]),
            "close":     round(float(tail_data["Close"].iloc[i]), 2),
        })

    return {
        "ticker":               symbol,
        "as_of_date":           today,
        "action":               action,         # "BUY" | "SELL" | "HOLD"
        "macd_line":            round(current_macd, 4),
        "signal_line":          round(current_signal, 4),
        "histogram":            round(current_hist, 4),
        "histogram_direction":  hist_dir,
        "fast_period":          fast_period,
        "slow_period":          slow_period,
        "signal_period":        signal_period,
        "using_cached":         using_cached,
        "confidence":           confidence,
        "regime":               regime,
        "optimizer_provider":   opt_prov,
        "optimized_at":         opt_at,
        "latest_close":         round(latest_close, 2),
        "price_change_pct":     price_change_pct,
        "macd_history":         macd_history,
        "error":                None,
    }


def compute_macd_signals_batch(tickers: list[str]) -> list[Dict[str, Any]]:
    """Run compute_macd_signal for a list of tickers. Errors per-ticker don't abort others."""
    return [compute_macd_signal(t) for t in tickers]


def _error_response(symbol: str, today: str, msg: str) -> Dict[str, Any]:
    logger.error("MACD signal error for %s: %s", symbol, msg)
    return {
        "ticker":              symbol,
        "as_of_date":          today,
        "action":              None,
        "macd_line":           None,
        "signal_line":         None,
        "histogram":           None,
        "histogram_direction": None,
        "fast_period":         None,
        "slow_period":         None,
        "signal_period":       None,
        "using_cached":        False,
        "confidence":          None,
        "regime":              None,
        "optimizer_provider":  None,
        "optimized_at":        None,
        "latest_close":        None,
        "price_change_pct":    None,
        "macd_history":        [],
        "error":               msg,
    }
