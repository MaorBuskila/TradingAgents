"""
sniper_signal.py
----------------
Precision Sniper: load cached WFO params → compute latest bar signal, SL/TP ladder.

EMA params come from ema_cache.db (set by ema_optimizer_algo).
DT thresholds come from sniper_cache.db (set by sniper_dt_optimizer).
min_score is hardcoded to 5.0 (grade B+ veto — not optimizable via simple Sharpe).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import pandas as pd

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.ema_cache import get_ema_params
from tradingagents.dataflows.sniper_cache import get_sniper_params
from tradingagents.quant_ml.indicators.stockstats_utils import _clean_dataframe, yf_retry
from tradingagents.quant_ml.indicators.vol_regime import compute_vol_regime
from tradingagents.quant_ml.risk.structure_sl import compute_stop
from tradingagents.quant_ml.risk.tp_ladder import compute_tps
from tradingagents.quant_ml.signals.sniper_features import build_sniper_feature_frame

logger = logging.getLogger(__name__)

MIN_SCORE = 5.0   # grade B+ veto — hardcoded, not optimized
SL_MULT   = 1.5   # default ATR multiplier for stop-loss


def _load_price(symbol: str, as_of: str) -> pd.DataFrame:
    config = get_config()
    today = pd.Timestamp.today()
    start = today - pd.DateOffset(years=15)
    start_str, end_str = start.strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")
    path = os.path.join(config["data_cache_dir"], f"{symbol}-YFin-data-{start_str}-{end_str}.csv")
    if os.path.exists(path):
        data = pd.read_csv(path, on_bad_lines="skip")
    else:
        import yfinance as yf

        os.makedirs(config["data_cache_dir"], exist_ok=True)
        data = yf_retry(
            lambda: yf.download(
                symbol, start=start_str, end=end_str,
                multi_level_index=False, progress=False, auto_adjust=True,
            )
        )
        data = data.reset_index()
        data.to_csv(path, index=False)
    data = _clean_dataframe(data)
    data = data.sort_values("Date").reset_index(drop=True)
    data = data[data["Date"] <= pd.to_datetime(as_of)].copy()
    return data


def compute_sniper_signal(ticker: str, as_of_date: Optional[str] = None) -> dict[str, Any]:
    """Return today's Sniper action, grades, SL/TP, vol regime, optional DT gate."""
    t = ticker.upper().strip()
    if not as_of_date:
        as_of_date = pd.Timestamp.today().strftime("%Y-%m-%d")

    # EMA params from ema_cache; DT thresholds from sniper_cache
    ema = get_ema_params(t) or {}
    dt  = get_sniper_params(t) or {}

    ema_f  = int(ema.get("optimal_fast",  9))
    ema_s  = int(ema.get("optimal_slow",  21))
    ema_t  = int(ema.get("optimal_trend", 55))
    th_a   = float(dt["threshold_a"]) if dt.get("threshold_a") is not None else None
    th_b   = float(dt["threshold_b"]) if dt.get("threshold_b") is not None else None

    try:
        raw = _load_price(t, as_of_date)
    except Exception as exc:
        return {"ticker": t, "as_of_date": as_of_date, "error": str(exc)}

    if len(raw) < 80:
        return {"ticker": t, "as_of_date": as_of_date, "error": "insufficient price history"}

    full = build_sniper_feature_frame(
        raw, fast=ema_f, slow=ema_s, trend=ema_t, vol_mult=1.2
    ).dropna(subset=["bull_score", "ema_f", "rsi", "vwap"])

    if full.empty:
        return {"ticker": t, "as_of_date": as_of_date, "error": "warmup incomplete"}

    last_i = len(full) - 1
    row    = full.iloc[last_i]
    close  = float(row["Close"])
    bull   = float(row["bull_score"])
    bear   = float(row["bear_score"])

    regime_s = compute_vol_regime(raw.tail(120)).iloc[-1] if len(raw) >= 60 else "NORMAL"

    action = "HOLD"
    if bool(row["cross_up"]) and bull >= MIN_SCORE:
        action = "BUY"
    elif bool(row["cross_dn"]) and bear >= MIN_SCORE:
        action = "SELL"

    entry = float(row["Close"])
    hist  = full.iloc[: last_i + 1]
    sl = tp1 = tp2 = tp3 = None
    if action == "BUY":
        sl = compute_stop("long",  entry, hist, entry_idx=last_i, atr_mult=SL_MULT)
        tp1, tp2, tp3 = compute_tps("long",  entry, sl)
    elif action == "SELL":
        sl = compute_stop("short", entry, hist, entry_idx=last_i, atr_mult=SL_MULT)
        tp1, tp2, tp3 = compute_tps("short", entry, sl)

    return {
        "ticker":       t,
        "as_of_date":   as_of_date,
        "action":       action,
        "bull_score":   round(bull, 4),
        "bear_score":   round(bear, 4),
        "bull_grade":   str(row["bull_grade"]),
        "bear_grade":   str(row["bear_grade"]),
        "vol_regime":   str(regime_s),
        "latest_close": round(close, 4),
        "ema_fast":     ema_f,
        "ema_slow":     ema_s,
        "ema_trend":    ema_t,
        "min_score":    MIN_SCORE,
        "stop_loss":    round(sl,  4) if sl  is not None else None,
        "tp1":          round(tp1, 4) if tp1 is not None else None,
        "tp2":          round(tp2, 4) if tp2 is not None else None,
        "tp3":          round(tp3, 4) if tp3 is not None else None,
        "threshold_a":  th_a,
        "threshold_b":  th_b,
        "ema_oos_sharpe": float(ema.get("oos_sharpe", 0)) if ema else None,
        "ema_confidence": str(ema.get("confidence", "UNKNOWN")) if ema else "UNKNOWN",
        "dt_oos_sharpe": float(dt["dt_oos_sharpe"]) if dt.get("dt_oos_sharpe") is not None else None,
        "last_rsi":      round(float(row["rsi"]),      2) if "rsi"      in row.index else None,
        "htf_bias":      round(float(row["htf_bias"]), 4) if "htf_bias" in row.index else None,
        "error":         None,
    }
