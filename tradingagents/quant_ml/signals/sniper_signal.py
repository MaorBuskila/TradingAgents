"""
sniper_signal.py
----------------
Precision Sniper: load cached WFO params → compute latest bar signal, SL/TP ladder.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import pandas as pd

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.sniper_cache import get_sniper_params
from tradingagents.quant_ml.indicators.stockstats_utils import _clean_dataframe, yf_retry
from tradingagents.quant_ml.indicators.vol_regime import compute_vol_regime
from tradingagents.quant_ml.risk.structure_sl import compute_stop
from tradingagents.quant_ml.risk.tp_ladder import compute_tps
from tradingagents.quant_ml.signals.sniper_features import build_sniper_feature_frame

logger = logging.getLogger(__name__)


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

    cached = get_sniper_params(t)
    ema_f = int(cached["optimal_ema_fast"]) if cached else 9
    ema_s = int(cached["optimal_ema_slow"]) if cached else 21
    ema_t = int(cached["optimal_ema_trend"]) if cached else 55
    min_sc = float(cached["optimal_min_score"]) if cached else 5.0
    sl_m = float(cached["optimal_sl_mult"]) if cached else 1.5
    vol_m = float(cached["optimal_vol_mult"]) if cached else 1.2
    th_a = float(cached["threshold_a"]) if cached and cached.get("threshold_a") is not None else None
    th_b = float(cached["threshold_b"]) if cached and cached.get("threshold_b") is not None else None

    try:
        raw = _load_price(t, as_of_date)
    except Exception as exc:
        return {"ticker": t, "as_of_date": as_of_date, "error": str(exc)}

    if len(raw) < 80:
        return {"ticker": t, "as_of_date": as_of_date, "error": "insufficient price history"}

    full = build_sniper_feature_frame(
        raw, fast=ema_f, slow=ema_s, trend=ema_t, vol_mult=vol_m
    ).dropna(
        subset=["bull_score", "ema_f", "rsi", "vwap"],
    )

    if full.empty:
        return {"ticker": t, "as_of_date": as_of_date, "error": "warmup incomplete"}

    last_i = len(full) - 1
    row = full.iloc[last_i]
    close = float(row["Close"])
    bull = float(row["bull_score"])
    bear = float(row["bear_score"])
    bgrade = str(row["bull_grade"])
    bear_grade = str(row["bear_grade"])

    regime_s = compute_vol_regime(raw.tail(120)).iloc[-1] if len(raw) >= 60 else "NORMAL"

    action = "HOLD"
    if bool(row["cross_up"]) and bull >= min_sc:
        action = "BUY"
    elif bool(row["cross_dn"]) and bear >= min_sc:
        action = "SELL"

    entry_i = last_i
    entry = float(row["Close"])
    hist = full.iloc[: entry_i + 1]
    sl = tp1 = tp2 = tp3 = None
    if action == "BUY":
        sl = compute_stop("long", entry, hist, entry_idx=entry_i, atr_mult=sl_m)
        tp1, tp2, tp3 = compute_tps("long", entry, sl)
    elif action == "SELL":
        sl = compute_stop("short", entry, hist, entry_idx=entry_i, atr_mult=sl_m)
        tp1, tp2, tp3 = compute_tps("short", entry, sl)

    out: dict[str, Any] = {
        "ticker": t,
        "as_of_date": as_of_date,
        "action": action,
        "bull_score": round(bull, 4),
        "bear_score": round(bear, 4),
        "bull_grade": bgrade,
        "bear_grade": bear_grade,
        "vol_regime": str(regime_s),
        "latest_close": round(close, 4),
        "using_cached_params": cached is not None,
        "ema_fast": ema_f,
        "ema_slow": ema_s,
        "ema_trend": ema_t,
        "min_score": min_sc,
        "sl_mult": sl_m,
        "vol_mult": vol_m,
        "stop_loss": round(sl, 4) if sl is not None else None,
        "tp1": round(tp1, 4) if tp1 is not None else None,
        "tp2": round(tp2, 4) if tp2 is not None else None,
        "tp3": round(tp3, 4) if tp3 is not None else None,
        "confidence": str(cached.get("confidence")) if cached else "UNKNOWN",
        "classical_oos_sharpe": float(cached["classical_oos_sharpe"]) if cached else None,
        "threshold_a": th_a,
        "threshold_b": th_b,
        "dt_oos_sharpe": float(cached["dt_oos_sharpe"]) if cached and cached.get("dt_oos_sharpe") is not None else None,
        "error": None,
    }
    out["dt_gate_note"] = "DT thresholds apply to ML layer; call run_sniper_dt_optimizer for live probs."
    return out
