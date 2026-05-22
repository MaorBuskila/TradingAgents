"""
experiment_runner.py
--------------------
Orchestrates RSI DT Feature Lab runs: fetches data once, runs each selected
RuleSet through the WFO machinery in parallel, persists results to SQLite.
"""

from __future__ import annotations

import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from tradingagents.quant_ml.optimizers.rsi_dt_optimizer import (
    _build_features_extended,
    _triple_barrier_labels,
    _run_wfo,
    _rolling_quantile,
    _oos_backtest_stats,
    _confidence_tier,
    _sentiment_today,
    FETCH_YEARS,
    WARMUP_BARS,
    MIN_USABLE_BARS,
    ROLLING_THRESHOLD_WINDOW,
    THRESHOLD_PERCENTILE,
    HIGH_VOL_PCT,
)
from tradingagents.quant_ml.indicators.adx_di import compute_adx_di
from tradingagents.quant_ml.indicators.volume_burst import compute_volume_burst
from tradingagents.quant_ml.indicators.stockstats_utils import yf_retry
from tradingagents.quant_ml.experiments.rule_sets import RuleSet, PREDEFINED, make_custom
from tradingagents.quant_ml.experiments.experiment_db import save_experiment_result

log = logging.getLogger("rsi_dt_feature_lab")


def run_single_rule_set(
    symbol: str,
    curr_date: str,
    rule_set: RuleSet,
    df: pd.DataFrame,
    sentiment_today: float | None,
    *,
    is_days: int,
    oos_days: int,
    label_horizon: int,
    tp_mult: float,
    sl_mult: float,
    rsi_period: int,
    run_id: str,
) -> dict:
    """Run WFO for one rule set. df is raw OHLCV already fetched by orchestrator."""
    log.info("[feature_lab] rule_set=%s symbol=%s", rule_set.name, symbol)

    feat_full = _build_features_extended(df, rsi_period=rsi_period)
    feat_full = feat_full.iloc[WARMUP_BARS:].copy()

    labels_full = _triple_barrier_labels(
        close=feat_full["price"],
        high=feat_full["high"],
        low=feat_full["low"],
        atr=feat_full["atr_14"],
        horizon=label_horizon,
        tp_mult=tp_mult,
        sl_mult=sl_mult,
    )

    feature_cols = list(rule_set.feature_cols)
    mask = labels_full.notna() & feat_full[feature_cols].notna().all(axis=1)
    feat_df = feat_full[mask].copy()
    labels = labels_full[mask].astype(int)

    if len(feat_df) < is_days + oos_days:
        raise ValueError(
            f"[{rule_set.name}] Too few labeled bars for {symbol}: {len(feat_df)} "
            f"(need {is_days + oos_days})"
        )

    wfo = _run_wfo(feat_df, labels, is_days=is_days, oos_days=oos_days, feature_cols=feature_cols)

    threshold_a = _rolling_quantile(
        wfo["oos_preds_a"].tolist(), ROLLING_THRESHOLD_WINDOW, THRESHOLD_PERCENTILE,
    )
    threshold_b = _rolling_quantile(
        wfo["oos_preds_b"].tolist(), ROLLING_THRESHOLD_WINDOW, THRESHOLD_PERCENTILE,
    )

    oos_sharpe, hit_rate, trade_count = _oos_backtest_stats(
        wfo["oos_preds_a"], threshold_a, wfo["oos_labels"], wfo["oos_returns"],
    )
    agreement = (wfo["oos_preds_a"] > threshold_a) & (wfo["oos_preds_b"] > threshold_b)
    agreement_rate = float(agreement.mean()) if len(agreement) else 0.0
    avg_oos_acc = float(np.mean([s["oos_acc"] for s in wfo["slides"]])) if wfo["slides"] else 0.0
    n_slides = len(wfo["slides"])
    confidence = _confidence_tier(oos_sharpe, agreement_rate)

    # Today's signal with only the veto rules this rule set declares
    today_row = feat_full.iloc[[-1]]
    price_today = float(today_row["price"].iloc[0])
    ema200_today = float(today_row["ema200"].iloc[0])

    gate_results: dict[str, bool] = {}
    if "ema200" in rule_set.veto_rules:
        gate_results["ema200"] = price_today > ema200_today
    if "sentiment" in rule_set.veto_rules:
        gate_results["sentiment"] = (sentiment_today is None) or (sentiment_today > 0.0)
    if "volume_burst" in rule_set.veto_rules:
        gate_results["volume_burst"] = bool(compute_volume_burst(df).iloc[-1])
    if "adx_gt_20" in rule_set.veto_rules:
        gate_results["adx_gt_20"] = float(compute_adx_di(df)["adx"].iloc[-1]) > 20.0

    atr_norm_today = float(today_row["atr_norm_14"].iloc[0])
    atr_norm_hist = feat_df["atr_norm_14"].to_numpy(dtype=float)
    atr_pct_rank = float((atr_norm_hist < atr_norm_today).mean() * 100) if len(atr_norm_hist) else 50.0
    size_mult = 0.5 if atr_pct_rank > HIGH_VOL_PCT else 1.0

    row = {
        "run_id": run_id,
        "symbol": symbol,
        "date": curr_date,
        "rule_set_name": rule_set.name,
        "features_used": feature_cols,
        "veto_rules": list(rule_set.veto_rules),
        "oos_sharpe": round(oos_sharpe, 4),
        "hit_rate": round(hit_rate, 4),
        "trade_count": int(trade_count),
        "confidence": confidence,
        "n_slides": int(n_slides),
        "avg_oos_acc": round(avg_oos_acc, 4),
        "threshold_a": round(threshold_a, 4),
        "threshold_b": round(threshold_b, 4),
        "gate_results": gate_results,
        "size_mult": round(size_mult, 2),
        "ran_at": datetime.utcnow().isoformat(),
    }

    try:
        save_experiment_result(row)
    except Exception as exc:
        log.warning("[feature_lab] DB save failed for %s/%s: %s", symbol, rule_set.name, exc)

    return row


def run_feature_lab(
    symbol: str,
    curr_date: str,
    rule_set_names: list[str],
    custom_features: list[str] | None = None,
    custom_vetos: list[str] | None = None,
    *,
    is_days: int = 1000,
    oos_days: int = 20,
    label_horizon: int = 10,
    tp_mult: float = 2.0,
    sl_mult: float = 1.0,
    rsi_period: int = 14,
    cancel_event: threading.Event | None = None,
    run_id: str | None = None,
) -> list[dict]:
    """
    Fetch OHLCV once, run selected rule sets in parallel, return results
    sorted by oos_sharpe descending.
    """
    if run_id is None:
        run_id = str(uuid.uuid4())

    curr_ts = pd.to_datetime(curr_date)
    start_date = (curr_ts - pd.DateOffset(years=FETCH_YEARS)).strftime("%Y-%m-%d")
    end_date = (curr_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    df = yf_retry(lambda: yf.download(
        symbol, start=start_date, end=end_date,
        interval="1d", auto_adjust=True, progress=False,
    ))
    if df is None or df.empty:
        raise ValueError(f"No data for {symbol}")
    df = df[df.index <= curr_ts].copy()
    if len(df) < MIN_USABLE_BARS:
        raise ValueError(
            f"Insufficient data for {symbol}: {len(df)} bars (need {MIN_USABLE_BARS})"
        )

    sentiment = _sentiment_today(symbol)

    rule_sets: list[RuleSet] = []
    for name in rule_set_names:
        if name == "CUSTOM":
            if custom_features:
                rule_sets.append(make_custom(custom_features, custom_vetos or []))
        elif name in PREDEFINED:
            rule_sets.append(PREDEFINED[name])
        else:
            log.warning("[feature_lab] Unknown rule set '%s', skipping", name)

    if not rule_sets:
        raise ValueError("No valid rule sets selected")

    results: list[dict] = []
    errors: list[str] = []

    with ThreadPoolExecutor(max_workers=min(len(rule_sets), 4)) as pool:
        futures = {}
        for rs in rule_sets:
            if cancel_event and cancel_event.is_set():
                break
            fut = pool.submit(
                run_single_rule_set,
                symbol, curr_date, rs, df, sentiment,
                is_days=is_days, oos_days=oos_days, label_horizon=label_horizon,
                tp_mult=tp_mult, sl_mult=sl_mult, rsi_period=rsi_period, run_id=run_id,
            )
            futures[fut] = rs.name

        for fut in as_completed(futures):
            rs_name = futures[fut]
            try:
                results.append(fut.result())
            except Exception as exc:
                log.error("[feature_lab] %s failed: %s", rs_name, exc)
                errors.append(f"{rs_name}: {exc}")

    results.sort(key=lambda r: r["oos_sharpe"], reverse=True)
    return results
