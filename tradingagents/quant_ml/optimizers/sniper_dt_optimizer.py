"""
sniper_dt_optimizer.py
========================
Walk-forward + Triple Barrier + two-model XGBoost ensemble for Precision Sniper.

Post-model veto: bull composite score >= 5 (grade B+) — not a training feature.
"""

from __future__ import annotations

import logging
import os
import warnings
from datetime import datetime, timedelta

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from tradingagents.dataflows.macd_cache import get_macd_params
from tradingagents.dataflows.rsi_cache import get_rsi_params
from tradingagents.dataflows.sniper_cache import get_sniper_params
from tradingagents.quant_ml.indicators.stockstats_utils import yf_retry
from tradingagents.quant_ml.indicators.vol_regime import compute_vol_regime_ratio
from tradingagents.quant_ml.optimizers.macd_optimizer_algo import _calc_macd
from tradingagents.quant_ml.optimizers.rsi_dt_optimizer import (
    _make_model,
    _spw,
    _triple_barrier_labels,
)
from tradingagents.quant_ml.optimizers.rsi_optimizer_algo import _calc_rsi
from tradingagents.quant_ml.signals.sniper_features import (
    build_sniper_base_indicators,
    finalize_sniper_frame,
)

warnings.filterwarnings("ignore", category=UserWarning)
log = logging.getLogger("sniper_dt_optimizer")

FETCH_YEARS = 8
WARMUP_BARS = 210
MIN_USABLE_BARS = 1400
CACHE_TTL_DAYS = 7
MODEL_B_TRAIN_BARS = 1500
THRESHOLD_GRID = [0.50, 0.55, 0.60, 0.65]

# 19-feature set — 14 original + 5 new (marked NEW):
#   ema_fs_ratio       NEW  continuous fast/slow EMA ratio (direction + magnitude)
#   close_ema_f_dist   NEW  close / ema_f − 1  (proximity to fast EMA)
#   rsi_opt            NEW  RSI on WFO-optimised period for this symbol (from rsi_cache.db)
#   macd_hist_opt      NEW  MACD histogram on WFO-optimised (fast,slow,signal) from macd_cache.db
#   bull_score         NEW  continuous 0-10 sniper composite score (was only a binary veto)
#   bear_score         NEW  continuous 0-10 sniper bear composite score
ML_COLS = [
    # EMA structure
    "cross_recency",
    "ema_trend_dist",
    "ema_fs_ratio",        # NEW
    "close_ema_f_dist",    # NEW
    # RSI
    "rsi_13",
    "rsi_slope5",
    "rsi_opt",             # NEW — WFO-optimised period from rsi_cache.db
    # MACD
    "macd_hist",
    "macd_gap",
    "macd_hist_opt",       # NEW — WFO-optimised params from macd_cache.db
    # Price anchors
    "vwap_dist",
    "bb_pct_b",
    # Trend strength
    "adx_14",
    "di_spread",
    # Volume / volatility
    "vol_ratio",
    "vol_regime_ratio",
    "atr_norm",
    # Higher-timeframe
    "htf_bias_f",
    # Composite scores (continuous, not just binary veto)
    "bull_score",          # NEW
    "bear_score",          # NEW
]


def _cache_is_fresh(cached: dict | None) -> bool:
    if not cached or not cached.get("optimized_at"):
        return False
    try:
        ts = str(cached["optimized_at"])
        try:
            dt = datetime.fromisoformat(ts)
        except ValueError:
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        return datetime.utcnow() - dt < timedelta(days=CACHE_TTL_DAYS)
    except Exception:
        return False


def _build_ml_features(raw: pd.DataFrame, symbol: str = "") -> pd.DataFrame:
    """Feature matrix with all 20 ML_COLS.

    New vs original 14:
      ema_fs_ratio     — ema_f / ema_s − 1 (continuous crossover direction)
      close_ema_f_dist — close / ema_f − 1 (proximity to fast EMA)
      rsi_opt          — RSI on WFO-optimised period (rsi_cache.db); falls back to RSI(14)
      macd_hist_opt    — MACD histogram on WFO-optimised params (macd_cache.db); falls back to (12,26,9)
      bull_score       — continuous 0-10 sniper confluence score (was binary veto only)
      bear_score       — continuous 0-10 sniper bear confluence score
    """
    base = build_sniper_base_indicators(raw)
    d = finalize_sniper_frame(base, fast=9, slow=21, trend=55, vol_mult=1.2)
    close = d["Close"].astype(float)

    # ── original 14 features ──────────────────────────────────────────────────
    cu = d["cross_up"].to_numpy(dtype=bool)
    rec = np.zeros(len(d), dtype=float)
    last = 0.0
    for i in range(len(d)):
        if cu[i]:
            last = 0.0
        else:
            last = min(last + 1.0, 500.0)
        rec[i] = last
    d["cross_recency"] = rec
    d["ema_trend_dist"] = close / d["ema_t"].replace(0, np.nan) - 1.0
    d["rsi_13"] = d["rsi"].astype(float)
    d["rsi_slope5"] = d["rsi"].diff(5)
    d["macd_hist"] = d["macd_hist"].astype(float)
    d["macd_gap"] = (d["macd_line"] - d["macd_signal"]).astype(float)
    d["vwap_dist"] = (close - d["vwap"]) / close.replace(0, np.nan)
    mid = close.rolling(20, min_periods=20).mean()
    std = close.rolling(20, min_periods=20).std()
    upper = mid + 2 * std
    lower = mid - 2 * std
    d["bb_pct_b"] = (close - lower) / (upper - lower + 1e-12)
    d["adx_14"] = d["adx"].astype(float)
    d["di_spread"] = (d["plus_di"] - d["minus_di"]).astype(float)
    vol = d["Volume"].astype(float)
    d["vol_ratio"] = vol / vol.rolling(20, min_periods=20).mean().replace(0, np.nan)
    d["vol_regime_ratio"] = compute_vol_regime_ratio(d)
    prev = close.shift(1)
    tr = pd.concat(
        [
            (d["High"].astype(float) - d["Low"].astype(float)).abs(),
            (d["High"].astype(float) - prev).abs(),
            (d["Low"].astype(float) - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1.0 / 14, adjust=False).mean()
    d["atr_norm"] = atr / close.replace(0, np.nan)
    d["htf_bias_f"] = d["htf_bias"].astype(float)

    # ── NEW: ema_fs_ratio — continuous fast/slow EMA ratio ───────────────────
    d["ema_fs_ratio"] = d["ema_f"].astype(float) / d["ema_s"].replace(0, np.nan).astype(float) - 1.0

    # ── NEW: close_ema_f_dist — proximity of close to fast EMA ──────────────
    d["close_ema_f_dist"] = close / d["ema_f"].replace(0, np.nan).astype(float) - 1.0

    # ── NEW: rsi_opt — RSI on WFO-optimised period ───────────────────────────
    rsi_period = 14  # fallback
    if symbol:
        try:
            cached_rsi = get_rsi_params(symbol)
            if cached_rsi and cached_rsi.get("optimal_period"):
                rsi_period = int(cached_rsi["optimal_period"])
        except Exception:
            pass
    if rsi_period == 13:
        # identical to rsi_13 already computed — reuse it
        d["rsi_opt"] = d["rsi_13"]
    else:
        d["rsi_opt"] = _calc_rsi(close, rsi_period)

    # ── NEW: macd_hist_opt — MACD histogram on WFO-optimised params ──────────
    macd_fast, macd_slow, macd_sig = 12, 26, 9  # fallback
    if symbol:
        try:
            cached_macd = get_macd_params(symbol)
            if cached_macd and cached_macd.get("optimal_fast"):
                macd_fast = int(cached_macd["optimal_fast"])
                macd_slow = int(cached_macd["optimal_slow"])
                macd_sig  = int(cached_macd["optimal_signal"])
        except Exception:
            pass
    if (macd_fast, macd_slow, macd_sig) == (12, 26, 9):
        # identical to already-computed macd_hist — reuse it
        d["macd_hist_opt"] = d["macd_hist"]
    else:
        _, _, opt_hist = _calc_macd(close, macd_fast, macd_slow, macd_sig)
        d["macd_hist_opt"] = opt_hist

    # ── NEW: bull_score / bear_score as continuous inputs ────────────────────
    # finalize_sniper_frame already calls compute_sniper_score_series and
    # concatenates bull_score + bear_score onto d — they are present already.
    d["bull_score"] = d["bull_score"].astype(float)
    d["bear_score"] = d["bear_score"].astype(float)

    return d


def _oos_dual(
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    ta: float,
    tb: float,
    labels: np.ndarray,
    rets: np.ndarray,
    grade_ok: np.ndarray,
) -> tuple[float, float, int]:
    take = (preds_a > ta) & (preds_b > tb) & grade_ok
    if not take.any():
        return 0.0, 0.0, 0
    tr = rets[take]
    wins = labels[take] == 1
    hit_rate = float(wins.mean()) if len(wins) else 0.0
    if tr.std(ddof=0) < 1e-12:
        sharpe = 0.0
    else:
        sharpe = float(np.sqrt(252) * tr.mean() / tr.std(ddof=0))
    return sharpe, hit_rate, int(take.sum())


def _confidence_tier(oos_sharpe: float, agreement_rate: float) -> str:
    if oos_sharpe > 1.0 and agreement_rate >= 0.60:
        return "HIGH"
    if oos_sharpe > 0.5:
        return "MEDIUM"
    return "LOW"


def _run_wfo(
    feat_df: pd.DataFrame,
    labels: pd.Series,
    bull_score: pd.Series,
    is_days: int,
    oos_days: int,
) -> dict:
    X = feat_df[ML_COLS].to_numpy(dtype=float)
    y = labels.to_numpy(dtype=float)
    grade_ok_full = (bull_score.astype(float) >= 5.0).to_numpy()
    n = len(X)

    b_end = min(MODEL_B_TRAIN_BARS, max(n - is_days - oos_days, is_days))
    X_b_train, y_b_train = X[:b_end], y[:b_end]
    scaler_b = StandardScaler()
    X_b_scaled = scaler_b.fit_transform(X_b_train)
    model_b = _make_model(_spw(y_b_train))
    model_b.fit(X_b_scaled, y_b_train)

    oos_preds_a: list[float] = []
    oos_preds_b: list[float] = []
    oos_labels: list[float] = []
    oos_returns: list[float] = []
    oos_grade: list[bool] = []
    slides: list[dict] = []

    close = feat_df["Close"].astype(float).to_numpy()
    returns = np.concatenate([[0.0], np.diff(close) / np.clip(close[:-1], 1e-12, None)])

    slide_idx = 0
    start = 0
    while start + is_days + oos_days <= n:
        is_start, is_end = start, start + is_days
        oos_start, oos_end = is_end, min(is_end + oos_days, n)

        X_tr, y_tr = X[is_start:is_end], y[is_start:is_end]
        X_te, y_te = X[oos_start:oos_end], y[oos_start:oos_end]

        scaler_a = StandardScaler()
        X_tr_scaled = scaler_a.fit_transform(X_tr)
        X_te_scaled_a = scaler_a.transform(X_te)
        X_te_scaled_b = scaler_b.transform(X_te)

        model_a = _make_model(_spw(y_tr))
        model_a.fit(X_tr_scaled, y_tr)

        preds_a = model_a.predict_proba(X_te_scaled_a)[:, 1]
        preds_b = model_b.predict_proba(X_te_scaled_b)[:, 1]

        oos_preds_a.extend(preds_a.tolist())
        oos_preds_b.extend(preds_b.tolist())
        oos_labels.extend(y_te.tolist())
        oos_returns.extend(returns[oos_start:oos_end].tolist())
        oos_grade.extend(grade_ok_full[oos_start:oos_end].tolist())

        train_acc = float((model_a.predict(X_tr_scaled) == y_tr).mean())
        oos_acc = float(((preds_a > 0.5).astype(int) == y_te.astype(int)).mean()) if len(y_te) else 0.0
        slides.append({
            "slide_idx": slide_idx,
            "is_start": int(is_start),
            "is_end": int(is_end),
            "oos_start": int(oos_start),
            "oos_end": int(oos_end),
            "train_acc": round(train_acc, 4),
            "oos_acc": round(oos_acc, 4),
        })
        slide_idx += 1
        start += oos_days

    return {
        "oos_preds_a": np.array(oos_preds_a),
        "oos_preds_b": np.array(oos_preds_b),
        "oos_labels": np.array(oos_labels),
        "oos_returns": np.array(oos_returns),
        "oos_grade": np.array(oos_grade),
        "slides": slides,
        "model_b": model_b,
        "scaler_b": scaler_b,
        "b_end": b_end,
    }


def run_sniper_dt_optimizer(
    symbol: str,
    curr_date: str,
    *,
    is_days: int = 1000,
    oos_days: int = 20,
    label_horizon: int = 10,
    tp_mult: float = 1.5,
    sl_mult: float = 1.0,
    force_reoptimize: bool = False,
    **_kwargs,
) -> dict:
    log.info("[sniper_dt] WFO for %s @ %s", symbol, curr_date)

    is_days = max(int(is_days), 500)
    oos_days = max(int(oos_days), 5)

    curr_ts = pd.to_datetime(curr_date)
    start_date = (curr_ts - pd.DateOffset(years=FETCH_YEARS)).strftime("%Y-%m-%d")
    end_date = (curr_ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    raw_dl = yf_retry(lambda: yf.download(
        symbol, start=start_date, end=end_date,
        interval="1d", auto_adjust=True, progress=False,
        multi_level_index=False,
    ))
    if raw_dl is None or raw_dl.empty:
        raise ValueError(f"No data for {symbol}")
    if isinstance(raw_dl.columns, pd.MultiIndex):
        raw_dl.columns = raw_dl.columns.get_level_values(0)
    raw_dl = raw_dl[raw_dl.index <= curr_ts].copy()
    raw = raw_dl.reset_index()
    if "Date" not in raw.columns and "index" in raw.columns:
        raw = raw.rename(columns={"index": "Date"})

    if len(raw) < MIN_USABLE_BARS:
        raise ValueError(f"Insufficient data for {symbol}: {len(raw)} bars")

    feat_full = _build_ml_features(raw, symbol=symbol).iloc[WARMUP_BARS:].copy()
    labels_full = _triple_barrier_labels(
        close=feat_full["Close"].squeeze(),
        high=feat_full["High"].squeeze(),
        low=feat_full["Low"].squeeze(),
        atr=(feat_full["atr_norm"] * feat_full["Close"]).squeeze(),
        horizon=int(label_horizon),
        tp_mult=float(tp_mult),
        sl_mult=float(sl_mult),
    )

    mask = labels_full.notna() & feat_full[ML_COLS].notna().all(axis=1)
    feat_df = feat_full[mask].copy()
    labels = labels_full[mask].astype(int)
    bull_score = feat_df["bull_score"].astype(float)

    if len(feat_df) < is_days + oos_days:
        raise ValueError(f"Too few labeled bars for {symbol}: {len(feat_df)}")

    cached = None if force_reoptimize else get_sniper_params(symbol)
    reuse = _cache_is_fresh(cached) and cached is not None

    slides_out: list = []
    if reuse:
        log.info("[sniper_dt] cache hit for %s", symbol)
        wfo = None
    else:
        wfo = _run_wfo(feat_df, labels, bull_score, is_days=is_days, oos_days=oos_days)

    avg_oos_acc = 0.0
    if wfo is not None and len(wfo["oos_preds_a"]):
        pa, pb = wfo["oos_preds_a"], wfo["oos_preds_b"]
        yv = wfo["oos_labels"]
        rv = wfo["oos_returns"]
        gv = wfo["oos_grade"]
        best_sh, best_ta, best_tb = -1e9, 0.55, 0.55
        best_hr = 0.0
        for ta in THRESHOLD_GRID:
            for tb in THRESHOLD_GRID:
                sh, hr, _ = _oos_dual(pa, pb, ta, tb, yv, rv, gv)
                if sh > best_sh:
                    best_sh, best_ta, best_tb, best_hr = sh, ta, tb, hr
        threshold_a, threshold_b = best_ta, best_tb
        oos_sharpe = float(best_sh)
        oos_hit_rate = float(best_hr)
        oos_trade_count = int(
            ((pa > threshold_a) & (pb > threshold_b) & gv).sum()
        )
        agreement = (pa > threshold_a) & (pb > threshold_b)
        agreement_rate = float(agreement.mean()) if len(agreement) else 0.0
        avg_oos_acc = float(np.mean([s["oos_acc"] for s in wfo["slides"]])) if wfo["slides"] else 0.0
        n_slides = len(wfo["slides"])
        slides_out = wfo["slides"]
    else:
        threshold_a = float(cached["threshold_a"]) if cached and cached.get("threshold_a") is not None else 0.55
        threshold_b = float(cached["threshold_b"]) if cached and cached.get("threshold_b") is not None else 0.55
        oos_sharpe = float(cached["dt_oos_sharpe"]) if cached and cached.get("dt_oos_sharpe") is not None else 0.0
        oos_hit_rate = float(cached["dt_oos_hit_rate"]) if cached and cached.get("dt_oos_hit_rate") is not None else 0.0
        oos_trade_count = 0
        agreement_rate = 0.55
        avg_oos_acc = 0.0
        n_slides = 0
        slides_out = []

    confidence = _confidence_tier(oos_sharpe, agreement_rate)

    X_all = feat_df[ML_COLS].to_numpy(dtype=float)
    y_all = labels.to_numpy(dtype=float)
    tail = min(is_days, len(X_all))
    X_tail, y_tail = X_all[-tail:], y_all[-tail:]

    scaler_final_a = StandardScaler()
    X_tail_scaled = scaler_final_a.fit_transform(X_tail)
    model_final_a = _make_model(_spw(y_tail))
    model_final_a.fit(X_tail_scaled, y_tail)

    if wfo is not None:
        model_final_b = wfo["model_b"]
        scaler_final_b = wfo["scaler_b"]
    else:
        b_end = min(MODEL_B_TRAIN_BARS, max(len(X_all) - is_days, is_days))
        scaler_final_b = StandardScaler()
        X_b_scaled = scaler_final_b.fit_transform(X_all[:b_end])
        model_final_b = _make_model(_spw(y_all[:b_end]))
        model_final_b.fit(X_b_scaled, y_all[:b_end])

    today_row = feat_df.iloc[[-1]]
    X_today = today_row[ML_COLS].to_numpy(dtype=float)
    prob_a = float(model_final_a.predict_proba(scaler_final_a.transform(X_today))[0, 1])
    prob_b = float(model_final_b.predict_proba(scaler_final_b.transform(X_today))[0, 1])
    bull_today = float(today_row["bull_score"].iloc[0])
    grade_ok = bull_today >= 5.0
    last_signal = int(
        prob_a > threshold_a and prob_b > threshold_b and grade_ok
    )

    return {
        "symbol": symbol,
        "curr_date": curr_date,
        "last_signal": last_signal,
        "last_prob_a": round(prob_a, 4),
        "last_prob_b": round(prob_b, 4),
        "threshold_a": round(threshold_a, 4),
        "threshold_b": round(threshold_b, 4),
        "oos_sharpe": round(oos_sharpe, 3),
        "oos_hit_rate": round(oos_hit_rate, 3),
        "oos_trade_count": int(oos_trade_count),
        "confidence": confidence,
        "avg_oos_acc": round(avg_oos_acc, 4),
        "n_slides": int(n_slides),
        "feature_names": ML_COLS,
        "bull_score_today": round(bull_today, 4),
        "grade_veto_ok": bool(grade_ok),
        "is_days": is_days,
        "oos_days": oos_days,
        "label_horizon": int(label_horizon),
        "slides": slides_out,
        "cache_hit": bool(reuse),
        "tp_mult": float(tp_mult),
        "sl_mult": float(sl_mult),
    }
