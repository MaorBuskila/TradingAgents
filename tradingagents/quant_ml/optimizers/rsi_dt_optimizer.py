"""
rsi_dt_optimizer.py
===================
DT-filtered RSI optimizer — Walk-Forward + Triple Barrier + two-model ensemble.

Pipeline:
  1. Fetch ~2000 trading days (8y) of daily OHLC via yfinance.
  2. Build features: rsi, ema_ratio, macd_hist, atr_norm (sentiment is NOT a
     training feature — it is applied as a post-model veto on today's signal
     to prevent FinBERT leakage into historical rows).
  3. Triple Barrier labels: 2.0*ATR upper / 1.0*ATR lower, 10-bar horizon.
  4. WFO Model A (sliding): train=1000, step=20, fresh StandardScaler per fold,
     scale_pos_weight auto-computed per fold to handle label imbalance.
  5. Model B (fixed, crash-aware): trained on the oldest 1500 bars (which
     span known stress regimes), scored on the same OOS test slices.
  6. Dynamic thresholds: rolling 85th percentile over the last ~250 OOS
     predictions per model.
  7. Final retrain on most recent data, predict today.
  8. LONG only when prob_a > threshold_a AND prob_b > threshold_b AND
     sentiment_today > 0 AND price_today > ema200_today.

MacOS stability: KMP_DUPLICATE_LIB_OK + OMP_NUM_THREADS=1 + n_jobs=1.
"""

from __future__ import annotations

import logging
import os
import warnings
from datetime import datetime, timedelta

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import yfinance as yf
import torch
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from tradingagents.quant_ml.indicators.stockstats_utils import yf_retry
from tradingagents.dataflows.rsi_dt_cache import (
    get_rsi_dt_params,
    upsert_rsi_dt_params,
)

warnings.filterwarnings("ignore", category=UserWarning)
log = logging.getLogger("rsi_dt_optimizer")

MODELS = {"tokenizer": None, "model": None}

FETCH_YEARS = 8
WARMUP_BARS = 210
MIN_USABLE_BARS = 1400
CACHE_TTL_DAYS = 7
ROLLING_THRESHOLD_WINDOW = 250
THRESHOLD_PERCENTILE = 0.85
MODEL_B_TRAIN_BARS = 1500
HIGH_VOL_PCT = 80.0


# ─────────────────────────────────────────────────────────────────────────────
# FinBERT (post-model sentiment veto only — NOT a training feature)
# ─────────────────────────────────────────────────────────────────────────────

def _load_finbert():
    if MODELS["model"] is None:
        name = "ProsusAI/finbert"
        MODELS["tokenizer"] = AutoTokenizer.from_pretrained(name)
        MODELS["model"] = AutoModelForSequenceClassification.from_pretrained(name)
        MODELS["model"].to("cpu")
    return MODELS["tokenizer"], MODELS["model"]


def _sentiment_today(symbol: str) -> float | None:
    """FinBERT sentiment score for the most recent headlines. Positive - Negative.
    Returns None on network/API failure so callers can skip the veto rather than block."""
    try:
        tokenizer, model = _load_finbert()
        news = yf.Ticker(symbol).news or []
        headlines = [item.get("title", "") for item in news][:10]
        headlines = [h for h in headlines if h]
        if not headlines:
            return None
        inputs = tokenizer(headlines, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            scores = (probs[:, 0] - probs[:, 1]).numpy().ravel()
        return float(np.mean(scores))
    except Exception as exc:
        log.warning("[rsi_dt] sentiment fetch failed for %s: %s", symbol, exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────

def _build_features(df: pd.DataFrame, rsi_period: int) -> pd.DataFrame:
    c = df["Close"].squeeze()
    h = df["High"].squeeze()
    lo = df["Low"].squeeze()

    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / rsi_period, adjust=False).mean()
    loss = (-delta).clip(lower=0).ewm(alpha=1 / rsi_period, adjust=False).mean()
    rsi = 100 - (100 / (1 + (gain / loss.replace(0, np.nan))))

    ema50 = c.ewm(span=50, adjust=False).mean()
    ema200 = c.ewm(span=200, adjust=False).mean()
    ema_ratio = ema50 / ema200.replace(0, np.nan)

    ema_fast = c.ewm(span=12, adjust=False).mean()
    ema_slow = c.ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    prev_close = c.shift()
    tr = pd.concat([h - lo, (h - prev_close).abs(), (lo - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()
    atr_norm = atr / c.replace(0, np.nan)

    return pd.DataFrame({
        "rsi_14": rsi,
        "ema_ratio_50_200": ema_ratio,
        "macd_hist": macd_hist,
        "atr_norm_14": atr_norm,
        "price": c,
        "ema200": ema200,
        "atr_14": atr,
        "high": h,
        "low": lo,
    })


# ─────────────────────────────────────────────────────────────────────────────
# Triple Barrier labels
# ─────────────────────────────────────────────────────────────────────────────

def _triple_barrier_labels(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    atr: pd.Series,
    horizon: int,
    tp_mult: float,
    sl_mult: float,
) -> pd.Series:
    c = close.to_numpy(dtype=float)
    h = high.to_numpy(dtype=float)
    lo = low.to_numpy(dtype=float)
    a = atr.to_numpy(dtype=float)
    n = len(c)
    labels = np.full(n, np.nan)

    for t in range(n - horizon):
        if np.isnan(a[t]) or np.isnan(c[t]):
            continue
        tp = c[t] + tp_mult * a[t]
        sl = c[t] - sl_mult * a[t]
        label = 0
        for k in range(1, horizon + 1):
            if h[t + k] >= tp:
                label = 1
                break
            if lo[t + k] <= sl:
                label = 0
                break
        labels[t] = label

    return pd.Series(labels, index=close.index, name="label")


# ─────────────────────────────────────────────────────────────────────────────
# Model helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_model(scale_pos_weight: float) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.08,
        n_jobs=1,
        tree_method="hist",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbosity=0,
    )


def _spw(y: np.ndarray) -> float:
    pos = float(y.sum())
    neg = float(len(y) - pos)
    return max(neg / max(pos, 1.0), 1.0)


def _rolling_quantile(preds: list[float], window: int, q: float) -> float:
    if not preds:
        return 0.55
    tail = preds[-window:] if len(preds) > window else preds
    return float(np.quantile(tail, q))


def _confidence_tier(oos_sharpe: float, agreement_rate: float) -> str:
    if oos_sharpe > 1.0 and agreement_rate >= 0.60:
        return "HIGH"
    if oos_sharpe > 0.5:
        return "MEDIUM"
    return "LOW"


def _oos_backtest_stats(
    preds: np.ndarray,
    threshold: float,
    labels: np.ndarray,
    close_returns: np.ndarray,
) -> tuple[float, float, int]:
    """Sharpe + hit rate + trade count on OOS predictions gated by threshold."""
    if len(preds) == 0:
        return 0.0, 0.0, 0
    take = preds > threshold
    if not take.any():
        return 0.0, 0.0, 0
    trade_returns = close_returns[take]
    wins = labels[take] == 1
    hit_rate = float(wins.mean()) if len(wins) else 0.0
    if trade_returns.std(ddof=0) == 0:
        sharpe = 0.0
    else:
        sharpe = float(np.sqrt(252) * trade_returns.mean() / trade_returns.std(ddof=0))
    return sharpe, hit_rate, int(take.sum())


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


# ─────────────────────────────────────────────────────────────────────────────
# WFO core
# ─────────────────────────────────────────────────────────────────────────────

ML_COLS = ["rsi_14", "ema_ratio_50_200", "macd_hist", "atr_norm_14"]


def _run_wfo(
    feat_df: pd.DataFrame,
    labels: pd.Series,
    is_days: int,
    oos_days: int,
) -> dict:
    """Walk-forward for Model A (sliding) + Model B (fixed crash window)."""
    X = feat_df[ML_COLS].to_numpy(dtype=float)
    y = labels.to_numpy(dtype=float)
    n = len(X)

    # Model B trains once on oldest MODEL_B_TRAIN_BARS bars.
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
    slides: list[dict] = []

    close = feat_df["price"].to_numpy(dtype=float)
    # forward-return proxy for OOS trade stats: next-bar return at entry
    returns = np.concatenate([[0.0], np.diff(close) / close[:-1]])

    slide_idx = 0
    # Model A starts its first training window at index 0; first OOS fold begins at is_days.
    # We continue stepping by oos_days until we run out of test bars.
    start = 0
    while start + is_days + oos_days <= n:
        is_start = start
        is_end = start + is_days
        oos_start = is_end
        oos_end = min(is_end + oos_days, n)

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
        "slides": slides,
        "model_b": model_b,
        "scaler_b": scaler_b,
        "b_end": b_end,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_rsi_dt_optimizer(
    symbol: str,
    curr_date: str,
    *,
    is_days: int = 1000,
    oos_days: int = 20,
    label_horizon: int = 10,
    tp_mult: float = 2.0,
    sl_mult: float = 1.0,
    rsi_period: int = 14,
    rsi_upper: float = 70.0,
    rsi_lower: float = 30.0,
    force_reoptimize: bool = False,
    **_kwargs,
) -> dict:
    log.info("[rsi_dt] WFO+TripleBarrier optimizer for %s @ %s", symbol, curr_date)

    is_days = max(int(is_days), 500)
    oos_days = max(int(oos_days), 5)

    # 1. Load ~2000 trading days
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

    # 2. Features (no sentiment in history)
    feat_full = _build_features(df, rsi_period=int(rsi_period))
    feat_full = feat_full.iloc[WARMUP_BARS:].copy()

    # 3. Triple Barrier labels
    labels_full = _triple_barrier_labels(
        close=feat_full["price"],
        high=feat_full["high"],
        low=feat_full["low"],
        atr=feat_full["atr_14"],
        horizon=int(label_horizon),
        tp_mult=float(tp_mult),
        sl_mult=float(sl_mult),
    )

    # Keep rows with valid labels AND no NaN in features
    mask = labels_full.notna() & feat_full[ML_COLS].notna().all(axis=1)
    feat_df = feat_full[mask].copy()
    labels = labels_full[mask].astype(int)

    if len(feat_df) < is_days + oos_days:
        raise ValueError(
            f"Too few labeled bars for {symbol}: {len(feat_df)} "
            f"(need at least {is_days + oos_days})"
        )

    # 4. WFO: cache-aware
    cached = None if force_reoptimize else get_rsi_dt_params(symbol)
    reuse_cache = _cache_is_fresh(cached)

    if reuse_cache:
        log.info("[rsi_dt] reusing cached thresholds for %s", symbol)
        wfo = None
    else:
        wfo = _run_wfo(feat_df, labels, is_days=is_days, oos_days=oos_days)

    # 5. Thresholds (rolling 85-pctile on last ~250 OOS preds per model)
    if reuse_cache:
        threshold_a = float(cached["threshold_a"])
        threshold_b = float(cached["threshold_b"])
    else:
        threshold_a = _rolling_quantile(
            wfo["oos_preds_a"].tolist(), ROLLING_THRESHOLD_WINDOW, THRESHOLD_PERCENTILE,
        )
        threshold_b = _rolling_quantile(
            wfo["oos_preds_b"].tolist(), ROLLING_THRESHOLD_WINDOW, THRESHOLD_PERCENTILE,
        )

    # 6. Final retrain on most recent data, predict today
    X_all = feat_df[ML_COLS].to_numpy(dtype=float)
    y_all = labels.to_numpy(dtype=float)
    tail = min(is_days, len(X_all))
    X_tail, y_tail = X_all[-tail:], y_all[-tail:]

    scaler_final_a = StandardScaler()
    X_tail_scaled = scaler_final_a.fit_transform(X_tail)
    model_final_a = _make_model(_spw(y_tail))
    model_final_a.fit(X_tail_scaled, y_tail)

    # Model B: refit on oldest MODEL_B_TRAIN_BARS if we ran WFO;
    # otherwise rebuild from the full feature frame (cache path).
    if wfo is not None:
        model_final_b = wfo["model_b"]
        scaler_final_b = wfo["scaler_b"]
    else:
        b_end = min(MODEL_B_TRAIN_BARS, max(len(X_all) - is_days, is_days))
        scaler_final_b = StandardScaler()
        X_b_scaled = scaler_final_b.fit_transform(X_all[:b_end])
        model_final_b = _make_model(_spw(y_all[:b_end]))
        model_final_b.fit(X_b_scaled, y_all[:b_end])

    # Today's feature row: last available row in the full (pre-label-mask) frame.
    today_row = feat_full.iloc[[-1]]
    X_today = today_row[ML_COLS].to_numpy(dtype=float)
    prob_a = float(model_final_a.predict_proba(scaler_final_a.transform(X_today))[0, 1])
    prob_b = float(model_final_b.predict_proba(scaler_final_b.transform(X_today))[0, 1])

    # 7. Sentiment only for TODAY (post-model veto)
    sentiment_today = _sentiment_today(symbol)

    # 8. Signal logic
    price_today = float(today_row["price"].iloc[0])
    ema200_today = float(today_row["ema200"].iloc[0])
    macro_up = price_today > ema200_today
    model_agree = prob_a > threshold_a and prob_b > threshold_b
    sentiment_ok = (sentiment_today is None) or (sentiment_today > 0.0)
    last_signal = int(model_agree and sentiment_ok and macro_up)

    # 9. OOS metrics + confidence
    if wfo is not None and len(wfo["oos_preds_a"]):
        oos_sharpe, hit_rate, trade_count = _oos_backtest_stats(
            wfo["oos_preds_a"], threshold_a,
            wfo["oos_labels"], wfo["oos_returns"],
        )
        agreement = (
            (wfo["oos_preds_a"] > threshold_a) &
            (wfo["oos_preds_b"] > threshold_b)
        )
        agreement_rate = float(agreement.mean()) if len(agreement) else 0.0
        avg_oos_acc = float(np.mean([s["oos_acc"] for s in wfo["slides"]])) if wfo["slides"] else 0.0
        n_slides = len(wfo["slides"])
        slides_out = wfo["slides"]
    else:
        oos_sharpe = float(cached["oos_sharpe"]) if cached else 0.0
        hit_rate = float(cached["oos_hit_rate"]) if cached else 0.0
        trade_count = int(cached["oos_trade_count"]) if cached else 0
        agreement_rate = 0.6 if cached and cached.get("confidence") == "HIGH" else 0.5
        avg_oos_acc = float(cached.get("avg_oos_acc", 0.0)) if cached else 0.0
        n_slides = int(cached["n_slides"]) if cached else 0
        slides_out = []

    confidence = _confidence_tier(oos_sharpe, agreement_rate)

    # 10. Size mult — halve position in high-vol regime
    atr_norm_today = float(today_row["atr_norm_14"].iloc[0])
    atr_norm_hist = feat_df["atr_norm_14"].to_numpy(dtype=float)
    atr_pct_rank = float((atr_norm_hist < atr_norm_today).mean() * 100) if len(atr_norm_hist) else 50.0
    size_mult = 0.5 if atr_pct_rank > HIGH_VOL_PCT else 1.0

    # 11. Cache (only on fresh WFO run)
    if wfo is not None:
        try:
            upsert_rsi_dt_params(
                ticker=symbol,
                threshold_a=threshold_a,
                threshold_b=threshold_b,
                oos_sharpe=oos_sharpe,
                oos_hit_rate=hit_rate,
                oos_trade_count=trade_count,
                confidence=confidence,
                training_days=is_days,
                step_days=oos_days,
                n_slides=n_slides,
                tp_mult=tp_mult,
                sl_mult=sl_mult,
                label_horizon=int(label_horizon),
                avg_oos_acc=avg_oos_acc,
            )
        except Exception as exc:
            log.warning("[rsi_dt] cache upsert failed for %s: %s", symbol, exc)

    rsi_today = float(today_row["rsi_14"].iloc[0])

    return {
        "symbol": symbol,
        "curr_date": curr_date,
        "last_signal": last_signal,
        "last_prob_a": round(prob_a, 4),
        "last_prob_b": round(prob_b, 4),
        "last_size_mult": round(size_mult, 2),
        "last_atr_pct": round(atr_pct_rank, 2),
        "last_rsi": round(rsi_today, 2),
        "last_sentiment": round(sentiment_today, 3) if sentiment_today is not None else None,
        "last_atr_norm": round(atr_norm_today, 5),
        "threshold_a": round(threshold_a, 4),
        "threshold_b": round(threshold_b, 4),
        "oos_sharpe": round(oos_sharpe, 3),
        "oos_hit_rate": round(hit_rate, 3),
        "oos_trade_count": int(trade_count),
        "confidence": confidence,
        "avg_oos_acc": round(avg_oos_acc, 4),
        "n_slides": int(n_slides),
        "feature_names": ML_COLS,
        "rsi_params": {"period": int(rsi_period), "upper": float(rsi_upper), "lower": float(rsi_lower)},
        "rsi_params_nondefault": bool(rsi_period != 14 or rsi_upper != 70.0 or rsi_lower != 30.0),
        "is_days": is_days,
        "oos_days": oos_days,
        "label_horizon": int(label_horizon),
        "min_prob_threshold": round(max(threshold_a, threshold_b), 4),
        "slides": slides_out,
        "cache_hit": bool(reuse_cache),
    }
