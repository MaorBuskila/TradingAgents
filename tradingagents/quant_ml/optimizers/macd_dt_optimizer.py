"""
macd_dt_optimizer.py
====================
Hybrid XGBoost + FinBERT Sentiment + MACD Momentum Optimizer.
Fixed: MacOS stability (Segfault 11) and Additive Signal Fusion.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Annotated

# --- MACOS STABILITY FIX ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import yfinance as yf
import torch
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from tradingagents.quant_ml.indicators.stockstats_utils import yf_retry, _clean_dataframe
from tradingagents.dataflows.config import get_config

warnings.filterwarnings("ignore", category=UserWarning)
log = logging.getLogger("macd_dt_optimizer")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Global Model Cache (FinBERT)
# ─────────────────────────────────────────────────────────────────────────────
MODELS = {"tokenizer": None, "model": None}

def _load_finbert():
    if MODELS["model"] is None:
        name = "ProsusAI/finbert"
        MODELS["tokenizer"] = AutoTokenizer.from_pretrained(name)
        MODELS["model"] = AutoModelForSequenceClassification.from_pretrained(name)
        MODELS["model"].to("cpu") # Stability on Mac
    return MODELS["tokenizer"], MODELS["model"]

# ─────────────────────────────────────────────────────────────────────────────
# 2. Strategy Constants
# ─────────────────────────────────────────────────────────────────────────────
IS_DAYS          = 150   
LABEL_HORIZON    = 10    
MIN_PROB         = 0.55  
SENTIMENT_EXIT   = -0.70 
SCORE_THRESHOLD  = 2     
REGIME_WINDOW    = 20    

MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9

# ─────────────────────────────────────────────────────────────────────────────
# 3. Sentiment & Feature Engineering
# ─────────────────────────────────────────────────────────────────────────────

def _get_finbert_sentiment(symbol: str, dates: pd.DatetimeIndex) -> pd.Series:
    tokenizer, model = _load_finbert()
    ticker = yf.Ticker(symbol)
    news = ticker.news
    s_series = pd.Series(0.0, index=dates)
    
    if not news: return s_series

    headlines = [item.get('title', '') for item in news][:15]
    if not headlines: return s_series

    inputs = tokenizer(headlines, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        # Score = Positive - Negative
        scores = (probs[:, 0] - probs[:, 1]).numpy().ravel()
    
    avg_score = float(np.mean(scores))
    s_series.iloc[-1] = avg_score
    return s_series.ffill().fillna(0.0)

def _build_macd_features(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    c = df["Close"].squeeze()
    h, lo = df["High"].squeeze(), df["Low"].squeeze()

    # MACD Calculation
    ema_fast = c.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = c.ewm(span=MACD_SLOW, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=MACD_SIGNAL, adjust=False).mean()
    hist = macd_line - signal_line

    # Indicators
    macd_slope = hist.diff(3)
    tr = pd.concat([h-lo, (h-c.shift()).abs(), (lo-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=14).mean()
    regime = c.pct_change().rolling(REGIME_WINDOW).mean()
    sentiment = _get_finbert_sentiment(symbol, df.index)

    return pd.DataFrame({
        "macd_hist": hist,
        "macd_slope": macd_slope,
        "signal_gap": macd_line - signal_line,
        "atr": atr,
        "regime": regime,
        "sentiment": sentiment,
        "price": c,
        "ema200": c.ewm(span=200).mean(),
        "macd_line": macd_line,
        "macd_signal": signal_line
    }).ffill().bfill().fillna(0)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Main Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def run_dt_optimizer(symbol: str, curr_date: str, **kwargs) -> dict:
    log.info(f"[macd_dt] Running Hybrid Optimizer for {symbol}")

    # A. Data Loading
    df = yf_retry(lambda: yf.download(symbol, period="2y", interval="1d", auto_adjust=True, progress=False))
    df = df[df.index <= pd.to_datetime(curr_date)].copy()
    
    if len(df) < IS_DAYS + LABEL_HORIZON:
        raise ValueError(f"History too short for {symbol}")

    feat_df = _build_macd_features(df, symbol)
    labels = (df["Close"].shift(-LABEL_HORIZON) > df["Close"]).astype(int).squeeze()
    
    # B. XGBoost Training (MacOS Stable)
    ml_cols = ["macd_hist", "macd_slope", "signal_gap", "atr", "regime", "sentiment"]
    train_idx = len(df) - IS_DAYS - LABEL_HORIZON
    
    X_train = feat_df[ml_cols].iloc[train_idx : -LABEL_HORIZON].values
    y_train = labels.iloc[train_idx : -LABEL_HORIZON].values.ravel()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = XGBClassifier(
        n_estimators=100, 
        max_depth=4, 
        learning_rate=0.05, 
        n_jobs=1, # Prevents Segfault 11
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # C. Prediction & Scoring Logic
    X_curr = scaler.transform(feat_df[ml_cols].iloc[[-1]].values)
    prob_xgb = float(model.predict_proba(X_curr)[0, 1])
    
    last = feat_df.iloc[-1]
    
    # Additive Score Fusion
    score = 0
    if prob_xgb >= MIN_PROB: score += 1
    if last['macd_line'] > last['macd_signal']: score += 1 # Trend Confirmation
    if last['macd_slope'] > 0: score += 1                # Momentum Confirmation
    
    # D. Signal Logic
    is_bullish = last['regime'] > 0
    final_signal = 1 if (score >= SCORE_THRESHOLD and is_bullish and last['price'] > last['ema200']) else 0
    
    # Sentiment Override
    if last['sentiment'] < SENTIMENT_EXIT:
        final_signal = 0
        log.warning(f"MACD VETO: Negative news for {symbol}")

    return {
        "symbol": symbol,
        "curr_date": curr_date,
        "last_signal": int(final_signal),
        "last_prob_a": round(prob_xgb, 4),
        "hybrid_score": int(score),
        "last_sentiment": round(float(last['sentiment']), 3),
        "last_regime": "BULL" if is_bullish else "BEAR",
        "last_rsi": 0.0, # Field compatibility
        "n_slides": 1,
        "feature_names": ml_cols
    }