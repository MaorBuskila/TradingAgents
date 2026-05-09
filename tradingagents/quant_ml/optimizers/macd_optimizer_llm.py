"""
macd_optimizer_llm.py
=====================
LLM-Enhanced MACD Optimizer — mirrors rsi_optimizer_llm.py

How it differs from the pure algo optimizer:
  1. Pre-compute 5 structured regime signals using Python math
  2. Pass those clean signals to the LLM (reasoning, not raw numbers)
  3. LLM classifies the market regime and NARROWS the 3D search grid
  4. Run the same WFO algo, but on a smaller, smarter grid
  5. LLM writes a plain-English explanation of its reasoning

The 5 regime signals we compute (MACD-specific vs RSI):
  1. Volatility Percentile   — where is today's ATR vs its 1-year history?
  2. Trend Strength          — distance from 50-SMA as %
  3. MACD Histogram (12/26/9)— current default MACD histogram value
  4. Histogram Direction     — expanding_bullish / expanding_bearish / contracting
  5. Recent 10-day return    — is the stock going up or down right now?

The LLM then outputs:
  - regime: trending_bull | trending_bear | mean_reverting | high_volatility | choppy
  - fast_range:   [min, max]
  - slow_range:   [min, max]
  - signal_range: [min, max]
  - reasoning: plain English
"""

import json
import numpy as np
import pandas as pd
from typing import Annotated

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.llm_invoke import invoke_chat_model_human_message

from tradingagents.quant_ml.optimizers.macd_optimizer_algo import (
    _load_price_data,
    _calc_macd,
    _generate_positions,
    _calc_sharpe,
)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Pre-compute MACD-specific regime signals
# ─────────────────────────────────────────────────────────────────────────────

def _compute_regime_signals(data: pd.DataFrame) -> dict:
    """
    Compute 5 structured signals that describe current market conditions.
    MACD-specific: includes default MACD histogram state alongside the
    standard ATR/SMA signals used in the RSI lab.

    Returns:
      volatility_percentile:   0-100, ATR-14 vs 1yr history
      trend_direction:         "up" / "down" / "flat"
      trend_strength_pct:      % distance from 50-day SMA
      price_above_sma50:       bool
      price_above_sma200:      bool
      macd_histogram_current:  default MACD(12,26,9) histogram value
      histogram_direction:     "expanding_bullish" | "expanding_bearish" | "contracting"
      macd_signal_spread_pct:  (macd_line - signal_line) / price × 100
      recent_10d_return_pct:   10-day price return %
    """
    closes = data["Close"]
    highs  = data["High"]
    lows   = data["Low"]
    n      = len(closes)

    # ── Signal 1: Volatility Percentile via ATR-14 ─────────────────────────
    high_low   = highs - lows
    high_close = (highs - closes.shift(1)).abs()
    low_close  = (lows  - closes.shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr14      = true_range.rolling(14).mean()

    lookback_vol = min(252, n - 14)
    if lookback_vol > 0:
        atr_window    = atr14.iloc[-lookback_vol:]
        current_atr   = atr14.iloc[-1]
        vol_percentile = float((atr_window < current_atr).mean() * 100)
    else:
        vol_percentile = 50.0

    # ── Signal 2: Trend Strength via SMA distance ─────────────────────────
    sma50  = closes.rolling(50).mean()
    sma200 = closes.rolling(200).mean()

    current_price  = float(closes.iloc[-1])
    current_sma50  = float(sma50.iloc[-1])  if not pd.isna(sma50.iloc[-1])  else current_price
    current_sma200 = float(sma200.iloc[-1]) if not pd.isna(sma200.iloc[-1]) else current_price

    trend_strength_pct = round((current_price - current_sma50) / current_sma50 * 100, 2)
    if trend_strength_pct > 3.0:
        trend_direction = "up"
    elif trend_strength_pct < -3.0:
        trend_direction = "down"
    else:
        trend_direction = "flat"

    price_above_sma50  = current_price > current_sma50
    price_above_sma200 = current_price > current_sma200

    # ── Signal 3: Default MACD(12, 26, 9) histogram state ────────────────
    macd_line, signal_line, histogram = _calc_macd(closes, 12, 26, 9)

    hist_current = float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else 0.0
    hist_prev    = float(histogram.iloc[-2]) if (n >= 2 and not pd.isna(histogram.iloc[-2])) else hist_current

    if hist_current > 0:
        if hist_current > abs(hist_prev):
            hist_dir = "expanding_bullish"
        else:
            hist_dir = "contracting"
    elif hist_current < 0:
        if abs(hist_current) > abs(hist_prev):
            hist_dir = "expanding_bearish"
        else:
            hist_dir = "contracting"
    else:
        hist_dir = "contracting"

    # MACD/signal spread as % of price (normalised so it's comparable across tickers)
    macd_val   = float(macd_line.iloc[-1])   if not pd.isna(macd_line.iloc[-1])   else 0.0
    signal_val = float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0.0
    spread_pct = round((macd_val - signal_val) / current_price * 100, 4) if current_price > 0 else 0.0

    # ── Signal 4: Recent 10-day return ────────────────────────────────────
    if n >= 11:
        recent_return = round((current_price / float(closes.iloc[-11]) - 1) * 100, 2)
    else:
        recent_return = 0.0

    return {
        "volatility_percentile":   round(vol_percentile, 1),
        "trend_direction":         trend_direction,
        "trend_strength_pct":      trend_strength_pct,
        "price_above_sma50":       price_above_sma50,
        "price_above_sma200":      price_above_sma200,
        "macd_histogram_current":  round(hist_current, 4),
        "histogram_direction":     hist_dir,
        "macd_signal_spread_pct":  spread_pct,
        "recent_10d_return_pct":   recent_return,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — LLM regime classification + grid narrowing
# ─────────────────────────────────────────────────────────────────────────────

def _ask_llm_for_grid(
    symbol: str,
    signals: dict,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    logs: list | None = None,
) -> dict:
    """
    Send pre-computed MACD regime signals to the LLM to narrow the search grid.
    Falls back to heuristic if LLM is unavailable.
    """
    if logs is None:
        logs = []

    if llm_provider is None:
        llm_provider = get_config().get("llm_provider", "openai")

    prompt = f"""You are an expert quantitative analyst. I am running a Walk-Forward
MACD parameter optimization for {symbol}. Narrow the search space based on current
market conditions before running the grid search.

Pre-computed regime signals for {symbol}:

  Volatility Percentile:       {signals['volatility_percentile']}%
    (100% = highest vol ever seen. 50% = average. 0% = lowest ever.)

  Trend Direction:             {signals['trend_direction']}
  Trend Strength (50-SMA):     {signals['trend_strength_pct']:+.2f}%
  Price above 50-day SMA:      {signals['price_above_sma50']}
  Price above 200-day SMA:     {signals['price_above_sma200']}

  Default MACD(12,26,9) Histogram: {signals['macd_histogram_current']}
  Histogram Direction:         {signals['histogram_direction']}
  MACD/Signal Spread (% price):{signals['macd_signal_spread_pct']}%

  Recent 10-day Return:        {signals['recent_10d_return_pct']:+.2f}%

MACD optimization logic:
- High volatility     → SHORTER fast/slow periods (faster EMA response to rapid swings)
- Strong trend        → LONGER periods (avoid premature crossover false signals)
- Mean-reverting/flat → STANDARD periods (12/26 range), tighter signal line
- Large histogram     → already in a strong move, longer periods may suit better
- Contracting histogram → potential crossover imminent, shorter signal period helps

Respond ONLY with valid JSON in this exact format:
{{
  "regime": "trending_bull|trending_bear|mean_reverting|high_volatility|choppy",
  "fast_range":   [min_fast, max_fast],
  "slow_range":   [min_slow, max_slow],
  "signal_range": [min_signal, max_signal],
  "reasoning": "2-3 sentences explaining your choice"
}}

Rules:
- fast_range:   integers only, min >= 6, max <= 16
- slow_range:   integers only, min >= 18, max <= 34
- signal_range: integers only, min >= 5, max <= 13
- fast must always be < slow (enforce this in your response)
- A range covering ~30-40% of the full grid is ideal"""

    import os

    if llm_provider == "heuristic":
        logs.append("⚡ Heuristic mode — skipping LLM call")
        fallback = _heuristic_grid(signals)
        fallback["token_usage"] = {"model": "heuristic", "prompt_tokens": 0,
                                   "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0}
        fallback["logs"] = logs
        return fallback

    logs.append(f"🔍 Selected provider: {llm_provider}")
    key_map = {
        "google":     ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
        "openai":     ["OPENAI_API_KEY"],
        "anthropic":  ["ANTHROPIC_API_KEY"],
        "xai":        ["XAI_API_KEY"],
        "openrouter": ["OPENROUTER_API_KEY"],
    }
    required_keys = key_map.get(llm_provider, [])
    if required_keys:
        found_key = any(os.getenv(k) for k in required_keys)
        if found_key:
            found_name = next(k for k in required_keys if os.getenv(k))
            logs.append(f"✅ API key found: {found_name}")
        else:
            logs.append(f"❌ No API key found! Checked: {', '.join(required_keys)}")

    try:
        logs.append(f"📡 Sending MACD regime signals to {llm_provider} LLM...")
        logs.append(
            f"   vol={signals['volatility_percentile']}%, "
            f"trend={signals['trend_direction']}, "
            f"hist={signals['macd_histogram_current']:.4f} ({signals['histogram_direction']})"
        )

        llm_response, token_usage = invoke_chat_model_human_message(
            prompt, provider=llm_provider, model=llm_model
        )

        logs.append(
            f"✅ LLM responded ({token_usage['total_tokens']} tokens, "
            f"${token_usage['cost_usd']:.6f})"
        )
        logs.append(f"📄 Raw LLM response: {llm_response[:300]}{'...' if len(llm_response) > 300 else ''}")

        result = json.loads(llm_response)

        # Validate and clamp ranges
        result["fast_range"] = [
            max(6,  int(result["fast_range"][0])),
            min(16, int(result["fast_range"][1])),
        ]
        result["slow_range"] = [
            max(18, int(result["slow_range"][0])),
            min(34, int(result["slow_range"][1])),
        ]
        result["signal_range"] = [
            max(5,  int(result["signal_range"][0])),
            min(13, int(result["signal_range"][1])),
        ]

        logs.append(
            f"🎯 Grid narrowed: fast {result['fast_range']}, "
            f"slow {result['slow_range']}, signal {result['signal_range']}"
        )
        logs.append(f"🌐 Regime: {result.get('regime', 'unknown')}")

        result["token_usage"] = token_usage
        result["logs"] = logs
        return result

    except Exception as e:
        logs.append(f"❌ LLM call failed: {type(e).__name__}: {e}")
        logs.append("⚡ Falling back to heuristic grid narrowing")
        fallback = _heuristic_grid(signals)
        fallback["token_usage"] = {
            "model": "heuristic", "prompt_tokens": 0,
            "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0,
        }
        fallback["logs"] = logs
        return fallback


def _heuristic_grid(signals: dict) -> dict:
    """
    Fallback grid narrowing — same logic the LLM would use, as explicit rules.
    """
    vol   = signals["volatility_percentile"]
    trend = signals["trend_direction"]
    hist  = signals["macd_histogram_current"]
    hist_dir = signals["histogram_direction"]

    if vol > 70:
        regime       = "high_volatility"
        fast_range   = [6, 10]
        slow_range   = [18, 24]
        signal_range = [5, 9]
        reasoning = (
            f"Volatility is at the {vol:.0f}th percentile (high). "
            "Short MACD periods (fast 6-10, slow 18-24) respond faster to rapid price swings. "
            "Tight signal range (5-9) keeps crossovers sensitive."
        )
    elif trend in ("up", "down") and abs(signals["trend_strength_pct"]) > 5:
        regime       = f"trending_{'bull' if trend == 'up' else 'bear'}"
        fast_range   = [10, 16]
        slow_range   = [24, 34]
        signal_range = [7, 13]
        reasoning = (
            f"Strong {trend}trend detected ({abs(signals['trend_strength_pct']):.1f}% from 50-SMA). "
            "Longer MACD periods (fast 10-16, slow 24-34) avoid premature crossover signals. "
            "Wider signal range (7-13) allows trend confirmation."
        )
    else:
        regime       = "mean_reverting"
        fast_range   = [8, 14]
        slow_range   = [20, 30]
        signal_range = [6, 11]
        reasoning = (
            "Market appears to be ranging (flat trend, moderate volatility). "
            "Standard MACD periods (fast 8-14, slow 20-30) balance responsiveness. "
            "Signal range 6-11 covers the classic 9-period while allowing flexibility."
        )

    return {
        "regime":       regime,
        "fast_range":   fast_range,
        "slow_range":   slow_range,
        "signal_range": signal_range,
        "reasoning":    reasoning,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Run the optimizer with the LLM-narrowed grid
# ─────────────────────────────────────────────────────────────────────────────

def run_llm_optimizer(
    symbol: Annotated[str, "ticker symbol, e.g. AAPL"],
    curr_date: Annotated[str, "current date YYYY-MM-DD"],
    is_days: Annotated[int, "in-sample window in trading days"] = 180,
    oos_days: Annotated[int, "out-of-sample window in trading days"] = 90,
    llm_provider: Annotated[
        str | None,
        "LLM provider (openai, google, anthropic, xai, openrouter, ollama, heuristic). "
        "Defaults to config['llm_provider']."
    ] = None,
    llm_model: Annotated[
        str | None,
        "specific model to use. Defaults to config['quick_think_llm']."
    ] = None,
) -> dict:
    """
    LLM-Enhanced Walk-Forward MACD Optimization.

    Same math as the pure algo optimizer, but the 3D search grid is narrowed
    by LLM regime classification before running.

    Returns same structure as run_algo_optimizer() PLUS:
      regime:          market regime classified by LLM
      llm_reasoning:   plain English explanation from LLM
      llm_grid:        the narrowed grid LLM chose
      llm_available:   True if LLM was used, False if heuristic fallback
      regime_signals:  the 5 computed signals passed to LLM
      debug_logs:      step-by-step execution log
      token_usage:     LLM cost/token metadata
    """
    data = _load_price_data(symbol)

    curr_dt = pd.to_datetime(curr_date)
    data = data[data["Date"] <= curr_dt].copy()
    closes = data["Close"].reset_index(drop=True)
    total_days = len(closes)

    warmup = 50
    min_required = is_days + oos_days + warmup
    if total_days < min_required:
        raise ValueError(
            f"Not enough data: {total_days} rows, need at least {min_required}."
        )

    oos_start = total_days - oos_days
    is_start  = max(0, oos_start - is_days)

    logs: list[str] = []
    logs.append(f"📊 Loaded {total_days} trading days for {symbol} up to {curr_date}")
    logs.append(f"📅 IS window: {is_days} days | OOS window: {oos_days} days")

    # ── Compute regime signals ─────────────────────────────────────────────
    signals = _compute_regime_signals(data)
    logs.append("🧮 Regime signals computed:")
    logs.append(f"   Volatility: {signals['volatility_percentile']}th percentile")
    logs.append(f"   Trend: {signals['trend_direction']} ({signals['trend_strength_pct']:+.2f}% from 50-SMA)")
    logs.append(f"   MACD hist: {signals['macd_histogram_current']:.4f} ({signals['histogram_direction']})")
    logs.append(f"   MACD/signal spread: {signals['macd_signal_spread_pct']:.4f}%")
    logs.append(f"   10d return: {signals['recent_10d_return_pct']:+.2f}%")
    logs.append(f"   Above 50-SMA: {signals['price_above_sma50']} | Above 200-SMA: {signals['price_above_sma200']}")

    # ── Ask LLM for narrowed grid ──────────────────────────────────────────
    llm_available = (llm_provider != "heuristic")
    grid = _ask_llm_for_grid(symbol, signals, llm_provider=llm_provider, llm_model=llm_model, logs=logs)
    if grid.get("token_usage", {}).get("model") == "heuristic" and llm_provider not in (None, "heuristic"):
        llm_available = False

    f_min, f_max = grid["fast_range"]
    s_min, s_max = grid["slow_range"]
    g_min, g_max = grid["signal_range"]

    fast_periods   = list(range(f_min, f_max + 1))
    slow_periods   = list(range(s_min, s_max + 1, 2))   # step 2 to keep it fast
    signal_periods = list(range(g_min, g_max + 1))

    logs.append(
        f"🔬 Starting grid search: "
        f"{len(fast_periods)} fast × {len(slow_periods)} slow × {len(signal_periods)} signal"
    )

    # ── Grid search on narrowed space ─────────────────────────────────────
    best_is_sharpe = -np.inf
    best_params    = {"fast": 12, "slow": 26, "signal": 9}
    combos_tested  = 0
    param_sharpes: list[dict] = []

    for fast in fast_periods:
        for slow in slow_periods:
            if fast >= slow:
                continue

            for sig in signal_periods:
                combos_tested += 1

                _, _, hist_full = _calc_macd(closes, fast, slow, sig)
                positions_full  = _generate_positions(hist_full)

                is_closes    = closes.iloc[is_start:oos_start]
                is_positions = positions_full.iloc[is_start:oos_start]
                is_sharpe    = _calc_sharpe(is_closes, is_positions)

                oos_closes    = closes.iloc[oos_start:]
                oos_positions = positions_full.iloc[oos_start:]
                oos_sharpe    = _calc_sharpe(oos_closes, oos_positions)

                param_sharpes.append({
                    "fast": fast, "slow": slow, "signal": sig,
                    "is_sharpe": round(is_sharpe, 4),
                    "oos_sharpe": round(oos_sharpe, 4),
                })

                if is_sharpe > best_is_sharpe:
                    best_is_sharpe = is_sharpe
                    best_params = {"fast": fast, "slow": slow, "signal": sig}

    logs.append(f"✅ Grid search done — {combos_tested} combos tested")
    logs.append(
        f"🏆 Best IS params: fast={best_params['fast']}, slow={best_params['slow']}, "
        f"signal={best_params['signal']}, IS Sharpe={round(best_is_sharpe, 3)}"
    )

    # ── OOS validation ────────────────────────────────────────────────────
    _, _, best_hist = _calc_macd(closes, best_params["fast"], best_params["slow"], best_params["signal"])
    best_pos        = _generate_positions(best_hist)
    best_oos_sharpe = _calc_sharpe(closes.iloc[oos_start:], best_pos.iloc[oos_start:])

    logs.append(f"📊 OOS Sharpe: {round(best_oos_sharpe, 3)}")

    # ── Baseline MACD(12, 26, 9) ──────────────────────────────────────────
    _, _, default_hist = _calc_macd(closes, 12, 26, 9)
    default_pos = _generate_positions(default_hist)
    default_is_sharpe  = _calc_sharpe(
        closes.iloc[is_start:oos_start], default_pos.iloc[is_start:oos_start]
    )
    default_oos_sharpe = _calc_sharpe(
        closes.iloc[oos_start:], default_pos.iloc[oos_start:]
    )

    # ── Confidence ────────────────────────────────────────────────────────
    if best_is_sharpe > 0 and best_oos_sharpe > 0:
        ratio = best_oos_sharpe / best_is_sharpe
        if best_oos_sharpe >= 0.5 and ratio >= 0.4:
            confidence = "HIGH"
        elif best_oos_sharpe >= 0.2 and ratio >= 0.2:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"
    elif best_oos_sharpe > 0.3:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        # Best parameters found
        "optimal_fast":       best_params["fast"],
        "optimal_slow":       best_params["slow"],
        "optimal_signal":     best_params["signal"],
        # Performance
        "is_sharpe":          round(best_is_sharpe, 4),
        "oos_sharpe":         round(best_oos_sharpe, 4),
        "confidence":         confidence,
        # Meta
        "combos_tested":      combos_tested,
        "is_days":            is_days,
        "oos_days":           oos_days,
        # Baseline
        "default_is_sharpe":  round(default_is_sharpe, 4),
        "default_oos_sharpe": round(default_oos_sharpe, 4),
        # LLM-specific outputs
        "regime":             grid["regime"],
        "llm_reasoning":      grid["reasoning"],
        "llm_available":      llm_available,
        "llm_grid": {
            "fast_range":   grid["fast_range"],
            "slow_range":   grid["slow_range"],
            "signal_range": grid["signal_range"],
        },
        "regime_signals":     signals,
        "debug_logs":         logs,
        "token_usage":        grid.get("token_usage", {
            "model": "heuristic", "prompt_tokens": 0,
            "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0,
        }),
        # Top-20 for chart
        "param_sharpes": sorted(param_sharpes, key=lambda x: x["is_sharpe"], reverse=True)[:20],
    }
