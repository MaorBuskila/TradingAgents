"""
rsi_optimizer_llm.py
====================
LLM-Enhanced RSI Optimizer — Option B

How it differs from the pure algo optimizer:
  1. Pre-compute 4 structured regime signals using Python math (we're good at that)
  2. Pass those clean signals to the LLM (it's good at reasoning, bad at raw numbers)
  3. LLM classifies the market regime and NARROWS the search grid
  4. Run the same WFO algo, but on a smaller, smarter grid
  5. LLM writes a plain-English explanation of its reasoning

Why Option B (pre-compute signals, LLM classifies) vs Option A (give LLM raw prices):
  - LLMs are bad at reading tables of 180 raw price numbers accurately
  - LLMs are GOOD at reasoning: "ATR is at 80th percentile = high vol → use faster RSI"
  - Pre-computing gives the LLM clean, meaningful inputs → better decisions

The 4 regime signals we compute:
  1. Volatility Percentile — where is today's ATR vs its 1-year history?
       e.g. 80th percentile = much higher vol than usual
  2. Trend Strength — is price trending or ranging?
       Uses distance from 50-SMA as a percentage
  3. Current RSI-14 level — are we already overbought/oversold?
  4. Recent 10-day return — is the stock going up or down right now?

The LLM then outputs:
  - regime: one of [trending_bull, trending_bear, mean_reverting, high_volatility, choppy]
  - period_range: [min_period, max_period] to search
  - upper_range: [min, max] for overbought threshold
  - lower_range: [min, max] for oversold threshold
  - reasoning: plain English explanation

We then run the same grid search + WFO but only within the LLM's suggested ranges.
"""

import json
import numpy as np
import pandas as pd
from typing import Annotated

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.llm_invoke import invoke_chat_model_human_message

from tradingagents.quant_ml.optimizers.rsi_optimizer_algo import (
    _load_price_data,
    _calc_rsi,
    _generate_positions,
    _calc_sharpe,
)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Pre-compute regime signals (Python math, not LLM)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_regime_signals(data: pd.DataFrame) -> dict:
    """
    Compute 4 structured signals that describe current market conditions.
    These are passed to the LLM as clean, interpretable numbers — NOT raw prices.

    Returns a dict with:
      volatility_percentile: 0-100, where current ATR sits vs 1yr history
      trend_direction: "up", "down", or "flat"
      trend_strength_pct: how far price is from its 50-day SMA (%)
      rsi14_current: current RSI-14 value
      recent_10d_return_pct: price return over last 10 trading days (%)
      price_above_sma50: True/False
      price_above_sma200: True/False
    """
    closes = data["Close"]
    highs  = data["High"]
    lows   = data["Low"]

    n = len(closes)

    # ── Signal 1: Volatility Percentile using ATR ──────────────────────────
    # ATR (Average True Range) = average of the daily price swing over 14 days
    # We compare today's ATR to its value over the past year (252 days)
    # to understand if we're in a high or low volatility regime
    high_low   = highs - lows
    high_close = (highs - closes.shift(1)).abs()
    low_close  = (lows - closes.shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr14 = true_range.rolling(14).mean()

    lookback_vol = min(252, n - 14)
    if lookback_vol > 0:
        atr_window = atr14.iloc[-(lookback_vol):]
        current_atr = atr14.iloc[-1]
        vol_percentile = float((atr_window < current_atr).mean() * 100)
    else:
        vol_percentile = 50.0  # not enough data, assume median

    # ── Signal 2: Trend Strength via SMA distance ─────────────────────────
    # How far is the current price from its 50-day average?
    # Large positive % = strong uptrend. Large negative % = strong downtrend. ~0% = ranging.
    sma50 = closes.rolling(50).mean()
    sma200 = closes.rolling(200).mean()

    current_price = float(closes.iloc[-1])
    current_sma50 = float(sma50.iloc[-1]) if not pd.isna(sma50.iloc[-1]) else current_price
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

    # ── Signal 3: Current RSI-14 value ────────────────────────────────────
    rsi14 = _calc_rsi(closes, 14)
    rsi14_current = round(float(rsi14.iloc[-1]), 1)

    # ── Signal 4: Recent 10-day price return ──────────────────────────────
    if n >= 11:
        recent_return = (current_price / float(closes.iloc[-11]) - 1) * 100
        recent_return = round(recent_return, 2)
    else:
        recent_return = 0.0

    return {
        "volatility_percentile": round(vol_percentile, 1),
        "trend_direction": trend_direction,
        "trend_strength_pct": trend_strength_pct,
        "price_above_sma50": price_above_sma50,
        "price_above_sma200": price_above_sma200,
        "rsi14_current": rsi14_current,
        "recent_10d_return_pct": recent_return,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — LLM regime classification + grid narrowing
# ─────────────────────────────────────────────────────────────────────────────

def _ask_llm_for_grid(symbol: str, signals: dict, llm_provider: str | None = None, llm_model: str | None = None, logs: list | None = None) -> dict:
    """
    Send the pre-computed regime signals to the LLM.
    llm_provider: any provider supported by create_llm_client, or "heuristic" to skip LLM.
                  Defaults to config["llm_provider"].
    logs: list to append debug messages to (passed in from run_llm_optimizer)
    Returns dict with: regime, period_range, upper_range, lower_range, reasoning, token_usage
    """
    if logs is None:
        logs = []
    if llm_provider is None:
        llm_provider = get_config().get("llm_provider", "openai")
    prompt = f"""You are an expert quantitative analyst. I am about to run a Walk-Forward
RSI parameter optimization for {symbol}. Before running an expensive full grid search,
I want you to intelligently narrow the search space based on current market conditions.

Here are the pre-computed market regime signals for {symbol} right now:

  Volatility Percentile: {signals['volatility_percentile']}%
    (100% = highest volatility ever seen. 50% = average. 0% = lowest ever.)

  Trend Direction: {signals['trend_direction']}
  Trend Strength (% distance from 50-day SMA): {signals['trend_strength_pct']}%
  Price above 50-day SMA: {signals['price_above_sma50']}
  Price above 200-day SMA: {signals['price_above_sma200']}

  Current RSI-14: {signals['rsi14_current']}
  Recent 10-day Return: {signals['recent_10d_return_pct']}%

Based on these signals, please:
1. Classify the current market regime for this stock
2. Recommend a NARROWED RSI parameter search grid

IMPORTANT RSI optimization logic:
- High volatility → use SHORTER RSI periods (faster response, catches rapid reversals)
- Strong trend → use LONGER periods (avoid premature reversal signals) and WIDER thresholds
- Mean-reverting / ranging market → shorter periods with TIGHTER thresholds work better
- RSI already extreme (>70 or <30) → the current regime is already signaling a condition

Respond ONLY with valid JSON in this exact format:
{{
  "regime": "one of: trending_bull | trending_bear | mean_reverting | high_volatility | choppy",
  "period_range": [min_period, max_period],
  "upper_range": [min_upper, max_upper],
  "lower_range": [min_lower, max_lower],
  "reasoning": "2-3 sentences explaining your choice"
}}

Rules for the ranges:
- period_range: integers only, min >= 2, max <= 28
- upper_range: multiples of 5, between 60 and 85
- lower_range: multiples of 5, between 15 and 40
- upper must always be > lower
- The narrower the range, the faster the optimization but the higher the risk of missing the best params
  A range covering ~30-40% of the full grid is ideal."""

    import os

    # Short-circuit: user explicitly chose heuristic
    if llm_provider == "heuristic":
        logs.append("⚡ Heuristic mode selected — skipping LLM call")
        fallback = _heuristic_grid(signals)
        fallback["token_usage"] = {"model": "heuristic", "prompt_tokens": 0,
                                   "completion_tokens": 0, "total_tokens": 0, "cost_usd": 0.0}
        fallback["logs"] = logs
        return fallback

    # Check API key presence before calling (best-effort logging, not a hard gate)
    logs.append(f"🔍 Selected provider: {llm_provider}")
    key_map = {
        "google":    ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
        "openai":    ["OPENAI_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY"],
        "xai":       ["XAI_API_KEY"],
        "openrouter":["OPENROUTER_API_KEY"],
    }
    required_keys = key_map.get(llm_provider, [])
    if required_keys:
        found_key = any(os.getenv(k) for k in required_keys)
        if found_key:
            found_name = next(k for k in required_keys if os.getenv(k))
            logs.append(f"✅ API key found: {found_name} ({'*' * 8 + os.getenv(found_name, '')[-4:]})")
        else:
            logs.append(f"❌ No API key found! Checked: {', '.join(required_keys)}")

    try:
        logs.append(f"📡 Sending regime signals to {llm_provider} LLM...")
        logs.append(f"   Signals: vol={signals['volatility_percentile']}%, "
                    f"trend={signals['trend_direction']}, RSI14={signals['rsi14_current']}")

        llm_response, token_usage = invoke_chat_model_human_message(
            prompt, provider=llm_provider, model=llm_model
        )

        logs.append(f"✅ LLM responded ({token_usage['total_tokens']} tokens, "
                    f"${token_usage['cost_usd']:.6f})")
        if token_usage['total_tokens'] == 0:
            logs.append(f"   ⚠️ Token count is 0 — metadata keys: {token_usage.get('_meta_keys', [])}")
        logs.append(f"📄 Raw LLM response: {llm_response[:300]}{'...' if len(llm_response) > 300 else ''}")

        result = json.loads(llm_response)

        # Validate and sanitize the response
        result["period_range"] = [
            max(2,  int(result["period_range"][0])),
            min(28, int(result["period_range"][1])),
        ]
        result["upper_range"] = [
            max(60, int(result["upper_range"][0])),
            min(85, int(result["upper_range"][1])),
        ]
        result["lower_range"] = [
            max(15, int(result["lower_range"][0])),
            min(40, int(result["lower_range"][1])),
        ]
        logs.append(f"🎯 Grid narrowed: periods {result['period_range']}, "
                    f"OB {result['upper_range']}, OS {result['lower_range']}")
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
    Fallback grid narrowing when no LLM is available.
    Applies the same logic the LLM would use, but as explicit rules.
    Useful for testing and as a safety net.
    """
    vol = signals["volatility_percentile"]
    trend = signals["trend_direction"]

    if vol > 70:
        # High volatility → short periods, tighter thresholds
        regime = "high_volatility"
        period_range = [4, 12]
        upper_range  = [70, 80]
        lower_range  = [20, 30]
        reasoning = (
            f"Volatility is at the {vol:.0f}th percentile (high). "
            "Short RSI periods (4-12) respond faster to rapid price swings. "
            "Tighter thresholds (70-80 / 20-30) reduce false signals in choppy conditions."
        )
    elif trend in ("up", "down") and abs(signals["trend_strength_pct"]) > 5:
        # Strong trend → longer periods, wider thresholds
        regime = f"trending_{trend}{'bull' if trend == 'up' else 'bear'}"
        period_range = [12, 22]
        upper_range  = [70, 85]
        lower_range  = [15, 30]
        reasoning = (
            f"Strong {trend}trend detected (price is {abs(signals['trend_strength_pct']):.1f}% "
            f"from 50-SMA). Longer RSI periods (12-22) avoid premature reversal signals. "
            "Wider thresholds (70-85 / 15-30) only trigger on extreme conditions."
        )
    else:
        # Ranging / mean-reverting market
        regime = "mean_reverting"
        period_range = [6, 18]
        upper_range  = [65, 80]
        lower_range  = [20, 35]
        reasoning = (
            "Market appears to be ranging (flat trend, moderate volatility). "
            "Medium RSI periods (6-18) balance responsiveness with noise reduction. "
            "Standard thresholds (65-80 / 20-35) work well in mean-reverting conditions."
        )

    return {
        "regime": regime,
        "period_range": period_range,
        "upper_range": upper_range,
        "lower_range": lower_range,
        "reasoning": reasoning,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Run the optimizer with the LLM-narrowed grid
# ─────────────────────────────────────────────────────────────────────────────

def run_llm_optimizer(
    symbol: Annotated[str, "ticker symbol, e.g. AAPL"],
    curr_date: Annotated[str, "current date YYYY-MM-DD"],
    is_days: Annotated[int, "in-sample window in trading days"] = 180,
    oos_days: Annotated[int, "out-of-sample window in trading days"] = 90,
    llm_provider: Annotated[str | None, "LLM provider (openai, google, anthropic, xai, openrouter, ollama, heuristic). Defaults to config['llm_provider']."] = None,
    llm_model: Annotated[str | None, "specific model to use. Defaults to config['quick_think_llm']."] = None,
) -> dict:
    """
    LLM-Enhanced Walk-Forward RSI Optimization.

    Same math as the pure algo optimizer, but the search grid is narrowed
    by LLM regime classification before running.

    Returns same structure as run_algo_optimizer() PLUS:
      regime: market regime classified by LLM
      llm_reasoning: plain English explanation from LLM
      llm_grid: the narrowed grid LLM chose
      llm_available: True if LLM was used, False if heuristic fallback was used
    """
    # Load data
    data = _load_price_data(symbol)

    # Filter to curr_date
    curr_dt = pd.to_datetime(curr_date)
    data = data[data["Date"] <= curr_dt].copy()
    closes = data["Close"].reset_index(drop=True)
    total_days = len(closes)

    min_required = is_days + oos_days + 30
    if total_days < min_required:
        raise ValueError(
            f"Not enough data: {total_days} rows, need at least {min_required}."
        )

    oos_start = total_days - oos_days
    is_start  = max(0, oos_start - is_days)

    # ── Compute regime signals ─────────────────────────────────────────────
    logs: list[str] = []
    logs.append(f"📊 Loaded {total_days} trading days for {symbol} up to {curr_date}")
    logs.append(f"📅 IS window: {is_days} days | OOS window: {oos_days} days")

    signals = _compute_regime_signals(data)
    logs.append(f"🧮 Regime signals computed:")
    logs.append(f"   Volatility: {signals['volatility_percentile']}th percentile")
    logs.append(f"   Trend: {signals['trend_direction']} ({signals['trend_strength_pct']:+.2f}% from 50-SMA)")
    logs.append(f"   RSI-14: {signals['rsi14_current']} | 10d return: {signals['recent_10d_return_pct']:+.2f}%")
    logs.append(f"   Above 50-SMA: {signals['price_above_sma50']} | Above 200-SMA: {signals['price_above_sma200']}")

    # ── Ask LLM for narrowed grid ──────────────────────────────────────────
    llm_available = (llm_provider != "heuristic")
    grid = _ask_llm_for_grid(symbol, signals, llm_provider=llm_provider, llm_model=llm_model, logs=logs)
    if grid.get("token_usage", {}).get("model") == "heuristic" and llm_provider not in (None, "heuristic"):
        llm_available = False

    # ── Build the narrowed grid from LLM suggestions ───────────────────────
    p_min, p_max = grid["period_range"]
    u_min, u_max = grid["upper_range"]
    l_min, l_max = grid["lower_range"]

    # Build ranges — step 1 for periods, step 5 for thresholds
    periods = list(range(p_min, p_max + 1))
    uppers  = list(range(u_min, u_max + 5, 5))
    lowers  = list(range(l_min, l_max + 5, 5))
    logs.append(f"🔬 Starting grid search: {len(periods)} periods × {len(uppers)} OB × {len(lowers)} OS thresholds")

    # ── Grid search on narrowed space ─────────────────────────────────────
    best_is_sharpe = -np.inf
    best_params = {"period": 14, "upper": 70, "lower": 30}
    combos_tested = 0
    period_best: dict[int, dict] = {}

    # Pre-cache RSI for each period in the narrowed range
    rsi_cache: dict[int, pd.Series] = {}
    for period in periods:
        rsi_cache[period] = _calc_rsi(closes, period)

    for period in periods:
        rsi_full = rsi_cache[period]
        period_best_sharpe = -np.inf
        period_best_oos = 0.0

        for upper in uppers:
            for lower in lowers:
                if upper <= lower:
                    continue

                combos_tested += 1
                positions_full = _generate_positions(rsi_full, upper, lower)

                is_closes    = closes.iloc[is_start:oos_start]
                is_positions = positions_full.iloc[is_start:oos_start]
                is_sharpe    = _calc_sharpe(is_closes, is_positions)

                if is_sharpe > period_best_sharpe:
                    period_best_sharpe = is_sharpe
                    oos_closes    = closes.iloc[oos_start:]
                    oos_positions = positions_full.iloc[oos_start:]
                    period_best_oos = _calc_sharpe(oos_closes, oos_positions)

                if is_sharpe > best_is_sharpe:
                    best_is_sharpe = is_sharpe
                    best_params = {"period": period, "upper": upper, "lower": lower}

        period_best[period] = {
            "period": period,
            "is_sharpe": round(period_best_sharpe, 4),
            "oos_sharpe": round(period_best_oos, 4),
        }

    logs.append(f"✅ Grid search done — {combos_tested} combos tested")
    logs.append(f"🏆 Best IS params: period={best_params['period']}, OB={best_params['upper']}, OS={best_params['lower']}, IS Sharpe={round(best_is_sharpe,3)}")

    # ── OOS validation ────────────────────────────────────────────────────
    best_rsi = rsi_cache[best_params["period"]]
    best_pos = _generate_positions(best_rsi, best_params["upper"], best_params["lower"])
    oos_closes    = closes.iloc[oos_start:]
    oos_positions = best_pos.iloc[oos_start:]
    best_oos_sharpe = _calc_sharpe(oos_closes, oos_positions)

    # ── Baseline ──────────────────────────────────────────────────────────
    default_rsi = _calc_rsi(closes, 14)
    default_pos = _generate_positions(default_rsi, 70.0, 30.0)
    default_is_sharpe  = _calc_sharpe(closes.iloc[is_start:oos_start], default_pos.iloc[is_start:oos_start])
    default_oos_sharpe = _calc_sharpe(closes.iloc[oos_start:], default_pos.iloc[oos_start:])

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
        "optimal_period": best_params["period"],
        "optimal_upper":  best_params["upper"],
        "optimal_lower":  best_params["lower"],
        # Performance
        "is_sharpe":  round(best_is_sharpe, 4),
        "oos_sharpe": round(best_oos_sharpe, 4),
        "confidence": confidence,
        # Meta
        "combos_tested":      combos_tested,
        "is_days":            is_days,
        "oos_days":           oos_days,
        # Baseline
        "default_is_sharpe":  round(default_is_sharpe, 4),
        "default_oos_sharpe": round(default_oos_sharpe, 4),
        # LLM-specific outputs
        "regime":         grid["regime"],
        "llm_reasoning":  grid["reasoning"],
        "llm_available":  llm_available,
        "llm_grid": {
            "period_range": grid["period_range"],
            "upper_range":  grid["upper_range"],
            "lower_range":  grid["lower_range"],
        },
        "regime_signals": signals,
        "debug_logs":     logs,
        "token_usage":    grid.get("token_usage", {
            "model": "heuristic",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
        }),
        # Chart data
        "period_sharpes": sorted(period_best.values(), key=lambda x: x["period"]),
    }
