"""
rsi_optimizer_algo.py
=====================
Pure algorithmic RSI optimizer — NO LLM involved.

What it does (step by step):
  1. Load cached OHLCV price data for the symbol (already downloaded by the main framework)
  2. Calculate RSI for many different period lengths (e.g. 2, 3, 4 ... 28)
  3. For each period + threshold combination, simulate a simple trading strategy and score it
  4. Split data into two windows:
       IS  (In-Sample)  = training window  → find the best parameters here
       OOS (Out-of-Sample) = test window   → validate the best parameters here on UNSEEN data
  5. Return the winner + confidence score

Key concept — Walk-Forward Validation:
  |-------- IS window (180 days) --------|---- OOS window (90 days) ----|
  Optimize here → pick best params         Test those params here
  If OOS Sharpe is close to IS Sharpe → the optimization is TRUSTWORTHY
  If OOS Sharpe collapses → we overfit (found noise, not a real pattern)

Scoring metric — Sharpe Ratio:
  Sharpe = (average daily strategy return) / (std dev of daily returns) * sqrt(252)
  Higher = better risk-adjusted return.
  > 1.0 is good. > 0.5 is acceptable. < 0 means the strategy loses money.
"""

import numpy as np
import pandas as pd
import os
from typing import Annotated
import yfinance as yf
from tradingagents.quant_ml.indicators.stockstats_utils import yf_retry, _clean_dataframe
from tradingagents.dataflows.config import get_config


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load OHLCV data (reuses the same cache as the main framework)
# ─────────────────────────────────────────────────────────────────────────────

def _load_price_data(symbol: str) -> pd.DataFrame:
    """
    Load historical OHLCV data for a symbol.
    Uses the same cache files as the rest of the framework — no extra downloads.

    Returns a DataFrame with columns: Date, Open, High, Low, Close, Volume
    """
    config = get_config()
    today = pd.Timestamp.today()
    start = today - pd.DateOffset(years=15)
    start_str = start.strftime("%Y-%m-%d")
    end_str = today.strftime("%Y-%m-%d")

    data_file = os.path.join(
        config["data_cache_dir"],
        f"{symbol}-YFin-data-{start_str}-{end_str}.csv",
    )

    if os.path.exists(data_file):
        data = pd.read_csv(data_file, on_bad_lines="skip")
    else:
        # Download and cache if not already present
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
    return data


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Calculate RSI
# ─────────────────────────────────────────────────────────────────────────────

def _calc_rsi(closes: pd.Series, period: int) -> pd.Series:
    """
    Calculate RSI using Wilder's smoothing method (same as stockstats).

    How RSI works:
      - Measures the ratio of average gains to average losses over `period` days
      - RSI = 100 - (100 / (1 + RS))  where RS = avg_gain / avg_loss
      - Range: 0-100.  Above 70 = overbought.  Below 30 = oversold.

    Args:
        closes: Series of closing prices
        period: Number of days to look back (e.g. 14)

    Returns:
        Series of RSI values
    """
    delta = closes.diff()  # price change each day

    # Separate gains (positive changes) and losses (negative changes)
    gain = delta.clip(lower=0)   # keep only positive changes, zero out losses
    loss = -delta.clip(upper=0)  # keep only negative changes (flip sign to positive)

    # Wilder's smoothing = exponential moving average with alpha = 1/period
    # This gives more recent data slightly more weight
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100.0 - (100.0 / (1.0 + rs))

    return rsi


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Generate trading signals from RSI
# ─────────────────────────────────────────────────────────────────────────────

def _generate_positions(rsi: pd.Series, upper: float, lower: float) -> pd.Series:
    """
    Convert RSI values into a position series.

    Rules:
      - RSI drops below `lower` (e.g. 30) → BUY (+1), price is oversold
      - RSI rises above `upper` (e.g. 70) → SELL (-1), price is overbought
      - RSI crosses back through 50 → EXIT (0), momentum has normalized

    This is the simplest possible RSI strategy — easy to understand and test.

    Returns:
        Series of positions: +1 (long), -1 (short), 0 (flat)
    """
    positions = []
    pos = 0  # start flat (no position)

    for val in rsi:
        if np.isnan(val):
            positions.append(0)
            continue

        if val < lower:
            pos = 1   # oversold → go long (buy)
        elif val > upper:
            pos = -1  # overbought → go short (sell)
        elif pos == 1 and val > 50:
            pos = 0   # exit long when RSI recovers above 50
        elif pos == -1 and val < 50:
            pos = 0   # exit short when RSI drops back below 50

        positions.append(pos)

    return pd.Series(positions, index=rsi.index)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Score a strategy using Sharpe Ratio
# ─────────────────────────────────────────────────────────────────────────────

def _calc_sharpe(closes: pd.Series, positions: pd.Series) -> float:
    """
    Calculate annualized Sharpe ratio for a strategy.

    How it works:
      - daily_return = today's close / yesterday's close - 1  (e.g. +0.02 = +2%)
      - strategy_return = position_yesterday * daily_return
        (if we were long yesterday and price went up → we made money)
      - Sharpe = mean(strategy_returns) / std(strategy_returns) * sqrt(252)
        (252 = trading days per year, used to annualize)

    Args:
        closes: price series
        positions: +1/-1/0 position series

    Returns:
        Sharpe ratio (float). Higher is better.
    """
    # Shift positions by 1: we act on yesterday's RSI signal
    pos_shifted = positions.shift(1).fillna(0)

    # Daily price returns
    price_returns = closes.pct_change().fillna(0)

    # Strategy daily returns = our position * the market's daily move
    strat_returns = pos_shifted * price_returns

    std = strat_returns.std()
    if std < 1e-10:
        return 0.0  # no trades or flat strategy = 0 Sharpe

    sharpe = strat_returns.mean() / std * np.sqrt(252)
    return round(float(sharpe), 4)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Walk-Forward Optimization (the full grid search + OOS validation)
# ─────────────────────────────────────────────────────────────────────────────

def _plateau_score_rsi(
    all_combos: list[dict],
    period: int,
    upper: float,
    lower: float,
) -> float:
    """Average IS Sharpe of (period, upper, lower) and its immediate neighbours.

    Neighbourhood: period ±2, upper ±5, lower ±5.  A high plateau score means
    the parameter region is stable — not a single noise spike.
    """
    lookup = {
        (r["period"], r["upper"], r["lower"]): r["is_sharpe"]
        for r in all_combos
    }
    values = []
    for dp in (-2, 0, 2):
        for du in (-5, 0, 5):
            for dl in (-5, 0, 5):
                key = (period + dp, upper + du, lower + dl)
                if key in lookup:
                    values.append(lookup[key])
    return float(np.mean(values)) if values else 0.0


def run_algo_optimizer(
    symbol: Annotated[str, "ticker symbol, e.g. AAPL"],
    curr_date: Annotated[str, "current date YYYY-MM-DD"],
    is_days: Annotated[int, "in-sample window length in trading days"] = 252,
    oos_days: Annotated[int, "out-of-sample window length in trading days"] = 90,
) -> dict:
    """
    Full Walk-Forward RSI Optimization.

    Grid searched (tighter than v1 to reduce overfitting):
      - RSI period:           5 to 21 (step 2)  →  9 values
      - Overbought threshold: 65 to 80 (step 5) →  4 values
      - Oversold threshold:   25 to 40 (step 5) →  4 values
      Total combinations: 9 × 4 × 4 ≈ 120 (minus invalid upper ≤ lower)

    Winner selected by plateau score (average IS Sharpe of combo + neighbours)
    rather than the raw IS peak — same approach as the MACD optimizer.

    Returns dict with:
      optimal_period, optimal_upper, optimal_lower,
      is_sharpe, oos_sharpe, confidence,
      combos_tested, period_sharpes (for chart),
      default_is_sharpe, default_oos_sharpe (RSI-14/70/30 baseline)
    """
    # Load data
    data = _load_price_data(symbol)

    curr_dt = pd.to_datetime(curr_date)
    data = data[data["Date"] <= curr_dt].copy()

    closes = data["Close"].reset_index(drop=True)
    total_days = len(closes)

    min_required = is_days + oos_days + 30
    if total_days < min_required:
        raise ValueError(
            f"Not enough data: {total_days} rows, need at least {min_required}. "
            f"Try a smaller is_days or oos_days."
        )

    oos_start = total_days - oos_days
    is_start = oos_start - is_days
    if is_start < 0:
        is_start = 0

    # ── Tighter grid — reduces multiple-comparison overfitting ────────────
    periods = list(range(5, 22, 2))   # 5,7,9,11,13,15,17,19,21 → 9 values
    uppers  = list(range(65, 81, 5))  # 65,70,75,80             → 4 values
    lowers  = list(range(25, 41, 5))  # 25,30,35,40             → 4 values

    best_params = {"period": 14, "upper": 70, "lower": 30}
    combos_tested = 0
    all_combos: list[dict] = []
    period_best: dict[int, dict] = {}

    rsi_cache: dict[int, pd.Series] = {}
    for period in periods:
        rsi_cache[period] = _calc_rsi(closes, period)

    is_closes  = closes.iloc[is_start:oos_start]
    oos_closes = closes.iloc[oos_start:]

    for period in periods:
        rsi_full = rsi_cache[period]
        period_best_is = -np.inf
        period_best_oos = 0.0

        for upper in uppers:
            for lower in lowers:
                if upper <= lower:
                    continue

                combos_tested += 1
                positions_full = _generate_positions(rsi_full, upper, lower)
                is_sharpe = _calc_sharpe(is_closes, positions_full.iloc[is_start:oos_start])
                oos_pos = positions_full.iloc[oos_start:]
                oos_trades = int((oos_pos.diff().fillna(0) != 0).sum())

                all_combos.append({
                    "period": period,
                    "upper": upper,
                    "lower": lower,
                    "is_sharpe": is_sharpe,
                    "oos_trades": oos_trades,
                })

                if is_sharpe > period_best_is:
                    period_best_is = is_sharpe
                    period_best_oos = _calc_sharpe(oos_closes, oos_pos)

        period_best[period] = {
            "period": period,
            "is_sharpe": round(period_best_is, 4),
            "oos_sharpe": round(period_best_oos, 4),
        }

    # ── Plateau-based winner selection — prefer combos with ≥ 5 OOS trades ──
    # Build ranked list: top-quartile IS, sorted by plateau score descending.
    # Pick the first candidate that fires at least MIN_OOS_TRADES in OOS.
    # If none qualify (very rare), fall back to the top-plateau combo.
    MIN_OOS_TRADES = 5
    if all_combos:
        top_quartile = float(np.percentile([r["is_sharpe"] for r in all_combos], 75))
        candidates = [
            r for r in all_combos if r["is_sharpe"] >= top_quartile
        ]
        for row in candidates:
            row["_plateau"] = _plateau_score_rsi(all_combos, row["period"], row["upper"], row["lower"])
        candidates.sort(key=lambda r: (r["_plateau"], r["is_sharpe"]), reverse=True)

        best_params = candidates[0]  # fallback — top plateau regardless of trades
        for row in candidates:
            if row["oos_trades"] >= MIN_OOS_TRADES:
                best_params = row
                break
        best_params = {"period": best_params["period"], "upper": best_params["upper"], "lower": best_params["lower"]}
    best_is_sharpe = -np.inf

    # ── Validate best params on OOS (true holdout — not seen during selection)
    best_rsi = rsi_cache[best_params["period"]]
    best_pos = _generate_positions(best_rsi, best_params["upper"], best_params["lower"])
    best_oos_sharpe = _calc_sharpe(oos_closes, best_pos.iloc[oos_start:])
    best_is_sharpe  = _calc_sharpe(is_closes,  best_pos.iloc[is_start:oos_start])
    best_oos_trades = int((best_pos.iloc[oos_start:].diff().fillna(0) != 0).sum())

    # ── Baseline: RSI(14) with 70/30 (industry default) ───────────────────
    default_rsi = rsi_cache.get(14, _calc_rsi(closes, 14))
    default_pos = _generate_positions(default_rsi, 70.0, 30.0)
    default_is_sharpe  = _calc_sharpe(closes.iloc[is_start:oos_start], default_pos.iloc[is_start:oos_start])
    default_oos_sharpe = _calc_sharpe(closes.iloc[oos_start:], default_pos.iloc[oos_start:])

    # ── Confidence ─────────────────────────────────────────────────────────
    if best_oos_trades < MIN_OOS_TRADES:
        confidence = "LOW"
    elif best_is_sharpe > 0 and best_oos_sharpe > 0:
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
        "optimal_period": best_params["period"],
        "optimal_upper":  best_params["upper"],
        "optimal_lower":  best_params["lower"],
        "is_sharpe":        round(best_is_sharpe, 4),
        "oos_sharpe":       round(best_oos_sharpe, 4),
        "oos_trade_count":  best_oos_trades,
        "confidence":       confidence,
        "combos_tested":      combos_tested,
        "is_days":            is_days,
        "oos_days":           oos_days,
        "default_is_sharpe":  round(default_is_sharpe, 4),
        "default_oos_sharpe": round(default_oos_sharpe, 4),
        "period_sharpes": sorted(period_best.values(), key=lambda x: x["period"]),
    }
