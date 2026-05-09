"""
macd_optimizer_algo.py
======================
Pure algorithmic MACD optimizer — NO LLM involved.

What it does (step by step):
  1. Load cached OHLCV price data for the symbol
  2. Optionally resample to weekly bars (freq='W') — uses sqrt(52) annualiser
  3. Calculate MACD for many different (fast, slow, signal) combinations
  4. For each combination, simulate a crossover trading strategy and score it
     with transaction costs deducted on every position change
  5. Split data into two windows:
       IS  (In-Sample)  = training window  → find the best parameters here
       OOS (Out-of-Sample) = test window   → validate the best parameters here
  6. Select winner by "parameter plateau" (avg Sharpe of combo + neighbours)
     rather than the raw peak, to avoid over-fitting a single outlier bar
  7. Reject low sample-size results: OOS trade count < 30 → confidence = LOW
  8. Return winner + Sharpe / Sortino / Calmar metrics + confidence score

MACD signal logic — Signal Line Crossover:
  histogram = macd_line - signal_line
  histogram flips negative → positive  →  BUY  (bullish crossover)
  histogram flips positive → negative  →  SELL (bearish crossover)
  hold current position otherwise

Grid searched:
  fast_period:   6 to 16  (step 1) → 11 values
  slow_period:  18 to 34  (step 2) →  9 values
  signal_period: 5 to 13  (step 1) →  9 values
  Constraint: fast_period < slow_period
  Total valid combos: ~713
"""

import logging
import numpy as np
import pandas as pd
import os
from typing import Annotated
import yfinance as yf
from tradingagents.quant_ml.indicators.stockstats_utils import yf_retry, _clean_dataframe
from tradingagents.dataflows.config import get_config
from tradingagents.quant_ml.walk_forward.wfo_analyzer import analyze_wfo

log = logging.getLogger("macd_optimizer")


# ─────────────────────────────────────────────────────────────────────────────
# Debug log capture — collects formatted records into a list so they can be
# returned in the result dict and displayed in the UI Debug Log panel.
# ─────────────────────────────────────────────────────────────────────────────

class _ListHandler(logging.Handler):
    """
    A logging Handler that appends each formatted record to an in-memory list.
    Attach to a logger for the duration of run_algo_optimizer, then detach.
    """
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[str] = []
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(self.format(record))


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load OHLCV data
# ─────────────────────────────────────────────────────────────────────────────

def _load_price_data(symbol: str, freq: str = "D") -> pd.DataFrame:
    """
    Load historical OHLCV data for a symbol, with optional weekly resampling.

    Args:
        symbol: Ticker symbol (e.g. "AAPL")
        freq:   'D' for daily (default), 'W' for weekly
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
        log.debug("[data] cache hit: %s", data_file)
        data = pd.read_csv(data_file, on_bad_lines="skip")
    else:
        log.debug("[data] cache miss — downloading %s (%s → %s)", symbol, start_str, end_str)
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
        log.debug("[data] saved to cache: %s", data_file)

    data = _clean_dataframe(data)
    data = data.sort_values("Date").reset_index(drop=True)
    log.debug("[data] raw rows after clean: %d  date range: %s → %s",
              len(data), data["Date"].iloc[0], data["Date"].iloc[-1])

    # ── Weekly resampling ────────────────────────────────────────────────────
    if freq == "W":
        data["Date"] = pd.to_datetime(data["Date"])
        data = (
            data.resample("W", on="Date")
            .agg({"Open": "first", "High": "max", "Low": "min",
                  "Close": "last", "Volume": "sum"})
            .dropna()
            .reset_index()
        )
        log.debug("[data] resampled to weekly: %d bars", len(data))

    return data


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Calculate MACD
# ─────────────────────────────────────────────────────────────────────────────

def _calc_macd(
    closes: pd.Series,
    fast: int,
    slow: int,
    signal: int,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Standard MACD via exponential moving averages.

    Returns:
        (macd_line, signal_line, histogram) — all as pd.Series
    """
    ema_fast    = closes.ewm(span=fast,   adjust=False).mean()
    ema_slow    = closes.ewm(span=slow,   adjust=False).mean()
    macd_line   = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram   = macd_line - signal_line
    return macd_line, signal_line, histogram


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Generate trading signals (vectorized)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_positions(histogram: pd.Series) -> pd.Series:
    """
    Convert MACD histogram into a position series via crossover detection.

    Vectorized implementation — ~15x faster than the previous Python loop.

    Rules:
      histogram shifts from ≤0 to >0  →  BUY  (+1)
      histogram shifts from ≥0 to <0  →  SELL (-1)
      hold current position between crossovers

    Returns:
        Series of positions: +1 (long), -1 (short), 0 (flat/initial)
    """
    prev = histogram.shift(1)
    buy_signal  = (histogram > 0) & (prev <= 0)
    sell_signal = (histogram < 0) & (prev >= 0)

    positions = pd.Series(np.nan, index=histogram.index)
    positions[buy_signal]  = 1
    positions[sell_signal] = -1

    return positions.ffill().fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Score a strategy: Sharpe, Sortino, Calmar
# ─────────────────────────────────────────────────────────────────────────────

def _calc_metrics(
    closes: pd.Series,
    positions: pd.Series,
    ann_factor: float = 252.0,
    transaction_cost: float = 0.001,
) -> dict:
    """
    Compute risk-adjusted return metrics for a strategy.

    Args:
        closes:           Closing price series
        positions:        Position series (+1 / -1 / 0)
        ann_factor:       Trading periods per year (252 daily, 52 weekly)
        transaction_cost: Round-trip cost fraction deducted on every change
                          (default 0.1% — covers commission + slippage)

    Returns dict with:
        sharpe:   Annualised Sharpe Ratio  (penalises all volatility)
        sortino:  Annualised Sortino Ratio (penalises only downside vol)
        calmar:   Annualised return / Maximum Drawdown  (tail-risk aware)
        trade_count: number of position changes in the window
    """
    pos_shifted   = positions.shift(1).fillna(0)
    price_returns = closes.pct_change().fillna(0)
    strat_returns = pos_shifted * price_returns

    # ── Transaction costs ────────────────────────────────────────────────────
    trades = positions.diff().fillna(0) != 0
    trade_count = int(trades.sum())
    strat_returns = strat_returns - trades.astype(float) * transaction_cost

    sqrt_ann = np.sqrt(ann_factor)
    mean_r   = strat_returns.mean()
    std_r    = strat_returns.std()

    # Sharpe
    sharpe = round(float(mean_r / std_r * sqrt_ann), 4) if std_r > 1e-10 else 0.0

    # Sortino — downside deviation only
    downside = strat_returns[strat_returns < 0]
    down_std = downside.std()
    sortino = round(float(mean_r / down_std * sqrt_ann), 4) if down_std > 1e-10 else 0.0

    # Calmar — annualised return / max drawdown
    cum_returns = (1 + strat_returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdown    = (cum_returns - rolling_max) / rolling_max
    max_dd      = abs(drawdown.min())
    ann_return  = float(mean_r * ann_factor)
    calmar = round(ann_return / max_dd, 4) if max_dd > 1e-10 else 0.0

    return {
        "sharpe":      sharpe,
        "sortino":     sortino,
        "calmar":      calmar,
        "trade_count": trade_count,
    }


# Thin wrapper kept for backward-compat with wfo_analyzer calls
def _calc_sharpe(
    closes: pd.Series,
    positions: pd.Series,
    ann_factor: float = 252.0,
    transaction_cost: float = 0.001,
) -> float:
    return _calc_metrics(closes, positions, ann_factor, transaction_cost)["sharpe"]


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Parameter stability: plateau score
# ─────────────────────────────────────────────────────────────────────────────

def _plateau_score(
    param_sharpes: list[dict],
    fast: int,
    slow: int,
    sig: int,
) -> float:
    """
    Return the average IS Sharpe of (fast, slow, sig) and its immediate
    neighbours (fast ±1, slow ±2, sig ±1).  A high plateau score means the
    parameter region is genuinely profitable, not a single-row outlier.
    """
    lookup = {
        (r["fast"], r["slow"], r["signal"]): r["is_sharpe"]
        for r in param_sharpes
    }
    values = []
    for df in (-1, 0, 1):
        for ds in (-2, 0, 2):
            for dsg in (-1, 0, 1):
                key = (fast + df, slow + ds, sig + dsg)
                if key in lookup:
                    values.append(lookup[key])
    return float(np.mean(values)) if values else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Walk-Forward Optimization
# ─────────────────────────────────────────────────────────────────────────────

def run_algo_optimizer(
    symbol: Annotated[str, "ticker symbol, e.g. AAPL"],
    curr_date: Annotated[str, "current date YYYY-MM-DD"],
    is_days: Annotated[int, "in-sample window length in trading periods"] = 180,
    oos_days: Annotated[int, "out-of-sample window length in trading periods"] = 90,
    freq: Annotated[str, "'D' for daily (default) or 'W' for weekly bars"] = "D",
    transaction_cost: Annotated[float, "round-trip cost fraction per trade (default 0.001 = 0.1%)"] = 0.001,
) -> dict:
    """
    Full Walk-Forward MACD Optimization with:
      - Vectorized signal generation (~15x faster)
      - Transaction costs deducted on every position change
      - OOS trade count guard (< 30 trades → LOW confidence)
      - Parameter plateau scoring (robustness over raw peak)
      - Weekly resampling support (freq='W', sqrt(52) annualiser)
      - Sharpe + Sortino + Calmar metrics

    Returns dict with:
      optimal_fast, optimal_slow, optimal_signal,
      is_sharpe, oos_sharpe, oos_sortino, oos_calmar,
      oos_trade_count, confidence,
      combos_tested, param_sharpes,
      default_is_sharpe, default_oos_sharpe,
      wfo_analysis
    """
    ann_factor = 52.0 if freq == "W" else 252.0

    # ── Attach list handler to capture all debug output for the UI ────────────
    _handler = _ListHandler()
    _wfo_log = logging.getLogger("wfo_analyzer")
    # Loggers default to WARNING — explicitly set to DEBUG so records reach the handler
    _prev_level     = log.level
    _prev_wfo_level = _wfo_log.level
    log.setLevel(logging.DEBUG)
    _wfo_log.setLevel(logging.DEBUG)
    log.addHandler(_handler)
    _wfo_log.addHandler(_handler)

    log.debug("[optimizer] ── run_algo_optimizer START ──────────────────────────")
    log.debug("[optimizer] symbol=%s  curr_date=%s  freq=%s  is_days=%d  oos_days=%d  tx_cost=%.4f",
              symbol, curr_date, freq, is_days, oos_days, transaction_cost)

    data = _load_price_data(symbol, freq=freq)

    curr_dt = pd.to_datetime(curr_date)
    data = data[data["Date"] <= curr_dt].copy()

    closes = data["Close"].reset_index(drop=True)
    total_days = len(closes)
    log.debug("[optimizer] bars available after date filter: %d", total_days)

    # MACD warmup = slow_period + signal_period (max: 34 + 13 = 47)
    warmup = 50
    min_required = is_days + oos_days + warmup
    if total_days < min_required:
        log.debug("[optimizer] ABORT — not enough data (%d < %d required)", total_days, min_required)
        raise ValueError(
            f"Not enough data: {total_days} bars, need at least {min_required}. "
            f"Try smaller is_days or oos_days."
        )

    oos_start = total_days - oos_days
    is_start  = max(0, oos_start - is_days)
    log.debug("[optimizer] windows — is_start=%d  oos_start=%d  oos_end=%d",
              is_start, oos_start, total_days)

    # Grid
    fast_periods   = list(range(6, 17))       # 6–16, step 1
    slow_periods   = list(range(18, 35, 2))   # 18–34, step 2
    signal_periods = list(range(5, 14))       # 5–13, step 1

    best_is_sharpe  = -np.inf
    best_plateau    = -np.inf
    best_params     = {"fast": 12, "slow": 26, "signal": 9}
    combos_tested   = 0

    param_sharpes: list[dict] = []

    log.debug("[grid] starting grid search — fast=%s  slow=%s  signal=%s",
              fast_periods, slow_periods, signal_periods)

    # ── First pass: compute IS/OOS Sharpe for every combo ───────────────────
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
                is_m         = _calc_metrics(is_closes, is_positions, ann_factor, transaction_cost)

                oos_closes    = closes.iloc[oos_start:]
                oos_positions = positions_full.iloc[oos_start:]
                oos_m         = _calc_metrics(oos_closes, oos_positions, ann_factor, transaction_cost)

                log.debug("[grid] (%2d,%2d,%2d)  IS sharpe=%-7.4f  OOS sharpe=%-7.4f  OOS trades=%d",
                          fast, slow, sig,
                          is_m["sharpe"], oos_m["sharpe"], oos_m["trade_count"])

                param_sharpes.append({
                    "fast":        fast,
                    "slow":        slow,
                    "signal":      sig,
                    "is_sharpe":   is_m["sharpe"],
                    "oos_sharpe":  oos_m["sharpe"],
                    "oos_sortino": oos_m["sortino"],
                    "oos_calmar":  oos_m["calmar"],
                    "oos_trades":  oos_m["trade_count"],
                })

    log.debug("[grid] done — %d combos evaluated", combos_tested)

    # ── Second pass: select winner by plateau score ──────────────────────────
    for row in param_sharpes:
        if row["is_sharpe"] > best_is_sharpe:
            best_is_sharpe = row["is_sharpe"]

    # Only consider top-quartile IS performers for plateau comparison
    threshold = np.percentile([r["is_sharpe"] for r in param_sharpes], 75)
    log.debug("[plateau] top-quartile IS sharpe threshold: %.4f  (raw peak: %.4f)",
              threshold, best_is_sharpe)

    for row in param_sharpes:
        if row["is_sharpe"] < threshold:
            continue
        plateau = _plateau_score(param_sharpes, row["fast"], row["slow"], row["signal"])
        log.debug("[plateau] (%2d,%2d,%2d) IS=%.4f  plateau=%.4f",
                  row["fast"], row["slow"], row["signal"], row["is_sharpe"], plateau)
        if plateau > best_plateau:
            best_plateau = plateau
            best_params  = {"fast": row["fast"], "slow": row["slow"], "signal": row["signal"]}

    log.debug("[plateau] winner: fast=%d  slow=%d  signal=%d  plateau_score=%.4f",
              best_params["fast"], best_params["slow"], best_params["signal"], best_plateau)

    # ── OOS metrics for best params ──────────────────────────────────────────
    _, _, best_hist = _calc_macd(
        closes, best_params["fast"], best_params["slow"], best_params["signal"]
    )
    best_pos     = _generate_positions(best_hist)
    best_oos_pos = best_pos.iloc[oos_start:]
    best_oos_m   = _calc_metrics(closes.iloc[oos_start:], best_oos_pos, ann_factor, transaction_cost)
    best_is_m    = _calc_metrics(closes.iloc[is_start:oos_start], best_pos.iloc[is_start:oos_start], ann_factor, transaction_cost)

    best_is_sharpe  = best_is_m["sharpe"]
    best_oos_sharpe = best_oos_m["sharpe"]
    oos_trade_count = best_oos_m["trade_count"]

    log.debug("[result] best IS  — sharpe=%.4f", best_is_sharpe)
    log.debug("[result] best OOS — sharpe=%.4f  sortino=%.4f  calmar=%.4f  trades=%d",
              best_oos_sharpe, best_oos_m["sortino"], best_oos_m["calmar"], oos_trade_count)

    # ── Baseline: MACD(12, 26, 9) ────────────────────────────────────────────
    _, _, default_hist = _calc_macd(closes, 12, 26, 9)
    default_pos = _generate_positions(default_hist)
    default_is_sharpe  = _calc_sharpe(
        closes.iloc[is_start:oos_start], default_pos.iloc[is_start:oos_start],
        ann_factor, transaction_cost
    )
    default_oos_sharpe = _calc_sharpe(
        closes.iloc[oos_start:], default_pos.iloc[oos_start:],
        ann_factor, transaction_cost
    )
    log.debug("[baseline] MACD(12,26,9) — IS sharpe=%.4f  OOS sharpe=%.4f",
              default_is_sharpe, default_oos_sharpe)

    # ── Confidence ───────────────────────────────────────────────────────────
    if oos_trade_count < 30:
        # Insufficient sample — statistical significance too low
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

    log.debug("[confidence] oos_trade_count=%d  confidence=%s", oos_trade_count, confidence)
    log.debug("[optimizer] ── run_algo_optimizer END ────────────────────────────")

    # ── Detach handler and restore original logger levels ─────────────────────
    log.removeHandler(_handler)
    _wfo_log.removeHandler(_handler)
    log.setLevel(_prev_level)
    _wfo_log.setLevel(_prev_wfo_level)
    debug_logs: list[str] = _handler.records

    # ── WFO statistical significance analysis ────────────────────────────────
    bar_frequency = "weekly" if freq == "W" else "daily"
    wfo = analyze_wfo(
        is_days=is_days,
        oos_days=oos_days,
        is_sharpe=best_is_sharpe,
        oos_sharpe=best_oos_sharpe,
        oos_positions=best_oos_pos.tolist(),
        bar_frequency=bar_frequency,
    )

    return {
        "optimal_fast":        best_params["fast"],
        "optimal_slow":        best_params["slow"],
        "optimal_signal":      best_params["signal"],
        "is_sharpe":           round(best_is_sharpe, 4),
        "oos_sharpe":          round(best_oos_sharpe, 4),
        "oos_sortino":         round(best_oos_m["sortino"], 4),
        "oos_calmar":          round(best_oos_m["calmar"], 4),
        "oos_trade_count":     oos_trade_count,
        "confidence":          confidence,
        "combos_tested":       combos_tested,
        "is_days":             is_days,
        "oos_days":            oos_days,
        "freq":                freq,
        "transaction_cost":    transaction_cost,
        "default_is_sharpe":   round(default_is_sharpe, 4),
        "default_oos_sharpe":  round(default_oos_sharpe, 4),
        # Top-20 combos by IS Sharpe for UI chart
        "param_sharpes": sorted(param_sharpes, key=lambda x: x["is_sharpe"], reverse=True)[:20],
        # WFO significance analysis
        "wfo_analysis": wfo.to_dict(),
        # Debug log lines captured during this run — displayed in UI debug panel
        "debug_logs": debug_logs,
    }
