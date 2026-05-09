"""
tests/test_integration_oos.py
==============================
Integration tests for the MACD and RSI Walk-Forward Optimizers.

These tests:
  1. Run the full optimizer pipeline end-to-end on real price data (yfinance).
  2. Display actual results — optimal parameters, IS/OOS Sharpe, and
     simulated dollar P&L from the OOS window — in a readable Rich table.
  3. Sweep across multiple historical dates (OOS walk-forward sweep) to see
     how the optimizer and its signals hold up over time.

No LLM is required — these are pure algorithmic tests.

Run:
    # Single MACD/RSI run
    pytest tests/test_integration_oos.py::test_macd_optimizer -v -s -m integration
    pytest tests/test_integration_oos.py::test_rsi_optimizer  -v -s -m integration

    # Full OOS walk-forward sweep (slower — runs optimizer N times)
    pytest tests/test_integration_oos.py::test_oos_sweep -v -s -m integration

    # All integration tests
    pytest tests/test_integration_oos.py -v -s -m integration

    # Override ticker / dates via environment variables:
    TEST_TICKER=NVDA TEST_DATE=2024-06-15 OOS_N_DATES=4 pytest ... -m integration
"""

import math
import os
import json
import pytest
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path

# ── Rich display ─────────────────────────────────────────────────────────────
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box

# ── Framework imports ─────────────────────────────────────────────────────────
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.dataflows.config import set_config
from tradingagents.quant_ml.optimizers.macd_optimizer_algo import (
    run_algo_optimizer as run_macd_optimizer,
    _load_price_data as _macd_load_price,
    _calc_macd,
    _generate_positions as _macd_positions,
    _calc_metrics,
)
from tradingagents.quant_ml.optimizers.rsi_optimizer_algo import (
    run_algo_optimizer as run_rsi_optimizer,
    _load_price_data as _rsi_load_price,
    _calc_rsi,
    _generate_positions as _rsi_positions,
    _calc_sharpe as _rsi_sharpe,
)

console = Console()

# ─────────────────────────────────────────────────────────────────────────────
# Constants / env-var driven configuration
# ─────────────────────────────────────────────────────────────────────────────

NOTIONAL       = float(os.getenv("TEST_NOTIONAL",   "10000"))   # $ starting capital
IS_DAYS        = int(os.getenv("TEST_IS_DAYS",      "100"))
OOS_DAYS       = int(os.getenv("TEST_OOS_DAYS",     "30"))
OOS_N_DATES    = int(os.getenv("OOS_N_DATES",       "6"))
SLIPPAGE       = float(os.getenv("TEST_SLIPPAGE",   "0.001"))  # 10 bps per side
COMMISSION     = float(os.getenv("TEST_COMMISSION",  "0.0"))   # flat $ per trade

_default_ticker = "SPY"
_default_date   = (datetime.today() - timedelta(days=90)).strftime("%Y-%m-%d")

TICKER  = os.getenv("TEST_TICKER", _default_ticker)
DATE    = os.getenv("TEST_DATE",   _default_date)

# OOS sweep window
_oos_start_default = (datetime.today() - timedelta(days=540)).strftime("%Y-%m-%d")  # ~18 months ago
_oos_end_default   = (datetime.today() - timedelta(days=90)).strftime("%Y-%m-%d")   # leave 90d of future data
OOS_START = os.getenv("OOS_START", _oos_start_default)
OOS_END   = os.getenv("OOS_END",   _oos_end_default)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _setup_config():
    cfg = DEFAULT_CONFIG.copy()
    set_config(cfg)
    Path(cfg["data_cache_dir"]).mkdir(parents=True, exist_ok=True)
    return cfg

def _save_results(filename: str, data: dict):
    out_dir = Path("results/integration")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    console.print(f"[dim]Results saved → {path}[/dim]")

def _business_dates(start: str, end: str, n: int) -> list[str]:
    bdays = pd.bdate_range(start=start, end=end)
    if len(bdays) < n: return [d.strftime("%Y-%m-%d") for d in bdays]
    indices = np.linspace(0, len(bdays) - 1, n, dtype=int)
    return [bdays[i].strftime("%Y-%m-%d") for i in indices]

def _pnl_color(val: float) -> str:
    return "green" if val > 0 else ("red" if val < 0 else "white")


def _decision_color(signal: str) -> str:
    """Map a signal word to a Rich colour."""
    return {"BUY": "green", "OVERWEIGHT": "green",
            "SELL": "red",  "UNDERWEIGHT": "red",
            "HOLD": "yellow"}.get(signal.upper(), "white")


def _compute_kelly_fraction(
    closes: pd.Series,
    positions: pd.Series,
    is_start: int,
    is_end: int,
) -> tuple[float, float, float, float]:
    """
    Compute Kelly fraction from IS-window strategy returns.

    Returns (kelly_f, half_kelly, win_rate, win_loss_ratio).
    Half-Kelly is clamped to [0.05, 1.0]; negative Kelly returns (0, 0, ...).
    """
    is_closes = closes.iloc[is_start:is_end].reset_index(drop=True)
    is_pos    = positions.iloc[is_start:is_end].reset_index(drop=True)

    price_ret = is_closes.pct_change().fillna(0)
    pos_shifted = is_pos.shift(1).fillna(0)
    strat_ret = pos_shifted * price_ret

    active = strat_ret[pos_shifted != 0]
    if len(active) < 5:
        return (0.0, 0.0, 0.0, 0.0)

    winners = active[active > 0]
    losers  = active[active < 0]

    win_rate = len(winners) / len(active) if len(active) else 0.0
    avg_win  = float(winners.mean()) if len(winners) else 0.0
    avg_loss = float(abs(losers.mean())) if len(losers) else 0.0
    wl_ratio = avg_win / avg_loss if avg_loss > 0 else 10.0

    kelly_f = win_rate - (1 - win_rate) / wl_ratio if wl_ratio > 0 else 0.0

    if kelly_f <= 0:
        return (kelly_f, 0.0, win_rate, wl_ratio)

    half_kelly = max(0.05, min(1.0, kelly_f / 2))
    return (round(kelly_f, 4), round(half_kelly, 4), round(win_rate, 4), round(wl_ratio, 4))


_POS_LABELS = {1: "BUY (Long)", -1: "SELL (Short)", 0: "EXIT (Flat)"}

def _simulate_oos(
    oos_closes: pd.Series,
    oos_positions: pd.Series,
    oos_dates: pd.Series,
    notional: float,
    kelly_f: float,
    slippage: float = SLIPPAGE,
    commission: float = COMMISSION,
    oos_indicator: pd.Series | None = None,
    bh_notional: float | None = None,
) -> dict:
    """
    Share-based day-by-day OOS simulation with Kelly sizing.

    Tracks cash, shares, entry_price. On position changes, buys/sells whole
    shares with slippage + commission. Returns final_equity, trade_log, etc.

    bh_notional: starting equity for the B&H comparison column (defaults to notional).
    """
    n = len(oos_closes)
    if n == 0:
        return {"final_equity": notional, "pnl_pct": 0.0, "pnl_dollar": 0.0,
                "win_rate": 0.0, "max_dd_pct": 0.0, "n_trades": 0, "kelly_f": kelly_f,
                "equity_curve": [], "trade_log": []}

    if bh_notional is None:
        bh_notional = notional

    cash        = notional
    shares      = 0
    pos_dir     = 0        # +1 long, -1 short, 0 flat
    entry_price = 0.0
    n_trades    = 0

    # B&H: buy whole shares on day 0
    bh_price0   = float(oos_closes.iloc[0])
    bh_shares   = math.floor(bh_notional / (bh_price0 * (1 + slippage)))
    bh_cash     = bh_notional - bh_shares * bh_price0 * (1 + slippage)

    equity_arr  = []
    bh_eq_arr   = []
    trade_log   = []

    for i in range(n):
        price    = float(oos_closes.iloc[i])
        new_pos  = int(oos_positions.iloc[i])
        old_pos  = pos_dir

        # Position change?
        if new_pos != old_pos:
            cost_this_trade = 0.0

            # 1) Close existing position
            if old_pos == 1 and shares > 0:
                proceeds = shares * price * (1 - slippage) - commission
                cost_this_trade += shares * price * slippage + commission
                cash += proceeds
                shares = 0
            elif old_pos == -1 and shares > 0:
                # Close short: profit = shares * (entry_price - price), minus slippage
                close_cost = shares * price * slippage + commission
                cost_this_trade += close_cost
                pnl_short = shares * (entry_price - price)
                cash += shares * entry_price + pnl_short - close_cost
                shares = 0

            # 2) Open new position
            if new_pos != 0 and kelly_f > 0:
                investable = cash * kelly_f
                buy_price  = price * (1 + slippage)
                new_shares = math.floor(investable / buy_price) if buy_price > 0 else 0
                if new_shares > 0:
                    open_cost = new_shares * price * slippage + commission
                    cost_this_trade += open_cost
                    cash -= new_shares * buy_price + commission
                    shares = new_shares
                    entry_price = price

            pos_dir = new_pos
            n_trades += 1

            # Equity after this trade
            if pos_dir == 1:
                eq = cash + shares * price
            elif pos_dir == -1:
                eq = cash + shares * (2 * entry_price - price)
            else:
                eq = cash

            indicator_val = None
            if oos_indicator is not None and i < len(oos_indicator):
                v = float(oos_indicator.iloc[i])
                indicator_val = round(v, 2) if not np.isnan(v) else None

            bh_eq = bh_cash + bh_shares * price

            trade_log.append({
                "date":       str(oos_dates.iloc[i])[:10],
                "action":     _POS_LABELS.get(new_pos, "HOLD"),
                "pos":        new_pos,
                "price":      round(price, 2),
                "indicator":  indicator_val,
                "shares":     shares,
                "cash":       round(cash, 2),
                "cost":       round(cost_this_trade, 2),
                "rsi_equity": round(eq, 2),
                "bh_equity":  round(bh_eq, 2),
                "rsi_pnl":    round(eq - notional, 2),
                "bh_pnl":     round(bh_eq - bh_notional, 2),
                "alpha":      round(eq - bh_eq, 2),
            })

        # Mark-to-market equity
        if pos_dir == 1:
            eq = cash + shares * price
        elif pos_dir == -1:
            eq = cash + shares * (2 * entry_price - price)
        else:
            eq = cash

        equity_arr.append(eq)
        bh_eq_arr.append(bh_cash + bh_shares * price)

    final_equity = equity_arr[-1] if equity_arr else notional
    pnl_dollar   = final_equity - notional
    pnl_pct      = (pnl_dollar / notional) * 100 if notional > 0 else 0.0

    # Max drawdown
    eq_series   = pd.Series(equity_arr)
    rolling_max = eq_series.cummax()
    dd = (eq_series - rolling_max) / rolling_max
    max_dd_pct = float(abs(dd.min()) * 100) if len(dd) else 0.0

    # Win rate (days with positive strategy return)
    eq_s = pd.Series(equity_arr)
    daily_ret = eq_s.pct_change().fillna(0)
    winning = int((daily_ret > 0).sum())
    total_active = int((daily_ret != 0).sum())
    win_rate = (winning / total_active * 100) if total_active > 0 else 0.0

    equity_curve = [
        {"date": str(oos_dates.iloc[i])[:10], "equity": round(equity_arr[i], 2)}
        for i in range(n)
    ]

    return {
        "final_equity":  round(final_equity, 2),
        "pnl_pct":       round(pnl_pct, 2),
        "pnl_dollar":    round(pnl_dollar, 2),
        "win_rate":      round(win_rate, 1),
        "max_dd_pct":    round(max_dd_pct, 2),
        "n_trades":      n_trades,
        "kelly_f":       kelly_f,
        "equity_curve":  equity_curve,
        "trade_log":     trade_log,
    }


def _compute_macd_oos_pnl(
    symbol: str,
    curr_date: str,
    fast: int,
    slow: int,
    signal: int,
    oos_days: int,
    notional: float = NOTIONAL,
    slippage: float = SLIPPAGE,
    commission: float = COMMISSION,
    bh_notional: float | None = None,
) -> dict:
    """Share-based MACD OOS P&L with Kelly sizing."""
    data = _macd_load_price(symbol, freq="D")
    curr_dt = pd.to_datetime(curr_date)
    data = data[data["Date"] <= curr_dt].copy()
    closes = data["Close"].reset_index(drop=True)
    dates  = data["Date"].reset_index(drop=True)

    total     = len(closes)
    oos_start = total - oos_days
    is_start  = max(0, oos_start - IS_DAYS)

    _, _, hist = _calc_macd(closes, fast, slow, signal)
    positions  = _macd_positions(hist)

    kelly_f, half_kelly, _, _ = _compute_kelly_fraction(closes, positions, is_start, oos_start)

    oos_closes    = closes.iloc[oos_start:].reset_index(drop=True)
    oos_positions = positions.iloc[oos_start:].reset_index(drop=True)
    oos_dates     = dates.iloc[oos_start:].reset_index(drop=True)
    oos_hist      = hist.iloc[oos_start:].reset_index(drop=True)

    return _simulate_oos(
        oos_closes, oos_positions, oos_dates,
        notional, half_kelly, slippage, commission,
        oos_indicator=oos_hist, bh_notional=bh_notional,
    )


def _compute_rsi_oos_pnl(
    symbol: str,
    curr_date: str,
    period: int,
    upper: float,
    lower: float,
    oos_days: int,
    notional: float = NOTIONAL,
    slippage: float = SLIPPAGE,
    commission: float = COMMISSION,
    bh_notional: float | None = None,
) -> dict:
    """Share-based RSI OOS P&L with Kelly sizing."""
    data = _rsi_load_price(symbol)
    curr_dt = pd.to_datetime(curr_date)
    data = data[data["Date"] <= curr_dt].copy()
    closes = data["Close"].reset_index(drop=True)
    dates  = data["Date"].reset_index(drop=True)

    total     = len(closes)
    oos_start = total - oos_days
    is_start  = max(0, oos_start - IS_DAYS)

    rsi_full  = _calc_rsi(closes, period)
    positions = _rsi_positions(rsi_full, upper, lower)

    kelly_f, half_kelly, _, _ = _compute_kelly_fraction(closes, positions, is_start, oos_start)

    oos_closes    = closes.iloc[oos_start:].reset_index(drop=True)
    oos_positions = positions.iloc[oos_start:].reset_index(drop=True)
    oos_dates     = dates.iloc[oos_start:].reset_index(drop=True)
    oos_rsi       = rsi_full.iloc[oos_start:].reset_index(drop=True)

    return _simulate_oos(
        oos_closes, oos_positions, oos_dates,
        notional, half_kelly, slippage, commission,
        oos_indicator=oos_rsi, bh_notional=bh_notional,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — MACD single optimizer run
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.timeout(300)
def test_macd_optimizer():
    """
    Run the MACD walk-forward optimizer for one ticker/date.
    Displays: optimal params, IS/OOS Sharpe, OOS P&L, confidence, WFO metrics.
    """
    _setup_config()

    console.print()
    console.rule(f"[bold cyan]MACD Optimizer — {TICKER}  @  {DATE}[/bold cyan]")

    result = run_macd_optimizer(TICKER, DATE, is_days=IS_DAYS, oos_days=OOS_DAYS)

    pnl = _compute_macd_oos_pnl(
        TICKER, DATE,
        fast=result["optimal_fast"],
        slow=result["optimal_slow"],
        signal=result["optimal_signal"],
        oos_days=OOS_DAYS,
    )

    # ── Parameters table ──────────────────────────────────────────────────────
    t = Table(title="MACD Optimization Results", box=box.ROUNDED, show_lines=True)
    t.add_column("Metric",          style="bold white", min_width=28)
    t.add_column("Optimized",       style="cyan",       min_width=18)
    t.add_column("Default (12,26,9)", style="dim",      min_width=18)

    t.add_row("Parameters (fast, slow, signal)",
              f"{result['optimal_fast']}, {result['optimal_slow']}, {result['optimal_signal']}",
              "12, 26, 9")
    t.add_row("IS  Sharpe",
              f"{result['is_sharpe']:+.4f}",
              f"{result['default_is_sharpe']:+.4f}")
    t.add_row("OOS Sharpe",
              f"[{'green' if result['oos_sharpe'] > 0 else 'red'}]{result['oos_sharpe']:+.4f}[/]",
              f"[{'green' if result['default_oos_sharpe'] > 0 else 'red'}]{result['default_oos_sharpe']:+.4f}[/]")
    t.add_row("OOS Sortino",   f"{result['oos_sortino']:+.4f}", "—")
    t.add_row("OOS Calmar",    f"{result['oos_calmar']:+.4f}",  "—")
    t.add_row("OOS Trades",    str(result["oos_trade_count"]),  "—")
    t.add_row("Combos Tested", str(result["combos_tested"]),    "713")
    t.add_row("IS  Window",    f"{result['is_days']} bars",     f"{IS_DAYS} bars")
    t.add_row("OOS Window",    f"{result['oos_days']} bars",    f"{OOS_DAYS} bars")

    confidence_color = {"HIGH": "green", "MEDIUM": "yellow", "LOW": "red"}.get(result["confidence"], "white")
    t.add_row("Confidence",
              f"[{confidence_color}]{result['confidence']}[/{confidence_color}]", "—")

    console.print(t)

    # ── WFO analysis ──────────────────────────────────────────────────────────
    wfo = result.get("wfo_analysis", {})
    if wfo:
        wt = Table(title="Walk-Forward Statistical Analysis", box=box.SIMPLE_HEAVY, show_lines=False)
        wt.add_column("Metric",  style="bold white", min_width=28)
        wt.add_column("Value",   style="cyan",       min_width=18)

        wt.add_row("WFE (OOS/IS Sharpe ratio)",
                   f"{wfo.get('wfe', 0):.4f}  "
                   f"({'[green]efficient[/]' if wfo.get('wfe', 0) >= 0.7 else '[yellow]degraded[/]' if wfo.get('wfe', 0) >= 0.3 else '[red]poor[/]'})")
        wt.add_row("Confidence Score (0-100)",
                   f"{wfo.get('confidence_score', 0):.1f}")
        wt.add_row("Confidence Label",
                   str(wfo.get("confidence_label", "—")))
        wt.add_row("Sample Risk",
                   str(wfo.get("sample_risk", "—")))
        console.print(wt)

    # ── P&L table ─────────────────────────────────────────────────────────────
    pt = Table(title=f"OOS P&L  (notional ${NOTIONAL:,.0f})", box=box.ROUNDED, show_lines=True)
    pt.add_column("Metric",      style="bold white", min_width=28)
    pt.add_column("Value",       min_width=18)

    pnl_color = _pnl_color(pnl["pnl_dollar"])
    pt.add_row("Kelly Fraction (half)",
               f"{pnl.get('kelly_f', 0):.2f}")
    pt.add_row("Cumulative Return",
               f"[{pnl_color}]{pnl['pnl_pct']:+.2f}%[/{pnl_color}]")
    pt.add_row("Dollar P&L",
               f"[{pnl_color}]${pnl['pnl_dollar']:+,.2f}[/{pnl_color}]")
    pt.add_row("Final Equity",
               f"${pnl.get('final_equity', NOTIONAL + pnl['pnl_dollar']):,.2f}")
    pt.add_row("Win Rate (active days)",
               f"{pnl['win_rate']:.1f}%")
    pt.add_row("Max Drawdown",
               f"[red]{pnl['max_dd_pct']:.2f}%[/red]")
    pt.add_row("Position Changes",
               str(pnl["n_trades"]))

    console.print(pt)

    # ── Top-5 parameter combos ─────────────────────────────────────────────────
    top5 = result["param_sharpes"][:5]
    if top5:
        ct = Table(title="Top 5 MACD Combos (by IS Sharpe)", box=box.SIMPLE, show_lines=False)
        ct.add_column("Rank",        style="dim",   min_width=6)
        ct.add_column("fast",        style="cyan",  min_width=6)
        ct.add_column("slow",        style="cyan",  min_width=6)
        ct.add_column("signal",      style="cyan",  min_width=8)
        ct.add_column("IS Sharpe",   style="white", min_width=10)
        ct.add_column("OOS Sharpe",  min_width=10)
        for i, row in enumerate(top5, 1):
            oos_c = "green" if row["oos_sharpe"] > 0 else "red"
            ct.add_row(str(i),
                       str(row["fast"]), str(row["slow"]), str(row["signal"]),
                       f"{row['is_sharpe']:+.4f}",
                       f"[{oos_c}]{row['oos_sharpe']:+.4f}[/{oos_c}]")
        console.print(ct)

    # ── Assertions ────────────────────────────────────────────────────────────
    assert result["optimal_fast"]   > 0
    assert result["optimal_slow"]   > result["optimal_fast"]
    assert result["optimal_signal"] > 0
    assert result["combos_tested"]  > 0
    assert result["confidence"] in ("HIGH", "MEDIUM", "LOW")

    _save_results(
        f"{TICKER}_{DATE}_macd_single.json",
        {"optimizer": result, "pnl": {k: v for k, v in pnl.items() if k != "equity_curve"}},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — RSI single optimizer run
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.timeout(300)
def test_rsi_optimizer():
    """
    Run the RSI walk-forward optimizer for one ticker/date.
    Displays: optimal params, IS/OOS Sharpe, OOS P&L, confidence.
    """
    _setup_config()

    console.print()
    console.rule(f"[bold magenta]RSI Optimizer — {TICKER}  @  {DATE}[/bold magenta]")

    result = run_rsi_optimizer(TICKER, DATE, is_days=IS_DAYS, oos_days=OOS_DAYS)

    pnl = _compute_rsi_oos_pnl(
        TICKER, DATE,
        period=result["optimal_period"],
        upper=result["optimal_upper"],
        lower=result["optimal_lower"],
        oos_days=OOS_DAYS,
    )

    # ── Parameters table ──────────────────────────────────────────────────────
    t = Table(title="RSI Optimization Results", box=box.ROUNDED, show_lines=True)
    t.add_column("Metric",               style="bold white", min_width=28)
    t.add_column("Optimized",            style="magenta",    min_width=18)
    t.add_column("Default (14/70/30)",   style="dim",        min_width=18)

    t.add_row("RSI Period",
              str(result["optimal_period"]), "14")
    t.add_row("Overbought Threshold",
              str(result["optimal_upper"]), "70")
    t.add_row("Oversold Threshold",
              str(result["optimal_lower"]), "30")
    t.add_row("IS  Sharpe",
              f"{result['is_sharpe']:+.4f}",
              f"{result['default_is_sharpe']:+.4f}")
    t.add_row("OOS Sharpe",
              f"[{'green' if result['oos_sharpe'] > 0 else 'red'}]{result['oos_sharpe']:+.4f}[/]",
              f"[{'green' if result['default_oos_sharpe'] > 0 else 'red'}]{result['default_oos_sharpe']:+.4f}[/]")
    t.add_row("Combos Tested", str(result["combos_tested"]), "972")
    t.add_row("IS  Window",    f"{result['is_days']} bars",  f"{IS_DAYS} bars")
    t.add_row("OOS Window",    f"{result['oos_days']} bars", f"{OOS_DAYS} bars")

    confidence_color = {"HIGH": "green", "MEDIUM": "yellow", "LOW": "red"}.get(result["confidence"], "white")
    t.add_row("Confidence",
              f"[{confidence_color}]{result['confidence']}[/{confidence_color}]", "—")

    console.print(t)

    # ── P&L table ─────────────────────────────────────────────────────────────
    pt = Table(title=f"OOS P&L  (notional ${NOTIONAL:,.0f})", box=box.ROUNDED, show_lines=True)
    pt.add_column("Metric",  style="bold white", min_width=28)
    pt.add_column("Value",   min_width=18)

    pnl_color = _pnl_color(pnl["pnl_dollar"])
    pt.add_row("Kelly Fraction (half)",
               f"{pnl.get('kelly_f', 0):.2f}")
    pt.add_row("Cumulative Return",
               f"[{pnl_color}]{pnl['pnl_pct']:+.2f}%[/{pnl_color}]")
    pt.add_row("Dollar P&L",
               f"[{pnl_color}]${pnl['pnl_dollar']:+,.2f}[/{pnl_color}]")
    pt.add_row("Final Equity",
               f"${pnl.get('final_equity', NOTIONAL + pnl['pnl_dollar']):,.2f}")
    pt.add_row("Win Rate (active days)",
               f"{pnl['win_rate']:.1f}%")
    pt.add_row("Max Drawdown",
               f"[red]{pnl['max_dd_pct']:.2f}%[/red]")
    pt.add_row("Trades",
               str(pnl["n_trades"]))

    console.print(pt)

    # ── Period Sharpe chart (text-based) ──────────────────────────────────────
    ps = result.get("period_sharpes", [])
    if ps:
        st = Table(title="RSI Period Scan (best IS Sharpe per period)",
                   box=box.SIMPLE, show_lines=False)
        st.add_column("Period",     style="dim",     min_width=8)
        st.add_column("IS Sharpe",  style="white",   min_width=12)
        st.add_column("OOS Sharpe", min_width=12)
        st.add_column("Bar",        min_width=30)

        best_is = max((r["is_sharpe"] for r in ps), default=1)
        for row in ps:
            bar_len = max(0, int(row["is_sharpe"] / max(best_is, 0.01) * 25))
            bar_str = "█" * bar_len
            oos_c   = "green" if row["oos_sharpe"] > 0 else "red"
            marker  = " ◀ optimal" if row["period"] == result["optimal_period"] else ""
            st.add_row(
                f"{row['period']}{marker}",
                f"{row['is_sharpe']:+.4f}",
                f"[{oos_c}]{row['oos_sharpe']:+.4f}[/{oos_c}]",
                f"[cyan]{bar_str}[/cyan]",
            )
        console.print(st)

    # ── Assertions ────────────────────────────────────────────────────────────
    assert 2 <= result["optimal_period"] <= 28
    assert result["optimal_upper"] > result["optimal_lower"]
    assert result["combos_tested"] > 0
    assert result["confidence"] in ("HIGH", "MEDIUM", "LOW")

    _save_results(
        f"{TICKER}_{DATE}_rsi_single.json",
        {"optimizer": result, "pnl": {k: v for k, v in pnl.items() if k != "equity_curve"}},
    )



def _print_rsi_trade_log(sweep_date: str, trade_log: list[dict], params: str, kelly_f: float = 0):
    """Print a detailed action-day table for one RSI OOS window."""
    if not trade_log:
        return
    kelly_str = f"  Kelly={kelly_f:.2f}" if kelly_f > 0 else "  Kelly=SKIP"
    t = Table(
        title=f"RSI Action Days — {sweep_date}  (params {params}{kelly_str})",
        box=box.SIMPLE_HEAVY, show_lines=False,
    )
    t.add_column("#",           style="dim",    min_width=4)
    t.add_column("Date",        style="white",  min_width=12)
    t.add_column("Action",      min_width=14)
    t.add_column("Price",       style="cyan",   min_width=10, justify="right")
    t.add_column("RSI",         style="magenta", min_width=7, justify="right")
    t.add_column("Shares",      style="white",  min_width=7, justify="right")
    t.add_column("Cash",        style="dim",    min_width=10, justify="right")
    t.add_column("Cost",        style="dim",    min_width=8, justify="right")
    t.add_column("RSI Equity",  min_width=12, justify="right")
    t.add_column("B&H Equity",  style="dim",   min_width=12, justify="right")
    t.add_column("Alpha $",     min_width=10, justify="right")

    for idx, entry in enumerate(trade_log, 1):
        action_c  = {"BUY (Long)": "green", "SELL (Short)": "red", "EXIT (Flat)": "yellow"}.get(entry["action"], "white")
        rsi_pnl_c = _pnl_color(entry["rsi_pnl"])
        alpha_c   = _pnl_color(entry["alpha"])
        ind_str   = f"{entry['indicator']:.1f}" if entry.get("indicator") is not None else "—"

        t.add_row(
            str(idx),
            entry["date"],
            f"[{action_c}]{entry['action']}[/{action_c}]",
            f"${entry['price']:,.2f}",
            ind_str,
            str(entry.get("shares", "—")),
            f"${entry.get('cash', 0):,.2f}",
            f"${entry.get('cost', 0):,.2f}",
            f"[{rsi_pnl_c}]${entry['rsi_equity']:,.2f}[/{rsi_pnl_c}]",
            f"${entry['bh_equity']:,.2f}",
            f"[{alpha_c}]${entry['alpha']:+,.2f}[/{alpha_c}]",
        )
    console.print(t)


def _compute_bh_oos_pnl(
    symbol: str,
    curr_date: str,
    oos_days: int,
    notional: float = NOTIONAL,
    slippage: float = SLIPPAGE,
) -> dict:
    """Share-based Buy & Hold for the OOS window. Returns final_equity for chaining."""
    data = _macd_load_price(symbol, freq="D")
    curr_dt = pd.to_datetime(curr_date)
    data = data[data["Date"] <= curr_dt].copy()
    closes = data["Close"].reset_index(drop=True)

    oos_start  = len(closes) - oos_days
    oos_closes = closes.iloc[oos_start:].reset_index(drop=True)

    if len(oos_closes) < 2:
        return {"pnl_pct": 0.0, "pnl_dollar": 0.0, "max_dd_pct": 0.0, "final_equity": notional}

    price0 = float(oos_closes.iloc[0])
    shares = math.floor(notional / (price0 * (1 + slippage)))
    cash   = notional - shares * price0 * (1 + slippage)

    equity = cash + shares * oos_closes.astype(float)
    final_equity = float(equity.iloc[-1])
    pnl_dollar   = final_equity - notional
    pnl_pct      = (pnl_dollar / notional) * 100 if notional > 0 else 0.0

    rolling_max = equity.cummax()
    dd = (equity - rolling_max) / rolling_max
    max_dd_pct = float(abs(dd.min()) * 100) if len(dd) else 0.0

    return {
        "pnl_pct":      round(pnl_pct, 2),
        "pnl_dollar":   round(pnl_dollar, 2),
        "max_dd_pct":   round(max_dd_pct, 2),
        "final_equity": round(final_equity, 2),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — OOS walk-forward sweep (both MACD and RSI across N dates)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.timeout(1200)
def test_oos_sweep():
    """
    Run MACD and RSI optimizers across OOS_N_DATES with continuous equity
    (no reset between windows) and Kelly position sizing.
    """
    _setup_config()
    sweep_dates = _business_dates(OOS_START, OOS_END, OOS_N_DATES)

    console.print()
    console.rule(f"[bold yellow]OOS Walk-Forward Sweep vs Benchmark — {TICKER}[/bold yellow]")
    console.print(f"  [dim]Notional: ${NOTIONAL:,.0f}  |  Slippage: {SLIPPAGE*100:.1f}bps/side  |  Sizing: Half-Kelly[/dim]")

    macd_rows, rsi_rows = [], []
    rsi_trade_logs = []

    # Continuous equity accumulators
    macd_equity = NOTIONAL
    rsi_equity  = NOTIONAL
    bh_equity   = NOTIONAL

    for i, curr_date in enumerate(sweep_dates, 1):
        console.print(f"  [dim]({i}/{len(sweep_dates)})[/dim] Processing [bold]{curr_date}[/bold]  "
                       f"[dim]MACD=${macd_equity:,.0f}  RSI=${rsi_equity:,.0f}  B&H=${bh_equity:,.0f}[/dim]")

        # B&H for this window (chains from previous equity)
        bh_p = _compute_bh_oos_pnl(TICKER, curr_date, oos_days=OOS_DAYS, notional=bh_equity)
        bh_equity = bh_p["final_equity"]

        # MACD (chains from previous equity)
        try:
            mr = run_macd_optimizer(TICKER, curr_date, is_days=IS_DAYS, oos_days=OOS_DAYS)
            mp = _compute_macd_oos_pnl(
                TICKER, curr_date, mr["optimal_fast"], mr["optimal_slow"],
                mr["optimal_signal"], OOS_DAYS, notional=macd_equity, bh_notional=bh_equity,
            )
            macd_equity = mp["final_equity"]
            macd_rows.append({
                "date": curr_date,
                "fast": mr["optimal_fast"], "slow": mr["optimal_slow"], "signal": mr["optimal_signal"],
                "oos_sharpe": mr["oos_sharpe"], "pnl_pct": mp["pnl_pct"], "pnl_dollar": mp["pnl_dollar"],
                "max_dd": mp["max_dd_pct"], "bh_pnl_pct": bh_p["pnl_pct"],
                "alpha": mp["pnl_pct"] - bh_p["pnl_pct"],
                "confidence": mr["confidence"], "is_sharpe": mr["is_sharpe"], "win_rate": mp["win_rate"],
                "kelly_f": mp["kelly_f"], "running_equity": macd_equity,
            })
        except Exception as e:
            macd_rows.append({"date": curr_date, "error": str(e)})

        # RSI (chains from previous equity)
        try:
            rr = run_rsi_optimizer(TICKER, curr_date, is_days=IS_DAYS, oos_days=OOS_DAYS)
            rp = _compute_rsi_oos_pnl(
                TICKER, curr_date, rr["optimal_period"], rr["optimal_upper"],
                rr["optimal_lower"], OOS_DAYS, notional=rsi_equity, bh_notional=bh_equity,
            )
            rsi_equity = rp["final_equity"]
            rsi_rows.append({
                "date": curr_date,
                "period": rr["optimal_period"], "upper": rr["optimal_upper"], "lower": rr["optimal_lower"],
                "oos_sharpe": rr["oos_sharpe"], "pnl_pct": rp["pnl_pct"], "pnl_dollar": rp["pnl_dollar"],
                "max_dd": rp["max_dd_pct"], "bh_pnl_pct": bh_p["pnl_pct"],
                "alpha": rp["pnl_pct"] - bh_p["pnl_pct"],
                "confidence": rr["confidence"], "is_sharpe": rr["is_sharpe"], "win_rate": rp["win_rate"],
                "n_trades": rp["n_trades"], "kelly_f": rp["kelly_f"], "running_equity": rsi_equity,
            })
            rsi_trade_logs.append({
                "date": curr_date,
                "params": f"{rr['optimal_period']}/{rr['optimal_upper']}/{rr['optimal_lower']}",
                "kelly_f": rp["kelly_f"],
                "trade_log": rp["trade_log"],
            })
        except Exception as e:
            rsi_rows.append({"date": curr_date, "error": str(e)})

    # Display results
    _print_sweep_table(f"MACD vs B&H — {TICKER}", macd_rows, [("Params", lambda r: f"{r['fast']},{r['slow']},{r['signal']}")])
    _print_summary_panel("MACD", macd_rows)

    _print_sweep_table(f"RSI vs B&H — {TICKER}", rsi_rows, [("Params", lambda r: f"{r['period']}/{r['upper']}/{r['lower']}")])
    _print_summary_panel("RSI", rsi_rows)

    # RSI per-action-day detail
    console.print()
    console.rule("[bold magenta]RSI Action-Day Detail — RSI Algo vs Buy & Hold[/bold magenta]")
    for log_entry in rsi_trade_logs:
        _print_rsi_trade_log(
            log_entry["date"], log_entry["trade_log"],
            log_entry["params"], log_entry.get("kelly_f", 0),
        )
        console.print()

    # Assertions
    valid_macd = [r for r in macd_rows if "error" not in r]
    valid_rsi  = [r for r in rsi_rows  if "error" not in r]

    assert len(valid_macd) > 0, "All MACD optimizer runs failed — check data / config"
    assert len(valid_rsi)  > 0, "All RSI optimizer runs failed — check data / config"

    _save_results(
        f"{TICKER}_oos_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        {"ticker": TICKER, "oos_start": OOS_START, "oos_end": OOS_END,
         "is_days": IS_DAYS, "oos_days": OOS_DAYS, "notional": NOTIONAL,
         "slippage": SLIPPAGE, "sizing": "half-kelly",
         "final_macd_equity": macd_equity, "final_rsi_equity": rsi_equity,
         "final_bh_equity": bh_equity,
         "macd": macd_rows, "rsi": rsi_rows,
         "rsi_trade_logs": rsi_trade_logs},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Display helpers used by test_oos_sweep
# ─────────────────────────────────────────────────────────────────────────────

def _print_sweep_table(title: str, rows: list[dict], param_cols: list):
    t = Table(title=title, box=box.ROUNDED, show_lines=True)
    t.add_column("Date",        style="white",  min_width=12)
    for col_name, _ in param_cols:
        t.add_column(col_name,  style="cyan",   min_width=14)
    t.add_column("Kelly f",     style="dim",    min_width=8)
    t.add_column("IS Sharpe",   style="white",  min_width=10)
    t.add_column("OOS Sharpe",  min_width=10)
    t.add_column("Confidence",  min_width=10)
    t.add_column("Win P&L %",   min_width=10)
    t.add_column("Window P&L $", min_width=12)
    t.add_column("Running $",   min_width=12)
    t.add_column("Max DD",      min_width=8)

    for r in rows:
        if "error" in r:
            n_extra = 7 + len(param_cols)
            t.add_row(r["date"], *["—"] * n_extra, f"[red]ERR[/red]")
            continue

        conf_c   = {"HIGH": "green", "MEDIUM": "yellow", "LOW": "red"}.get(r["confidence"], "white")
        oos_c    = "green" if r["oos_sharpe"] > 0 else "red"
        pnl_c    = _pnl_color(r["pnl_dollar"])
        max_dd_c = "red" if r["max_dd"] > 10 else "yellow" if r["max_dd"] > 5 else "green"
        run_eq   = r.get("running_equity", 0)
        run_c    = _pnl_color(run_eq - NOTIONAL)
        kelly    = r.get("kelly_f", 0)
        kelly_s  = f"{kelly:.2f}" if kelly > 0 else "SKIP"

        t.add_row(
            r["date"],
            *[fn(r) for _, fn in param_cols],
            kelly_s,
            f"{r['is_sharpe']:+.4f}",
            f"[{oos_c}]{r['oos_sharpe']:+.4f}[/{oos_c}]",
            f"[{conf_c}]{r['confidence']}[/{conf_c}]",
            f"[{pnl_c}]{r['pnl_pct']:+.2f}%[/{pnl_c}]",
            f"[{pnl_c}]${r['pnl_dollar']:+,.2f}[/{pnl_c}]",
            f"[{run_c}]${run_eq:,.2f}[/{run_c}]",
            f"[{max_dd_c}]{r['max_dd']:.2f}%[/{max_dd_c}]",
        )

    console.print(t)


def _print_summary_panel(label: str, rows: list[dict]):
    valid = [r for r in rows if "error" not in r]
    if not valid:
        return

    total_pnl      = sum(r["pnl_dollar"] for r in valid)
    total_alpha    = sum(r["alpha"] for r in valid)
    beat_bh        = sum(1 for r in valid if r["alpha"] > 0)
    beat_rate      = (beat_bh / len(valid)) * 100
    avg_oos_sharpe = np.mean([r["oos_sharpe"] for r in valid])
    avg_is_sharpe  = np.mean([r["is_sharpe"]  for r in valid])
    final_eq       = valid[-1].get("running_equity", NOTIONAL + total_pnl)
    total_ret      = (final_eq / NOTIONAL - 1) * 100

    pnl_c   = _pnl_color(total_pnl)
    alpha_c = _pnl_color(total_alpha)
    ret_c   = _pnl_color(total_ret)

    text = Text()
    text.append(f" {label} Summary  ({len(valid)}/{len(rows)} windows)\n\n", style="bold white")
    text.append(f"  Final Equity      : ", style="white")
    text.append(f"${final_eq:,.2f}", style=ret_c)
    text.append(f"  ({total_ret:+.1f}% total return)\n", style="dim")
    text.append(f"  Cumulative P&L    : ", style="white")
    text.append(f"${total_pnl:+,.2f}\n", style=pnl_c)
    text.append(f"  Beats B&H Rate    : ", style="white")
    text.append(f"{beat_rate:.1f}% ", style="green" if beat_rate > 50 else "yellow")
    text.append(f"({beat_bh}/{len(valid)} windows)\n", style="dim")
    text.append(f"  Avg IS  Sharpe    : {avg_is_sharpe:+.4f}\n", style="white")
    text.append(f"  Avg OOS Sharpe    : ", style="white")
    text.append(f"{avg_oos_sharpe:+.4f}\n", style="green" if avg_oos_sharpe > 0 else "red")

    console.print(Panel(text, border_style="cyan", padding=(0, 2)))
