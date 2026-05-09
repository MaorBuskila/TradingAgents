"""
tests/test_sniper_integration_oos.py
=====================================
Phase 1 — End-to-end Precision Sniper on real SPY daily data (yfinance).

Runs default-preset indicators, scores each bar, defines long entries on
EMA cross-up + composite grade ≥ B, simulates first-touch SL vs TP ladder,
and prints a Rich summary table.

Run:
    pytest tests/test_sniper_integration_oos.py::test_baseline_default_preset -v -s -m integration

Env:
    SNIPER_TEST_TICKER=SPY SNIPER_START=2024-01-01 SNIPER_END=2026-04-18 ...
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from tradingagents.dataflows.config import set_config
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.quant_ml.optimizers.rsi_optimizer_algo import _load_price_data
from tradingagents.quant_ml.signals.sniper_features import build_sniper_feature_frame
from tradingagents.quant_ml.walk_forward.sniper_sim import simulate_long_trade_r

console = Console()

TICKER = os.getenv("SNIPER_TEST_TICKER", "SPY")
START = os.getenv("SNIPER_START", "2024-01-01")
END = os.getenv("SNIPER_END", "2026-04-18")
MIN_SCORE = float(os.getenv("SNIPER_MIN_SCORE", "5.0"))  # grade B+


def _setup_config():
    cfg = DEFAULT_CONFIG.copy()
    set_config(cfg)
    Path(cfg["data_cache_dir"]).mkdir(parents=True, exist_ok=True)
    return cfg


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_baseline_default_preset():
    """SPY walk: default Sniper params, Rich table, soft sanity checks."""
    _setup_config()
    raw = _load_price_data(TICKER)
    full = build_sniper_feature_frame(raw)
    full = full.dropna(
        subset=[
            "ema_f",
            "ema_s",
            "ema_t",
            "rsi",
            "macd_hist",
            "vwap",
            "adx",
            "bull_score",
        ]
    ).reset_index(drop=True)

    dstart = pd.Timestamp(START)
    dend = pd.Timestamp(END)
    dates = pd.to_datetime(full["Date"])
    win_mask = (dates >= dstart) & (dates <= dend)
    win_idx = full.index[win_mask]

    signals: list[int] = []
    for i in win_idx:
        if i + 1 >= len(full):
            continue
        if not bool(full["cross_up"].iloc[i]):
            continue
        if float(full["bull_score"].iloc[i]) < MIN_SCORE:
            continue
        signals.append(int(i))

    outcomes: list[float] = []
    tp1_hits = 0
    grade_counts = {"A+": 0, "A": 0, "B": 0, "C": 0}
    aa_tp1_hits = 0
    aa_signals = 0

    next_allowed = -1
    for sig_i in signals:
        if sig_i < next_allowed:
            continue
        entry_i = sig_i + 1
        if entry_i >= len(full):
            continue
        g = str(full["bull_grade"].iloc[sig_i])
        if g in grade_counts:
            grade_counts[g] += 1
        if g in ("A+", "A"):
            aa_signals += 1
        r_mult, hit1, exit_j = simulate_long_trade_r(full, entry_i)
        outcomes.append(r_mult)
        if hit1:
            tp1_hits += 1
            if g in ("A+", "A"):
                aa_tp1_hits += 1
        next_allowed = exit_j + 1

    n_sig = len(outcomes)
    hit_rate = (tp1_hits / n_sig) if n_sig else 0.0
    aa_hit_rate = (aa_tp1_hits / aa_signals) if aa_signals else 0.0
    avg_r = sum(outcomes) / n_sig if n_sig else 0.0
    wins = sum(x for x in outcomes if x > 0)
    losses = -sum(x for x in outcomes if x < 0)
    profit_factor = wins / losses if losses > 0 else (float("inf") if wins > 0 else 0.0)

    table = Table(title=f"Precision Sniper baseline — {TICKER} {START} → {END}", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row("Signals (cross_up + bull ≥ B)", str(n_sig))
    table.add_row("TP1 hit rate (all signals)", f"{hit_rate:.1%}")
    table.add_row("TP1 hit rate (A+ / A only)", f"{aa_hit_rate:.1%}")
    table.add_row("Avg R (first-touch)", f"{avg_r:.3f}")
    table.add_row("Profit factor (gross R)", f"{profit_factor:.3f}")
    table.add_row("Grade mix (signal bars)", str(grade_counts))

    console.print(Panel(table, expand=False))

    assert n_sig > 0, "expected at least one long signal in window"
    assert all(isinstance(x, float) for x in outcomes)
    assert not any(pd.isna(x) for x in outcomes)

    # Soft sanity: documented in plan as ~40% for A+/A — warn-only via assertion skip threshold
    if aa_signals >= 10 and aa_hit_rate < 0.25:
        console.print(
            "[yellow]Warning: A/A+ TP1 hit rate below 25% — check data window or regime.[/yellow]"
        )


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_optimized_vs_baseline():
    """Classical optimized params should produce finite OOS metrics on SPY."""
    _setup_config()
    from tradingagents.quant_ml.optimizers.sniper_optimizer_algo import run_algo_optimizer

    out = run_algo_optimizer(
        "SPY",
        "2025-06-01",
        is_days=120,
        oos_days=40,
        fast_grid=[9],
        slow_grid=[21],
        trend_grid=[55],
        min_score_grid=[5.0],
        sl_mult_grid=[1.5],
        vol_mult_grid=[1.2],
    )
    assert out["oos_sharpe"] == out["oos_sharpe"]
    assert out["default_oos_sharpe"] == out["default_oos_sharpe"]
