"""
sniper_optimizer_algo.py
========================
Thin wrapper — delegates EMA stack optimization to ema_optimizer_algo.

Previously contained a custom R-multiple simulation (grid over EMA params +
min_score + sl_mult + vol_mult). That logic is replaced by the standard
annualized-Sharpe approach used by rsi_optimizer_algo and macd_optimizer_algo.

Kept as a separate module so existing callers (API, tests) don't need updating.
"""

from __future__ import annotations

from tradingagents.quant_ml.optimizers.ema_optimizer_algo import run_algo_optimizer


__all__ = ["run_algo_optimizer"]
