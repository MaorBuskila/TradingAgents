"""Unit tests for Sniper classical WFO (small grid, no heavy parallel)."""

from __future__ import annotations

import pytest

from tradingagents.dataflows.config import set_config
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.quant_ml.optimizers.sniper_optimizer_algo import run_algo_optimizer


def _cfg():
    cfg = DEFAULT_CONFIG.copy()
    set_config(cfg)


@pytest.mark.integration
@pytest.mark.timeout(300)
def test_sniper_algo_small_grid_spy():
    _cfg()
    out = run_algo_optimizer(
        "SPY",
        "2025-06-01",
        is_days=120,
        oos_days=40,
        fast_grid=[9],
        slow_grid=[21],
        trend_grid=[55],
        min_score_grid=[5.0, 6.0],
        sl_mult_grid=[1.5],
        vol_mult_grid=[1.2],
    )
    assert out["optimal_ema_fast"] == 9
    assert out["optimal_ema_slow"] == 21
    assert out["optimal_ema_trend"] == 55
    assert out["combos_tested"] == 2
    assert out["oos_sharpe"] == out["oos_sharpe"]  # finite
    assert abs(float(out["oos_sharpe"])) < 1e4


def test_grid_requires_slow_gt_fast():
    _cfg()
    with pytest.raises(ValueError, match="No valid EMA"):
        run_algo_optimizer(
            "SPY",
            "2025-06-01",
            is_days=120,
            oos_days=40,
            fast_grid=[30],
            slow_grid=[21],
            trend_grid=[55],
            min_score_grid=[5.0],
            sl_mult_grid=[1.5],
            vol_mult_grid=[1.2],
        )
