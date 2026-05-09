"""
sniper_cache.py
---------------
SQLite cache for Precision Sniper classical WFO params + DT thresholds.

DB: data/sniper_cache.db
Table: sniper_params_cache
"""

from __future__ import annotations

import sqlite3
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Optional


def _project_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path(__file__).resolve().parent


_DB_PATH = _project_root() / "data" / "sniper_cache.db"


def _get_conn() -> sqlite3.Connection:
    encoded = urllib.parse.quote(str(_DB_PATH), safe="/:")
    uri = f"file:{encoded}?nolock=1"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_table() -> None:
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sniper_params_cache (
            ticker              TEXT PRIMARY KEY,
            optimal_ema_fast    INTEGER NOT NULL,
            optimal_ema_slow      INTEGER NOT NULL,
            optimal_ema_trend     INTEGER NOT NULL,
            optimal_min_score     REAL    NOT NULL,
            optimal_sl_mult       REAL    NOT NULL,
            optimal_vol_mult      REAL    NOT NULL,
            classical_is_sharpe   REAL    NOT NULL,
            classical_oos_sharpe REAL   NOT NULL,
            confidence            TEXT    NOT NULL,
            threshold_a           REAL,
            threshold_b           REAL,
            dt_oos_sharpe         REAL,
            dt_oos_hit_rate       REAL,
            training_days         INTEGER,
            test_days             INTEGER,
            optimizer_provider    TEXT,
            optimized_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


_ensure_table()


def get_sniper_params(ticker: str) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM sniper_params_cache WHERE ticker = ?",
        (ticker.upper().strip(),),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def upsert_sniper_params(
    ticker: str,
    optimal_ema_fast: int,
    optimal_ema_slow: int,
    optimal_ema_trend: int,
    optimal_min_score: float,
    optimal_sl_mult: float,
    optimal_vol_mult: float,
    classical_is_sharpe: float,
    classical_oos_sharpe: float,
    confidence: str,
    threshold_a: float | None = None,
    threshold_b: float | None = None,
    dt_oos_sharpe: float | None = None,
    dt_oos_hit_rate: float | None = None,
    training_days: int = 180,
    test_days: int = 90,
    optimizer_provider: str = "algo",
) -> None:
    conn = _get_conn()
    conn.execute("""
        INSERT INTO sniper_params_cache (
            ticker, optimal_ema_fast, optimal_ema_slow, optimal_ema_trend,
            optimal_min_score, optimal_sl_mult, optimal_vol_mult,
            classical_is_sharpe, classical_oos_sharpe, confidence,
            threshold_a, threshold_b, dt_oos_sharpe, dt_oos_hit_rate,
            training_days, test_days, optimizer_provider, optimized_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(ticker) DO UPDATE SET
            optimal_ema_fast     = excluded.optimal_ema_fast,
            optimal_ema_slow     = excluded.optimal_ema_slow,
            optimal_ema_trend    = excluded.optimal_ema_trend,
            optimal_min_score    = excluded.optimal_min_score,
            optimal_sl_mult      = excluded.optimal_sl_mult,
            optimal_vol_mult     = excluded.optimal_vol_mult,
            classical_is_sharpe  = excluded.classical_is_sharpe,
            classical_oos_sharpe = excluded.classical_oos_sharpe,
            confidence           = excluded.confidence,
            threshold_a          = excluded.threshold_a,
            threshold_b          = excluded.threshold_b,
            dt_oos_sharpe        = excluded.dt_oos_sharpe,
            dt_oos_hit_rate      = excluded.dt_oos_hit_rate,
            training_days        = excluded.training_days,
            test_days            = excluded.test_days,
            optimizer_provider   = excluded.optimizer_provider,
            optimized_at         = CURRENT_TIMESTAMP
    """, (
        ticker.upper().strip(),
        optimal_ema_fast, optimal_ema_slow, optimal_ema_trend,
        optimal_min_score, optimal_sl_mult, optimal_vol_mult,
        classical_is_sharpe, classical_oos_sharpe, confidence,
        threshold_a, threshold_b, dt_oos_sharpe, dt_oos_hit_rate,
        training_days, test_days, optimizer_provider,
    ))
    conn.commit()
    conn.close()


def list_all_sniper_cached() -> list:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM sniper_params_cache ORDER BY ticker ASC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_sniper_params(ticker: str) -> bool:
    conn = _get_conn()
    cur = conn.execute(
        "DELETE FROM sniper_params_cache WHERE ticker = ?",
        (ticker.upper().strip(),),
    )
    conn.commit()
    n = cur.rowcount
    conn.close()
    return n > 0
