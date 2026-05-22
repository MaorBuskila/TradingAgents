"""
ema_cache.py
------------
Persistent SQLite cache for optimized EMA stack parameters per ticker.

Each ticker stores one row — the latest optimization result.
On cache hit the sniper pipeline skips the EMA optimizer, saving compute time.
Force a refresh by calling upsert_ema_params() directly.

Table: ema_params_cache
"""

import sqlite3
import os
import urllib.parse
from pathlib import Path
from typing import Optional, Dict, Any

def _project_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path(__file__).resolve().parent


_DB_PATH = _project_root() / "data" / "ema_cache.db"


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
        CREATE TABLE IF NOT EXISTS ema_params_cache (
            ticker          TEXT PRIMARY KEY,
            optimal_fast    INTEGER NOT NULL,
            optimal_slow    INTEGER NOT NULL,
            optimal_trend   INTEGER NOT NULL,
            is_sharpe       REAL    NOT NULL,
            oos_sharpe      REAL    NOT NULL,
            confidence      TEXT    NOT NULL,
            combos_tested   INTEGER,
            training_days   INTEGER,
            test_days       INTEGER,
            optimized_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


_ensure_table()


def get_ema_params(ticker: str) -> Optional[Dict[str, Any]]:
    """Return cached EMA params for ticker, or None if not found."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM ema_params_cache WHERE ticker = ?",
        (ticker.upper().strip(),)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def upsert_ema_params(
    ticker: str,
    optimal_fast: int,
    optimal_slow: int,
    optimal_trend: int,
    is_sharpe: float,
    oos_sharpe: float,
    confidence: str,
    combos_tested: int = 0,
    training_days: int = 180,
    test_days: int = 90,
) -> None:
    """Insert or replace EMA optimization result for a ticker."""
    conn = _get_conn()
    conn.execute("""
        INSERT INTO ema_params_cache
            (ticker, optimal_fast, optimal_slow, optimal_trend,
             is_sharpe, oos_sharpe, confidence, combos_tested,
             training_days, test_days, optimized_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(ticker) DO UPDATE SET
            optimal_fast   = excluded.optimal_fast,
            optimal_slow   = excluded.optimal_slow,
            optimal_trend  = excluded.optimal_trend,
            is_sharpe      = excluded.is_sharpe,
            oos_sharpe     = excluded.oos_sharpe,
            confidence     = excluded.confidence,
            combos_tested  = excluded.combos_tested,
            training_days  = excluded.training_days,
            test_days      = excluded.test_days,
            optimized_at   = CURRENT_TIMESTAMP
    """, (
        ticker.upper().strip(),
        optimal_fast, optimal_slow, optimal_trend,
        is_sharpe, oos_sharpe, confidence, combos_tested,
        training_days, test_days,
    ))
    conn.commit()
    conn.close()


def list_all_cached_tickers() -> list:
    """Return all tickers with cached EMA params."""
    conn = _get_conn()
    rows = conn.execute("""
        SELECT ticker, optimal_fast, optimal_slow, optimal_trend,
               oos_sharpe, confidence, optimized_at
        FROM ema_params_cache
        ORDER BY ticker ASC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_ema_params(ticker: str) -> bool:
    """Remove cached params for a ticker. Returns True if a row was deleted."""
    conn = _get_conn()
    cursor = conn.execute(
        "DELETE FROM ema_params_cache WHERE ticker = ?",
        (ticker.upper().strip(),)
    )
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    return affected > 0
