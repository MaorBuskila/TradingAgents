"""
sniper_cache.py
---------------
SQLite cache for Precision Sniper DT ensemble thresholds and performance metrics.

EMA stack params (fast/slow/trend) are now stored in ema_cache.db via ema_cache.py.
This table only holds the DT-layer outputs produced by sniper_dt_optimizer.

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


_CURRENT_COLUMNS = {
    "ticker", "threshold_a", "threshold_b",
    "dt_oos_sharpe", "dt_oos_hit_rate",
    "training_days", "test_days", "optimized_at",
}

_CREATE_DDL = """
    CREATE TABLE IF NOT EXISTS sniper_params_cache (
        ticker          TEXT PRIMARY KEY,
        threshold_a     REAL,
        threshold_b     REAL,
        dt_oos_sharpe   REAL,
        dt_oos_hit_rate REAL,
        training_days   INTEGER,
        test_days       INTEGER,
        optimized_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
"""


def _ensure_table() -> None:
    conn = _get_conn()
    conn.execute(_CREATE_DDL)
    conn.commit()

    # Migrate: if the table has legacy EMA columns (moved to ema_cache.db),
    # recreate it with only the current schema to avoid NOT NULL violations.
    cols = {row[1] for row in conn.execute("PRAGMA table_info(sniper_params_cache)")}
    if not cols.issubset(_CURRENT_COLUMNS):
        conn.execute("ALTER TABLE sniper_params_cache RENAME TO sniper_params_cache_old")
        conn.execute(_CREATE_DDL.replace("IF NOT EXISTS ", ""))
        shared = cols & _CURRENT_COLUMNS - {"optimized_at"}
        col_list = ", ".join(shared)
        conn.execute(f"""
            INSERT INTO sniper_params_cache ({col_list})
            SELECT {col_list} FROM sniper_params_cache_old
        """)
        conn.execute("DROP TABLE sniper_params_cache_old")
        conn.commit()

    conn.close()


_ensure_table()


def get_sniper_params(ticker: str) -> Optional[Dict[str, Any]]:
    """Return cached DT params for ticker, or None if not found."""
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM sniper_params_cache WHERE ticker = ?",
        (ticker.upper().strip(),),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def upsert_sniper_params(
    ticker: str,
    threshold_a: float | None = None,
    threshold_b: float | None = None,
    dt_oos_sharpe: float | None = None,
    dt_oos_hit_rate: float | None = None,
    training_days: int = 1000,
    test_days: int = 20,
) -> None:
    """Insert or replace DT optimization result for a ticker."""
    conn = _get_conn()
    conn.execute("""
        INSERT INTO sniper_params_cache (
            ticker, threshold_a, threshold_b,
            dt_oos_sharpe, dt_oos_hit_rate,
            training_days, test_days, optimized_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(ticker) DO UPDATE SET
            threshold_a     = excluded.threshold_a,
            threshold_b     = excluded.threshold_b,
            dt_oos_sharpe   = excluded.dt_oos_sharpe,
            dt_oos_hit_rate = excluded.dt_oos_hit_rate,
            training_days   = excluded.training_days,
            test_days       = excluded.test_days,
            optimized_at    = CURRENT_TIMESTAMP
    """, (
        ticker.upper().strip(),
        threshold_a, threshold_b,
        dt_oos_sharpe, dt_oos_hit_rate,
        training_days, test_days,
    ))
    conn.commit()
    conn.close()


def list_all_sniper_cached() -> list:
    """Return all tickers with cached DT params."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM sniper_params_cache ORDER BY ticker ASC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_sniper_params(ticker: str) -> bool:
    """Remove cached params for a ticker. Returns True if a row was deleted."""
    conn = _get_conn()
    cur = conn.execute(
        "DELETE FROM sniper_params_cache WHERE ticker = ?",
        (ticker.upper().strip(),),
    )
    conn.commit()
    n = cur.rowcount
    conn.close()
    return n > 0
