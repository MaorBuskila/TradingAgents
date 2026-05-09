"""
rsi_dt_cache.py
---------------
Persistent SQLite cache for the DT-filtered RSI optimizer (WFO + Triple
Barrier + two-model ensemble).

Separate table from rsi_params_cache because the DT variant produces a
fundamentally different output: dynamic probability thresholds and OOS
track-record stats rather than grid-search RSI parameters.

Table: rsi_dt_params_cache
"""

import sqlite3
import urllib.parse
from pathlib import Path
from typing import Any, Dict, Optional


def _project_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path(__file__).resolve().parent


_DB_PATH = _project_root() / "data" / "rsi_cache.db"


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
        CREATE TABLE IF NOT EXISTS rsi_dt_params_cache (
            ticker          TEXT PRIMARY KEY,
            threshold_a     REAL    NOT NULL,
            threshold_b     REAL    NOT NULL,
            oos_sharpe      REAL    NOT NULL,
            oos_hit_rate    REAL    NOT NULL,
            oos_trade_count INTEGER NOT NULL,
            confidence      TEXT    NOT NULL,
            training_days   INTEGER,
            step_days       INTEGER,
            n_slides        INTEGER,
            tp_mult         REAL,
            sl_mult         REAL,
            label_horizon   INTEGER,
            avg_oos_acc     REAL    DEFAULT 0.0,
            optimized_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Idempotent migration for existing DBs that predate avg_oos_acc
    try:
        conn.execute("ALTER TABLE rsi_dt_params_cache ADD COLUMN avg_oos_acc REAL DEFAULT 0.0")
    except Exception:
        pass  # column already exists
    conn.commit()
    conn.close()


_ensure_table()


def get_rsi_dt_params(ticker: str) -> Optional[Dict[str, Any]]:
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM rsi_dt_params_cache WHERE ticker = ?",
        (ticker.upper().strip(),),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def upsert_rsi_dt_params(
    ticker: str,
    threshold_a: float,
    threshold_b: float,
    oos_sharpe: float,
    oos_hit_rate: float,
    oos_trade_count: int,
    confidence: str,
    training_days: int,
    step_days: int,
    n_slides: int,
    tp_mult: float,
    sl_mult: float,
    label_horizon: int,
    avg_oos_acc: float = 0.0,
) -> None:
    conn = _get_conn()
    conn.execute("""
        INSERT INTO rsi_dt_params_cache
            (ticker, threshold_a, threshold_b, oos_sharpe, oos_hit_rate,
             oos_trade_count, confidence, training_days, step_days, n_slides,
             tp_mult, sl_mult, label_horizon, avg_oos_acc, optimized_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(ticker) DO UPDATE SET
            threshold_a     = excluded.threshold_a,
            threshold_b     = excluded.threshold_b,
            oos_sharpe      = excluded.oos_sharpe,
            oos_hit_rate    = excluded.oos_hit_rate,
            oos_trade_count = excluded.oos_trade_count,
            confidence      = excluded.confidence,
            training_days   = excluded.training_days,
            step_days       = excluded.step_days,
            n_slides        = excluded.n_slides,
            tp_mult         = excluded.tp_mult,
            sl_mult         = excluded.sl_mult,
            label_horizon   = excluded.label_horizon,
            avg_oos_acc     = excluded.avg_oos_acc,
            optimized_at    = CURRENT_TIMESTAMP
    """, (
        ticker.upper().strip(),
        threshold_a, threshold_b, oos_sharpe, oos_hit_rate,
        oos_trade_count, confidence, training_days, step_days, n_slides,
        tp_mult, sl_mult, label_horizon, avg_oos_acc,
    ))
    conn.commit()
    conn.close()


def delete_rsi_dt_params(ticker: str) -> bool:
    conn = _get_conn()
    cursor = conn.execute(
        "DELETE FROM rsi_dt_params_cache WHERE ticker = ?",
        (ticker.upper().strip(),),
    )
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    return affected > 0
