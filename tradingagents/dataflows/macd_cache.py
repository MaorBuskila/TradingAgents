"""
macd_cache.py
-------------
Persistent SQLite cache for optimized MACD parameters per ticker.

Each ticker stores one row — the latest optimization result.
On cache hit the graph skips the optimizer entirely.
Force a refresh by calling upsert_macd_params() directly or setting
force_reoptimize=True in the optimizer node.

Table: macd_params_cache
"""

import sqlite3
import os
import urllib.parse
from pathlib import Path
from typing import Optional, Dict, Any

# Dedicated MACD cache DB — lives in data/ at project root.
def _project_root() -> Path:
    """Walk up from this file until pyproject.toml is found (project root)."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path(__file__).resolve().parent  # fallback

_DB_PATH = _project_root() / "data" / "macd_cache.db"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    """
    Open a connection to the MACD cache SQLite DB.

    Uses a URI connection with nolock=1 + journal_mode=OFF so it works on
    both standard macOS/Linux filesystems and FUSE/virtiofs mounts (e.g.
    macOS VM shares, Docker volume mounts) where POSIX byte-range locking
    is not supported.
    """
    encoded = urllib.parse.quote(str(_DB_PATH), safe="/:")
    uri = f"file:{encoded}?nolock=1"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_table() -> None:
    """Create macd_params_cache table if it doesn't exist (idempotent)."""
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS macd_params_cache (
            ticker              TEXT PRIMARY KEY,
            optimal_fast        INTEGER NOT NULL,
            optimal_slow        INTEGER NOT NULL,
            optimal_signal      INTEGER NOT NULL,
            oos_sharpe          REAL    NOT NULL,
            is_sharpe           REAL    NOT NULL,
            confidence          TEXT    NOT NULL,
            regime              TEXT,
            training_days       INTEGER,
            test_days           INTEGER,
            optimizer_provider  TEXT,
            model_used          TEXT,
            optimized_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


# Run on import so the table always exists
_ensure_table()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_macd_params(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Load cached MACD params for a ticker.

    Returns a dict with keys:
        optimal_fast, optimal_slow, optimal_signal,
        oos_sharpe, is_sharpe, confidence, regime,
        training_days, test_days, optimizer_provider, model_used, optimized_at

    Returns None if no cache entry exists.
    """
    conn = _get_conn()
    row = conn.execute(
        "SELECT * FROM macd_params_cache WHERE ticker = ?",
        (ticker.upper().strip(),)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def upsert_macd_params(
    ticker: str,
    optimal_fast: int,
    optimal_slow: int,
    optimal_signal: int,
    oos_sharpe: float,
    is_sharpe: float,
    confidence: str,
    regime: str = "",
    training_days: int = 180,
    test_days: int = 90,
    optimizer_provider: str = "algo",
    model_used: str = "",
) -> None:
    """
    Insert or replace MACD optimization result for a ticker.
    optimized_at is automatically set to current UTC timestamp.
    """
    conn = _get_conn()
    conn.execute("""
        INSERT INTO macd_params_cache
            (ticker, optimal_fast, optimal_slow, optimal_signal,
             oos_sharpe, is_sharpe, confidence, regime,
             training_days, test_days, optimizer_provider, model_used,
             optimized_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(ticker) DO UPDATE SET
            optimal_fast       = excluded.optimal_fast,
            optimal_slow       = excluded.optimal_slow,
            optimal_signal     = excluded.optimal_signal,
            oos_sharpe         = excluded.oos_sharpe,
            is_sharpe          = excluded.is_sharpe,
            confidence         = excluded.confidence,
            regime             = excluded.regime,
            training_days      = excluded.training_days,
            test_days          = excluded.test_days,
            optimizer_provider = excluded.optimizer_provider,
            model_used         = excluded.model_used,
            optimized_at       = CURRENT_TIMESTAMP
    """, (
        ticker.upper().strip(),
        optimal_fast, optimal_slow, optimal_signal,
        oos_sharpe, is_sharpe, confidence, regime,
        training_days, test_days, optimizer_provider, model_used,
    ))
    conn.commit()
    conn.close()


def list_all_cached_tickers() -> list:
    """Return all tickers that have cached MACD params, with their metadata."""
    conn = _get_conn()
    rows = conn.execute("""
        SELECT ticker, optimal_fast, optimal_slow, optimal_signal,
               oos_sharpe, confidence, regime, optimizer_provider, optimized_at
        FROM macd_params_cache
        ORDER BY ticker ASC
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_macd_params(ticker: str) -> bool:
    """Remove cached params for a ticker. Returns True if a row was deleted."""
    conn = _get_conn()
    cursor = conn.execute(
        "DELETE FROM macd_params_cache WHERE ticker = ?",
        (ticker.upper().strip(),)
    )
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    return affected > 0
