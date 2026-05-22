"""
experiment_db.py
----------------
SQLite persistence for RSI DT Feature Lab experiment runs.
Stored in data/rsi_dt_experiments.db, separate from rsi_cache.db.
"""

from __future__ import annotations

import json
import sqlite3
import urllib.parse
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path(__file__).resolve().parent


_DB_PATH = _project_root() / "data" / "rsi_dt_experiments.db"


def _get_conn() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    encoded = urllib.parse.quote(str(_DB_PATH), safe="/:")
    uri = f"file:{encoded}?nolock=1"
    conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_table() -> None:
    conn = _get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rsi_dt_experiments (
            run_id          TEXT NOT NULL,
            symbol          TEXT NOT NULL,
            date            TEXT NOT NULL,
            rule_set_name   TEXT NOT NULL,
            features_used   TEXT NOT NULL,
            veto_rules      TEXT NOT NULL,
            oos_sharpe      REAL NOT NULL,
            hit_rate        REAL NOT NULL,
            trade_count     INTEGER NOT NULL,
            confidence      TEXT NOT NULL,
            n_slides        INTEGER NOT NULL,
            avg_oos_acc     REAL NOT NULL,
            ran_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (run_id, rule_set_name)
        )
    """)
    conn.commit()
    conn.close()


_ensure_table()


def save_experiment_result(row: dict[str, Any]) -> None:
    conn = _get_conn()
    conn.execute("""
        INSERT OR REPLACE INTO rsi_dt_experiments
            (run_id, symbol, date, rule_set_name, features_used, veto_rules,
             oos_sharpe, hit_rate, trade_count, confidence, n_slides, avg_oos_acc)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        row["run_id"],
        row["symbol"],
        row["date"],
        row["rule_set_name"],
        json.dumps(row["features_used"]),
        json.dumps(row["veto_rules"]),
        row["oos_sharpe"],
        row["hit_rate"],
        row["trade_count"],
        row["confidence"],
        row["n_slides"],
        row["avg_oos_acc"],
    ))
    conn.commit()
    conn.close()


def get_experiment_runs(symbol: str, limit: int = 50) -> list[dict[str, Any]]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM rsi_dt_experiments WHERE symbol = ? ORDER BY ran_at DESC LIMIT ?",
        (symbol.upper().strip(), min(limit, 200)),
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["features_used"] = json.loads(d["features_used"])
        d["veto_rules"] = json.loads(d["veto_rules"])
        result.append(d)
    return result


def get_experiment_run(run_id: str) -> list[dict[str, Any]]:
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM rsi_dt_experiments WHERE run_id = ? ORDER BY oos_sharpe DESC",
        (run_id,),
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["features_used"] = json.loads(d["features_used"])
        d["veto_rules"] = json.loads(d["veto_rules"])
        result.append(d)
    return result
