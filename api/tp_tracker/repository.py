from __future__ import annotations

from typing import Any, Dict, List, Optional

from api.database import get_db_connection


def create_tracker(row: Dict[str, Any]) -> int:
    """INSERT into position_targets. Returns new id."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO position_targets
           (ticker, side, entry, sl, trail_sl, tp1, tp2, tp3,
            tp1_hit, tp2_hit, tp3_hit, trail_active, closed,
            rsi_at_entry, htf_bias, vol_regime, bull_score, as_of_date)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            row["ticker"],
            row["side"],
            row["entry"],
            row["sl"],
            row.get("trail_sl", row["sl"]),
            row["tp1"],
            row["tp2"],
            row["tp3"],
            int(row.get("tp1_hit", 0)),
            int(row.get("tp2_hit", 0)),
            int(row.get("tp3_hit", 0)),
            int(row.get("trail_active", 0)),
            int(row.get("closed", 0)),
            row.get("rsi_at_entry"),
            row.get("htf_bias"),
            row.get("vol_regime"),
            row.get("bull_score"),
            row.get("as_of_date"),
        ),
    )
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return int(row_id)


def get_tracker_by_id(tracker_id: int) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM position_targets WHERE id = ?", (tracker_id,))
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def get_tracker(ticker: str) -> Optional[Dict[str, Any]]:
    """Return the most recent non-closed tracker for ticker."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT * FROM position_targets
           WHERE ticker = ? AND closed = 0
           ORDER BY created_at DESC LIMIT 1""",
        (ticker.upper(),),
    )
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def list_all_trackers(include_closed: bool = False) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    if include_closed:
        cursor.execute("SELECT * FROM position_targets ORDER BY created_at DESC")
    else:
        cursor.execute(
            "SELECT * FROM position_targets WHERE closed = 0 ORDER BY created_at DESC"
        )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def update_tracker(tracker_id: int, fields: Dict[str, Any]) -> None:
    """Update arbitrary fields. Always stamps updated_at."""
    if not fields:
        return
    fields = dict(fields)
    fields["updated_at"] = "CURRENT_TIMESTAMP"
    # Build SET clause — updated_at gets the SQL function, others get ?
    set_parts = []
    values = []
    for k, v in fields.items():
        if k == "updated_at":
            set_parts.append("updated_at = CURRENT_TIMESTAMP")
        else:
            set_parts.append(f"{k} = ?")
            values.append(v)
    values.append(tracker_id)
    sql = f"UPDATE position_targets SET {', '.join(set_parts)} WHERE id = ?"
    conn = get_db_connection()
    conn.execute(sql, values)
    conn.commit()
    conn.close()


def close_tracker(tracker_id: int, exit_reason: str) -> None:
    conn = get_db_connection()
    conn.execute(
        "UPDATE position_targets SET closed = 1, exit_reason = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
        (exit_reason, tracker_id),
    )
    conn.commit()
    conn.close()
