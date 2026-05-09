from typing import List, Dict, Any, Optional

from ..database import get_db_connection


def get_all_positions() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM positions")
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def add_position(ticker: str, quantity: float, cost_basis: float) -> int:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO positions (ticker, quantity, cost_basis) VALUES (?, ?, ?)",
        (ticker.upper(), quantity, cost_basis),
    )
    conn.commit()
    pos_id = cursor.lastrowid
    conn.close()
    return pos_id


def update_position(pos_id: int, ticker: str, quantity: float, cost_basis: float):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE positions SET ticker = ?, quantity = ?, cost_basis = ? WHERE id = ?",
        (ticker.upper(), quantity, cost_basis, pos_id),
    )
    conn.commit()
    conn.close()


def delete_position(pos_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM positions WHERE id = ?", (pos_id,))
    conn.commit()
    conn.close()


def patch_position_meta(
    pos_id: int,
    exchange: Optional[str],
    target_weight: Optional[float],
    notes: Optional[str],
    follow_ticker: Optional[str] = None,
    manual_price: Optional[float] = None,
    category: Optional[str] = None,
):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE positions SET exchange=?, target_weight=?, notes=?, follow_ticker=?, manual_price=?, category=? WHERE id=?",
        (exchange, target_weight, notes, follow_ticker.upper().strip() if follow_ticker else None, manual_price, category, pos_id),
    )
    conn.commit()
    conn.close()


def get_lots_for_position(position_id: int) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM position_lots WHERE position_id=? ORDER BY purchased_at ASC, id ASC",
        (position_id,),
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def add_lot(position_id: int, purchased_at: str, price_per_share: float, quantity: float, notes: Optional[str]) -> int:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO position_lots (position_id, purchased_at, price_per_share, quantity, notes) VALUES (?,?,?,?,?)",
        (position_id, purchased_at, price_per_share, quantity, notes),
    )
    conn.commit()
    lot_id = cursor.lastrowid
    _sync_position_from_lots(cursor, position_id)
    conn.commit()
    conn.close()
    return lot_id


def update_lot(lot_id: int, purchased_at: str, price_per_share: float, quantity: float, notes: Optional[str]):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE position_lots SET purchased_at=?, price_per_share=?, quantity=?, notes=? WHERE id=?",
        (purchased_at, price_per_share, quantity, notes, lot_id),
    )
    cursor.execute("SELECT position_id FROM position_lots WHERE id=?", (lot_id,))
    row = cursor.fetchone()
    if row:
        _sync_position_from_lots(cursor, row["position_id"])
    conn.commit()
    conn.close()


def delete_lot(lot_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT position_id FROM position_lots WHERE id=?", (lot_id,))
    row = cursor.fetchone()
    cursor.execute("DELETE FROM position_lots WHERE id=?", (lot_id,))
    if row:
        _sync_position_from_lots(cursor, row["position_id"])
    conn.commit()
    conn.close()


def _sync_position_from_lots(cursor, position_id: int):
    """Recompute weighted average cost_basis and total quantity from lots."""
    cursor.execute(
        "SELECT SUM(price_per_share * quantity) as cost_sum, SUM(quantity) as qty_sum FROM position_lots WHERE position_id=?",
        (position_id,),
    )
    row = cursor.fetchone()
    if row and row["qty_sum"] and row["qty_sum"] > 0:
        avg_cost = row["cost_sum"] / row["qty_sum"]
        cursor.execute(
            "UPDATE positions SET cost_basis=?, quantity=? WHERE id=?",
            (avg_cost, row["qty_sum"], position_id),
        )


def update_prices(prices: Dict[str, float]):
    conn = get_db_connection()
    cursor = conn.cursor()
    for ticker, price in prices.items():
        cursor.execute(
            "UPDATE positions SET current_price = ?, last_updated = CURRENT_TIMESTAMP WHERE ticker = ?",
            (price, ticker.upper()),
        )
    conn.commit()
    conn.close()
