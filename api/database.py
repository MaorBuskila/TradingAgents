import sqlite3
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional

def _project_root() -> Path:
    """Walk up from this file until pyproject.toml is found (project root)."""
    for candidate in Path(__file__).resolve().parents:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path(__file__).resolve().parent  # fallback

DB_PATH = str(_project_root() / "data" / "portfolio.db")
CATALOG_DATA_PATH = Path(__file__).resolve().parent / "catalog_data.json"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            quantity REAL NOT NULL,
            cost_basis REAL NOT NULL,
            current_price REAL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            exchange TEXT DEFAULT 'US',
            target_weight REAL,
            notes TEXT
        )
    ''')
    # Migrate existing positions table to add new columns if they don't exist
    for col, definition in [
        ("exchange",      "TEXT DEFAULT 'US'"),
        ("target_weight", "REAL"),
        ("notes",         "TEXT"),
        ("follow_ticker", "TEXT"),
        ("manual_price",  "REAL"),
    ]:
        try:
            cursor.execute(f"ALTER TABLE positions ADD COLUMN {col} {definition}")
        except Exception:
            pass
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS position_lots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            position_id INTEGER NOT NULL REFERENCES positions(id) ON DELETE CASCADE,
            purchased_at DATE NOT NULL,
            price_per_share REAL NOT NULL,
            quantity REAL NOT NULL,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_lots_position ON position_lots(position_id)"
    )
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS catalog_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL UNIQUE,
            name TEXT,
            category TEXT NOT NULL,
            asset_type TEXT NOT NULL DEFAULT 'stock',
            source TEXT NOT NULL DEFAULT 'seed'
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS catalog_favorites (
            ticker TEXT PRIMARY KEY
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS youtube_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id TEXT NOT NULL,
            url TEXT NOT NULL,
            title TEXT,
            transcript_chars INTEGER,
            summary_en TEXT NOT NULL,
            summary_he TEXT NOT NULL,
            provider TEXT,
            model TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_youtube_summaries_created ON youtube_summaries(created_at)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_youtube_summaries_video ON youtube_summaries(video_id)"
    )
    # RSI optimization cache — one row per ticker, upserted on every re-optimization.
    # NOTE: the authoritative copy lives in rsi_params.db (tradingagents/dataflows/rsi_cache.py).
    # This mirror table keeps the API layer aware of cached params for future dashboard display.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rsi_params_cache (
            ticker             TEXT PRIMARY KEY,
            optimal_period     INTEGER NOT NULL,
            optimal_upper      REAL    NOT NULL,
            optimal_lower      REAL    NOT NULL,
            oos_sharpe         REAL    NOT NULL,
            is_sharpe          REAL    NOT NULL,
            confidence         TEXT    NOT NULL,
            regime             TEXT,
            training_days      INTEGER,
            test_days          INTEGER,
            optimizer_provider TEXT,
            model_used         TEXT,
            optimized_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    seed_catalog_if_needed()


def seed_catalog_if_needed() -> None:
    """Merge seed symbols from catalog_data.json (INSERT OR IGNORE). Safe to run on every startup."""
    if not CATALOG_DATA_PATH.is_file():
        return
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    with open(CATALOG_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    for row in data.get("symbols", []):
        cursor.execute(
            """INSERT OR IGNORE INTO catalog_items (ticker, name, category, asset_type, source)
               VALUES (?, ?, ?, ?, 'seed')""",
            (
                row["ticker"].upper().strip(),
                row.get("name") or "",
                row["category"].strip(),
                (row.get("asset_type") or "stock").lower(),
            ),
        )
    conn.commit()
    conn.close()


def load_category_labels() -> Dict[str, str]:
    if not CATALOG_DATA_PATH.is_file():
        return {}
    with open(CATALOG_DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return dict(data.get("category_labels") or {})


def list_catalog_rows(
    category: Optional[str] = None,
    q: Optional[str] = None,
    favorites_only: bool = False,
) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    sql = """
        SELECT c.id, c.ticker, c.name, c.category, c.asset_type, c.source,
               CASE WHEN f.ticker IS NOT NULL THEN 1 ELSE 0 END AS is_favorite
        FROM catalog_items c
        LEFT JOIN catalog_favorites f ON c.ticker = f.ticker
        WHERE 1=1
    """
    params: List[Any] = []
    if category:
        sql += " AND c.category = ?"
        params.append(category.strip())
    if q:
        like = f"%{q.strip()}%"
        sql += " AND (UPPER(c.ticker) LIKE UPPER(?) OR c.name LIKE ?)"
        params.extend([like, like])
    if favorites_only:
        sql += " AND f.ticker IS NOT NULL"
    sql += " ORDER BY c.ticker ASC"
    cursor.execute(sql, params)
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def sort_catalog_rows(
    rows: List[Dict[str, Any]],
    sort: str = "ticker",
    order: str = "asc",
) -> List[Dict[str, Any]]:
    reverse = order.lower() == "desc"
    sk = sort.lower()

    def key_ticker(r):
        return (r.get("ticker") or "").upper()

    def key_name(r):
        return (r.get("name") or "").lower()

    def key_category(r):
        return (r.get("category") or "").lower()

    def key_asset(r):
        return (r.get("asset_type") or "").lower()

    if sk == "favorite":
        # favorites first by default (asc); desc puts non-favorites first
        fav_first = not reverse
        if fav_first:
            rows = sorted(
                rows,
                key=lambda r: (0 if r.get("is_favorite") else 1, key_ticker(r)),
            )
        else:
            rows = sorted(
                rows,
                key=lambda r: (1 if r.get("is_favorite") else 0, key_ticker(r)),
            )
    elif sk == "name":
        rows = sorted(rows, key=key_name, reverse=reverse)
    elif sk == "category":
        rows = sorted(rows, key=key_category, reverse=reverse)
    elif sk == "asset_type":
        rows = sorted(rows, key=key_asset, reverse=reverse)
    else:
        rows = sorted(rows, key=key_ticker, reverse=reverse)
    return rows


def set_catalog_favorite(ticker: str, favorite: bool) -> None:
    t = ticker.upper().strip()
    conn = get_db_connection()
    cursor = conn.cursor()
    if favorite:
        cursor.execute("INSERT OR IGNORE INTO catalog_favorites (ticker) VALUES (?)", (t,))
    else:
        cursor.execute("DELETE FROM catalog_favorites WHERE ticker = ?", (t,))
    conn.commit()
    conn.close()


def add_catalog_item(ticker: str, name: str, category: str, asset_type: str = "stock") -> int:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO catalog_items (ticker, name, category, asset_type, source)
           VALUES (?, ?, ?, ?, 'user')""",
        (ticker.upper().strip(), name.strip(), category.strip(), asset_type.lower().strip()),
    )
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id


def category_counts() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT category, COUNT(*) AS cnt FROM catalog_items GROUP BY category ORDER BY category"
    )
    rows = [{"category": r["category"], "count": r["cnt"]} for r in cursor.fetchall()]
    conn.close()
    return rows

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def insert_youtube_summary(
    video_id: str,
    url: str,
    title: Optional[str],
    transcript_chars: int,
    summary_en: str,
    summary_he: str,
    provider: str,
    model: str,
) -> int:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO youtube_summaries
           (video_id, url, title, transcript_chars, summary_en, summary_he, provider, model)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            video_id,
            url,
            title,
            transcript_chars,
            summary_en,
            summary_he,
            provider,
            model,
        ),
    )
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return int(row_id)


def update_youtube_summary_he(row_id: int, summary_he: str) -> bool:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE youtube_summaries SET summary_he = ? WHERE id = ?",
        (summary_he, row_id),
    )
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    return affected > 0


def list_youtube_summaries(limit: int = 50) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT id, video_id, url, title, transcript_chars, provider, model, created_at
           FROM youtube_summaries
           ORDER BY datetime(created_at) DESC
           LIMIT ?""",
        (limit,),
    )
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows


def get_youtube_summary_by_id(row_id: int) -> Optional[Dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """SELECT id, video_id, url, title, transcript_chars, summary_en, summary_he,
                  provider, model, created_at
           FROM youtube_summaries WHERE id = ?""",
        (row_id,),
    )
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None
