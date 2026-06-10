"""
SQLite persistence for cafes, sales history, and column mappings.
Survives server restarts — sellers upload once, then add daily sales manually.
"""

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "loveovers.db")

SALES_COLUMNS = [
    "date", "item", "sold_qty", "produced_qty", "price",
    "discount_pct", "weather", "day_of_week",
]


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def get_db():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS cafes (
                cafe_id TEXT PRIMARY KEY,
                cafe_name TEXT NOT NULL,
                column_mapping TEXT,
                assessment_id TEXT,
                items_json TEXT,
                metrics_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_trained_at TEXT
            );

            CREATE TABLE IF NOT EXISTS sales_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                cafe_id TEXT NOT NULL,
                sale_date TEXT NOT NULL,
                item TEXT NOT NULL,
                sold_qty REAL NOT NULL,
                produced_qty REAL,
                price REAL DEFAULT 0,
                discount_pct REAL DEFAULT 0,
                weather TEXT DEFAULT 'Unknown',
                day_of_week TEXT,
                source TEXT NOT NULL DEFAULT 'upload',
                created_at TEXT NOT NULL,
                UNIQUE (cafe_id, sale_date, item),
                FOREIGN KEY (cafe_id) REFERENCES cafes(cafe_id)
            );

            CREATE INDEX IF NOT EXISTS idx_sales_cafe_date
                ON sales_records(cafe_id, sale_date);
        """)


def upsert_cafe(
    cafe_id: str,
    cafe_name: str,
    column_mapping: dict | None = None,
    assessment_id: str | None = None,
    items: list | None = None,
    metrics: dict | None = None,
    trained: bool = False,
):
    now = datetime.now().isoformat()
    with get_db() as conn:
        row = conn.execute(
            "SELECT cafe_id FROM cafes WHERE cafe_id = ?", (cafe_id,)
        ).fetchone()
        mapping_json = json.dumps(column_mapping) if column_mapping else None
        items_json = json.dumps(items) if items else None
        metrics_json = json.dumps(metrics) if metrics else None
        last_trained = now if trained else None

        if row:
            conn.execute(
                """
                UPDATE cafes SET
                    cafe_name = COALESCE(?, cafe_name),
                    column_mapping = COALESCE(?, column_mapping),
                    assessment_id = COALESCE(?, assessment_id),
                    items_json = COALESCE(?, items_json),
                    metrics_json = COALESCE(?, metrics_json),
                    updated_at = ?,
                    last_trained_at = COALESCE(?, last_trained_at)
                WHERE cafe_id = ?
                """,
                (
                    cafe_name, mapping_json, assessment_id, items_json,
                    metrics_json, now, last_trained, cafe_id,
                ),
            )
        else:
            conn.execute(
                """
                INSERT INTO cafes (
                    cafe_id, cafe_name, column_mapping, assessment_id,
                    items_json, metrics_json, created_at, updated_at, last_trained_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cafe_id, cafe_name, mapping_json, assessment_id,
                    items_json, metrics_json, now, now, last_trained,
                ),
            )


def get_cafe(cafe_id: str) -> dict | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM cafes WHERE cafe_id = ?", (cafe_id,)
        ).fetchone()
    if not row:
        return None
    return _cafe_row_to_dict(row)


def list_cafes() -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM cafes ORDER BY updated_at DESC"
        ).fetchall()
    return [_cafe_row_to_dict(r) for r in rows]


def _cafe_row_to_dict(row) -> dict:
    d = dict(row)
    if d.get("column_mapping"):
        d["column_mapping"] = json.loads(d["column_mapping"])
    if d.get("items_json"):
        d["items"] = json.loads(d["items_json"])
    if d.get("metrics_json"):
        d["metrics"] = json.loads(d["metrics_json"])
    return d


def save_sales_dataframe(cafe_id: str, df: pd.DataFrame, source: str = "upload"):
    """Upsert standardized sales rows (columns: date, item, sold_qty, ...)."""
    now = datetime.now().isoformat()
    rows = []
    for _, r in df.iterrows():
        sale_date = pd.to_datetime(r["date"]).strftime("%Y-%m-%d")
        rows.append((
            cafe_id,
            sale_date,
            str(r["item"]),
            float(r["sold_qty"]),
            float(r.get("produced_qty", r["sold_qty"] * 1.12)),
            float(r.get("price", 0) or 0),
            float(r.get("discount_pct", 0) or 0),
            str(r.get("weather", "Unknown")),
            str(r.get("day_of_week", "")),
            source,
            now,
        ))

    with get_db() as conn:
        conn.executemany(
            """
            INSERT INTO sales_records (
                cafe_id, sale_date, item, sold_qty, produced_qty,
                price, discount_pct, weather, day_of_week, source, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(cafe_id, sale_date, item) DO UPDATE SET
                sold_qty = excluded.sold_qty,
                produced_qty = excluded.produced_qty,
                price = excluded.price,
                discount_pct = excluded.discount_pct,
                weather = excluded.weather,
                day_of_week = excluded.day_of_week,
                source = excluded.source,
                created_at = excluded.created_at
            """,
            rows,
        )


def add_daily_sales(
    cafe_id: str,
    sale_date: str,
    entries: list[dict],
    day_of_week: str | None = None,
    weather: str = "Unknown",
    default_discount: float = 0,
) -> dict:
    """
    Manual daily entry. Each entry: {item, sold_qty, produced_qty?, price?}
    """
    now = datetime.now().isoformat()
    saved = []
    with get_db() as conn:
        for e in entries:
            item = e["item"]
            sold = float(e["sold_qty"])
            produced = float(e.get("produced_qty", sold * 1.12))
            price = float(e.get("price", 0) or 0)
            discount = float(e.get("discount_pct", default_discount) or 0)
            dow = e.get("day_of_week") or day_of_week or ""
            wthr = e.get("weather", weather)

            conn.execute(
                """
                INSERT INTO sales_records (
                    cafe_id, sale_date, item, sold_qty, produced_qty,
                    price, discount_pct, weather, day_of_week, source, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'manual', ?)
                ON CONFLICT(cafe_id, sale_date, item) DO UPDATE SET
                    sold_qty = excluded.sold_qty,
                    produced_qty = excluded.produced_qty,
                    price = excluded.price,
                    discount_pct = excluded.discount_pct,
                    weather = excluded.weather,
                    day_of_week = excluded.day_of_week,
                    source = 'manual',
                    created_at = excluded.created_at
                """,
                (
                    cafe_id, sale_date, item, sold, produced,
                    price, discount, wthr, dow, now,
                ),
            )
            saved.append({"item": item, "sold_qty": sold, "sale_date": sale_date})

    return {"saved": len(saved), "entries": saved}


def load_sales_dataframe(cafe_id: str) -> pd.DataFrame:
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT sale_date AS date, item, sold_qty, produced_qty, price,
                   discount_pct, weather, day_of_week, source
            FROM sales_records
            WHERE cafe_id = ?
            ORDER BY sale_date, item
            """,
            (cafe_id,),
        ).fetchall()

    if not rows:
        return pd.DataFrame(columns=SALES_COLUMNS + ["source"])

    df = pd.DataFrame([dict(r) for r in rows])
    df["date"] = pd.to_datetime(df["date"])
    return df


def sales_summary(cafe_id: str) -> dict:
    with get_db() as conn:
        stats = conn.execute(
            """
            SELECT
                COUNT(*) AS total_rows,
                COUNT(DISTINCT sale_date) AS total_days,
                COUNT(DISTINCT item) AS total_items,
                MIN(sale_date) AS first_date,
                MAX(sale_date) AS last_date,
                SUM(CASE WHEN source = 'manual' THEN 1 ELSE 0 END) AS manual_rows,
                SUM(CASE WHEN source = 'upload' THEN 1 ELSE 0 END) AS upload_rows
            FROM sales_records WHERE cafe_id = ?
            """,
            (cafe_id,),
        ).fetchone()
    return dict(stats) if stats else {}


def get_recent_sales(cafe_id: str, limit: int = 50) -> list[dict]:
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT sale_date, item, sold_qty, produced_qty, weather,
                   day_of_week, source, created_at
            FROM sales_records
            WHERE cafe_id = ?
            ORDER BY sale_date DESC, item
            LIMIT ?
            """,
            (cafe_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


init_db()
