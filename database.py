"""SQLite persistence layer for ReFind AI.

All SQL operations use parameterized queries to prevent SQL injection and keep
data handling clean, reliable, and persistent across application restarts.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Union

from utils import DEFAULT_DB_PATH, REPORT_TYPES, STATUS_OPTIONS, clean_text

PathLike = Union[str, Path]


def _resolve_db_path(db_path: Optional[PathLike] = None) -> Path:
    path = Path(db_path) if db_path is not None else Path(DEFAULT_DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_connection(db_path: Optional[PathLike] = None) -> sqlite3.Connection:
    """Open a SQLite connection with Row access and lock timeout enabled."""
    resolved = _resolve_db_path(db_path)
    conn = sqlite3.connect(str(resolved), timeout=10.0)
    conn.row_factory = sqlite3.Row
    return conn


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def init_db(db_path: Optional[PathLike] = None) -> Path:
    """Create the SQLite database file, reports table, app_meta table, and indexes.

    Non-destructive: preserves all existing tables and records.
    """
    resolved = _resolve_db_path(db_path)
    with get_connection(resolved) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                category TEXT NOT NULL,
                report_type TEXT NOT NULL CHECK (report_type IN ('lost', 'found')),
                location TEXT NOT NULL,
                report_date TEXT NOT NULL,
                additional_details TEXT NOT NULL DEFAULT '',
                contact_label TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'recovered')),
                image_path TEXT DEFAULT NULL,
                is_seed INTEGER NOT NULL DEFAULT 0,
                seed_key TEXT UNIQUE DEFAULT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS app_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_reports_type_status ON reports (report_type, status)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_reports_category ON reports (category)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_reports_created_at ON reports (created_at DESC)"
        )
        conn.commit()
    return resolved


def get_meta(key: str, db_path: Optional[PathLike] = None) -> Optional[str]:
    """Read a metadata value from app_meta, or None if not set."""
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT value FROM app_meta WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row is not None else None


def set_meta(key: str, value: str, db_path: Optional[PathLike] = None) -> None:
    """Upsert a metadata key/value pair into app_meta."""
    with get_connection(db_path) as conn:
        conn.execute(
            """
            INSERT INTO app_meta (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, str(value)),
        )
        conn.commit()


def find_duplicate_report(
    *,
    title: str,
    description: str,
    report_type: str,
    location: str,
    report_date: str,
    db_path: Optional[PathLike] = None,
) -> Optional[dict[str, Any]]:
    """Return an existing matching active report if an identical submission already exists."""
    clean_title = clean_text(title).lower()
    clean_desc = clean_text(description).lower()
    clean_type = clean_text(report_type).lower()
    clean_loc = clean_text(location).lower()
    clean_date = clean_text(report_date)

    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT * FROM reports
            WHERE report_type = ?
              AND status = 'active'
              AND LOWER(TRIM(title)) = ?
              AND LOWER(TRIM(description)) = ?
              AND LOWER(TRIM(location)) = ?
              AND report_date = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (clean_type, clean_title, clean_desc, clean_loc, clean_date),
        ).fetchone()
        return dict(row) if row is not None else None


def insert_report(
    *,
    title: str,
    description: str,
    category: str,
    report_type: str,
    location: str,
    report_date: str,
    additional_details: str = "",
    contact_label: str = "",
    status: str = "active",
    image_path: Optional[str] = None,
    is_seed: bool = False,
    seed_key: Optional[str] = None,
    db_path: Optional[PathLike] = None,
) -> int:
    """Insert a new lost or found report and return its generated row ID."""
    clean_title = clean_text(title)
    clean_desc = clean_text(description)
    clean_cat = clean_text(category) or "Other"
    clean_type = clean_text(report_type).lower()
    clean_loc = clean_text(location)
    clean_date = clean_text(report_date)
    clean_extra = clean_text(additional_details)
    clean_contact = clean_text(contact_label) or "Campus Lost & Found Desk (Demo)"
    clean_status = clean_text(status).lower() or "active"

    if not clean_title:
        raise ValueError("Item title is required.")
    if not clean_desc:
        raise ValueError("Item description is required.")
    if clean_type not in REPORT_TYPES:
        raise ValueError(f"report_type must be one of {REPORT_TYPES}.")
    if clean_status not in STATUS_OPTIONS:
        raise ValueError(f"status must be one of {STATUS_OPTIONS}.")
    if not clean_loc:
        raise ValueError("Location is required.")
    if not clean_date:
        raise ValueError("Report date is required.")

    now_iso = _utc_now_iso()
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO reports (
                title, description, category, report_type, location, report_date,
                additional_details, contact_label, status, image_path,
                is_seed, seed_key, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                clean_title,
                clean_desc,
                clean_cat,
                clean_type,
                clean_loc,
                clean_date,
                clean_extra,
                clean_contact,
                clean_status,
                image_path,
                1 if is_seed else 0,
                seed_key,
                now_iso,
                now_iso,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def insert_seed_report(record: dict[str, Any], db_path: Optional[PathLike] = None) -> bool:
    """Idempotently insert a synthetic seed record using its unique seed_key."""
    seed_key = record.get("seed_key")
    if not seed_key:
        raise ValueError("Seed records must define a unique 'seed_key'.")

    now_iso = _utc_now_iso()
    with get_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO reports (
                title, description, category, report_type, location, report_date,
                additional_details, contact_label, status, image_path,
                is_seed, seed_key, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
            """,
            (
                clean_text(record["title"]),
                clean_text(record["description"]),
                clean_text(record["category"]),
                clean_text(record["report_type"]).lower(),
                clean_text(record["location"]),
                clean_text(record["report_date"]),
                clean_text(record.get("additional_details", "")),
                clean_text(record.get("contact_label", "Campus Lost & Found Desk (Demo)")),
                clean_text(record.get("status", "active")).lower(),
                record.get("image_path"),
                seed_key,
                now_iso,
                now_iso,
            ),
        )
        conn.commit()
        return cursor.rowcount > 0


def get_report_by_id(
    report_id: int, db_path: Optional[PathLike] = None
) -> Optional[dict[str, Any]]:
    """Fetch a single report by primary key, or None if not found."""
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT * FROM reports WHERE id = ?", (int(report_id),)).fetchone()
        return dict(row) if row is not None else None


def get_all_reports(
    *,
    report_type: Optional[str] = None,
    category: Optional[str] = None,
    status: Optional[str] = None,
    search_query: Optional[str] = None,
    limit: Optional[int] = None,
    db_path: Optional[PathLike] = None,
) -> list[dict[str, Any]]:
    """Retrieve reports with optional parameterized filtering, most recently created first."""
    clauses: list[str] = []
    params: list[Any] = []

    if report_type and report_type.lower() in REPORT_TYPES:
        clauses.append("report_type = ?")
        params.append(report_type.lower())

    if category and category != "All":
        clauses.append("category = ?")
        params.append(category)

    if status and status.lower() in STATUS_OPTIONS:
        clauses.append("status = ?")
        params.append(status.lower())

    clean_query = clean_text(search_query)
    if clean_query:
        like_term = f"%{clean_query}%"
        clauses.append(
            "(title LIKE ? OR description LIKE ? OR location LIKE ? OR additional_details LIKE ? OR category LIKE ?)"
        )
        params.extend([like_term, like_term, like_term, like_term, like_term])

    sql = "SELECT * FROM reports"
    if clauses:
        sql += " WHERE " + " AND ".join(clauses)
    sql += " ORDER BY created_at DESC, id DESC"

    if limit is not None and int(limit) > 0:
        sql += " LIMIT ?"
        params.append(int(limit))

    with get_connection(db_path) as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()
        return [dict(row) for row in rows]


def update_report_status(
    report_id: int, status: str, db_path: Optional[PathLike] = None
) -> bool:
    """Update a report's status ('active' or 'recovered'). Returns True if updated."""
    clean_status = clean_text(status).lower()
    if clean_status not in STATUS_OPTIONS:
        raise ValueError(f"status must be one of {STATUS_OPTIONS}.")

    with get_connection(db_path) as conn:
        cursor = conn.execute(
            "UPDATE reports SET status = ?, updated_at = ? WHERE id = ?",
            (clean_status, _utc_now_iso(), int(report_id)),
        )
        conn.commit()
        return cursor.rowcount > 0


def delete_report(report_id: int, db_path: Optional[PathLike] = None) -> bool:
    """Delete a report by ID. Returns True if a record was removed."""
    with get_connection(db_path) as conn:
        cursor = conn.execute("DELETE FROM reports WHERE id = ?", (int(report_id),))
        conn.commit()
        return cursor.rowcount > 0


def delete_seed_reports(db_path: Optional[PathLike] = None) -> int:
    """Remove all synthetic demonstration records while preserving user submissions."""
    with get_connection(db_path) as conn:
        cursor = conn.execute("DELETE FROM reports WHERE is_seed = 1")
        conn.commit()
        return int(cursor.rowcount)


def count_reports(db_path: Optional[PathLike] = None) -> int:
    """Return total number of rows currently stored in the reports table."""
    with get_connection(db_path) as conn:
        row = conn.execute("SELECT COUNT(*) AS cnt FROM reports").fetchone()
        return int(row["cnt"]) if row else 0


def get_dashboard_stats(db_path: Optional[PathLike] = None) -> dict[str, int]:
    """Compute aggregate KPI counts for the campus dashboard."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS total_reports,
                COALESCE(SUM(CASE WHEN report_type = 'lost' THEN 1 ELSE 0 END), 0) AS total_lost,
                COALESCE(SUM(CASE WHEN report_type = 'found' THEN 1 ELSE 0 END), 0) AS total_found,
                COALESCE(SUM(CASE WHEN status = 'recovered' THEN 1 ELSE 0 END), 0) AS total_recovered,
                COALESCE(SUM(CASE WHEN status = 'active' THEN 1 ELSE 0 END), 0) AS active_reports,
                COALESCE(SUM(CASE WHEN report_type = 'lost' AND status = 'active' THEN 1 ELSE 0 END), 0) AS active_lost,
                COALESCE(SUM(CASE WHEN report_type = 'found' AND status = 'active' THEN 1 ELSE 0 END), 0) AS active_found
            FROM reports
            """
        ).fetchone()
        return {
            "total_reports": int(row["total_reports"]),
            "total_lost": int(row["total_lost"]),
            "total_found": int(row["total_found"]),
            "total_recovered": int(row["total_recovered"]),
            "active_reports": int(row["active_reports"]),
            "active_lost": int(row["active_lost"]),
            "active_found": int(row["active_found"]),
        }
