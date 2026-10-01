from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / ".data" / "legislative_bill_analyzer.sqlite3"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def resolve_db_path() -> Path:
    configured = os.getenv("APP_DB_PATH", "").strip()
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend([DEFAULT_DB, Path("/tmp/claire_yuan_legislative_bill_analyzer.sqlite3")])
    for candidate in candidates:
        try:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            with candidate.open("a", encoding="utf-8"):
                pass
            return candidate
        except OSError:
            continue
    raise OSError("No writable location is available for the prototype SQLite database.")


DB_PATH = resolve_db_path()


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH, timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS visits (
                session_hash TEXT PRIMARY KEY,
                first_seen_utc TEXT NOT NULL,
                last_seen_utc TEXT NOT NULL,
                view_count INTEGER NOT NULL DEFAULT 1
            );

            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                display_name TEXT NOT NULL DEFAULT '',
                password_hash TEXT NOT NULL,
                created_utc TEXT NOT NULL,
                last_login_utc TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS profiles (
                user_id INTEGER PRIMARY KEY,
                profile_json TEXT NOT NULL DEFAULT '{}',
                updated_utc TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS follows (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                bill_id TEXT NOT NULL,
                citation TEXT NOT NULL,
                title TEXT NOT NULL,
                official_url TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT '',
                latest_action_date TEXT NOT NULL DEFAULT '',
                latest_action_text TEXT NOT NULL DEFAULT '',
                created_utc TEXT NOT NULL,
                checked_utc TEXT NOT NULL,
                UNIQUE(user_id, bill_id),
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                bill_id TEXT NOT NULL,
                message TEXT NOT NULL,
                created_utc TEXT NOT NULL,
                is_read INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );
            """
        )


def _normalize_email(email: str) -> str:
    normalized = (email or "").strip().lower()
    if "@" not in normalized or normalized.startswith("@") or normalized.endswith("@"):
        raise ValueError("Enter a valid email address.")
    return normalized


def _hash_password(password: str, *, salt: bytes | None = None) -> str:
    if len(password or "") < 8:
        raise ValueError("Password must contain at least 8 characters.")
    salt = salt or secrets.token_bytes(16)
    iterations = 390_000
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def _verify_password(password: str, stored: str) -> bool:
    try:
        algorithm, iterations_text, salt_hex, digest_hex = stored.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        iterations = int(iterations_text)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def record_visit(session_id: str) -> int:
    init_db()
    session_hash = hashlib.sha256((session_id or "anonymous").encode("utf-8")).hexdigest()
    now = utc_now()
    with connect() as conn:
        existing = conn.execute("SELECT view_count FROM visits WHERE session_hash = ?", (session_hash,)).fetchone()
        if existing:
            conn.execute(
                "UPDATE visits SET last_seen_utc = ?, view_count = view_count + 1 WHERE session_hash = ?",
                (now, session_hash),
            )
        else:
            conn.execute(
                "INSERT INTO visits(session_hash, first_seen_utc, last_seen_utc, view_count) VALUES (?, ?, ?, 1)",
                (session_hash, now, now),
            )
        row = conn.execute("SELECT COUNT(*) AS count FROM visits").fetchone()
    return int(row["count"] if row else 0)


def visitor_stats() -> dict[str, int]:
    init_db()
    with connect() as conn:
        unique = conn.execute("SELECT COUNT(*) AS count FROM visits").fetchone()["count"]
        views = conn.execute("SELECT COALESCE(SUM(view_count), 0) AS count FROM visits").fetchone()["count"]
        users = conn.execute("SELECT COUNT(*) AS count FROM users").fetchone()["count"]
    return {"unique_sessions": int(unique), "page_views": int(views), "registered_users": int(users)}


def create_user(email: str, password: str, display_name: str = "") -> dict[str, Any]:
    init_db()
    normalized = _normalize_email(email)
    hashed = _hash_password(password)
    now = utc_now()
    try:
        with connect() as conn:
            cursor = conn.execute(
                "INSERT INTO users(email, display_name, password_hash, created_utc, last_login_utc) VALUES (?, ?, ?, ?, ?)",
                (normalized, (display_name or "").strip(), hashed, now, now),
            )
            user_id = int(cursor.lastrowid)
            conn.execute(
                "INSERT INTO profiles(user_id, profile_json, updated_utc) VALUES (?, '{}', ?)",
                (user_id, now),
            )
    except sqlite3.IntegrityError as exc:
        raise ValueError("An account with that email already exists.") from exc
    return {"id": user_id, "email": normalized, "display_name": (display_name or "").strip()}


def authenticate(email: str, password: str) -> dict[str, Any] | None:
    init_db()
    try:
        normalized = _normalize_email(email)
    except ValueError:
        return None
    with connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (normalized,)).fetchone()
        if not row or not _verify_password(password, row["password_hash"]):
            return None
        conn.execute("UPDATE users SET last_login_utc = ? WHERE id = ?", (utc_now(), row["id"]))
    return {"id": int(row["id"]), "email": row["email"], "display_name": row["display_name"]}


def save_profile(user_id: int, profile: Mapping[str, Any]) -> None:
    init_db()
    payload = json.dumps(dict(profile), ensure_ascii=False, default=str)
    now = utc_now()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO profiles(user_id, profile_json, updated_utc)
            VALUES (?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET profile_json = excluded.profile_json, updated_utc = excluded.updated_utc
            """,
            (int(user_id), payload, now),
        )


def get_profile(user_id: int) -> dict[str, Any]:
    init_db()
    with connect() as conn:
        row = conn.execute("SELECT profile_json FROM profiles WHERE user_id = ?", (int(user_id),)).fetchone()
    if not row:
        return {}
    try:
        value = json.loads(row["profile_json"])
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


def follow_bill(
    user_id: int,
    *,
    bill_id: str,
    citation: str,
    title: str,
    official_url: str = "",
    status: str = "",
    latest_action_date: str = "",
    latest_action_text: str = "",
) -> None:
    init_db()
    now = utc_now()
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO follows(
                user_id, bill_id, citation, title, official_url, status,
                latest_action_date, latest_action_text, created_utc, checked_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, bill_id) DO UPDATE SET
                citation = excluded.citation,
                title = excluded.title,
                official_url = excluded.official_url,
                status = excluded.status,
                latest_action_date = excluded.latest_action_date,
                latest_action_text = excluded.latest_action_text,
                checked_utc = excluded.checked_utc
            """,
            (
                int(user_id), bill_id, citation, title, official_url, status,
                latest_action_date, latest_action_text, now, now,
            ),
        )


def unfollow_bill(user_id: int, bill_id: str) -> None:
    init_db()
    with connect() as conn:
        conn.execute("DELETE FROM follows WHERE user_id = ? AND bill_id = ?", (int(user_id), bill_id))


def list_follows(user_id: int) -> list[dict[str, Any]]:
    init_db()
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM follows WHERE user_id = ? ORDER BY created_utc DESC",
            (int(user_id),),
        ).fetchall()
    return [dict(row) for row in rows]


def update_follow_and_notify(
    user_id: int,
    bill_id: str,
    *,
    new_status: str,
    latest_action_date: str,
    latest_action_text: str,
) -> bool:
    init_db()
    now = utc_now()
    changed = False
    with connect() as conn:
        row = conn.execute(
            "SELECT citation, status, latest_action_date, latest_action_text FROM follows WHERE user_id = ? AND bill_id = ?",
            (int(user_id), bill_id),
        ).fetchone()
        if not row:
            return False
        changed = (
            (new_status or "") != (row["status"] or "")
            or (latest_action_date or "") != (row["latest_action_date"] or "")
            or (latest_action_text or "") != (row["latest_action_text"] or "")
        )
        if changed:
            message = (
                f"{row['citation']} changed: status {row['status'] or 'unknown'} → {new_status or 'unknown'}; "
                f"latest action {latest_action_date or 'date unavailable'}: {latest_action_text or 'details unavailable'}"
            )
            conn.execute(
                "INSERT INTO notifications(user_id, bill_id, message, created_utc, is_read) VALUES (?, ?, ?, ?, 0)",
                (int(user_id), bill_id, message, now),
            )
        conn.execute(
            """
            UPDATE follows
            SET status = ?, latest_action_date = ?, latest_action_text = ?, checked_utc = ?
            WHERE user_id = ? AND bill_id = ?
            """,
            (new_status, latest_action_date, latest_action_text, now, int(user_id), bill_id),
        )
    return changed


def list_notifications(user_id: int, *, unread_only: bool = False) -> list[dict[str, Any]]:
    init_db()
    query = "SELECT * FROM notifications WHERE user_id = ?"
    params: list[Any] = [int(user_id)]
    if unread_only:
        query += " AND is_read = 0"
    query += " ORDER BY created_utc DESC LIMIT 200"
    with connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def mark_notifications_read(user_id: int) -> None:
    init_db()
    with connect() as conn:
        conn.execute("UPDATE notifications SET is_read = 1 WHERE user_id = ?", (int(user_id),))
