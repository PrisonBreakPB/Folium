"""Session prompt persistence - save and load system prompts per session."""

import sqlite3
from pathlib import Path

DB_DIR = Path.cwd() / "data"
DB_PATH = DB_DIR / "session_prompts.db"


def _get_connection() -> sqlite3.Connection:
    DB_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS session_prompts (
            session_id TEXT PRIMARY KEY,
            system_prompt TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.commit()
    return conn


def save_prompt(session_id: str, system_prompt: str) -> None:
    """Save or update system prompt for a session."""
    conn = _get_connection()
    try:
        conn.execute(
            """
            INSERT INTO session_prompts (session_id, system_prompt)
            VALUES (?, ?)
            ON CONFLICT(session_id) DO UPDATE SET system_prompt = excluded.system_prompt
            """,
            (session_id, system_prompt),
        )
        conn.commit()
    finally:
        conn.close()


def load_prompt(session_id: str) -> str | None:
    """Load system prompt for a session, or None if not found."""
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT system_prompt FROM session_prompts WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()
