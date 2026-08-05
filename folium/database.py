"""SQLite storage shared by sessions and observability."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

DB_DIR = Path(os.getcwd()) / "data"
DB_PATH = DB_DIR / "folium.db"


class FoliumConnection(sqlite3.Connection):
    """SQLite connection that closes after a transaction context ends."""

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


def get_connection(path: str | Path | None = None) -> sqlite3.Connection:
    """Open the Folium database and ensure its schema exists."""
    db_path = Path(path) if path is not None else DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), factory=FoliumConnection)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    _initialize_schema(conn)
    return conn


def _initialize_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            model TEXT NOT NULL,
            system_prompt TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            model_content TEXT,
            transcript_position INTEGER,
            context_position INTEGER,
            kind TEXT NOT NULL DEFAULT 'normal',
            tool_call_id TEXT,
            tool_name TEXT,
            tool_calls_json TEXT,
            metadata_json TEXT
        );

        CREATE TABLE IF NOT EXISTS traces (
            trace_id TEXT PRIMARY KEY,
            session_id TEXT REFERENCES sessions(id) ON DELETE CASCADE,
            turn_index INTEGER,
            model TEXT,
            name TEXT,
            status TEXT,
            started_at TEXT,
            ended_at TEXT,
            duration_ms INTEGER
        );

        CREATE TABLE IF NOT EXISTS trace_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT NOT NULL REFERENCES traces(trace_id) ON DELETE CASCADE,
            event_index INTEGER NOT NULL,
            timestamp TEXT,
            event_type TEXT,
            name TEXT,
            type TEXT,
            status TEXT,
            duration_ms INTEGER,
            span_id TEXT,
            parent_span_id TEXT,
            message_id INTEGER REFERENCES messages(id) ON DELETE SET NULL,
            tool_call_id TEXT,
            payload_json TEXT,
            error_json TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_messages_transcript
            ON messages(session_id, transcript_position);
        CREATE INDEX IF NOT EXISTS idx_messages_context
            ON messages(session_id, context_position);
        CREATE INDEX IF NOT EXISTS idx_traces_session_time
            ON traces(session_id, started_at DESC);
        CREATE INDEX IF NOT EXISTS idx_trace_events_trace_index
            ON trace_events(trace_id, event_index);
        """
    )
    conn.execute("PRAGMA user_version = 1")
    conn.commit()
