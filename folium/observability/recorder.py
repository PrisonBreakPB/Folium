"""Trace event recorders."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from ..database import get_connection


class NullRecorder:
    def record(self, event: dict[str, Any]) -> None:
        return None

    def flush(self, trace_id: str) -> None:
        return None

    def path_for(self, trace_id: str) -> Path | None:
        return None


class SQLiteRecorder:
    """Buffer one trace in memory, then commit it to SQLite atomically."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path) if db_path is not None else None
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._lock = threading.Lock()

    def record(self, event: dict[str, Any]) -> None:
        trace_id = event.get("trace_id")
        if not trace_id:
            return
        with self._lock:
            self._events.setdefault(str(trace_id), []).append(dict(event))

    def flush(self, trace_id: str) -> None:
        with self._lock:
            events = self._events.pop(trace_id, [])
        if not events:
            return

        started = next((event for event in events if event.get("event") == "span_start" and not event.get("parent_span_id")), events[0])
        ended = next((event for event in reversed(events) if event.get("event") == "span_end" and not event.get("parent_span_id")), None)
        session_id = started.get("session_id")
        with get_connection(self.db_path) as conn:
            if session_id:
                exists = conn.execute(
                    "SELECT 1 FROM sessions WHERE id = ?",
                    (session_id,),
                ).fetchone()
                if not exists:
                    conn.execute(
                        """
                        INSERT INTO sessions (id, model, system_prompt, created_at, updated_at)
                        VALUES (?, ?, '', ?, ?)
                        """,
                        (
                            session_id,
                            (started.get("metadata") or {}).get("model") or "unknown",
                            started.get("timestamp") or "",
                            started.get("timestamp") or "",
                        ),
                    )
            conn.execute(
                """
                INSERT INTO traces (
                    trace_id, session_id, turn_index, model, name, status,
                    started_at, ended_at, duration_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(trace_id) DO UPDATE SET
                    session_id = excluded.session_id,
                    turn_index = excluded.turn_index,
                    model = excluded.model,
                    name = excluded.name,
                    status = excluded.status,
                    started_at = excluded.started_at,
                    ended_at = excluded.ended_at,
                    duration_ms = excluded.duration_ms
                """,
                (
                    trace_id,
                    session_id,
                    started.get("turn_index"),
                    (started.get("metadata") or {}).get("model"),
                    started.get("name"),
                    (ended or {}).get("status", "unknown"),
                    started.get("timestamp"),
                    (ended or {}).get("timestamp"),
                    (ended or {}).get("duration_ms", 0),
                ),
            )
            conn.execute("DELETE FROM trace_events WHERE trace_id = ?", (trace_id,))
            conn.executemany(
                """
                INSERT INTO trace_events (
                    trace_id, event_index, timestamp, event_type, name, type,
                    status, duration_ms, span_id, parent_span_id, tool_call_id,
                    payload_json, error_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        trace_id,
                        index,
                        event.get("timestamp"),
                        event.get("event"),
                        event.get("name"),
                        event.get("type"),
                        event.get("status"),
                        event.get("duration_ms"),
                        event.get("span_id"),
                        event.get("parent_span_id"),
                        (event.get("metadata") or {}).get("tool_call_id"),
                        json.dumps(event.get("metadata") or {}, ensure_ascii=False, default=str),
                        json.dumps(event.get("error"), ensure_ascii=False, default=str)
                        if event.get("error") is not None
                        else None,
                    )
                    for index, event in enumerate(events)
                ],
            )
