"""SQLite trace query helpers."""

from __future__ import annotations

import json
from pathlib import Path

from ..database import get_connection


def _db_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.suffix == ".db" else path / "folium.db"


def list_traces(trace_dir: str | Path | None = None, limit: int = 20) -> list[dict]:
    with get_connection(_db_path(trace_dir)) as conn:
        rows = conn.execute(
            """
            SELECT
                t.trace_id, t.status, t.duration_ms, t.session_id, t.turn_index, t.started_at,
                SUM(CASE WHEN e.event_type = 'span_end' AND e.type = 'llm' THEN 1 ELSE 0 END) AS llm_calls,
                SUM(CASE WHEN e.event_type = 'span_end' AND e.type = 'tool' THEN 1 ELSE 0 END) AS tool_calls,
                SUM(CASE WHEN e.status = 'error' THEN 1 ELSE 0 END) AS errors,
                SUM(CASE WHEN e.event_type = 'llm_request_snapshot' THEN 1 ELSE 0 END) AS llm_request_snapshots,
                SUM(CASE WHEN e.event_type = 'llm_response_snapshot' THEN 1 ELSE 0 END) AS llm_response_snapshots,
                SUM(CASE WHEN e.event_type = 'context_snapshot' THEN 1 ELSE 0 END) AS context_snapshots
            FROM traces t
            LEFT JOIN trace_events e ON e.trace_id = t.trace_id
            GROUP BY t.trace_id
            ORDER BY t.started_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [_summary_row(row) for row in rows]


def read_trace_summary(trace_id: str, trace_dir: str | Path | None = None) -> dict | None:
    with get_connection(_db_path(trace_dir)) as conn:
        row = conn.execute(
            """
            SELECT
                t.trace_id, t.status, t.duration_ms, t.session_id, t.turn_index, t.started_at,
                SUM(CASE WHEN e.event_type = 'span_end' AND e.type = 'llm' THEN 1 ELSE 0 END) AS llm_calls,
                SUM(CASE WHEN e.event_type = 'span_end' AND e.type = 'tool' THEN 1 ELSE 0 END) AS tool_calls,
                SUM(CASE WHEN e.status = 'error' THEN 1 ELSE 0 END) AS errors,
                SUM(CASE WHEN e.event_type = 'llm_request_snapshot' THEN 1 ELSE 0 END) AS llm_request_snapshots,
                SUM(CASE WHEN e.event_type = 'llm_response_snapshot' THEN 1 ELSE 0 END) AS llm_response_snapshots,
                SUM(CASE WHEN e.event_type = 'context_snapshot' THEN 1 ELSE 0 END) AS context_snapshots
            FROM traces t
            LEFT JOIN trace_events e ON e.trace_id = t.trace_id
            WHERE t.trace_id = ?
            GROUP BY t.trace_id
            """,
            (trace_id,),
        ).fetchone()
        if row is None:
            return None
        summary = _summary_row(row)
        events = conn.execute(
            """
            SELECT * FROM trace_events
            WHERE trace_id = ? AND event_type = 'span_end'
            ORDER BY event_index
            """,
            (trace_id,),
        ).fetchall()
    summary["spans"] = [
        {
            "span_id": event["span_id"],
            "parent_span_id": event["parent_span_id"],
            "name": event["name"],
            "type": event["type"],
            "status": event["status"],
            "duration_ms": event["duration_ms"],
            "metadata": _load_json(event["payload_json"]),
            "error": _load_json(event["error_json"]),
        }
        for event in events
    ]
    return summary


def delete_traces_for_session(session_id: str, trace_dir: str | Path | None = None) -> int:
    """Delete traces for a session. Normal session deletion cascades automatically."""
    with get_connection(_db_path(trace_dir)) as conn:
        count = conn.execute(
            "SELECT COUNT(*) FROM traces WHERE session_id = ?",
            (session_id,),
        ).fetchone()[0]
        conn.execute("DELETE FROM traces WHERE session_id = ?", (session_id,))
    return count


def _summary_row(row) -> dict:
    return {
        "trace_id": row["trace_id"],
        "status": row["status"] or "unknown",
        "duration_ms": row["duration_ms"] or 0,
        "llm_calls": row["llm_calls"] or 0,
        "tool_calls": row["tool_calls"] or 0,
        "errors": row["errors"] or 0,
        "llm_request_snapshots": row["llm_request_snapshots"] or 0,
        "llm_response_snapshots": row["llm_response_snapshots"] or 0,
        "context_snapshots": row["context_snapshots"] or 0,
        "session_id": row["session_id"],
        "turn_index": row["turn_index"],
        "started_at": row["started_at"],
    }


def _load_json(value: str | None):
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None
