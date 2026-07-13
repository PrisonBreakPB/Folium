"""SQLite-backed session persistence."""

from __future__ import annotations

import copy
import json
import re
import time
import uuid
from typing import Any

from .database import get_connection
from .encoding import repair_mojibake_text

_SAFE_SESSION_RE = re.compile(r"[^A-Za-z0-9._-]+")
_SUMMARY_MARKER = "Another language model has started working on this problem"


def _normalize_session_id(session_id: str | None) -> str:
    if not session_id:
        return _new_session_id()
    name = session_id.strip().replace("\\", "/").split("/")[-1]
    name = _SAFE_SESSION_RE.sub("-", name).strip(".-_")
    return name or _new_session_id()


def _new_session_id() -> str:
    return f"session_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def new_session_id() -> str:
    """Create a session ID without persisting it."""
    return _new_session_id()


def ensure_session(session_id: str, model: str, system_prompt: str = "") -> str:
    """Create a session row when needed so traces can reference it."""
    now = _now()
    session_id = _normalize_session_id(session_id)
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO sessions (id, model, system_prompt, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                model = excluded.model,
                system_prompt = CASE
                    WHEN excluded.system_prompt <> '' THEN excluded.system_prompt
                    ELSE sessions.system_prompt
                END,
                updated_at = excluded.updated_at
            """,
            (session_id, model, system_prompt, now, now),
        )
    return session_id


def save_session(
    messages: list[dict],
    model: str,
    session_id: str | None = None,
    transcript: list[dict] | None = None,
    system_prompt: str | None = None,
) -> str:
    """Persist canonical transcript and current model context in one transaction."""
    session_id = _normalize_session_id(session_id)
    transcript = transcript if transcript is not None else copy.deepcopy(messages)
    now = _now()
    with get_connection() as conn:
        existing = conn.execute(
            "SELECT created_at, system_prompt FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        prompt = system_prompt if system_prompt is not None else (existing["system_prompt"] if existing else "")
        conn.execute(
            """
            INSERT INTO sessions (id, model, system_prompt, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                model = excluded.model,
                system_prompt = excluded.system_prompt,
                updated_at = excluded.updated_at
            """,
            (session_id, model, prompt, existing["created_at"] if existing else now, now),
        )
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        _insert_messages(conn, session_id, messages, transcript)
    return session_id


def load_session(session_id: str) -> tuple[list[dict], str, list[dict], str | None] | None:
    """Return current context, model, canonical transcript, and system prompt."""
    session_id = _normalize_session_id(session_id)
    with get_connection() as conn:
        session = conn.execute(
            "SELECT model, system_prompt FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if not session:
            return None
        rows = conn.execute(
            """
            SELECT * FROM messages
            WHERE session_id = ?
            ORDER BY
                CASE WHEN transcript_position IS NULL THEN 1 ELSE 0 END,
                transcript_position,
                id
            """,
            (session_id,),
        ).fetchall()

    transcript = [_row_to_message(row, use_model_content=False) for row in rows if row["transcript_position"] is not None]
    context_rows = sorted(
        (row for row in rows if row["context_position"] is not None),
        key=lambda row: row["context_position"],
    )
    messages = [_row_to_message(row, use_model_content=True) for row in context_rows]
    return messages, session["model"], transcript, session["system_prompt"] or None


def delete_session(session_id: str) -> bool:
    """Delete a session and all related messages, traces, and trace events."""
    session_id = _normalize_session_id(session_id)
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    return cursor.rowcount > 0


def calculate_session_stats(session_id: str) -> dict:
    """Aggregate persisted LLM token usage from trace events."""
    stats = {
        "prompt_tokens": 0,
        "cache_miss_tokens": 0,
        "cached_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
        "cache_hit_rate": 0,
    }
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT e.trace_id, e.event_index, e.event_type, e.span_id, e.payload_json
            FROM trace_events e
            JOIN traces t ON t.trace_id = e.trace_id
            WHERE t.session_id = ? AND e.event_type IN ('llm_result', 'llm_response_snapshot')
            ORDER BY e.trace_id, e.event_index
            """,
            (session_id,),
        ).fetchall()
    usage_by_span: dict[tuple[str, str], tuple[str, dict]] = {}
    for row in rows:
        try:
            metadata = json.loads(row["payload_json"] or "{}")
        except json.JSONDecodeError:
            continue

        span_id = row["span_id"] or f"event_{row['event_index']}"
        key = (row["trace_id"], span_id)
        existing = usage_by_span.get(key)
        if existing and (existing[0] == "llm_result" or row["event_type"] != "llm_result"):
            continue
        usage_by_span[key] = (row["event_type"], metadata)

    for _, metadata in usage_by_span.values():
        prompt = int(metadata.get("prompt_tokens") or 0)
        completion = int(metadata.get("completion_tokens") or 0)
        cached = int(metadata.get("cached_tokens") or 0)
        stats["prompt_tokens"] += prompt
        stats["completion_tokens"] += completion
        stats["cached_tokens"] += cached
        cost = metadata.get("cost")
        if isinstance(cost, int | float):
            stats["cost"] += cost
    stats["cache_miss_tokens"] = stats["prompt_tokens"] - stats["cached_tokens"]
    stats["total_tokens"] = stats["prompt_tokens"] + stats["completion_tokens"]
    stats["cache_hit_rate"] = (
        stats["cached_tokens"] / stats["prompt_tokens"]
        if stats["prompt_tokens"] > 0
        else 0
    )
    return stats


def list_sessions() -> list[dict]:
    """List non-empty saved sessions, newest first."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                s.id,
                s.model,
                s.created_at,
                s.updated_at,
                (
                    SELECT m.content
                    FROM messages m
                    WHERE m.session_id = s.id AND m.role = 'user'
                      AND m.transcript_position IS NOT NULL
                    ORDER BY m.transcript_position
                    LIMIT 1
                ) AS preview
            FROM sessions s
            WHERE EXISTS (
                SELECT 1 FROM messages m
                WHERE m.session_id = s.id AND m.transcript_position IS NOT NULL
            )
            ORDER BY s.updated_at DESC
            """
        ).fetchall()
    return [
        {
            "id": row["id"],
            "model": row["model"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "preview": repair_mojibake_text(row["preview"] or "")[:80],
        }
        for row in rows
    ]


def _insert_messages(conn, session_id: str, messages: list[dict], transcript: list[dict]) -> None:
    records: list[dict[str, Any]] = []
    transcript_records: list[dict[str, Any]] = []
    for position, message in enumerate(transcript):
        record = _message_record(message)
        record["transcript_position"] = position
        transcript_records.append(record)
        records.append(record)

    unmatched = list(transcript_records)
    context_records: list[dict[str, Any]] = []
    for position, message in enumerate(messages):
        record = _message_record(message)
        match = _find_matching_transcript_record(record, unmatched)
        if match is not None:
            match["context_position"] = position
            model_content = record["model_content"] or record["content"]
            if model_content != match["content"]:
                match["model_content"] = model_content
            unmatched.remove(match)
        else:
            record["context_position"] = position
            record["kind"] = _context_kind(record)
            context_records.append(record)

    for record in records + context_records:
        conn.execute(
            """
            INSERT INTO messages (
                session_id, role, content, model_content, transcript_position,
                context_position, kind, tool_call_id, tool_name, tool_calls_json,
                metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                record["role"],
                record["content"],
                record["model_content"],
                record.get("transcript_position"),
                record.get("context_position"),
                record["kind"],
                record.get("tool_call_id"),
                record.get("tool_name"),
                _dump_json(record.get("tool_calls")),
                _dump_json(record.get("metadata")),
            ),
        )


def _message_record(message: dict) -> dict[str, Any]:
    raw_content = message.get("_raw_content")
    content = str(raw_content if raw_content is not None else message.get("content") or "")
    visible = str(message.get("content") or "")
    metadata = {
        key: value
        for key, value in message.items()
        if key not in {
            "role", "content", "_raw_content", "tool_call_id", "name",
            "tool_calls", "_usage",
        }
    }
    return {
        "role": str(message.get("role") or "user"),
        "content": content,
        "model_content": visible if visible != content else None,
        "kind": "tool" if message.get("role") == "tool" else "normal",
        "tool_call_id": message.get("tool_call_id"),
        "tool_name": message.get("name"),
        "tool_calls": message.get("tool_calls"),
        "metadata": metadata or None,
    }


def _find_matching_transcript_record(record: dict, candidates: list[dict]) -> dict | None:
    for candidate in candidates:
        if candidate["role"] != record["role"]:
            continue
        if candidate.get("tool_call_id") != record.get("tool_call_id"):
            continue
        if candidate.get("tool_name") != record.get("tool_name"):
            continue
        return candidate
    return None


def _context_kind(record: dict) -> str:
    if record["role"] == "tool":
        return "tool"
    if _SUMMARY_MARKER in (record["model_content"] or record["content"]):
        return "context_summary"
    return "internal"


def _row_to_message(row, *, use_model_content: bool) -> dict:
    content = row["model_content"] if use_model_content and row["model_content"] is not None else row["content"]
    message = {"role": row["role"], "content": content}
    if row["tool_call_id"] is not None:
        message["tool_call_id"] = row["tool_call_id"]
    if row["tool_name"] is not None:
        message["name"] = row["tool_name"]
    tool_calls = _load_json(row["tool_calls_json"])
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    metadata = _load_json(row["metadata_json"])
    if isinstance(metadata, dict):
        message.update(metadata)
    return message


def _dump_json(value: Any) -> str | None:
    return json.dumps(value, ensure_ascii=False, default=str) if value is not None else None


def _load_json(value: str | None) -> Any:
    if value is None:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")
