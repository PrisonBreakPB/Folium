"""Idempotent import of legacy JSON sessions and JSONL traces."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .database import get_connection
from .session import save_session


def migrate_legacy_storage(root: str | Path | None = None) -> dict[str, int]:
    """Import old `conversations/` data without deleting the source files."""
    conversations = Path(root) if root is not None else Path.cwd() / "conversations"
    imported = {"sessions": 0, "traces": 0}
    if not conversations.exists():
        return imported

    for path in conversations.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            session_id = data.get("id") or path.stem
            with get_connection() as conn:
                exists = conn.execute(
                    "SELECT 1 FROM sessions WHERE id = ?",
                    (session_id,),
                ).fetchone()
            if exists:
                continue
            prompt = _legacy_prompt(session_id)
            save_session(
                data.get("messages") or [],
                data.get("model") or "unknown",
                session_id,
                transcript=data.get("transcript") or data.get("messages") or [],
                system_prompt=prompt,
            )
            imported["sessions"] += 1
        except (OSError, json.JSONDecodeError, TypeError):
            continue

    trace_root = conversations / "traces"
    for path in trace_root.glob("*.jsonl") if trace_root.exists() else []:
        if _import_trace(path):
            imported["traces"] += 1
    return imported


def _legacy_prompt(session_id: str) -> str:
    path = Path.cwd() / "data" / "session_prompts.db"
    if not path.exists():
        return ""
    try:
        with sqlite3.connect(path) as conn:
            row = conn.execute(
                "SELECT system_prompt FROM session_prompts WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return row[0] if row else ""
    except sqlite3.Error:
        return ""


def _import_trace(path: Path) -> bool:
    trace_id = path.stem
    with get_connection() as conn:
        if conn.execute("SELECT 1 FROM traces WHERE trace_id = ?", (trace_id,)).fetchone():
            return False
    events = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if isinstance(event, dict):
                events.append(event)
    except (OSError, json.JSONDecodeError):
        return False
    if not events:
        return False

    from .observability.recorder import SQLiteRecorder

    recorder = SQLiteRecorder()
    for event in events:
        recorder.record(event)
    recorder.flush(trace_id)
    return True
