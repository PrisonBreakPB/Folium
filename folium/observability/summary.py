"""Utilities for reading trace files."""

from __future__ import annotations

import json
from pathlib import Path

from .config import ObservabilityConfig


def list_traces(trace_dir: str | Path | None = None, limit: int = 20) -> list[dict]:
    root = Path(trace_dir) if trace_dir else ObservabilityConfig.from_env().trace_dir
    if not root.exists():
        return []
    files = sorted(root.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [_file_summary(path) for path in files[:limit]]


def read_trace_summary(trace_id: str, trace_dir: str | Path | None = None) -> dict | None:
    root = Path(trace_dir) if trace_dir else ObservabilityConfig.from_env().trace_dir
    path = root / f"{trace_id}.jsonl"
    if not path.exists():
        return None
    return _file_summary(path, include_spans=True)


def delete_traces_for_session(session_id: str, trace_dir: str | Path | None = None) -> int:
    """Delete trace files associated with a session. Returns deleted file count."""
    root = Path(trace_dir) if trace_dir else ObservabilityConfig.from_env().trace_dir
    if not root.exists():
        return 0

    deleted = 0
    for path in root.glob("*.jsonl"):
        if _trace_matches_session(path, session_id):
            path.unlink()
            deleted += 1
    return deleted


def _trace_matches_session(path: Path, session_id: str) -> bool:
    try:
        with path.open(encoding="utf-8") as fp:
            for line in fp:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("session_id") == session_id:
                    return True
    except OSError:
        return False
    return False


def _file_summary(path: Path, include_spans: bool = False) -> dict:
    starts: dict[str, dict] = {}
    spans: list[dict] = []
    summary = {
        "trace_id": path.stem,
        "path": str(path),
        "status": "unknown",
        "duration_ms": 0,
        "llm_calls": 0,
        "tool_calls": 0,
        "errors": 0,
        "llm_request_snapshots": 0,
        "llm_response_snapshots": 0,
        "context_snapshots": 0,
        "session_id": None,
        "turn_index": None,
        "started_at": None,
    }
    with path.open(encoding="utf-8") as fp:
        for line in fp:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "span_start":
                starts[event["span_id"]] = event
                if summary["started_at"] is None:
                    summary["started_at"] = event.get("timestamp")
                summary["session_id"] = summary["session_id"] or event.get("session_id")
                summary["turn_index"] = summary["turn_index"] if summary["turn_index"] is not None else event.get("turn_index")
            elif event.get("event") == "span_end":
                if event.get("type") == "llm":
                    summary["llm_calls"] += 1
                if event.get("type") == "tool":
                    summary["tool_calls"] += 1
                if event.get("status") == "error":
                    summary["errors"] += 1
                if event.get("parent_span_id") is None:
                    summary["status"] = event.get("status", "unknown")
                    summary["duration_ms"] = event.get("duration_ms", 0)
                if include_spans:
                    start = starts.get(event.get("span_id"), {})
                    spans.append({
                        "span_id": event.get("span_id"),
                        "parent_span_id": event.get("parent_span_id"),
                        "name": event.get("name"),
                        "type": event.get("type"),
                        "status": event.get("status"),
                        "duration_ms": event.get("duration_ms"),
                        "metadata": start.get("metadata", {}),
                        "error": event.get("error"),
                    })
            elif event.get("event") == "llm_request_snapshot":
                summary["llm_request_snapshots"] += 1
            elif event.get("event") == "llm_response_snapshot":
                summary["llm_response_snapshots"] += 1
            elif event.get("event") == "context_snapshot":
                summary["context_snapshots"] += 1
    if include_spans:
        summary["spans"] = spans
    return summary
