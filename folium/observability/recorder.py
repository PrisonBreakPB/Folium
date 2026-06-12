"""Trace event recorders."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any


class NullRecorder:
    def record(self, event: dict[str, Any]) -> None:
        return None

    def path_for(self, trace_id: str) -> Path | None:
        return None


class JSONLRecorder:
    def __init__(self, trace_dir: str | Path):
        self.trace_dir = Path(trace_dir)
        self._lock = threading.Lock()

    def path_for(self, trace_id: str) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in trace_id)
        return self.trace_dir / f"{safe}.jsonl"

    def record(self, event: dict[str, Any]) -> None:
        trace_id = event.get("trace_id")
        if not trace_id:
            return
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for(str(trace_id))
        line = json.dumps(event, ensure_ascii=False, default=str)
        with self._lock:
            with path.open("a", encoding="utf-8") as fp:
                fp.write(line + "\n")
