"""ID helpers for traces, spans, and artifacts."""

from __future__ import annotations

import time
import uuid


def new_trace_id() -> str:
    return f"trace_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def new_span_id() -> str:
    return f"span_{uuid.uuid4().hex[:12]}"


def new_artifact_id() -> str:
    return f"artifact_{uuid.uuid4().hex[:12]}"
