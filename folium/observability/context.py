"""Context-managed tracing helpers."""

from __future__ import annotations

import contextvars
import time
import traceback
from contextlib import contextmanager
from typing import Any, Iterator

from .config import ObservabilityConfig
from .ids import new_span_id, new_trace_id
from .recorder import JSONLRecorder, NullRecorder

_trace_id_var = contextvars.ContextVar("folium_trace_id", default=None)
_span_stack_var = contextvars.ContextVar("folium_span_stack", default=())
_span_status_var = contextvars.ContextVar("folium_span_status", default={})
_observer_var = contextvars.ContextVar("folium_observer", default=None)


class Observer:
    def __init__(self, config: ObservabilityConfig | None = None):
        self.config = config or ObservabilityConfig.from_env()
        self.recorder = (
            JSONLRecorder(self.config.trace_dir)
            if self.config.enabled
            else NullRecorder()
        )

    def record(self, event: dict[str, Any]) -> None:
        if not self.config.enabled:
            return
        self.recorder.record(event)


def active_observer() -> Observer:
    observer = _observer_var.get()
    if observer is None:
        observer = Observer()
        _observer_var.set(observer)
    return observer


def current_trace_id() -> str | None:
    return _trace_id_var.get()


def current_span_id() -> str | None:
    stack = _span_stack_var.get()
    return stack[-1] if stack else None


def mark_current_span_status(status: str) -> None:
    span_id = current_span_id()
    if not span_id:
        return
    statuses = dict(_span_status_var.get())
    statuses[span_id] = status
    _span_status_var.set(statuses)


@contextmanager
def observe_trace(
    name: str,
    span_type: str = "agent",
    *,
    trace_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    session_id: str | None = None,
    turn_index: int | None = None,
) -> Iterator[str]:
    observer = active_observer()
    tid = trace_id or new_trace_id()
    token_trace = _trace_id_var.set(tid)
    token_stack = _span_stack_var.set(())
    try:
        with span(
            name,
            span_type,
            metadata=metadata,
            session_id=session_id,
            turn_index=turn_index,
            force_trace_id=tid,
        ) as span_id:
            yield span_id
    finally:
        _span_stack_var.reset(token_stack)
        _trace_id_var.reset(token_trace)


@contextmanager
def span(
    name: str,
    span_type: str,
    *,
    metadata: dict[str, Any] | None = None,
    session_id: str | None = None,
    turn_index: int | None = None,
    force_trace_id: str | None = None,
) -> Iterator[str]:
    observer = active_observer()
    trace_id = force_trace_id or current_trace_id() or new_trace_id()
    stack = _span_stack_var.get()
    parent_span_id = stack[-1] if stack else None
    span_id = new_span_id()
    start = time.time()

    base = {
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "name": name,
        "type": span_type,
        "session_id": session_id,
        "turn_index": turn_index,
    }
    observer.record({
        "event": "span_start",
        "timestamp": _iso_time(start),
        "metadata": metadata or {},
        **base,
    })

    token_trace = _trace_id_var.set(trace_id)
    token_stack = _span_stack_var.set(stack + (span_id,))
    try:
        yield span_id
    except Exception as exc:
        end = time.time()
        observer.record({
            "event": "span_end",
            "timestamp": _iso_time(end),
            "status": "error",
            "duration_ms": int((end - start) * 1000),
            "error": {
                "type": exc.__class__.__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=5),
            },
            **base,
        })
        raise
    else:
        end = time.time()
        status = _span_status_var.get().get(span_id, "ok")
        observer.record({
            "event": "span_end",
            "timestamp": _iso_time(end),
            "status": status,
            "duration_ms": int((end - start) * 1000),
            **base,
        })
    finally:
        statuses = dict(_span_status_var.get())
        statuses.pop(span_id, None)
        _span_status_var.set(statuses)
        _span_stack_var.reset(token_stack)
        _trace_id_var.reset(token_trace)


def record_event(name: str, event_type: str, metadata: dict[str, Any] | None = None) -> None:
    trace_id = current_trace_id()
    if not trace_id:
        return
    active_observer().record({
        "event": event_type,
        "name": name,
        "trace_id": trace_id,
        "span_id": current_span_id(),
        "timestamp": _iso_time(time.time()),
        "metadata": metadata or {},
    })


def _iso_time(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts)) + f".{int((ts % 1) * 1000):03d}"
