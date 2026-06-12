"""Small helpers for safe trace payloads."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


_SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{12,}"),
    re.compile(r"(?i)(api[_-]?key|token|secret|password)\s*[:=]\s*['\"]?[^'\"\s,}]+"),
    re.compile(r"(?i)(authorization:\s*bearer\s+)[A-Za-z0-9._-]+"),
]


def stable_hash(value: Any) -> str:
    text = _to_text(value)
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def preview(value: Any, max_chars: int = 1000, redact: bool = True) -> str:
    text = _to_text(value)
    if redact:
        text = redact_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"... ({len(text)} chars total)"


def redact_text(text: str) -> str:
    redacted = text
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub(_replace_secret, redacted)
    return redacted


def compact_payload(
    value: Any,
    *,
    include_full: bool,
    max_preview_chars: int,
    redact: bool,
) -> dict:
    text = _to_text(value)
    payload = {
        "preview": preview(text, max_preview_chars, redact),
        "length": len(text),
        "hash": stable_hash(text),
    }
    if include_full:
        payload["value"] = redact_text(text) if redact else text
    return payload


def _replace_secret(match: re.Match) -> str:
    text = match.group(0)
    if ":" in text and text.lower().startswith("authorization"):
        return match.group(1) + "[REDACTED]"
    key = re.split(r"[:=]", text, maxsplit=1)[0]
    return f"{key}=[REDACTED]"


def _to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return str(value)
