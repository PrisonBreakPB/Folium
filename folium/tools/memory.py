"""Persistent long-term memory tool."""

from __future__ import annotations

import threading
from pathlib import Path

from .base import Tool
from ..prompt import MAX_MEMORY_CHARS, MEMORY_FILE


_MEMORY_LOCK = threading.Lock()
_MAX_ENTRY_CHARS = 500
_SECTIONS = {
    "user_preferences": "User Preferences",
    "long_term_context": "Long-Term Context",
    "confirmed_decisions": "Confirmed Decisions",
    "open_items": "Open Items",
}


class MemoryTool(Tool):
    name = "memory"
    description = (
        "Persist one durable user preference, stable research context, confirmed decision, "
        "or open item in long-term memory. Use only for information likely to remain useful "
        "across future tasks; never store secrets, full chat transcripts, temporary guesses, "
        "or raw tool output."
    )
    parameters = {
        "type": "object",
        "properties": {
            "section": {
                "type": "string",
                "description": (
                    "Memory category: user_preferences, long_term_context, "
                    "confirmed_decisions, or open_items."
                ),
            },
            "content": {
                "type": "string",
                "description": "One concise, durable fact to append.",
            },
        },
        "required": ["section", "content"],
    }

    def execute(self, section: str, content: str) -> str:
        heading = _SECTIONS.get(section)
        if heading is None:
            return "Error: section must be one of: " + ", ".join(_SECTIONS)

        entry = " ".join(content.split())
        if not entry:
            return "Error: content must not be empty"
        if len(entry) > _MAX_ENTRY_CHARS:
            return f"Error: content must be at most {_MAX_ENTRY_CHARS} characters"

        with _MEMORY_LOCK:
            memory = _read_memory(MEMORY_FILE)
            bullet = f"- {entry}"
            if bullet in memory.splitlines():
                return "Error: this memory entry already exists"

            updated = _append_to_section(memory, heading, bullet)
            if len(updated) > MAX_MEMORY_CHARS:
                return (
                    f"Error: memory would exceed the {MAX_MEMORY_CHARS}-character prompt limit; "
                    "do not add this entry"
                )

            MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
            MEMORY_FILE.write_text(updated, encoding="utf-8")

        return f"Saved long-term memory in {section}: {entry}"


def _read_memory(path: Path) -> str:
    try:
        memory = path.read_text(encoding="utf-8-sig")
    except FileNotFoundError:
        return _empty_memory()
    except OSError as exc:
        raise RuntimeError(f"could not read memory file: {exc}") from exc
    return memory.strip() or _empty_memory()


def _empty_memory() -> str:
    return """# Folium Memory

## User Preferences

## Long-Term Context

## Confirmed Decisions

## Open Items"""


def _append_to_section(memory: str, heading: str, bullet: str) -> str:
    lines = memory.splitlines()
    target = f"## {heading}"
    try:
        index = lines.index(target)
    except ValueError as exc:
        raise RuntimeError(f"memory file is missing required section '{heading}'") from exc

    insert_at = index + 1
    while insert_at < len(lines) and not lines[insert_at].startswith("## "):
        insert_at += 1
    while insert_at > index + 1 and not lines[insert_at - 1].strip():
        insert_at -= 1
    lines.insert(insert_at, bullet)
    return "\n".join(lines).rstrip() + "\n"
