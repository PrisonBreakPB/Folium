"""Persistent long-term memory tool."""

from __future__ import annotations

import hashlib
import os
import tempfile
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
        "Read or persist durable long-term memory. Append only stable user preferences, "
        "research context, confirmed decisions, or open items; never store secrets, full "
        "chat transcripts, temporary guesses, or raw tool output."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["read", "append"],
                "description": "read returns memory and its version; append adds one entry.",
            },
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
            "expected_version": {
                "type": "string",
                "description": "Version returned by read. Use for conflict-safe background appends.",
            },
        },
        "required": [],
    }

    def execute(
        self,
        section: str | None = None,
        content: str | None = None,
        action: str | None = None,
        expected_version: str | None = None,
    ) -> str:
        action = action or "append"
        if action == "read":
            if section is not None or content is not None or expected_version is not None:
                return "Error: read does not accept section, content, or expected_version"
            with _MEMORY_LOCK:
                memory = _read_memory(MEMORY_FILE)
            return f"Memory version: {_memory_version(memory)}\n\n{memory}"
        if action != "append":
            return "Error: action must be one of: read, append"
        if section is None or content is None:
            return "Error: append requires section and content"

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
            if expected_version is not None and expected_version != _memory_version(memory):
                return "Conflict: memory changed; read the latest memory before retrying."
            bullet = f"- {entry}"
            if bullet in memory.splitlines():
                return "Error: this memory entry already exists"

            updated = _append_to_section(memory, heading, bullet)
            if len(updated) > MAX_MEMORY_CHARS:
                return (
                    f"Error: memory would exceed the {MAX_MEMORY_CHARS}-character prompt limit; "
                    "do not add this entry"
                )

            _write_memory(MEMORY_FILE, updated)

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


def _memory_version(memory: str) -> str:
    return hashlib.sha256(memory.encode("utf-8")).hexdigest()


def _write_memory(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_file.write(content)
            temp_path = temp_file.name
        os.replace(temp_path, path)
    except OSError as exc:
        raise RuntimeError(f"could not write memory file: {exc}") from exc
    finally:
        if temp_path:
            try:
                Path(temp_path).unlink(missing_ok=True)
            except OSError:
                pass


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
