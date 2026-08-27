"""Persistent long-term memory tool."""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .base import Tool, ToolFailure, tool_failure
from ..prompt import MAX_MEMORY_CHARS, MEMORY_FILE


_MEMORY_LOCK = threading.Lock()
_MAX_ENTRY_CHARS = 500
_SECTIONS = {
    "user_preferences": "User Preferences",
    "long_term_context": "Long-Term Context",
    "confirmed_decisions": "Confirmed Decisions",
    "open_items": "Open Items",
}


class MemoryArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["read", "append"] = Field(default="append", description="read returns memory and its version; append adds one entry.")
    section: str | None = Field(default=None, description="Memory category: user_preferences, long_term_context, confirmed_decisions, or open_items.")
    content: str | None = Field(default=None, description="One concise, durable fact to append.")
    expected_version: str | None = Field(default=None, description="Version returned by read. Use for conflict-safe background appends.")


class MemoryTool(Tool):
    name = "memory"
    description = (
        "Read or persist durable long-term memory. Append only stable user preferences, "
        "research context, confirmed decisions, or open items; never store secrets, full "
        "chat transcripts, temporary guesses, or raw tool output."
    )
    args_model = MemoryArgs

    def execute(
        self,
        section: str | None = None,
        content: str | None = None,
        action: str | None = None,
        expected_version: str | None = None,
    ) -> str | ToolFailure:
        action = action or "append"
        if action == "read":
            if section is not None or content is not None or expected_version is not None:
                return tool_failure("invalid_read_arguments", "validation", "read does not accept section, content, or expected_version")
            try:
                with _MEMORY_LOCK:
                    memory = _read_memory(MEMORY_FILE)
            except Exception as exc:
                return tool_failure("memory_read_failed", "filesystem", str(exc))
            return f"Memory version: {_memory_version(memory)}\n\n{memory}"
        if action != "append":
            return tool_failure("invalid_action", "validation", "action must be one of: read, append")
        if section is None or content is None:
            return tool_failure("missing_memory_fields", "validation", "append requires section and content")

        heading = _SECTIONS.get(section)
        if heading is None:
            return tool_failure("invalid_memory_section", "validation", "section must be one of: " + ", ".join(_SECTIONS))

        entry = " ".join(content.split())
        if not entry:
            return tool_failure("empty_memory_content", "validation", "content must not be empty")
        if len(entry) > _MAX_ENTRY_CHARS:
            return tool_failure("memory_content_too_long", "validation", f"content must be at most {_MAX_ENTRY_CHARS} characters")

        with _MEMORY_LOCK:
            try:
                memory = _read_memory(MEMORY_FILE)
            except Exception as exc:
                return tool_failure("memory_read_failed", "filesystem", str(exc))
            if expected_version is not None and expected_version != _memory_version(memory):
                return tool_failure(
                    "memory_conflict",
                    "conflict",
                    "memory changed; read the latest memory before retrying.",
                    retryable=True,
                    content="Conflict: memory changed; read the latest memory before retrying.",
                )
            bullet = f"- {entry}"
            if bullet in memory.splitlines():
                return tool_failure("duplicate_memory_entry", "conflict", "this memory entry already exists")

            updated = _append_to_section(memory, heading, bullet)
            if len(updated) > MAX_MEMORY_CHARS:
                return tool_failure(
                    "memory_limit_exceeded",
                    "validation",
                    f"memory would exceed the {MAX_MEMORY_CHARS}-character prompt limit; do not add this entry",
                )

            try:
                _write_memory(MEMORY_FILE, updated)
            except Exception as exc:
                return tool_failure("memory_write_failed", "filesystem", str(exc))

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
