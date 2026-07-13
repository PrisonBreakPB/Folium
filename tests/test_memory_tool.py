from unittest import mock

from folium.prompt import MAX_MEMORY_CHARS
from folium.tools.memory import MemoryTool


def test_memory_tool_appends_to_requested_section(tmp_path):
    memory_file = tmp_path / "memory.md"
    with mock.patch("folium.tools.memory.MEMORY_FILE", memory_file):
        result = MemoryTool().execute(
            section="user_preferences",
            content="Prefer Chinese responses.",
        )

    assert result == "Saved long-term memory in user_preferences: Prefer Chinese responses."
    assert memory_file.read_text(encoding="utf-8") == """# Folium Memory

## User Preferences
- Prefer Chinese responses.

## Long-Term Context

## Confirmed Decisions

## Open Items
"""


def test_memory_tool_rejects_duplicate_entry(tmp_path):
    memory_file = tmp_path / "memory.md"
    memory_file.write_text(
        "# Folium Memory\n\n## User Preferences\n- Prefer Chinese responses.\n\n"
        "## Long-Term Context\n\n## Confirmed Decisions\n\n## Open Items\n",
        encoding="utf-8",
    )
    with mock.patch("folium.tools.memory.MEMORY_FILE", memory_file):
        result = MemoryTool().execute(
            section="user_preferences",
            content="Prefer Chinese responses.",
        )

    assert result == "Error: this memory entry already exists"


def test_memory_tool_rejects_unknown_section(tmp_path):
    with mock.patch("folium.tools.memory.MEMORY_FILE", tmp_path / "memory.md"):
        result = MemoryTool().execute(section="unknown", content="Keep this.")

    assert result.startswith("Error: section must be one of:")


def test_memory_tool_rejects_entries_beyond_prompt_limit(tmp_path):
    memory_file = tmp_path / "memory.md"
    template = """# Folium Memory

## User Preferences

## Long-Term Context

## Confirmed Decisions

## Open Items
"""
    memory_file.write_text(
        template + ("x" * (MAX_MEMORY_CHARS - len(template))),
        encoding="utf-8",
    )
    with mock.patch("folium.tools.memory.MEMORY_FILE", memory_file):
        result = MemoryTool().execute(
            section="user_preferences",
            content="Prefer Chinese responses.",
        )

    assert "memory would exceed" in result
    assert len(memory_file.read_text(encoding="utf-8")) == MAX_MEMORY_CHARS


def test_memory_tool_uses_host_path_when_copy_sandbox_is_enabled(tmp_path, monkeypatch):
    memory_file = tmp_path / "memory.md"
    monkeypatch.setenv("FOLIUM_SANDBOX_WORKSPACE_MODE", "copy")
    with mock.patch("folium.tools.memory.MEMORY_FILE", memory_file):
        result = MemoryTool().execute(
            section="confirmed_decisions",
            content="Use the Web UI by default.",
        )

    assert result.startswith("Saved long-term memory")
    assert "- Use the Web UI by default." in memory_file.read_text(encoding="utf-8")
