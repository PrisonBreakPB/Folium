"""Read persisted history for the current agent session."""

from __future__ import annotations

from .base import Tool, ToolOutput
from ..database import get_connection


_DEFAULT_SEARCH_LIMIT = 10
_MAX_SEARCH_LIMIT = 20
_DEFAULT_READ_CHARS = 6000
_MAX_READ_CHARS = 12000


class SessionHistoryTool(Tool):
    name = "session_history"
    description = (
        "Search the persisted transcript of the current session, including complete original "
        "tool outputs that may no longer be in the active context after compression. "
        "Use action='search' to find relevant message IDs, then action='read' to retrieve one "
        "message in full or in chunks. This tool can read only the current session."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "Operation: 'search' for matching history or 'read' for one message",
            },
            "query": {
                "type": "string",
                "description": "Optional keyword for search; omit it to list recent messages",
            },
            "message_id": {
                "type": "integer",
                "description": "Message ID returned by search; required for action='read'",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum search results, from 1 to 20; defaults to 10",
            },
            "offset": {
                "type": "integer",
                "description": "Character offset for action='read'; defaults to 0",
            },
            "max_chars": {
                "type": "integer",
                "description": "Maximum characters returned by action='read', up to 12000",
            },
        },
        "required": ["action"],
    }

    _parent_agent = None

    def execute(
        self,
        action: str,
        query: str | None = None,
        message_id: int | None = None,
        limit: int | None = None,
        offset: int | None = None,
        max_chars: int | None = None,
    ) -> str | ToolOutput:
        session_id = self._current_session_id()
        if not session_id:
            return "Error: session_history is unavailable because there is no active session"

        if action == "search":
            return self._search(session_id, query or "", limit)
        if action == "read":
            if message_id is None:
                return "Error: message_id is required for action='read'"
            return self._read(session_id, message_id, offset, max_chars)
        return "Error: action must be 'search' or 'read'"

    def _current_session_id(self) -> str | None:
        session_id = getattr(self._parent_agent, "session_id", None)
        return session_id if isinstance(session_id, str) and session_id else None

    def _search(self, session_id: str, query: str, limit: int | None) -> str | ToolOutput:
        result_limit = _bounded(limit, _DEFAULT_SEARCH_LIMIT, _MAX_SEARCH_LIMIT)
        clauses = [
            "session_id = ?",
            "transcript_position IS NOT NULL",
            "COALESCE(tool_name, '') <> 'session_history'",
        ]
        parameters: list[object] = [session_id]
        if query:
            pattern = f"%{_escape_like(query)}%"
            clauses.append("(content LIKE ? ESCAPE '\\' OR COALESCE(tool_name, '') LIKE ? ESCAPE '\\')")
            parameters.extend([pattern, pattern])

        where = " AND ".join(clauses)
        parameters.append(result_limit)
        with get_connection() as conn:
            rows = conn.execute(
                f"""
                SELECT * FROM (
                    SELECT id, transcript_position, role, content, tool_name, tool_call_id
                    FROM messages
                    WHERE {where}
                    ORDER BY transcript_position DESC, id DESC
                    LIMIT ?
                )
                ORDER BY transcript_position, id
                """,
                parameters,
            ).fetchall()

        if not rows:
            suffix = f" for {query!r}" if query else ""
            return f"No session history found{suffix}."

        header = f"Session history search returned {len(rows)} message(s)"
        if query:
            header += f" for {query!r}"
        lines = [header + ":"]
        for row in rows:
            tool = f", tool={row['tool_name']}" if row["tool_name"] else ""
            lines.extend([
                (
                    f"[message_id={row['id']}, position={row['transcript_position']}, "
                    f"role={row['role']}{tool}]"
                ),
                _snippet(row["content"], query),
            ])
        return ToolOutput(content="\n".join(lines), preview=header)

    def _read(
        self,
        session_id: str,
        message_id: int,
        offset: int | None,
        max_chars: int | None,
    ) -> str | ToolOutput:
        read_offset = max(0, offset or 0)
        char_limit = _bounded(max_chars, _DEFAULT_READ_CHARS, _MAX_READ_CHARS)
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT id, transcript_position, role, content, tool_name, tool_call_id
                FROM messages
                WHERE id = ? AND session_id = ? AND transcript_position IS NOT NULL
                  AND COALESCE(tool_name, '') <> 'session_history'
                """,
                (message_id, session_id),
            ).fetchone()

        if not row:
            return f"Error: message_id {message_id} was not found in the current session"

        content = row["content"]
        if read_offset >= len(content):
            return f"Error: offset {read_offset} is beyond message length {len(content)}"
        chunk = content[read_offset:read_offset + char_limit]
        end = read_offset + len(chunk)
        tool = f", tool={row['tool_name']}" if row["tool_name"] else ""
        header = (
            f"[message_id={row['id']}, position={row['transcript_position']}, "
            f"role={row['role']}{tool}, chars={read_offset}:{end}/{len(content)}]"
        )
        if end < len(content):
            chunk += f"\n\n... ({len(content) - end} more characters; call read with offset={end})"
        return ToolOutput(content=f"{header}\n{chunk}", preview=f"Read session history message {message_id}")


def _bounded(value: int | None, default: int, maximum: int) -> int:
    return max(1, min(value if value is not None else default, maximum))


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _snippet(content: str, query: str, limit: int = 320) -> str:
    if not content:
        return "(empty)"
    start = 0
    if query:
        start = content.casefold().find(query.casefold())
        start = max(0, start - 80) if start >= 0 else 0
    end = min(len(content), start + limit)
    prefix = "..." if start else ""
    suffix = "..." if end < len(content) else ""
    return prefix + content[start:end] + suffix
