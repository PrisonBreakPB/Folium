"""Restricted L1 atom memory tool used by the background memory agent.

Wraps the ``l1_atom`` table (and its FTS5 index) behind a small allow-list of
actions so the background agent can recall candidates for dedup and persist
extracted atoms without arbitrary database access.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .base import Tool, ToolOutput, tool_failure
from ..database import get_connection


class MemoryAtomArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    action: Literal["search", "read", "upsert"] = Field(
        description="Operation: 'search' the L1 atom memory, 'read' one atom, or 'upsert' an atom"
    )
    query: str | None = Field(
        default=None,
        description="Keyword query for action='search' (matched against atom content)",
    )
    topic: str | None = Field(
        default=None,
        description="Optional topic filter for 'search', or required topic for 'upsert'",
    )
    atom_id: int | None = Field(
        default=None,
        description="Atom id returned by search; required for action='read'",
    )
    content: str | None = Field(
        default=None,
        description="Atom content to store, required for action='upsert'",
    )
    session_id: str | None = Field(
        default=None,
        description="Optional source session id recorded on 'upsert'",
    )
    existing_id: int | None = Field(
        default=None,
        description="When set, 'upsert' updates this atom instead of inserting a new one",
    )
    limit: int | None = Field(
        default=None, description="Maximum results for 'search', from 1 to 20; defaults to 10"
    )


class MemoryAtomTool(Tool):
    name = "memory_atom"
    description = (
        "Access the project's L1 atom memory. action='search' recalls similar atoms by "
        "keyword; action='read' fetches one atom by id (for dedup comparison); "
        "action='upsert' stores or updates one atom (content + topic). "
        "Use search before upsert to avoid duplicating an existing atom."
    )
    args_model = MemoryAtomArgs
    retry_safe = False  # upsert mutates state

    MIN_TOPIC_LENGTH = 1
    MAX_TOPIC_LENGTH = 64
    MAX_CONTENT_LENGTH = 4000

    def execute(
        self,
        action: str,
        query: str | None = None,
        topic: str | None = None,
        atom_id: int | None = None,
        content: str | None = None,
        session_id: str | None = None,
        existing_id: int | None = None,
        limit: int | None = None,
    ) -> str | ToolOutput | ToolFailure:
        if action == "search":
            return self._search(query or "", topic, _bounded(limit, 10, 20))
        if action == "read":
            if atom_id is None:
                return tool_failure("missing_atom_id", "validation", "atom_id is required for action='read'")
            return self._read(atom_id)
        if action == "upsert":
            return self._upsert(content, topic, session_id, existing_id)
        return tool_failure("invalid_action", "validation", "action must be 'search', 'read', or 'upsert'")

    def _search(self, query: str, topic: str | None, limit: int) -> str | ToolOutput | ToolFailure:
        try:
            clauses: list[str] = []
            params: list[object] = []
            if query.strip():
                clauses.append("l1_atom_fts MATCH ?")
                params.append(self._fts_query(query))
            if topic:
                clauses.append("a.topic = ?")
                params.append(topic)
            sql = "SELECT a.id, a.content, a.topic, a.created_at, bm25(l1_atom_fts) AS score "
            sql += "FROM l1_atom_fts JOIN l1_atom a ON a.id = l1_atom_fts.rowid "
            if clauses:
                sql += "WHERE " + " AND ".join(clauses) + " "
            sql += "ORDER BY score, a.id LIMIT ?"
            params.append(limit)
            with get_connection() as conn:
                rows = conn.execute(sql, params).fetchall()
        except Exception as exc:  # noqa: BLE001 - surface DB/query errors to the agent
            return tool_failure("atom_search_failed", "database", f"atom search failed: {exc}")

        if not rows:
            return f"No matching atoms for {query!r}."

        lines = [f"Atom search returned {len(rows)} result(s):"]
        for row in rows:
            lines.append(f"[id={row['id']}, topic={row['topic']}, score={row['score']:.3f}] {row['content']}")
        return ToolOutput(content="\n".join(lines), preview=f"Atom search: {len(rows)} result(s)")

    def _read(self, atom_id: int) -> str | ToolOutput | ToolFailure:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT id, session_id, content, topic, created_at, updated_at FROM l1_atom WHERE id = ?",
                (atom_id,),
            ).fetchone()
        if not row:
            return tool_failure("atom_not_found", "resource", f"atom {atom_id} was not found")
        return ToolOutput(
            content=f"[id={row['id']}, topic={row['topic']}, created={row['created_at']}]\n{row['content']}",
            preview=f"Read atom {atom_id}",
        )

    def _upsert(
        self,
        content: str | None,
        topic: str | None,
        session_id: str | None,
        existing_id: int | None,
    ) -> str | ToolOutput | ToolFailure:
        if not content or not content.strip():
            return tool_failure("missing_content", "validation", "content is required for action='upsert'")
        if not topic:
            return tool_failure("missing_topic", "validation", "topic is required for action='upsert'")
        topic = topic.strip()
        content = content.strip()
        if not topic or len(topic) > self.MAX_TOPIC_LENGTH:
            return tool_failure("invalid_topic", "validation", "topic must be 1..%d chars" % self.MAX_TOPIC_LENGTH)
        if len(content) > self.MAX_CONTENT_LENGTH:
            content = content[: self.MAX_CONTENT_LENGTH]

        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with get_connection() as conn:
            if existing_id is not None:
                conn.execute(
                    "UPDATE l1_atom SET content = ?, topic = ?, session_id = ?, updated_at = ? WHERE id = ?",
                    (content, topic, session_id, now, existing_id),
                )
                if conn.total_changes == 0:
                    return tool_failure("atom_not_found", "resource", f"atom {existing_id} was not found")
                return ToolOutput(content=f"Updated atom {existing_id}: [{topic}] {content}", preview=f"Updated atom {existing_id}")
            cur = conn.execute(
                "INSERT INTO l1_atom (session_id, content, topic, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, content, topic, now, now),
            )
        return ToolOutput(content=f"Stored atom {cur.lastrowid}: [{topic}] {content}", preview=f"Stored atom {cur.lastrowid}")

    @staticmethod
    def _fts_query(query: str) -> str:
        quoted = '"' + query.replace('"', '""') + '"'
        return quoted


def _bounded(value: int | None, default: int, maximum: int) -> int:
    return max(1, min(value if value is not None else default, maximum))