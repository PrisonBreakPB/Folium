from types import SimpleNamespace

from folium import database
from folium.database import get_connection
from folium.session import save_session
from folium.tools.session_history import SessionHistoryTool


def _use_temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "folium.db")


def _tool_for_session(session_id: str) -> SessionHistoryTool:
    tool = SessionHistoryTool()
    tool._parent_agent = SimpleNamespace(session_id=session_id)
    return tool


def test_session_history_searches_only_current_session(tmp_path, monkeypatch):
    _use_temp_db(tmp_path, monkeypatch)
    raw_output = "PID controller full output: Kp=1.2, Ki=0.4, Kd=0.05"
    save_session(
        [
            {"role": "user", "content": "Tune a PID controller."},
            {
                "role": "tool",
                "tool_call_id": "call_pid",
                "name": "bash",
                "content": "[Content compacted to save context]",
                "_raw_content": raw_output,
            },
            {
                "role": "tool",
                "tool_call_id": "call_history",
                "name": "session_history",
                "content": "PID controller full output should not be indexed again",
            },
        ],
        "model-a",
        "session_a",
    )
    save_session(
        [{"role": "user", "content": "PID content from another session"}],
        "model-a",
        "session_b",
    )

    result = _tool_for_session("session_a").execute(action="search", query="PID")

    assert "message_id=" in result.content
    assert raw_output in result.content
    assert "another session" not in result.content
    assert "should not be indexed again" not in result.content


def test_session_history_reads_full_raw_tool_output_and_enforces_session(tmp_path, monkeypatch):
    _use_temp_db(tmp_path, monkeypatch)
    raw_output = "full tool output " * 1000
    save_session(
        [{
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "bash",
            "content": "[Content compacted to save context]",
            "_raw_content": raw_output,
        }],
        "model-a",
        "session_a",
    )
    save_session(
        [{"role": "user", "content": "private other session"}],
        "model-a",
        "session_b",
    )
    with get_connection() as conn:
        own_id = conn.execute(
            "SELECT id FROM messages WHERE session_id = ?",
            ("session_a",),
        ).fetchone()["id"]
        other_id = conn.execute(
            "SELECT id FROM messages WHERE session_id = ?",
            ("session_b",),
        ).fetchone()["id"]

    tool = _tool_for_session("session_a")
    first_chunk = tool.execute(action="read", message_id=own_id, max_chars=200)
    other_session = tool.execute(action="read", message_id=other_id)

    assert raw_output[:200] in first_chunk.content
    assert "more characters; call read with offset=200" in first_chunk.content
    assert "not found in the current session" in other_session


def test_session_history_requires_an_active_session():
    result = SessionHistoryTool().execute(action="search")

    assert "no active session" in result
