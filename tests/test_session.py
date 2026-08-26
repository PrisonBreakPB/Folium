from folium import database
from folium.database import get_connection
from folium.session import (
    calculate_session_stats,
    delete_session,
    ensure_session,
    get_session_workspace,
    load_session,
    list_sessions,
    save_session,
)


def _use_temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "folium.db")


def test_default_session_ids_do_not_collide(tmp_path, monkeypatch):
    _use_temp_db(tmp_path, monkeypatch)

    first_id = save_session([{"role": "user", "content": "first"}], "model-a")
    second_id = save_session([{"role": "user", "content": "second"}], "model-b")

    assert first_id != second_id
    assert load_session(first_id) == (
        [{"role": "user", "content": "first"}],
        "model-a",
        [{"role": "user", "content": "first"}],
        None,
    )
    assert load_session(second_id) == (
        [{"role": "user", "content": "second"}],
        "model-b",
        [{"role": "user", "content": "second"}],
        None,
    )


def test_session_keeps_transcript_and_model_context_separately(tmp_path, monkeypatch):
    _use_temp_db(tmp_path, monkeypatch)

    sid = save_session(
        [{"role": "user", "content": "injected skill instructions"}],
        "model-a",
        "session_a",
        transcript=[{"role": "user", "content": "original user message"}],
        system_prompt="test system prompt",
    )

    assert load_session(sid) == (
        [{"role": "user", "content": "injected skill instructions"}],
        "model-a",
        [{"role": "user", "content": "original user message"}],
        "test system prompt",
    )


def test_tool_raw_output_survives_context_trimming(tmp_path, monkeypatch):
    _use_temp_db(tmp_path, monkeypatch)
    raw_output = "full output " * 100
    visible_output = "[Content compacted to save context]"

    save_session(
        [{
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "bash",
            "content": visible_output,
            "_raw_content": raw_output,
        }],
        "model-a",
        "session_a",
        transcript=[{
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "bash",
            "content": visible_output,
            "_raw_content": raw_output,
        }],
    )

    messages, _, transcript, _ = load_session("session_a")
    assert messages[0]["content"] == visible_output
    assert transcript[0]["content"] == raw_output


def test_delete_session_cascades_to_traces(tmp_path, monkeypatch):
    _use_temp_db(tmp_path, monkeypatch)
    ensure_session("session_a", "model-a")
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO traces (trace_id, session_id, status) VALUES (?, ?, ?)",
            ("trace_a", "session_a", "ok"),
        )
        conn.execute(
            "INSERT INTO trace_events (trace_id, event_index, event_type) VALUES (?, ?, ?)",
            ("trace_a", 0, "span_start"),
        )

    assert delete_session("session_a") is True
    with get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM trace_events").fetchone()[0] == 0


def test_session_persists_workspace_path(tmp_path, monkeypatch):
    _use_temp_db(tmp_path, monkeypatch)
    workspace = tmp_path / "project"
    workspace.mkdir()

    sid = save_session(
        [{"role": "user", "content": "hello"}],
        "model-a",
        workspace_path=str(workspace),
    )

    assert get_session_workspace(sid) == str(workspace)
    assert list_sessions()[0]["workspace_path"] == str(workspace)


def test_session_stats_keeps_separate_llm_spans_with_identical_usage(tmp_path, monkeypatch):
    _use_temp_db(tmp_path, monkeypatch)
    ensure_session("session_a", "model-a")
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO traces (trace_id, session_id, status) VALUES (?, ?, ?)",
            ("trace_a", "session_a", "ok"),
        )
        conn.executemany(
            """
            INSERT INTO trace_events (
                trace_id, event_index, event_type, span_id, payload_json
            ) VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    "trace_a",
                    0,
                    "llm_result",
                    "span_first",
                    '{"prompt_tokens": 10, "completion_tokens": 2, "cached_tokens": 4}',
                ),
                (
                    "trace_a",
                    1,
                    "llm_result",
                    "span_second",
                    '{"prompt_tokens": 10, "completion_tokens": 2, "cached_tokens": 4}',
                ),
            ],
        )

    assert calculate_session_stats("session_a") == {
        "prompt_tokens": 20,
        "cache_miss_tokens": 12,
        "cached_tokens": 8,
        "completion_tokens": 4,
        "total_tokens": 24,
        "cost": 0.0,
        "cache_hit_rate": 0.4,
    }
