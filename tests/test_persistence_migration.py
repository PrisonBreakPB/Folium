import json
import sqlite3

from folium import database
from folium.observability.summary import list_traces
from folium.persistence_migration import migrate_legacy_storage
from folium.session import load_session


def test_migrates_legacy_sessions_prompts_and_traces(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "data" / "folium.db")

    conversations = tmp_path / "conversations"
    traces = conversations / "traces"
    traces.mkdir(parents=True)
    (conversations / "legacy_session.json").write_text(
        json.dumps(
            {
                "id": "legacy_session",
                "model": "legacy-model",
                "messages": [{"role": "user", "content": "compressed context"}],
                "transcript": [{"role": "user", "content": "original question"}],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (traces / "trace_legacy.jsonl").write_text(
        "\n".join(
            json.dumps(event)
            for event in [
                {
                    "event": "span_start",
                    "trace_id": "trace_legacy",
                    "session_id": "legacy_session",
                    "name": "user_task",
                    "type": "agent",
                    "timestamp": "2026-01-01T00:00:00.000",
                    "metadata": {"model": "legacy-model"},
                },
                {
                    "event": "span_end",
                    "trace_id": "trace_legacy",
                    "name": "user_task",
                    "type": "agent",
                    "status": "ok",
                    "duration_ms": 12,
                    "timestamp": "2026-01-01T00:00:00.012",
                    "metadata": {},
                },
            ]
        ),
        encoding="utf-8",
    )

    prompt_db = tmp_path / "data" / "session_prompts.db"
    prompt_db.parent.mkdir()
    with sqlite3.connect(prompt_db) as conn:
        conn.execute(
            "CREATE TABLE session_prompts (session_id TEXT PRIMARY KEY, system_prompt TEXT NOT NULL)"
        )
        conn.execute(
            "INSERT INTO session_prompts VALUES (?, ?)",
            ("legacy_session", "legacy system prompt"),
        )

    assert migrate_legacy_storage(conversations) == {"sessions": 1, "traces": 1}
    assert load_session("legacy_session") == (
        [{"role": "user", "content": "compressed context"}],
        "legacy-model",
        [{"role": "user", "content": "original question"}],
        "legacy system prompt",
    )
    assert list_traces(database.DB_PATH)[0]["trace_id"] == "trace_legacy"
    assert migrate_legacy_storage(conversations) == {"sessions": 0, "traces": 0}
