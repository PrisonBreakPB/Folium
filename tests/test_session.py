from folium import session as session_module
from folium import session_prompts
from folium.session import delete_session, load_session, save_session
from folium.session_prompts import load_prompt


def test_default_session_ids_do_not_collide(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)

    first_id = save_session([{"role": "user", "content": "first"}], "model-a")
    second_id = save_session([{"role": "user", "content": "second"}], "model-b")

    assert first_id != second_id
    assert load_session(first_id) == (
        [{"role": "user", "content": "first"}],
        "model-a",
        [{"role": "user", "content": "first"}],
    )
    assert load_session(second_id) == (
        [{"role": "user", "content": "second"}],
        "model-b",
        [{"role": "user", "content": "second"}],
    )


def test_session_saves_transcript_separately(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)

    sid = save_session(
        [{"role": "user", "content": "summarized context"}],
        "model-a",
        "session_a",
        transcript=[{"role": "user", "content": "original user message"}],
    )

    assert load_session(sid) == (
        [{"role": "user", "content": "summarized context"}],
        "model-a",
        [{"role": "user", "content": "original user message"}],
    )


def test_legacy_session_transcript_fallback_is_independent(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path)
    path = tmp_path / "legacy.json"
    path.write_text(
        '{"id":"legacy","model":"model-a","messages":[{"role":"user","content":"old"}]}',
        encoding="utf-8",
    )

    loaded = load_session("legacy")
    assert loaded is not None
    messages, model, transcript = loaded

    assert model == "model-a"
    assert transcript == messages
    assert transcript is not messages


def test_delete_session_deletes_system_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(session_module, "SESSIONS_DIR", tmp_path / "conversations")
    monkeypatch.setattr(session_prompts, "DB_DIR", tmp_path / "data")
    monkeypatch.setattr(session_prompts, "DB_PATH", tmp_path / "data" / "session_prompts.db")

    session_id = save_session(
        [{"role": "user", "content": "test"}],
        "model-a",
        "session_a",
        system_prompt="test system prompt",
    )

    assert load_prompt(session_id) == "test system prompt"
    assert delete_session(session_id) is True
    assert load_prompt(session_id) is None
