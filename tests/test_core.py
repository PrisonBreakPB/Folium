"""Tests for core modules: config, context, session, imports."""

import os
import pathlib
from unittest import mock

from folium import Agent, LLM, Config, ALL_TOOLS, __version__
from folium.config import DEFAULT_MAX_CONTEXT_TOKENS
from folium.context import ContextManager, DEFAULT_RESERVED_OUTPUT_TOKENS, SUMMARY_PREFIX, estimate_tokens
from folium.session import save_session, load_session, list_sessions
from folium.tools import get_tool


def test_version():
    assert __version__ == "0.3.0"


def test_public_api_exports():
    """Users should be able to import key classes from the top-level package."""
    assert Agent is not None
    assert LLM is not None
    assert Config is not None
    assert len(ALL_TOOLS) == 7


def test_config_from_env():
    os.environ["FOLIUM_MODEL"] = "test-model"
    c = Config.from_env()
    assert c.model == "test-model"
    del os.environ["FOLIUM_MODEL"]


def test_config_defaults():
    # temporarily clear relevant env vars
    saved = {}
    for k in ["FOLIUM_MODEL", "FOLIUM_MAX_TOKENS"]:
        if k in os.environ:
            saved[k] = os.environ.pop(k)

    c = Config.from_env()
    assert c.model == "gpt-4o"
    assert c.max_tokens == 4096
    assert c.max_context_tokens == DEFAULT_MAX_CONTEXT_TOKENS
    assert c.temperature == 0.0
    assert c.token_estimator == "deepseek"

    os.environ.update(saved)


# --- Context ---

def test_estimate_tokens():
    msgs = [{"role": "user", "content": "hello world"}]
    t = estimate_tokens(msgs)
    assert t > 0
    assert t < 100


def test_context_snip():
    ctx = ContextManager(max_tokens=3000)
    msgs = [
        {"role": "tool", "tool_call_id": "t1", "content": "x\n" * 1000},
    ]
    before = estimate_tokens(msgs)
    ctx._snip_tool_outputs(msgs)
    after = estimate_tokens(msgs)
    assert after < before


def test_context_reserves_output_tokens():
    ctx = ContextManager(max_tokens=100_000)

    assert ctx.reserved_output_tokens == DEFAULT_RESERVED_OUTPUT_TOKENS
    assert ctx.input_budget_tokens == 80_000
    assert ctx._snip_at == 48_000
    assert ctx._summarize_at == 56_000
    assert ctx._collapse_at == 72_000


def test_context_compress():
    ctx = ContextManager(max_tokens=2000)
    msgs = []
    for i in range(20):
        msgs.append({"role": "user", "content": f"msg {i} " + "a" * 200})
        msgs.append({"role": "tool", "tool_call_id": f"t{i}", "content": "b" * 2000})
    before = estimate_tokens(msgs)
    ctx.maybe_compress(msgs, None)
    after = estimate_tokens(msgs)
    assert after < before
    assert len(msgs) < 40  # should be compressed


class FakeLLM:
    def __init__(self):
        self.calls = []

    def chat(self, messages, tools=None, on_token=None):
        self.calls.append(messages)
        content = "summary"
        if "Existing summary:" in messages[-1]["content"]:
            content = "updated summary"
        return type("Resp", (), {"content": content})()


def test_context_compress_uses_incremental_summary_without_recent_tail():
    ctx = ContextManager(max_tokens=1000)
    llm = FakeLLM()
    msgs = [
        {"role": "user", "content": "first request"},
        {"role": "assistant", "content": "assistant detail"},
        {"role": "tool", "tool_call_id": "t1", "content": "tool output"},
        {"role": "user", "content": "latest request"},
    ]

    assert ctx._summarize_old(msgs, llm)

    assert [m["role"] for m in msgs] == ["user", "user", "user"]
    assert msgs[0]["content"] == "first request"
    assert msgs[1]["content"] == "latest request"
    assert msgs[0]["_protected"] is True
    assert msgs[2]["content"].startswith(f"{SUMMARY_PREFIX}\nsummary")
    assert not any(m.get("role") in {"assistant", "tool"} for m in msgs)


def test_context_compress_updates_existing_summary_with_delta_only():
    ctx = ContextManager(max_tokens=1000)
    llm = FakeLLM()
    msgs = [
        {"role": "user", "content": "old protected", "_protected": True},
        {"role": "user", "content": f"{SUMMARY_PREFIX}\nold summary"},
        {"role": "assistant", "content": "new assistant fact"},
        {"role": "user", "content": "new user request"},
    ]

    assert ctx._summarize_old(msgs, llm)

    update_prompt = llm.calls[-1][-1]["content"]
    assert "Existing summary:\nold summary" in update_prompt
    assert "new assistant fact" in update_prompt
    assert "new user request" in update_prompt
    assert "old protected" not in update_prompt
    assert msgs[-1]["content"].startswith(f"{SUMMARY_PREFIX}\nupdated summary")


def test_context_protected_users_use_token_budget_and_keep_latest_oversized():
    ctx = ContextManager(max_tokens=1000, protected_user_tokens=5)
    msgs = [
        {"role": "user", "content": "older"},
        {"role": "assistant", "content": "assistant"},
        {"role": "user", "content": "latest very long request"},
    ]

    with mock.patch("folium.context.estimate_text_tokens", side_effect=lambda text: 10 if "latest" in text else 1):
        protected = ctx._collect_protected_user_messages(msgs)

    assert len(protected) == 1
    assert protected[0]["content"] == "latest very long request"
    assert protected[0]["_protected"] is True


# --- Session ---

def test_session_save_load():
    msgs = [{"role": "user", "content": "test message"}]
    sid = save_session(msgs, "test-model", "pytest_test_session")
    loaded = load_session("pytest_test_session")
    assert loaded is not None
    assert loaded[0] == msgs
    assert loaded[1] == "test-model"
    # cleanup
    pathlib.Path.home().joinpath(".folium/sessions/pytest_test_session.json").unlink()


def test_session_name_is_sanitized():
    msgs = [{"role": "user", "content": "test message"}]
    sid = save_session(msgs, "test-model", "../Research Notes!")

    assert sid == "Research-Notes"
    path = pathlib.Path.home().joinpath(".folium/sessions/Research-Notes.json")
    assert path.exists()
    assert load_session("../Research Notes!") is not None
    path.unlink()


def test_session_not_found():
    assert load_session("nonexistent_session_id") is None


def test_list_sessions():
    sessions = list_sessions()
    assert isinstance(sessions, list)


# --- Cost estimation ---

def test_cost_estimation_known_model():
    from folium.llm import LLM
    llm = LLM.__new__(LLM)
    llm.model = "gpt-5.4"
    llm.total_prompt_tokens = 1_000_000
    llm.total_completion_tokens = 500_000
    cost = llm.estimated_cost
    assert cost is not None
    assert cost == 2.5 + 7.5  # $2.5/M in + $15/M out * 0.5M

def test_cost_estimation_unknown_model():
    from folium.llm import LLM
    llm = LLM.__new__(LLM)
    llm.model = "some-custom-model"
    llm.total_prompt_tokens = 1000
    llm.total_completion_tokens = 500
    assert llm.estimated_cost is None


# --- Changed files tracking ---

def test_edit_tracks_changed_files(tmp_path):
    from folium.tools.edit import _changed_files
    _changed_files.clear()
    edit = get_tool("edit_file")
    path = tmp_path / "sample.py"
    path.write_text("aaa\nbbb\n")
    edit.execute(file_path=str(path), old_string="aaa", new_string="zzz")
    assert any(str(path) in p for p in _changed_files)
    _changed_files.clear()


def test_write_tracks_changed_files(tmp_path):
    from folium.tools.edit import _changed_files
    _changed_files.clear()
    write = get_tool("write_file")
    path = tmp_path / "tracked.txt"
    write.execute(file_path=str(path), content="tracked\n")
    assert any("tracked" not in p and path.name in p for p in _changed_files) or len(_changed_files) > 0
    _changed_files.clear()
