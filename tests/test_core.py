"""Tests for core modules: config, context, session, imports."""

import os
import pathlib
from unittest import mock

from folium import Agent, LLM, Config, ALL_TOOLS, __version__
from folium.config import DEFAULT_MAX_CONTEXT_TOKENS
from folium.context import (
    ContextManager,
    DEFAULT_RESERVED_OUTPUT_TOKENS,
    SUMMARY_PREFIX,
    TOOL_COMPRESS_EXEMPT,
    TOOL_COMPRESSION_TIER,
    TOOL_OUTPUT_DEDUPE_PLACEHOLDER,
    TOOL_OUTPUT_DEDUPE_THRESHOLD_CHARS,
    TOOL_OUTPUT_TRIM_KEEP_CHARS,
    TOOL_OUTPUT_TRIM_MARKER,
    TOOL_OUTPUT_TRIM_THRESHOLD_CHARS,
    estimate_tokens,
)
from folium.session import save_session, load_session, list_sessions
from folium.tools import get_tool
from folium.tools.edit import EditFileTool
from folium.tools.write import WriteFileTool


def test_version():
    assert __version__ == "0.3.0"


def test_public_api_exports():
    """Users should be able to import key classes from the top-level package."""
    assert Agent is not None
    assert LLM is not None
    assert Config is not None
    assert len(ALL_TOOLS) == 16


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("FOLIUM_MODEL", "test-model")
    c = Config.from_env()
    assert c.model == "test-model"


def test_config_defaults(monkeypatch):
    # temporarily clear relevant env vars
    for k in ["FOLIUM_MODEL", "FOLIUM_MAX_TOKENS"]:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr("folium.config._load_dotenv", lambda: None)

    c = Config.from_env()
    assert c.model == "gpt-4o"
    assert c.max_tokens == 4096
    assert c.max_context_tokens == DEFAULT_MAX_CONTEXT_TOKENS
    assert c.temperature == 0.0
    assert c.token_estimator == "deepseek"


# --- Context ---

def test_estimate_tokens():
    msgs = [{"role": "user", "content": "hello world"}]
    t = estimate_tokens(msgs)
    assert t > 0
    assert t < 100


def test_context_snip():
    ctx = ContextManager(max_tokens=3000)
    content = "a" * TOOL_OUTPUT_TRIM_THRESHOLD_CHARS + "b"
    msgs = [
        {"role": "tool", "tool_call_id": "t1", "content": content},
    ]

    report = ctx._snip_tool_outputs(msgs, recent_tool_rounds_to_keep=0)

    trimmed = msgs[0]["content"]
    assert report["changed"] is True
    assert report["tools"] == [{"tool_call_id": "t1", "name": None}]
    assert TOOL_OUTPUT_TRIM_MARKER in trimmed
    assert trimmed.startswith("a" * TOOL_OUTPUT_TRIM_KEEP_CHARS)
    assert trimmed.endswith("a" * (TOOL_OUTPUT_TRIM_KEEP_CHARS - 1) + "b")
    assert len(trimmed) < len(content)


def test_context_snip_keeps_tool_outputs_at_threshold():
    content = "x" * TOOL_OUTPUT_TRIM_THRESHOLD_CHARS
    msgs = [{"role": "tool", "tool_call_id": "t1", "content": content}]

    report = ContextManager._snip_tool_outputs(msgs, recent_tool_rounds_to_keep=0)

    assert report["changed"] is False
    assert msgs[0]["content"] == content


def test_context_prune_only_handles_trimmed_outputs():
    old_snipped = "head\n... (10 lines, snipped to save context) ...\ntail"
    msgs = [{"role": "tool", "tool_call_id": "t1", "content": old_snipped}]

    report = ContextManager._prune_tool_outputs(msgs)

    assert report["changed"] is False
    assert msgs[0]["content"] == old_snipped


def test_context_prune_reports_tool_identity():
    msgs = [{
        "role": "tool",
        "tool_call_id": "t1",
        "name": "bash",
        "content": f"head\n... (5000 chars total, {TOOL_OUTPUT_TRIM_MARKER}) ...\ntail",
    }]

    report = ContextManager._prune_tool_outputs(msgs)

    assert report["changed"] is True
    assert report["tools"] == [{"tool_call_id": "t1", "name": "bash"}]


def test_context_prune_skips_secondary_tools():
    msgs = [{
        "role": "tool",
        "tool_call_id": "t1",
        "name": "read_file",
        "content": f"head\n... (5000 chars total, {TOOL_OUTPUT_TRIM_MARKER}) ...\ntail",
    }]

    report = ContextManager._prune_tool_outputs(msgs)

    assert report["changed"] is False
    assert TOOL_OUTPUT_TRIM_MARKER in msgs[0]["content"]


def test_context_dedupe_keeps_latest_full_tool_output():
    content = "duplicate output\n" + ("x" * TOOL_OUTPUT_DEDUPE_THRESHOLD_CHARS)
    msgs = [
        {"role": "tool", "name": "read_file", "tool_call_id": "oldest", "content": content},
        {"role": "tool", "name": "read_file", "tool_call_id": "middle", "content": content},
        {"role": "tool", "name": "read_file", "tool_call_id": "latest", "content": content},
    ]

    report = ContextManager._dedupe_tool_outputs(msgs)

    assert report["changed"] is True
    assert report["tools"] == [
        {"tool_call_id": "oldest", "name": "read_file"},
        {"tool_call_id": "middle", "name": "read_file"},
    ]
    assert msgs[0]["content"] == TOOL_OUTPUT_DEDUPE_PLACEHOLDER
    assert msgs[1]["content"] == TOOL_OUTPUT_DEDUPE_PLACEHOLDER
    assert msgs[2]["content"] == content


def test_context_dedupe_skips_short_non_string_and_exempt_outputs():
    TOOL_COMPRESS_EXEMPT.add("test_exempt")
    try:
        short = "s" * (TOOL_OUTPUT_DEDUPE_THRESHOLD_CHARS - 1)
        exempt = "exempt output\n" + ("x" * TOOL_OUTPUT_DEDUPE_THRESHOLD_CHARS)
        msgs = [
            {"role": "tool", "tool_call_id": "short_1", "content": short},
            {"role": "tool", "tool_call_id": "short_2", "content": short},
            {"role": "tool", "tool_call_id": "blocks_1", "content": [{"type": "text", "text": "same"}]},
            {"role": "tool", "tool_call_id": "blocks_2", "content": [{"type": "text", "text": "same"}]},
            {"role": "tool", "name": "test_exempt", "tool_call_id": "exempt_1", "content": exempt},
            {"role": "tool", "name": "test_exempt", "tool_call_id": "exempt_2", "content": exempt},
        ]

        report = ContextManager._dedupe_tool_outputs(msgs)

        assert report["changed"] is False
        assert msgs[0]["content"] == short
        assert msgs[1]["content"] == short
        assert msgs[2]["content"] == [{"type": "text", "text": "same"}]
        assert msgs[3]["content"] == [{"type": "text", "text": "same"}]
        assert msgs[4]["content"] == exempt
        assert msgs[5]["content"] == exempt
    finally:
        TOOL_COMPRESS_EXEMPT.discard("test_exempt")


def test_context_dedupe_runs_at_half_input_budget_before_snip():
    ctx = ContextManager(max_tokens=100_000)
    content = "duplicate output\n" + ("x" * 500)
    msgs = [
        {"role": "tool", "name": "read_file", "tool_call_id": "old", "content": content},
        {"role": "tool", "name": "read_file", "tool_call_id": "new", "content": content},
    ]

    report = ctx.maybe_compress(msgs, None, real_tokens=ctx._dedupe_at + 1)

    assert report["compressed"] is True
    assert [layer["name"] for layer in report["layers"]] == ["dedupe"]
    assert msgs[0]["content"] == TOOL_OUTPUT_DEDUPE_PLACEHOLDER
    assert msgs[1]["content"] == content


def test_context_snip_exempt_tool():
    """Exempt tools are never compressed."""
    TOOL_COMPRESS_EXEMPT.add("test_exempt")
    try:
        msgs = [{"role": "tool", "name": "test_exempt", "tool_call_id": "t1", "content": "x" * 5000}]
        report = ContextManager._snip_tool_outputs(msgs, recent_tool_rounds_to_keep=0)
        assert report["changed"] is False
        assert msgs[0]["content"] == "x" * 5000
    finally:
        TOOL_COMPRESS_EXEMPT.discard("test_exempt")


def test_context_snip_secondary_tool_skipped_at_low_usage():
    """Secondary tools are not trimmed when context usage is low."""
    msgs = [{"role": "tool", "name": "read_file", "tool_call_id": "t1", "content": "x" * 5000}]
    report = ContextManager._snip_tool_outputs(msgs, context_usage_ratio=0.5, recent_tool_rounds_to_keep=0)
    assert report["changed"] is False
    assert msgs[0]["content"] == "x" * 5000


def test_context_snip_secondary_tool_skipped_below_secondary_threshold():
    """Secondary tools are not trimmed before the 70% threshold."""
    msgs = [{"role": "tool", "name": "read_file", "tool_call_id": "t1", "content": "x" * 5000}]
    report = ContextManager._snip_tool_outputs(msgs, context_usage_ratio=0.65, recent_tool_rounds_to_keep=0)
    assert report["changed"] is False
    assert msgs[0]["content"] == "x" * 5000


def test_context_snip_secondary_tool_trimmed_at_secondary_threshold():
    """Secondary tools are trimmed when context usage reaches 70%."""
    msgs = [{"role": "tool", "name": "read_file", "tool_call_id": "t1", "content": "x" * 5000}]
    report = ContextManager._snip_tool_outputs(msgs, context_usage_ratio=0.7, recent_tool_rounds_to_keep=0)
    assert report["changed"] is True
    assert TOOL_OUTPUT_TRIM_MARKER in msgs[0]["content"]


def test_context_snip_primary_tool_always_trimmed():
    """Primary tools are trimmed regardless of context usage."""
    msgs = [{"role": "tool", "name": "bash", "tool_call_id": "t1", "content": "x" * 5000}]
    report = ContextManager._snip_tool_outputs(msgs, context_usage_ratio=0.3, recent_tool_rounds_to_keep=0)
    assert report["changed"] is True
    assert TOOL_OUTPUT_TRIM_MARKER in msgs[0]["content"]


def test_context_snip_protects_recent_tool_rounds():
    content = "x" * 5000
    msgs = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "old", "function": {"name": "bash"}}]},
        {"role": "tool", "name": "bash", "tool_call_id": "old", "content": content},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "recent1", "function": {"name": "bash"}}]},
        {"role": "tool", "name": "bash", "tool_call_id": "recent1", "content": content},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "recent2", "function": {"name": "bash"}}]},
        {"role": "tool", "name": "bash", "tool_call_id": "recent2", "content": content},
    ]

    report = ContextManager._snip_tool_outputs(
        msgs,
        context_usage_ratio=0.9,
        recent_tool_rounds_to_keep=2,
    )

    assert report["changed"] is True
    assert report["tools"] == [{"tool_call_id": "old", "name": "bash"}]
    assert TOOL_OUTPUT_TRIM_MARKER in msgs[1]["content"]
    assert msgs[3]["content"] == content
    assert msgs[5]["content"] == content


def test_context_reserves_output_tokens():
    ctx = ContextManager(max_tokens=100_000)

    assert ctx.reserved_output_tokens == DEFAULT_RESERVED_OUTPUT_TOKENS
    assert ctx.input_budget_tokens == 80_000
    assert ctx._dedupe_at == 40_000
    assert ctx._snip_at == 48_000
    assert ctx._secondary_snip_at == 56_000
    assert ctx._prune_at == 64_000
    assert ctx._summarize_at == 72_000


def test_context_compress():
    ctx = ContextManager(max_tokens=2000)
    msgs = []
    for i in range(20):
        msgs.append({"role": "user", "content": f"msg {i} " + "a" * 200})
        msgs.append({"role": "tool", "tool_call_id": f"t{i}", "content": "b" * 2000})
    before = estimate_tokens(msgs)
    report = ctx.maybe_compress(msgs, None)
    after = estimate_tokens(msgs)
    assert report["compressed"] is True
    assert report["layers"]
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


class FailingLLM:
    def chat(self, messages, tools=None, on_token=None):
        raise RuntimeError("boom")


def test_context_compress_uses_incremental_summary_without_recent_tail():
    ctx = ContextManager(max_tokens=1000)
    llm = FakeLLM()
    msgs = [
        {"role": "user", "content": "first request"},
        {"role": "assistant", "content": "assistant detail"},
        {"role": "tool", "tool_call_id": "t1", "content": "tool output"},
        {"role": "user", "content": "latest request"},
    ]

    report = ctx._summarize_old(msgs, llm)

    assert report["changed"] is True
    assert report["delta_message_count"] == 4
    assert report["protected_user_count"] == 2
    assert report["used_llm"] is True
    assert report["fallback_used"] is False
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

    report = ctx._summarize_old(msgs, llm)

    assert report["changed"] is True
    assert report["delta_message_count"] == 2
    assert report["protected_user_count"] == 2
    assert report["before_message_count"] == 4
    assert report["after_message_count"] == 3
    assert report["used_llm"] is True
    assert report["fallback_used"] is False
    update_prompt = llm.calls[-1][-1]["content"]
    assert "Existing summary:\nold summary" in update_prompt
    assert "new assistant fact" in update_prompt
    assert "new user request" in update_prompt
    assert "old protected" not in update_prompt
    assert msgs[-1]["content"].startswith(f"{SUMMARY_PREFIX}\nupdated summary")


def test_context_summarize_report_marks_fallback():
    ctx = ContextManager(max_tokens=1000)
    msgs = [
        {"role": "user", "content": "please inspect folium/context.py"},
        {"role": "tool", "tool_call_id": "t1", "name": "read_file", "content": "Error: missing file"},
    ]

    report = ctx._summarize_old(msgs, FailingLLM())

    assert report["changed"] is True
    assert report["delta_message_count"] == 2
    assert report["protected_user_count"] == 1
    assert report["used_llm"] is False
    assert report["fallback_used"] is True


def test_context_protected_users_use_token_budget_and_keep_latest_oversized():
    ctx = ContextManager(
        max_tokens=1000,
        protected_user_tokens=5,
        protected_initial_user_messages=0,
    )
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


def test_context_protects_initial_and_recent_user_messages():
    ctx = ContextManager(
        max_tokens=1000,
        protected_user_tokens=1,
        protected_initial_user_messages=2,
    )
    msgs = [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "assistant"},
        {"role": "user", "content": "middle"},
        {"role": "user", "content": "latest"},
    ]

    with mock.patch(
        "folium.context.estimate_text_tokens",
        side_effect=lambda text: 10 if text in {"first", "second"} else 1,
    ):
        protected = ctx._collect_protected_user_messages(msgs)

    assert [message["content"] for message in protected] == ["first", "second", "latest"]
    assert all(message["_protected"] is True for message in protected)


def test_context_summary_keeps_initial_and_recent_user_messages():
    ctx = ContextManager(
        max_tokens=1000,
        protected_user_tokens=1,
        protected_initial_user_messages=2,
    )
    msgs = [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "assistant detail"},
        {"role": "user", "content": "middle"},
        {"role": "user", "content": "latest"},
    ]

    with mock.patch("folium.context.estimate_text_tokens", return_value=1):
        report = ctx._summarize_old(msgs, FakeLLM())

    assert report["protected_user_count"] == 3
    assert [message["content"] for message in msgs[:-1]] == ["first", "second", "latest"]
    assert msgs[-1]["content"].startswith(f"{SUMMARY_PREFIX}\nsummary")


def test_context_initial_user_messages_ignore_token_budget():
    ctx = ContextManager(
        max_tokens=1000,
        protected_user_tokens=1,
        protected_initial_user_messages=2,
    )
    msgs = [
        {"role": "user", "content": "first"},
        {"role": "user", "content": "second"},
        {"role": "user", "content": "latest"},
    ]

    with mock.patch(
        "folium.context.estimate_text_tokens",
        side_effect=lambda text: 10 if text in {"first", "second"} else 1,
    ):
        protected = ctx._collect_protected_user_messages(msgs)

    assert [message["content"] for message in protected] == ["first", "second", "latest"]


# --- Session ---

def test_session_save_load(tmp_path, monkeypatch):
    from folium import database

    monkeypatch.setattr(database, "DB_PATH", tmp_path / "folium.db")
    msgs = [{"role": "user", "content": "test message"}]
    sid = save_session(msgs, "test-model", "pytest_test_session")
    loaded = load_session("pytest_test_session")
    assert loaded is not None
    assert loaded[0] == msgs
    assert loaded[1] == "test-model"


def test_session_name_is_sanitized(tmp_path, monkeypatch):
    from folium import database

    monkeypatch.setattr(database, "DB_PATH", tmp_path / "folium.db")
    msgs = [{"role": "user", "content": "test message"}]
    sid = save_session(msgs, "test-model", "../Research Notes!")

    assert sid == "Research-Notes"
    assert load_session("../Research Notes!") is not None


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
    path = tmp_path / "sample.py"
    path.write_text("aaa\nbbb\n")
    with mock.patch.dict(os.environ, {"FOLIUM_BASH_BACKEND": "local"}, clear=False):
        edit = EditFileTool()
        edit.execute(file_path=str(path), old_string="aaa", new_string="zzz")
    assert any(str(path) in p for p in _changed_files)
    _changed_files.clear()


def test_write_tracks_changed_files(tmp_path):
    from folium.tools.edit import _changed_files
    _changed_files.clear()
    path = tmp_path / "tracked.txt"
    with mock.patch.dict(os.environ, {"FOLIUM_BASH_BACKEND": "local"}, clear=False):
        write = WriteFileTool()
        write.execute(file_path=str(path), content="tracked\n")
    assert any("tracked" not in p and path.name in p for p in _changed_files) or len(_changed_files) > 0
    _changed_files.clear()
