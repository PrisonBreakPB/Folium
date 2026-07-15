import asyncio
import copy
import json
import threading
from unittest import mock

import pytest

from folium.database import get_connection
from folium.llm import LLMResponse, ToolCall
from folium.memory_maintenance import (
    MEMORY_MAINTENANCE_USER_PROMPT,
    MemoryAgent,
    MemoryMaintenanceResult,
    MemoryMaintenanceScheduler,
    TOOL_OUTPUT_TRIM_KEEP_CHARS,
    TOOL_OUTPUT_TRIM_MARKER,
    estimate_memory_maintenance_input_tokens,
)
from folium.observability.config import ObservabilityConfig
from folium.observability.context import Observer, _observer_var
from folium.token_estimator import estimate_text_tokens
from folium.tools.base import ToolValidationError
from folium.tools.memory import MemoryTool


def _version(read_output: str) -> str:
    return read_output.splitlines()[0].removeprefix("Memory version: ")


def _messages(turns: int = 1) -> list[dict]:
    messages = [{"role": "system", "content": "Main agent system prompt."}]
    for index in range(turns):
        messages.extend(
            [
                {"role": "user", "content": f"user {index}"},
                {"role": "assistant", "content": f"assistant {index}"},
            ]
        )
    return messages


def _visible_tools() -> list[dict]:
    return [
        MemoryTool().schema(),
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Run a shell command.",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
    ]


def _run_agent(agent, *, messages=None, visible_tools=None):
    return agent.run(
        session_id="session_test",
        turn_index=10,
        messages=messages or _messages(),
        visible_tools=visible_tools or _visible_tools(),
        input_tokens=100,
    )


def test_memory_tool_read_and_compare_and_swap_append(tmp_path):
    memory_file = tmp_path / "memory.md"
    tool = MemoryTool()
    with mock.patch("folium.tools.memory.MEMORY_FILE", memory_file):
        initial = tool.execute(action="read")
        version = _version(initial)
        saved = tool.execute(
            action="append",
            section="confirmed_decisions",
            content="Use a background memory maintainer.",
            expected_version=version,
        )
        stale = tool.execute(
            action="append",
            section="confirmed_decisions",
            content="This must not be written.",
            expected_version=version,
        )

    assert saved.startswith("Saved long-term memory")
    assert stale.startswith("Conflict:")
    contents = memory_file.read_text(encoding="utf-8")
    assert "Use a background memory maintainer." in contents
    assert "This must not be written." not in contents


def test_memory_tool_action_enum_is_validated():
    with pytest.raises(ToolValidationError, match="must be one of: read, append"):
        MemoryTool().validate_arguments({"action": "delete"})


class NoChangeLLM:
    model = "fake-memory-model"

    def __init__(self):
        self.requests = []
        self.trace_inputs = []

    def chat(self, messages, tools=None, on_token=None, trace_input=True):
        self.requests.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        self.trace_inputs.append(trace_input)
        return LLMResponse(content="NO_CHANGE", cached_tokens=17)


class ReadThenAppendLLM:
    model = "fake-memory-model"

    def __init__(self):
        self.calls = 0
        self.requests = []

    def chat(self, messages, tools=None, on_token=None, trace_input=True):
        self.calls += 1
        self.requests.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        if self.calls == 1:
            return LLMResponse(
                tool_calls=[ToolCall(id="read", name="memory", arguments={"action": "read"})],
                cached_tokens=3,
            )
        if self.calls == 2:
            return LLMResponse(
                tool_calls=[
                    ToolCall(
                        id="append",
                        name="memory",
                        arguments={
                            "action": "append",
                            "section": "user_preferences",
                            "content": "Prefer concise Chinese responses.",
                            "expected_version": _version(messages[-1]["content"]),
                        },
                    )
                ],
                cached_tokens=5,
            )
        return LLMResponse(content="UPDATED", cached_tokens=7)


class NonMemoryToolLLM:
    model = "fake-memory-model"

    def __init__(self):
        self.calls = 0
        self.requests = []

    def chat(self, messages, tools=None, on_token=None, trace_input=True):
        self.calls += 1
        self.requests.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        if self.calls == 1:
            return LLMResponse(
                tool_calls=[ToolCall(id="bash", name="bash", arguments={"command": "touch no"})]
            )
        return LLMResponse(content="NO_CHANGE")


class ReadForeverLLM:
    model = "fake-memory-model"

    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools=None, on_token=None, trace_input=True):
        self.calls += 1
        return LLMResponse(
            tool_calls=[ToolCall(id=str(self.calls), name="memory", arguments={"action": "read"})]
        )


def test_memory_agent_appends_final_prompt_and_preserves_full_visible_tools(tmp_path):
    llm = NoChangeLLM()
    messages = _messages()
    original_messages = copy.deepcopy(messages)
    visible_tools = _visible_tools()
    original_tools = copy.deepcopy(visible_tools)

    with mock.patch("folium.tools.memory.MEMORY_FILE", tmp_path / "memory.md"):
        result = _run_agent(MemoryAgent(llm), messages=messages, visible_tools=visible_tools)

    request_messages, request_tools = llm.requests[0]
    assert result.status == "NO_CHANGE"
    assert messages == original_messages
    assert visible_tools == original_tools
    assert request_messages[:-1] == original_messages
    assert request_messages[-1] == {
        "role": "user",
        "content": MEMORY_MAINTENANCE_USER_PROMPT,
    }
    assert request_tools == original_tools
    assert [item["function"]["name"] for item in request_tools] == ["memory", "bash"]
    assert result.cached_tokens == 17
    assert llm.trace_inputs == [False]


def test_memory_agent_rejects_non_memory_tool_calls_without_side_effects(tmp_path):
    llm = NonMemoryToolLLM()
    memory_file = tmp_path / "memory.md"

    with mock.patch("folium.tools.memory.MEMORY_FILE", memory_file):
        result = _run_agent(MemoryAgent(llm))

    assert result.status == "NO_CHANGE"
    assert result.rejected_tool_calls == 1
    assert not memory_file.exists()
    tool_result = llm.requests[1][0][-1]
    assert tool_result["name"] == "bash"
    assert "only 'memory' is permitted" in tool_result["content"]


def test_memory_agent_reads_then_writes_and_records_trace(tmp_path):
    memory_file = tmp_path / "memory.md"
    db_path = tmp_path / "folium.db"
    observer = Observer(ObservabilityConfig(database_path=db_path))
    token = _observer_var.set(observer)
    llm = ReadThenAppendLLM()
    try:
        with mock.patch("folium.tools.memory.MEMORY_FILE", memory_file):
            result = _run_agent(MemoryAgent(llm))
    finally:
        _observer_var.reset(token)

    assert result.status == "UPDATED"
    assert result.cached_tokens == 15
    assert "Prefer concise Chinese responses." in memory_file.read_text(encoding="utf-8")
    with get_connection(db_path) as conn:
        trace = conn.execute(
            "SELECT trace_id, name FROM traces WHERE session_id = ?",
            ("session_test",),
        ).fetchone()
        events = conn.execute(
            "SELECT event_type, name, payload_json FROM trace_events WHERE trace_id = ?",
            (trace["trace_id"],),
        ).fetchall()
    assert trace["name"] == "memory_maintenance"
    assert {
        "memory_maintenance_scheduled",
        "memory_maintenance_started",
        "memory_operation",
        "memory_maintenance_decision",
        "memory_maintenance_completed",
    }.issubset({event["name"] for event in events})
    metadata = [json.loads(event["payload_json"] or "{}") for event in events]
    scheduled = next(item for item in metadata if item.get("context_source"))
    assert scheduled["context_source"] == "completed_main_agent_messages"
    assert scheduled["message_count"] == 3
    assert scheduled["visible_tool_count"] == 2
    assert any(item.get("cached_tokens") == 15 for item in metadata)
    assert all("Folium Memory" not in json.dumps(item) for item in metadata)


def test_memory_agent_stops_at_max_steps(tmp_path):
    llm = ReadForeverLLM()
    with mock.patch("folium.tools.memory.MEMORY_FILE", tmp_path / "memory.md"):
        result = _run_agent(MemoryAgent(llm, max_steps=5))

    assert result.status == "NO_CHANGE"
    assert result.steps == 5
    assert llm.calls == 5


class BlockingRunner:
    def __init__(self, result):
        self.result = result
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self.kwargs = None

    def run(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        self.started.set()
        self.release.wait(timeout=2)
        return self.result


async def _turn(
    scheduler,
    session_id,
    *,
    turns=1,
    used_memory=False,
    main_prompt_tokens=0,
    main_completion_tokens=0,
    main_request_matches_memory_context=False,
):
    return await scheduler.on_turn_completed(
        session_id=session_id,
        messages=_messages(turns),
        visible_tools=_visible_tools(),
        main_agent_used_memory=used_memory,
        main_prompt_tokens=main_prompt_tokens,
        main_completion_tokens=main_completion_tokens,
        main_request_matches_memory_context=main_request_matches_memory_context,
    )


def test_scheduler_keeps_turns_completed_while_maintenance_runs():
    async def scenario():
        runner = BlockingRunner(MemoryMaintenanceResult("NO_CHANGE", 0, 1))
        scheduler = MemoryMaintenanceScheduler(
            lambda: runner,
            threshold=10,
            max_context_tokens=100_000,
        )
        for turn in range(10):
            scheduled = await _turn(scheduler, "session_a", turns=turn + 1)
        assert scheduled is True
        await asyncio.to_thread(runner.started.wait, 1)
        task = scheduler._states["session_a"].task
        for turn in range(2):
            await _turn(scheduler, "session_a", turns=11 + turn)
        runner.release.set()
        await task
        assert await scheduler.pending_turns("session_a") == 2

    asyncio.run(scenario())


def test_scheduler_resets_checkpoint_for_main_memory_and_keeps_failures_pending():
    async def scenario():
        successful_runner = BlockingRunner(MemoryMaintenanceResult("NO_CHANGE", 0, 1))
        scheduler = MemoryMaintenanceScheduler(
            lambda: successful_runner,
            threshold=3,
            max_context_tokens=100_000,
        )
        for turn in range(2):
            await _turn(scheduler, "session_main_memory", turns=turn + 1)
        assert await scheduler.pending_turns("session_main_memory") == 2
        assert not await _turn(
            scheduler,
            "session_main_memory",
            turns=3,
            used_memory=True,
        )
        assert await scheduler.pending_turns("session_main_memory") == 0

        failed_runner = BlockingRunner(MemoryMaintenanceResult("FAILED", 0, 0))
        failed = MemoryMaintenanceScheduler(
            lambda: failed_runner,
            threshold=3,
            max_context_tokens=100_000,
        )
        for turn in range(3):
            await _turn(failed, "session_failure", turns=turn + 1)
        await asyncio.to_thread(failed_runner.started.wait, 1)
        task = failed._states["session_failure"].task
        failed_runner.release.set()
        await task
        assert await failed.pending_turns("session_failure") == 3

    asyncio.run(scenario())


def test_scheduler_uses_main_api_usage_without_recounting_visible_tools():
    async def scenario():
        runner = BlockingRunner(MemoryMaintenanceResult("NO_CHANGE", 0, 1))
        scheduler = MemoryMaintenanceScheduler(
            lambda: runner,
            threshold=1,
            max_context_tokens=10_000,
            max_output_tokens=200,
        )
        messages = _messages()
        visible_tools = _visible_tools()
        assert await scheduler.on_turn_completed(
            session_id="session_real_usage",
            messages=messages,
            visible_tools=visible_tools,
            main_agent_used_memory=False,
            main_prompt_tokens=700,
            main_completion_tokens=30,
            main_request_matches_memory_context=True,
        )
        await asyncio.to_thread(runner.started.wait, 1)
        assert runner.kwargs["input_tokens"] == (
            700 + 30 + estimate_text_tokens(MEMORY_MAINTENANCE_USER_PROMPT)
        )
        assert runner.kwargs["context_source"] == "main_api_usage"
        assert runner.kwargs["tool_output_trimmed_count"] == 0
        assert runner.kwargs["messages"] == messages
        task = scheduler._states["session_real_usage"].task
        runner.release.set()
        await task
        assert await scheduler.pending_turns("session_real_usage") == 0

    asyncio.run(scenario())


def test_scheduler_falls_back_to_message_estimate_for_incompatible_main_request():
    async def scenario():
        runner = BlockingRunner(MemoryMaintenanceResult("NO_CHANGE", 0, 1))
        scheduler = MemoryMaintenanceScheduler(
            lambda: runner,
            threshold=1,
            max_context_tokens=100_000,
        )
        messages = _messages()
        visible_tools = _visible_tools()
        assert await scheduler.on_turn_completed(
            session_id="session_fallback",
            messages=messages,
            visible_tools=visible_tools,
            main_agent_used_memory=False,
            main_prompt_tokens=99_999,
            main_completion_tokens=99_999,
            main_request_matches_memory_context=False,
        )
        await asyncio.to_thread(runner.started.wait, 1)
        assert runner.kwargs["input_tokens"] == estimate_memory_maintenance_input_tokens(
            messages,
            visible_tools,
        )
        assert runner.kwargs["context_source"] == "estimated_messages"
        task = scheduler._states["session_fallback"].task
        runner.release.set()
        await task

    asyncio.run(scenario())


def test_scheduler_trims_all_large_tool_outputs_then_runs_without_rechecking_budget():
    async def scenario():
        runner = BlockingRunner(MemoryMaintenanceResult("NO_CHANGE", 0, 1))
        scheduler = MemoryMaintenanceScheduler(
            lambda: runner,
            threshold=1,
            max_context_tokens=1_000,
            max_output_tokens=100,
        )
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "first user request"},
            {
                "role": "assistant",
                "content": "first assistant output",
                "tool_calls": [{"id": "old_call", "name": "bash", "arguments": "{}"}],
            },
            {
                "role": "tool",
                "tool_call_id": "old_call",
                "name": "bash",
                "content": "old-start-" + ("a" * 5_000) + "-old-end",
            },
            {"role": "user", "content": "latest user request"},
            {
                "role": "assistant",
                "content": "latest assistant output",
                "tool_calls": [{"id": "new_call", "name": "bash", "arguments": "{}"}],
            },
            {
                "role": "tool",
                "tool_call_id": "new_call",
                "name": "bash",
                "content": "new-start-" + ("b" * 5_000) + "-new-end",
            },
        ]
        original_messages = copy.deepcopy(messages)
        assert await scheduler.on_turn_completed(
            session_id="session_trim",
            messages=messages,
            visible_tools=_visible_tools(),
            main_agent_used_memory=False,
            main_prompt_tokens=320,
            main_completion_tokens=10,
            main_request_matches_memory_context=True,
        )
        await asyncio.to_thread(runner.started.wait, 1)
        scheduled_messages = runner.kwargs["messages"]
        assert messages == original_messages
        assert [
            message for message in scheduled_messages if message["role"] != "tool"
        ] == [
            message for message in original_messages if message["role"] != "tool"
        ]
        tool_messages = [
            message for message in scheduled_messages if message["role"] == "tool"
        ]
        assert len(tool_messages) == 2
        assert runner.kwargs["tool_output_trimmed_count"] == 2
        assert runner.kwargs["tool_output_trimmed_characters"] > 0
        for message in tool_messages:
            assert TOOL_OUTPUT_TRIM_MARKER in message["content"]
            assert len(message["content"]) <= (
                TOOL_OUTPUT_TRIM_KEEP_CHARS * 2 + len(TOOL_OUTPUT_TRIM_MARKER) + 2
            )
        task = scheduler._states["session_trim"].task
        runner.release.set()
        await task
        assert await scheduler.pending_turns("session_trim") == 0

    asyncio.run(scenario())
