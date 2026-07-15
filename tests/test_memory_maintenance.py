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
)
from folium.observability.config import ObservabilityConfig
from folium.observability.context import Observer, _observer_var
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


def test_memory_agent_skips_context_limit_without_calling_the_model():
    llm = NoChangeLLM()
    result = MemoryAgent(llm).run(
        session_id="session_context_limit",
        turn_index=10,
        messages=_messages(),
        visible_tools=_visible_tools(),
        input_tokens=100,
        skip_reason="SKIPPED_CONTEXT_LIMIT",
    )

    assert result.status == "SKIPPED_CONTEXT_LIMIT"
    assert llm.requests == []


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


async def _turn(scheduler, session_id, *, turns=1, used_memory=False):
    return await scheduler.on_turn_completed(
        session_id=session_id,
        messages=_messages(turns),
        visible_tools=_visible_tools(),
        main_agent_used_memory=used_memory,
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


def test_scheduler_marks_context_limit_skip_as_checked():
    async def scenario():
        runner = BlockingRunner(MemoryMaintenanceResult("SKIPPED_CONTEXT_LIMIT", 0, 0))
        scheduler = MemoryMaintenanceScheduler(
            lambda: runner,
            threshold=1,
            max_context_tokens=10,
            max_output_tokens=2,
        )
        assert await scheduler.on_turn_completed(
            session_id="session_context_limit",
            messages=[{"role": "system", "content": "This message is deliberately too long."}],
            visible_tools=_visible_tools(),
            main_agent_used_memory=False,
        )
        await asyncio.to_thread(runner.started.wait, 1)
        assert runner.kwargs["skip_reason"] == "SKIPPED_CONTEXT_LIMIT"
        assert runner.kwargs["input_tokens"] + 2 > 10
        task = scheduler._states["session_context_limit"].task
        runner.release.set()
        await task
        assert await scheduler.pending_turns("session_context_limit") == 0

    asyncio.run(scenario())
