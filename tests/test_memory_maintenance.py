import asyncio
import json
import threading
from unittest import mock

import pytest

from folium.database import get_connection
from folium.llm import LLMResponse, ToolCall
from folium.memory_maintenance import (
    MemoryMaintenanceAgent,
    MemoryMaintenanceResult,
    MemoryMaintenanceScheduler,
    build_memory_maintenance_snapshot,
)
from folium.observability.config import ObservabilityConfig
from folium.observability.context import Observer, _observer_var
from folium.tools.base import ToolValidationError
from folium.tools.memory import MemoryTool


def _version(read_output: str) -> str:
    return read_output.splitlines()[0].removeprefix("Memory version: ")


def _transcript(turns: int) -> list[dict]:
    messages = []
    for index in range(turns):
        messages.extend(
            [
                {"role": "user", "content": f"user {index}"},
                {"role": "assistant", "content": f"assistant {index}"},
            ]
        )
    return messages


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


def test_snapshot_keeps_only_user_and_final_assistant_messages():
    snapshot = build_memory_maintenance_snapshot(
        [
            {"role": "user", "content": "Remember the project constraint."},
            {
                "role": "assistant",
                "content": "Calling a tool",
                "tool_calls": [{"id": "call_1"}],
            },
            {"role": "tool", "name": "read_file", "content": "raw tool output"},
            {"role": "assistant", "content": "The constraint is confirmed."},
        ]
    )

    assert "Remember the project constraint." in snapshot.content
    assert "The constraint is confirmed." in snapshot.content
    assert "Calling a tool" not in snapshot.content
    assert "raw tool output" not in snapshot.content


class NoChangeLLM:
    model = "fake-memory-model"

    def __init__(self):
        self.tools = []

    def chat(self, messages, tools=None, on_token=None):
        self.tools.append(tools)
        return LLMResponse(content="NO_CHANGE")


class ReadThenAppendLLM:
    model = "fake-memory-model"

    def __init__(self):
        self.calls = 0
        self.tools = []

    def chat(self, messages, tools=None, on_token=None):
        self.calls += 1
        self.tools.append(tools)
        if self.calls == 1:
            return LLMResponse(
                tool_calls=[ToolCall(id="read", name="memory", arguments={"action": "read"})]
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
                ]
            )
        return LLMResponse(content="UPDATED")


class ReadForeverLLM:
    model = "fake-memory-model"

    def __init__(self):
        self.calls = 0

    def chat(self, messages, tools=None, on_token=None):
        self.calls += 1
        return LLMResponse(
            tool_calls=[ToolCall(id=str(self.calls), name="memory", arguments={"action": "read"})]
        )


def test_maintenance_agent_can_choose_no_change(tmp_path):
    llm = NoChangeLLM()
    tool = MemoryTool()
    snapshot = build_memory_maintenance_snapshot(_transcript(1))
    with mock.patch("folium.tools.memory.MEMORY_FILE", tmp_path / "memory.md"):
        result = MemoryMaintenanceAgent(llm, tool).run(
            session_id="session_no_change",
            turn_index=10,
            snapshot=snapshot,
        )

    assert result.status == "NO_CHANGE"
    assert result.writes == 0
    assert len(llm.tools) == 1
    assert [schema["function"]["name"] for schema in llm.tools[0]] == ["memory"]


def test_maintenance_agent_reads_then_writes_and_records_trace(tmp_path):
    memory_file = tmp_path / "memory.md"
    db_path = tmp_path / "folium.db"
    observer = Observer(ObservabilityConfig(database_path=db_path))
    token = _observer_var.set(observer)
    try:
        with mock.patch("folium.tools.memory.MEMORY_FILE", memory_file):
            result = MemoryMaintenanceAgent(ReadThenAppendLLM()).run(
                session_id="session_trace",
                turn_index=10,
                snapshot=build_memory_maintenance_snapshot(_transcript(1)),
            )
    finally:
        _observer_var.reset(token)

    assert result.status == "UPDATED"
    assert "Prefer concise Chinese responses." in memory_file.read_text(encoding="utf-8")
    with get_connection(db_path) as conn:
        trace = conn.execute(
            "SELECT trace_id, name FROM traces WHERE session_id = ?",
            ("session_trace",),
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
    assert all("Folium Memory" not in json.dumps(item) for item in metadata)


def test_maintenance_agent_stops_at_max_steps(tmp_path):
    llm = ReadForeverLLM()
    with mock.patch("folium.tools.memory.MEMORY_FILE", tmp_path / "memory.md"):
        result = MemoryMaintenanceAgent(llm, max_steps=5).run(
            session_id="session_steps",
            turn_index=10,
            snapshot=build_memory_maintenance_snapshot(_transcript(1)),
        )

    assert result.status == "NO_CHANGE"
    assert result.steps == 5
    assert llm.calls == 5


class BlockingRunner:
    def __init__(self, result):
        self.result = result
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def run(self, **kwargs):
        self.calls += 1
        self.started.set()
        self.release.wait(timeout=2)
        return self.result


def test_scheduler_keeps_turns_completed_while_maintenance_runs():
    async def scenario():
        runner = BlockingRunner(MemoryMaintenanceResult("NO_CHANGE", 0, 1))
        scheduler = MemoryMaintenanceScheduler(lambda: runner, threshold=10)
        transcript = _transcript(10)
        for turn in range(10):
            scheduled = await scheduler.on_turn_completed(
                session_id="session_a",
                transcript=transcript[: (turn + 1) * 2],
                main_agent_used_memory=False,
            )
        assert scheduled is True
        await asyncio.to_thread(runner.started.wait, 1)
        task = scheduler._states["session_a"].task
        for turn in range(2):
            await scheduler.on_turn_completed(
                session_id="session_a",
                transcript=transcript + _transcript(turn + 1),
                main_agent_used_memory=False,
            )
        runner.release.set()
        await task
        assert await scheduler.pending_turns("session_a") == 2

    asyncio.run(scenario())


def test_scheduler_resets_checkpoint_for_main_memory_and_keeps_failures_pending():
    async def scenario():
        successful_runner = BlockingRunner(MemoryMaintenanceResult("NO_CHANGE", 0, 1))
        scheduler = MemoryMaintenanceScheduler(lambda: successful_runner, threshold=3)
        for _ in range(2):
            await scheduler.on_turn_completed(
                session_id="session_main_memory",
                transcript=_transcript(2),
                main_agent_used_memory=False,
            )
        assert await scheduler.pending_turns("session_main_memory") == 2
        assert not await scheduler.on_turn_completed(
            session_id="session_main_memory",
            transcript=_transcript(3),
            main_agent_used_memory=True,
        )
        assert await scheduler.pending_turns("session_main_memory") == 0

        failed_runner = BlockingRunner(MemoryMaintenanceResult("FAILED", 0, 0))
        failed = MemoryMaintenanceScheduler(lambda: failed_runner, threshold=3)
        for _ in range(3):
            await failed.on_turn_completed(
                session_id="session_failure",
                transcript=_transcript(3),
                main_agent_used_memory=False,
            )
        await asyncio.to_thread(failed_runner.started.wait, 1)
        task = failed._states["session_failure"].task
        failed_runner.release.set()
        await task
        assert await failed.pending_turns("session_failure") == 3

    asyncio.run(scenario())
