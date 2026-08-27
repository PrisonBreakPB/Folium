import unittest
import time
from pathlib import Path
from unittest import mock

from folium.agent import FINAL_ROUND_REMINDER, Agent, TOOL_PARAMETER_WARN, TOOL_EXECUTION_WARN, ToolExecutionResult
from folium.context import TOOL_OUTPUT_DEDUPE_PLACEHOLDER
from folium.llm import LLMResponse, ToolCall
from folium.skills import load_skills
from folium.skills.types import Skill
from folium.tools import ALL_TOOLS, get_tool
from folium.tools.agent import _sub_agent_tool
from pydantic import BaseModel, ConfigDict

from folium.tools.base import Tool, ToolError, ToolExecutionError, ToolFailure, ToolOutput, ToolValidationError
from folium.tools.todo import TodoTool


class _AnyArgs(BaseModel):
    model_config = ConfigDict(extra="allow")


class BlockingTool(Tool):
    name = "blocking"
    description = "Block longer than the agent tool timeout."
    args_model = _AnyArgs

    def execute(self):
        time.sleep(2)
        return "done"


class TimeoutEchoTool(Tool):
    name = "timeout_echo"
    description = "Echo the timeout value."
    args_model = _AnyArgs

    def execute(self, timeout: int):
        return f"timeout={timeout}"


class EchoTool(Tool):
    name = "echo_tool"
    description = "Return a short result."
    args_model = _AnyArgs

    def execute(self):
        return "ok"


class StructuredFailureTool(Tool):
    name = "structured_failure"
    description = "Return a structured tool failure."
    args_model = _AnyArgs

    def execute(self):
        return ToolFailure(
            ToolError(
                code="http_503",
                category="network",
                message="service unavailable",
                retryable=True,
                details={"http_status": 503},
            )
        )


class RaisedToolExecutionErrorTool(Tool):
    name = "raised_tool_error"
    description = "Raise a structured tool execution error."
    args_model = _AnyArgs

    def execute(self):
        raise ToolExecutionError(
            ToolError(
                code="permission_denied",
                category="permission",
                message="access denied",
                retryable=False,
            )
        )


class ToolValidationTests(unittest.TestCase):
    def test_all_tools_accept_minimal_valid_arguments(self):
        samples = {
            "bash": {"command": "echo hello"},
            "read_file": {"file_path": "README.md"},
            "write_file": {"file_path": "out.txt", "content": "hello"},
            "edit_file": {"file_path": "README.md", "old_string": "a", "new_string": "b"},
            "glob": {"pattern": "*.py"},
            "grep": {"pattern": "Folium"},
            "agent": {"task": "summarize this project"},
            "todo": {"items": [{"id": "1", "text": "Plan task", "status": "pending"}]},
            "web_search": {"query": "agent context compression"},
            "web_fetch": {"url": "https://example.com"},
            "pdf_fetch": {"url": "https://example.com/paper.pdf"},
            "paper_search": {"query": "machine learning"},
            "paper_validate": {"papers": [{"title": "Attention Is All You Need"}]},
            "arxiv_search": {"query": "event-triggered control"},
            "session_history": {"action": "search"},
            "memory": {"section": "user_preferences", "content": "Prefer Chinese responses."},
        }

        for tool in ALL_TOOLS:
            with self.subTest(tool=tool.name):
                self.assertEqual(tool.validate_arguments(samples[tool.name]), samples[tool.name])

    def test_missing_required_field_is_rejected(self):
        tool = get_tool("read_file")

        with self.assertRaises(ToolValidationError) as ctx:
            tool.validate_arguments({})

        self.assertIn("missing required field 'file_path'", str(ctx.exception))

    def test_unknown_field_is_rejected(self):
        tool = get_tool("bash")

        with self.assertRaises(ToolValidationError) as ctx:
            tool.validate_arguments({"command": "echo hello", "extra": "nope"})

        self.assertIn("unknown field 'extra'", str(ctx.exception))

    def test_wrong_type_is_rejected(self):
        tool = get_tool("bash")

        with self.assertRaises(ToolValidationError) as ctx:
            tool.validate_arguments({"command": "echo hello", "timeout": "slow"})

        self.assertIn("field 'timeout'", str(ctx.exception))

    def test_pydantic_tool_preserves_omitted_defaults(self):
        tool = get_tool("read_file")

        self.assertEqual(
            tool.validate_arguments({"file_path": "README.md"}),
            {"file_path": "README.md"},
        )

    def test_pydantic_tool_rejects_unknown_field(self):
        tool = get_tool("read_file")

        with self.assertRaises(ToolValidationError) as ctx:
            tool.validate_arguments({"file_path": "README.md", "extra": True})

        self.assertIn("unknown field 'extra'", str(ctx.exception))

    def test_pydantic_tool_rejects_strict_type(self):
        tool = get_tool("read_file")

        with self.assertRaises(ToolValidationError) as ctx:
            tool.validate_arguments({"file_path": "README.md", "offset": "2"})

        self.assertIn("field 'offset'", str(ctx.exception))

    def test_agent_surfaces_pydantic_validation_error(self):
        agent = Agent(llm=None)
        tc = ToolCall(id="call_1", name="read_file", arguments={"offset": "2"})

        result = agent._exec_tool(tc)

        self.assertEqual(result.status, "bad_arguments")
        self.assertEqual(result.error_code, "bad_arguments")
        self.assertEqual(result.error_category, "validation")
        self.assertIn("bad arguments for read_file", result.content)
        self.assertIn("field 'file_path'", result.content)
        self.assertIn("field 'offset'", result.content)

    def test_pydantic_tool_rejects_invalid_enum(self):
        tool = get_tool("arxiv_search")

        with self.assertRaises(ToolValidationError) as ctx:
            tool.validate_arguments({"query": "control", "sort": "citations"})

        self.assertIn("field 'sort'", str(ctx.exception))

    def test_pydantic_tool_surfaces_malformed_json_error(self):
        tool = get_tool("read_file")

        with self.assertRaises(ToolValidationError) as ctx:
            tool.validate_arguments('{"file_path":')

        self.assertIn("arguments JSON could not be parsed", str(ctx.exception))

    def test_agent_tool_execution_rejects_bad_arguments_before_execute(self):
        agent = Agent(llm=None)
        tc = ToolCall(id="call_1", name="bash", arguments={"timeout": 1})

        result = agent._exec_tool(tc)

        self.assertEqual(result.status, "bad_arguments")
        self.assertIn("bad arguments for bash", result.content)
        self.assertIn("missing required field 'command'", result.content)

    def test_agent_tool_execution_times_out(self):
        agent = Agent(llm=None, tools=[BlockingTool()], tool_timeout=1)
        tc = ToolCall(id="call_1", name="blocking", arguments={})

        result = agent._exec_tool(tc)

        self.assertEqual(result.status, "timeout")
        self.assertEqual(result.error_code, "timeout")
        self.assertEqual(result.error_category, "timeout")
        self.assertIn("timed out after 1s", result.content)

    def test_agent_clamps_tool_timeout_argument(self):
        agent = Agent(llm=None, tools=[TimeoutEchoTool()], tool_timeout=3)
        tc = ToolCall(id="call_1", name="timeout_echo", arguments={"timeout": 120})

        result = agent._exec_tool(tc)

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.content, "timeout=3")

    def test_agent_preserves_structured_tool_failure(self):
        agent = Agent(llm=None, tools=[StructuredFailureTool()])
        result = agent._exec_tool(ToolCall(id="call_1", name="structured_failure", arguments={}))

        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_code, "http_503")
        self.assertEqual(result.error_category, "network")
        self.assertTrue(result.retryable)
        self.assertEqual(result.error_details, {"http_status": 503})
        self.assertEqual(result.content, "service unavailable")

    def test_agent_converts_structured_tool_execution_exception(self):
        agent = Agent(llm=None, tools=[RaisedToolExecutionErrorTool()])
        result = agent._exec_tool(ToolCall(id="call_1", name="raised_tool_error", arguments={}))

        self.assertEqual(result.status, "error")
        self.assertEqual(result.error_code, "permission_denied")
        self.assertEqual(result.error_category, "permission")
        self.assertFalse(result.retryable)
        self.assertEqual(result.content, "access denied")

    def test_tool_output_content_is_not_parsed_as_error(self):
        class ErrorTextTool(Tool):
            name = "error_text"
            description = "Return successful content beginning with Error."
            args_model = _AnyArgs

            def execute(self):
                return ToolOutput(content="Error handling is documented here")

        agent = Agent(llm=None, tools=[ErrorTextTool()])
        result = agent._exec_tool(ToolCall(id="call_1", name="error_text", arguments={}))

        self.assertEqual(result.status, "ok")
        self.assertIsNone(result.error_code)

    def test_agent_stops_after_five_bad_tool_arguments(self):
        class BadToolLLM:
            model = "fake-model"

            def __init__(self):
                self.calls = 0

            @property
            def estimated_cost(self):
                return None

            def chat(self, messages, tools=None, on_token=None, **kwargs):
                self.calls += 1
                return LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(id=f"call_{self.calls}", name="write_file", arguments={})
                    ],
                )

        agent = Agent(llm=BadToolLLM(), max_rounds=50, max_bad_tool_calls=5)

        result = agent.chat("write a tex file")

        self.assertIn("I stopped retrying write_file", result)
        self.assertIn("parameter_validation_failure", result)
        self.assertEqual(agent.llm.calls, 5)
        self.assertEqual(len([m for m in agent.messages if m["role"] == "tool"]), 5)
        self.assertEqual(agent.messages[-1]["name"], "write_file")
        self.assertIn("missing required field 'file_path'", agent.messages[-1]["content"])
        # N1 warning injected exactly once
        self.assertEqual(len([m for m in agent.messages if m.get("content") == TOOL_PARAMETER_WARN]), 1)

    def test_agent_stops_after_five_tool_execution_errors(self):
        class FailingTool(Tool):
            name = "failing"
            description = "Always fails."
            args_model = _AnyArgs

            def execute(self):
                return "Error: always fails"

        class FailingLLM:
            model = "fake-model"

            def __init__(self):
                self.calls = 0

            @property
            def estimated_cost(self):
                return None

            def chat(self, messages, tools=None, on_token=None, **kwargs):
                self.calls += 1
                return LLMResponse(
                    content="",
                    tool_calls=[ToolCall(id=f"call_{self.calls}", name="failing", arguments="{}")],
                )

        agent = Agent(llm=FailingLLM(), tools=[FailingTool()], max_rounds=50, max_tool_errors=5)

        result = agent.chat("do something")

        self.assertIn("I stopped retrying failing", result)
        self.assertIn("execution_failure", result)
        self.assertEqual(agent.llm.calls, 5)
        self.assertEqual(len([m for m in agent.messages if m.get("content") == TOOL_EXECUTION_WARN]), 1)

    def test_failure_streaks_count_independently(self):
        advance = Agent._advance_failure_streaks

        # bad_arguments increments the parameter streak, resets the error streak
        self.assertEqual(advance(ToolExecutionResult("x", "bad_arguments"), 2, 3), (3, 0))
        # error increments the error streak, resets the parameter streak
        self.assertEqual(advance(ToolExecutionResult("x", "error"), 2, 3), (0, 4))
        # ok / timeout reset both
        self.assertEqual(advance(ToolExecutionResult("x", "ok"), 4, 4), (0, 0))
        self.assertEqual(advance(ToolExecutionResult("x", "timeout"), 4, 4), (0, 0))

    def test_agent_does_not_inject_todo_reminder_before_todos_exist(self):
        class NoTodoLLM:
            model = "fake-model"

            def __init__(self):
                self.calls = 0
                self.total_prompt_tokens = 0
                self.total_completion_tokens = 0
                self.total_cached_tokens = 0
                self.last_prompt_tokens = 0
                self.last_completion_tokens = 0

            @property
            def estimated_cost(self):
                return None

            def chat(self, messages, tools=None, on_token=None, **kwargs):
                self.calls += 1
                if self.calls <= 4:
                    return LLMResponse(
                        content="",
                        tool_calls=[ToolCall(id=f"call_{self.calls}", name="echo_tool", arguments={})],
                    )
                return LLMResponse(content="done")

        agent = Agent(llm=NoTodoLLM(), tools=[EchoTool(), TodoTool()], max_rounds=5)
        agent.chat("do a long task")

        reminders = [m for m in agent.messages if m.get("content") == "<reminder>Update your todos.</reminder>"]
        self.assertEqual(reminders, [])
        transcript_reminders = [
            m for m in agent.transcript
            if m.get("content") == "<reminder>Update your todos.</reminder>"
        ]
        self.assertEqual(transcript_reminders, [])

    def test_agent_injects_todo_reminder_after_existing_todos_go_stale(self):
        class StaleTodoLLM:
            model = "fake-model"

            def __init__(self):
                self.calls = 0
                self.total_prompt_tokens = 0
                self.total_completion_tokens = 0
                self.total_cached_tokens = 0
                self.last_prompt_tokens = 0
                self.last_completion_tokens = 0

            @property
            def estimated_cost(self):
                return None

            def chat(self, messages, tools=None, on_token=None, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return LLMResponse(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="todo_1",
                                name="todo",
                                arguments={
                                    "items": [
                                        {"id": "1", "text": "Inspect code", "status": "in_progress"}
                                    ]
                                },
                            )
                        ],
                    )
                if self.calls <= 5:
                    return LLMResponse(
                        content="",
                        tool_calls=[ToolCall(id=f"call_{self.calls}", name="echo_tool", arguments={})],
                    )
                return LLMResponse(content="done")

        agent = Agent(llm=StaleTodoLLM(), tools=[EchoTool(), TodoTool()], max_rounds=6)
        agent.chat("do a long task")

        reminders = [m for m in agent.messages if m.get("content") == "<reminder>Update your todos.</reminder>"]
        self.assertEqual(len(reminders), 1)
        transcript_reminders = [
            m for m in agent.transcript
            if m.get("content") == "<reminder>Update your todos.</reminder>"
        ]
        self.assertEqual(transcript_reminders, [])

    def test_final_round_disables_tools_and_requests_direct_response(self):
        class FinalRoundLLM:
            model = "fake-model"

            def __init__(self):
                self.calls = 0
                self.requests = []
                self.total_prompt_tokens = 0
                self.total_completion_tokens = 0
                self.total_cached_tokens = 0
                self.last_prompt_tokens = 0
                self.last_completion_tokens = 0

            @property
            def estimated_cost(self):
                return None

            def chat(self, messages, tools=None, on_token=None, **kwargs):
                self.calls += 1
                self.requests.append({"messages": messages, "tools": tools})
                if self.calls == 1:
                    return LLMResponse(
                        content="",
                        tool_calls=[ToolCall(id="call_1", name="echo_tool", arguments={})],
                    )
                return LLMResponse(content="final answer")

        llm = FinalRoundLLM()
        agent = Agent(llm=llm, tools=[EchoTool()], max_rounds=2)

        self.assertEqual(agent.chat("do a bounded task"), "final answer")
        self.assertIsNotNone(llm.requests[0]["tools"])
        self.assertIsNone(llm.requests[1]["tools"])
        self.assertEqual(llm.requests[1]["messages"][-1], {"role": "user", "content": FINAL_ROUND_REMINDER})
        self.assertFalse(any(m.get("content") == FINAL_ROUND_REMINDER for m in agent.messages))
        self.assertFalse(any(m.get("content") == FINAL_ROUND_REMINDER for m in agent.transcript))

    def test_final_round_tool_calls_are_not_executed(self):
        class StubbornLLM:
            model = "fake-model"

            def __init__(self):
                self.calls = 0
                self.total_prompt_tokens = 0
                self.total_completion_tokens = 0
                self.total_cached_tokens = 0
                self.last_prompt_tokens = 0
                self.last_completion_tokens = 0

            @property
            def estimated_cost(self):
                return None

            def chat(self, messages, tools=None, on_token=None, **kwargs):
                self.calls += 1
                return LLMResponse(
                    content="",
                    tool_calls=[ToolCall(id=f"call_{self.calls}", name="echo_tool", arguments={})],
                )

        agent = Agent(llm=StubbornLLM(), tools=[EchoTool()], max_rounds=1)

        self.assertEqual(agent.chat("do a bounded task"), "(reached maximum tool-call rounds)")
        self.assertFalse(any(m.get("role") == "assistant" and m.get("tool_calls") for m in agent.messages))
        self.assertFalse(any(m.get("role") == "tool" for m in agent.messages))

    def test_agent_emits_todo_update_event(self):
        class TodoLLM:
            model = "fake-model"

            def __init__(self):
                self.calls = 0
                self.total_prompt_tokens = 0
                self.total_completion_tokens = 0
                self.total_cached_tokens = 0
                self.last_prompt_tokens = 0
                self.last_completion_tokens = 0

            @property
            def estimated_cost(self):
                return None

            def chat(self, messages, tools=None, on_token=None, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    return LLMResponse(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="todo_1",
                                name="todo",
                                arguments={
                                    "items": [
                                        {"id": "1", "text": "Inspect code", "status": "in_progress"}
                                    ]
                                },
                            )
                        ],
                    )
                return LLMResponse(content="done")

        agent = Agent(llm=TodoLLM(), max_rounds=3)
        events = []
        agent.chat("inspect", on_event=events.append)

        todo_events = [e for e in events if e["type"] == "todo_update"]
        self.assertEqual(len(todo_events), 1)
        self.assertEqual(todo_events[0]["items"][0]["status"], "in_progress")

    def test_sub_agent_gets_independent_todo_manager(self):
        agent = Agent(llm=None)
        agent.todo_manager.update([
            {"id": "1", "text": "Parent task", "status": "in_progress"}
        ])

        sub_todo = _sub_agent_tool(agent.todo_tool)

        self.assertIsInstance(sub_todo, TodoTool)
        self.assertIsNot(sub_todo.manager, agent.todo_manager)
        self.assertEqual(sub_todo.manager.snapshot(), [])

    def test_context_compression_preserves_system_addendum(self):
        addendum = "You are the literature-searcher sub-agent."
        agent = Agent(llm=None, tools=[], skills=[], system_addendum=addendum)
        agent.context.maybe_compress = lambda messages, llm=None, real_tokens=None: {
            "compressed": True,
            "layers": [{"name": "test", "changed": True}],
        }

        agent._maybe_compress_observed("test")

        self.assertIn("# Sub-agent Instructions", agent._system)
        self.assertIn(addendum, agent._system)
        self.assertEqual(agent.skills, [])

    def test_context_compression_preserves_explicit_skill_scope(self):
        allowed_skill = next(
            skill for skill in load_skills()
            if skill.name == "control-literature-search"
        )
        agent = Agent(llm=None, tools=[], skills=[allowed_skill])
        agent.context.maybe_compress = lambda messages, llm=None, real_tokens=None: {
            "compressed": True,
            "layers": [{"name": "test", "changed": True}],
        }

        agent._maybe_compress_observed("test")

        self.assertEqual(
            [skill.name for skill in agent.skills],
            ["control-literature-search"],
        )

    def test_context_compression_rescans_default_skill_directory(self):
        initial = Skill(
            "initial",
            "Initial skill.",
            Path("skills/initial"),
            Path("skills/initial/SKILL.md"),
        )
        refreshed = Skill(
            "refreshed",
            "Refreshed skill.",
            Path("skills/refreshed"),
            Path("skills/refreshed/SKILL.md"),
        )

        with mock.patch("folium.agent.load_skills", side_effect=[[initial], [refreshed]]):
            agent = Agent(llm=None, tools=[])
            agent.context.maybe_compress = lambda messages, llm=None, real_tokens=None: {
                "compressed": True,
                "layers": [{"name": "test", "changed": True}],
            }

            agent._maybe_compress_observed("test")

        self.assertEqual([skill.name for skill in agent.skills], ["refreshed"])

    def test_agent_transcript_keeps_full_tool_output_after_context_compression(self):
        long_output = "x" * 9000

        class LongTool(Tool):
            name = "long_tool"
            description = "Return a long output."
            args_model = _AnyArgs

            def execute(self):
                return long_output

        class LongToolLLM:
            model = "fake-model"

            def __init__(self):
                self.calls = 0
                self.total_prompt_tokens = 0
                self.total_completion_tokens = 0
                self.total_cached_tokens = 0
                self.last_prompt_tokens = 1300000
                self.last_completion_tokens = 0

            @property
            def estimated_cost(self):
                return None

            def chat(self, messages, tools=None, on_token=None, **kwargs):
                self.calls += 1
                if self.calls <= 3:
                    return LLMResponse(
                        content="",
                        tool_calls=[ToolCall(id=f"call_{self.calls}", name="long_tool", arguments={})],
                    )
                return LLMResponse(content="done")

        agent = Agent(
            llm=LongToolLLM(),
            tools=[LongTool()],
            max_context_tokens=2_000_000,
            max_rounds=4,
        )
        agent.chat("run long tool")

        context_tool = next(
            m for m in agent.messages
            if m.get("role") == "tool" and m.get("tool_call_id") == "call_1"
        )
        transcript_tool = next(
            m for m in agent.transcript
            if m.get("role") == "tool" and m.get("tool_call_id") == "call_1"
        )
        self.assertEqual(context_tool["content"], TOOL_OUTPUT_DEDUPE_PLACEHOLDER)
        self.assertEqual(transcript_tool["content"], long_output)


if __name__ == "__main__":
    unittest.main()
