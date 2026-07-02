import unittest
import time

from folium.agent import Agent
from folium.llm import LLMResponse, ToolCall
from folium.tools import ALL_TOOLS, get_tool
from folium.tools.agent import _sub_agent_tool
from folium.tools.base import Tool, ToolValidationError
from folium.tools.todo import TodoTool


class BlockingTool(Tool):
    name = "blocking"
    description = "Block longer than the agent tool timeout."
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def execute(self):
        time.sleep(2)
        return "done"


class TimeoutEchoTool(Tool):
    name = "timeout_echo"
    description = "Echo the timeout value."
    parameters = {
        "type": "object",
        "properties": {
            "timeout": {"type": "integer"},
        },
        "required": ["timeout"],
    }

    def execute(self, timeout: int):
        return f"timeout={timeout}"


class EchoTool(Tool):
    name = "echo_tool"
    description = "Return a short result."
    parameters = {
        "type": "object",
        "properties": {},
        "required": [],
    }

    def execute(self):
        return "ok"


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

        self.assertIn("field 'timeout' must be integer", str(ctx.exception))

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
        self.assertIn("timed out after 1s", result.content)

    def test_agent_clamps_tool_timeout_argument(self):
        agent = Agent(llm=None, tools=[TimeoutEchoTool()], tool_timeout=3)
        tc = ToolCall(id="call_1", name="timeout_echo", arguments={"timeout": 120})

        result = agent._exec_tool(tc)

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.content, "timeout=3")

    def test_agent_stops_after_five_bad_tool_arguments(self):
        class BadToolLLM:
            model = "fake-model"

            def __init__(self):
                self.calls = 0

            @property
            def estimated_cost(self):
                return None

            def chat(self, messages, tools=None, on_token=None):
                self.calls += 1
                return LLMResponse(
                    content="",
                    tool_calls=[
                        ToolCall(id=f"call_{self.calls}", name="write_file", arguments={})
                    ],
                )

        agent = Agent(llm=BadToolLLM(), max_rounds=50, max_bad_tool_calls=5)

        result = agent.chat("write a tex file")

        self.assertEqual(result, "Consecutive 5 tool call failures, current task stopped.")
        self.assertEqual(agent.llm.calls, 5)
        self.assertEqual(len([m for m in agent.messages if m["role"] == "tool"]), 5)
        self.assertEqual(agent.messages[-1]["name"], "write_file")
        self.assertIn("missing required field 'file_path'", agent.messages[-1]["content"])

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

            def chat(self, messages, tools=None, on_token=None):
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

            def chat(self, messages, tools=None, on_token=None):
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

            def chat(self, messages, tools=None, on_token=None):
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

    def test_agent_transcript_keeps_full_tool_output_after_context_trim(self):
        long_output = "x" * 9000

        class LongTool(Tool):
            name = "long_tool"
            description = "Return a long output."
            parameters = {"type": "object", "properties": {}, "required": []}

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

            def chat(self, messages, tools=None, on_token=None):
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
        self.assertIn("trimmed to save context", context_tool["content"])
        self.assertEqual(transcript_tool["content"], long_output)


if __name__ == "__main__":
    unittest.main()
