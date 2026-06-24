import unittest
import time

from folium.agent import Agent
from folium.llm import LLMResponse, ToolCall
from folium.tools import ALL_TOOLS, get_tool
from folium.tools.base import Tool, ToolValidationError


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

        self.assertEqual(result, "连续 5 次工具调用失败，已停止当前任务。")
        self.assertEqual(agent.llm.calls, 5)
        self.assertEqual(len([m for m in agent.messages if m["role"] == "tool"]), 5)
        self.assertEqual(agent.messages[-1]["name"], "write_file")
        self.assertIn("missing required field 'file_path'", agent.messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
