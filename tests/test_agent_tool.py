import unittest

from folium.agent import Agent
from folium.llm import LLMResponse
from folium.tools import create_tools
from folium.tools.agent import AgentTool


class CaptureLLM:
    model = "fake-model"

    def __init__(self):
        self.messages = None
        self.tools = None

    def chat(self, messages, tools=None, on_token=None):
        self.messages = messages
        self.tools = tools or []
        return LLMResponse(content="done")


class AgentToolTests(unittest.TestCase):
    def test_unknown_agent_type_returns_error(self):
        parent = Agent(llm=CaptureLLM(), tools=[AgentTool()])
        tool = parent.tools[0]

        result = tool.execute(task="search papers", agent_type="unknown")

        self.assertIn("unknown agent_type", result)

    def test_literature_searcher_limits_tools_and_skills(self):
        llm = CaptureLLM()
        parent = Agent(llm=llm, tools=create_tools(), tool_timeout=60)
        tool = next(t for t in parent.tools if isinstance(t, AgentTool))

        result = tool.execute(
            task="Search event-triggered control papers.",
            agent_type="literature-searcher",
            output_format="papers",
            timeout=30,
        )

        self.assertIn("[Sub-agent completed: literature-searcher]", result)
        tool_names = {tool["function"]["name"] for tool in llm.tools}
        self.assertEqual(tool_names, {"read_file", "paper_search", "paper_validate", "arxiv_search", "web_search", "web_fetch"})

        system = llm.messages[0]["content"]
        self.assertIn("You are the literature-searcher sub-agent", system)
        self.assertIn("<name>control-literature-search</name>", system)
        self.assertNotIn("<name>deep-research</name>", system)

        user_task = llm.messages[1]["content"]
        self.assertIn("Search event-triggered control papers.", user_task)
        self.assertIn("[Required output format]", user_task)
        self.assertIn('"papers"', user_task)

    def test_context_other_than_none_is_rejected_for_now(self):
        parent = Agent(llm=CaptureLLM(), tools=[AgentTool()])
        tool = parent.tools[0]

        result = tool.execute(task="search papers", context="full")

        self.assertIn("only context='none' is supported", result)


if __name__ == "__main__":
    unittest.main()
