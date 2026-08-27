"""Tests for automatic tool-execution retry (retry_safe + retryable)."""

import random
import time
import unittest
from unittest import mock
from pydantic import BaseModel
from typing import ClassVar

from folium.agent import Agent
from folium.llm import LLMResponse, ToolCall
from folium.tools.base import Tool, ToolOutput, tool_failure


class _EmptyArgs(BaseModel):
    pass


class RetryableFlakyTool(Tool):
    """Fails retryably N times, then succeeds."""

    name = "retryable_flaky"
    description = "Fails a few times then succeeds."
    args_model: ClassVar[type[BaseModel]] = _EmptyArgs
    retry_safe = True

    def __init__(self, fail_before: int = 2):
        self.fail_before = fail_before
        self.calls = 0

    def execute(self, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_before:
            return tool_failure("transient", "network", "temporary failure", retryable=True)
        return ToolOutput(content=f"ok after {self.calls} calls")


class NeverRetryableTool(Tool):
    """Not retry_safe, even though its failure is marked retryable."""

    name = "write_fake"
    description = "Fake non-retry-safe tool."
    args_model: ClassVar[type[BaseModel]] = _EmptyArgs
    retry_safe = False

    def __init__(self):
        self.calls = 0

    def execute(self, **kwargs):
        self.calls += 1
        return tool_failure("transient", "network", "transient", retryable=True)


class DeterministicTool(Tool):
    """Failure is NOT retryable (deterministic)."""

    name = "deterministic_fail"
    description = "Deterministic failure, not retryable."
    args_model: ClassVar[type[BaseModel]] = _EmptyArgs
    retry_safe = True

    def __init__(self):
        self.calls = 0

    def execute(self, **kwargs):
        self.calls += 1
        return tool_failure("bad_input", "validation", "nope", retryable=False)


class _QuietLLM:
    model = "fake"

    def chat(self, messages, tools=None, on_token=None, **kwargs):
        return LLMResponse(content="done")


def _mk_agent(tool, max_retries=3):
    return Agent(llm=_QuietLLM(), tools=[tool], max_tool_retries=max_retries)


def _tc(tool: Tool, invocation=1):
    return ToolCall(id=f"tc_{invocation}", name=tool.name, arguments={})


class ToolRetryTests(unittest.TestCase):
    def _exec(self, agent, tc):
        # Avoid real backoff sleeps while still exercising the retry loop.
        with mock.patch.object(time, "sleep"):
            return agent._exec_tool(tc)

    def test_transient_tool_retries_then_succeeds(self):
        tool = RetryableFlakyTool(fail_before=2)
        agent = _mk_agent(tool, max_retries=3)

        result = self._exec(agent, _tc(tool))

        self.assertEqual(result.status, "ok")
        self.assertEqual(result.content, "ok after 3 calls")
        self.assertEqual(tool.calls, 3)
        self.assertEqual(result.attempts, 3)

    def test_hits_retry_cap_and_returns_last_failure(self):
        tool = RetryableFlakyTool(fail_before=99)  # always fails
        agent = _mk_agent(tool, max_retries=3)

        result = self._exec(agent, _tc(tool))

        self.assertEqual(result.status, "error")
        self.assertTrue(result.retryable)
        # 1 initial + max_retries retries = 4 executions.
        self.assertEqual(tool.calls, 4)
        self.assertEqual(result.attempts, 4)

    def test_non_retry_safe_tool_is_not_retried(self):
        tool = NeverRetryableTool()
        agent = _mk_agent(tool, max_retries=3)

        result = self._exec(agent, _tc(tool))

        self.assertEqual(tool.calls, 1)
        self.assertEqual(result.attempts, 1)

    def test_deterministic_failure_not_retried(self):
        tool = DeterministicTool()
        agent = _mk_agent(tool, max_retries=3)

        result = self._exec(agent, _tc(tool))

        self.assertEqual(tool.calls, 1)
        self.assertEqual(result.attempts, 1)

    def test_max_retries_zero_disables_retry(self):
        tool = RetryableFlakyTool(fail_before=99)
        agent = _mk_agent(tool, max_retries=0)

        result = self._exec(agent, _tc(tool))

        self.assertEqual(tool.calls, 1)
        self.assertEqual(result.attempts, 1)

    def test_retry_delay_is_exponential_with_jitter(self):
        agent = _mk_agent(RetryableFlakyTool(), max_retries=3)
        random.seed(7)
        computed = [agent._retry_delay(a) for a in (1, 2, 3, 4)]
        # base 0.5, factor 2, full jitter in [0, base*2^(n-1)].
        ratios = [c / (0.5 * (2 ** (a - 1))) for a, c in zip((1, 2, 3, 4), computed)]
        for r in ratios:
            self.assertGreaterEqual(r, 0.0)
            self.assertLess(r, 1.0)


if __name__ == "__main__":
    unittest.main()