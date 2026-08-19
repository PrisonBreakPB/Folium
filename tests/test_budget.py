"""Tests for the Agent-level budget (CostMeter) soft/hard enforcement."""

import pytest

from folium import Agent
from folium.cost_meter import CostMeter
from folium.llm import LLMResponse


class CapturingLLM:
    model = "gpt-4o"

    def __init__(self):
        self.calls = 0
        self.cheap_only = None
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    @property
    def estimated_cost(self):
        return None

    def chat(self, messages, tools=None, on_token=None, **kwargs):
        self.calls += 1
        self.cheap_only = kwargs.get("cheap_only", False)
        return LLMResponse(content="done", prompt_tokens=1, completion_tokens=1)


def make_agent(budget_usd=10.0):
    llm = CapturingLLM()
    agent = Agent(llm=llm, max_rounds=3, budget_usd=budget_usd, budget_soft_ratio=0.8)
    assert isinstance(agent._cost_meter, CostMeter)
    return agent, llm


def test_agent_wires_meter_onto_llm():
    agent, llm = make_agent()
    assert agent._cost_meter.enabled is True
    assert llm.meter is agent._cost_meter


def test_agent_reset_creates_fresh_meter():
    agent, _ = make_agent()
    first = agent._cost_meter
    first.record(5.0)
    agent.reset()
    assert agent._cost_meter is not first
    assert agent._cost_meter.spent() == 0.0
    assert agent.llm.meter is agent._cost_meter


def test_soft_budget_forces_cheap_only():
    agent, llm = make_agent()
    agent._cost_meter.record(8.0)  # ratio = 0.80 -> soft reached
    result = agent.chat("hello")
    assert result == "done"
    assert llm.calls == 1
    assert llm.cheap_only is True


def test_hard_budget_stops_before_calling_llm():
    agent, llm = make_agent()
    agent._cost_meter.record(10.0)  # ratio = 1.0 -> exhausted
    result = agent.chat("hello")
    assert "预算已用完" in result
    assert llm.calls == 0