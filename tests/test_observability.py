import json
import unittest

from folium.agent import Agent
from folium.llm import LLMResponse, ToolCall
from folium.observability.context import Observer, _observer_var
from folium.observability.config import ObservabilityConfig
from folium.observability.redaction import compact_payload
from folium.observability.summary import list_traces, read_trace_summary
class FakeLLM:
    model = "fake-model"

    def __init__(self, tool_calls=None):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.calls = 0
        self.tool_calls = tool_calls or [
            ToolCall(id="call_1", name="glob", arguments={"pattern": "AGENTS.md", "path": "."})
        ]

    @property
    def estimated_cost(self):
        return None

    def chat(self, messages, tools=None, on_token=None):
        from folium.observability import span
        from folium.observability.context import active_observer, current_span_id, current_trace_id

        self.calls += 1
        with span("fake_llm", "llm", metadata={"model": self.model}):
            if self.calls == 1:
                active_observer().record({
                    "event": "llm_result",
                    "trace_id": current_trace_id(),
                    "span_id": current_span_id(),
                    "name": "fake_llm",
                    "type": "llm",
                    "metadata": {"prompt_tokens": 10, "completion_tokens": 2, "tool_call_count": 1},
                })
                return LLMResponse(
                    content="",
                    tool_calls=self.tool_calls,
                    prompt_tokens=10,
                    completion_tokens=2,
                )
            if on_token:
                on_token("done")
            active_observer().record({
                "event": "llm_result",
                "trace_id": current_trace_id(),
                "span_id": current_span_id(),
                "name": "fake_llm",
                "type": "llm",
                "metadata": {"prompt_tokens": 8, "completion_tokens": 1, "tool_call_count": 0},
            })
            return LLMResponse(content="done", prompt_tokens=8, completion_tokens=1)


class ObservabilityTests(unittest.TestCase):
    def test_compact_payload_redacts_secret(self):
        payload = compact_payload(
            "OPENAI_API_KEY=sk-secretsecretsecret",
            include_full=True,
            max_preview_chars=100,
            redact=True,
        )
        self.assertNotIn("sk-secret", payload["preview"])
        self.assertNotIn("sk-secret", payload["value"])

    def test_agent_records_trace(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            trace_dir = Path(tmp)
            observer = Observer(ObservabilityConfig(trace_dir=trace_dir))
            token = _observer_var.set(observer)
            try:
                agent = Agent(llm=FakeLLM(), max_rounds=3)
                agent.session_id = "session_test"

                self.assertEqual(agent.chat("hello"), "done")

                traces = list_traces(trace_dir)
                self.assertEqual(len(traces), 1)
                summary = read_trace_summary(traces[0]["trace_id"], trace_dir)
                self.assertEqual(summary["status"], "ok")
                self.assertEqual(summary["session_id"], "session_test")
                self.assertEqual(summary["llm_calls"], 2)
                self.assertEqual(summary["tool_calls"], 1)

                lines = [
                    json.loads(line)
                    for line in (trace_dir / f"{traces[0]['trace_id']}.jsonl").read_text(encoding="utf-8").splitlines()
                ]
                self.assertTrue(any(e.get("event") == "tool_result" for e in lines))
                self.assertTrue(any(e.get("event") == "llm_result" for e in lines))
            finally:
                _observer_var.reset(token)

    def test_parallel_tool_calls_stay_in_same_trace(self):
        import tempfile
        from pathlib import Path

        tool_calls = [
            ToolCall(id="call_1", name="glob", arguments={"pattern": "AGENTS.md", "path": "."}),
            ToolCall(id="call_2", name="grep", arguments={"pattern": "Folium", "path": "README.md"}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            trace_dir = Path(tmp)
            observer = Observer(ObservabilityConfig(trace_dir=trace_dir))
            token = _observer_var.set(observer)
            try:
                agent = Agent(llm=FakeLLM(tool_calls=tool_calls), max_rounds=3)
                self.assertEqual(agent.chat("hello"), "done")

                traces = list_traces(trace_dir)
                self.assertEqual(len(traces), 1)
                summary = read_trace_summary(traces[0]["trace_id"], trace_dir)
                self.assertEqual(summary["tool_calls"], 2)
            finally:
                _observer_var.reset(token)


if __name__ == "__main__":
    unittest.main()
