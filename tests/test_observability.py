import json
import os
import unittest
from unittest import mock

from folium.agent import Agent
from folium.database import get_connection
from folium.llm import LLMResponse, ToolCall
from folium.observability.config import ObservabilityConfig
from folium.observability.context import Observer, _observer_var
from folium.observability.redaction import compact_payload
from folium.observability.summary import delete_traces_for_session, list_traces, read_trace_summary


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

    def chat(self, messages, tools=None, on_token=None, **kwargs):
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
            active_observer().record({
                "event": "llm_result",
                "trace_id": current_trace_id(),
                "span_id": current_span_id(),
                "name": "fake_llm",
                "type": "llm",
                "metadata": {"prompt_tokens": 8, "completion_tokens": 1, "tool_call_count": 0},
            })
            return LLMResponse(content="done", prompt_tokens=8, completion_tokens=1)


class NoToolLLM:
    model = "fake-model"
    total_prompt_tokens = 0
    total_completion_tokens = 0

    @property
    def estimated_cost(self):
        return None

    def chat(self, messages, tools=None, on_token=None, **kwargs):
        return LLMResponse(content="done", prompt_tokens=8, completion_tokens=1)


def _events(db_path, trace_id):
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT event_type, payload_json FROM trace_events WHERE trace_id = ? ORDER BY event_index",
            (trace_id,),
        ).fetchall()
    return [
        {"event": row["event_type"], "metadata": json.loads(row["payload_json"] or "{}")}
        for row in rows
    ]


class ObservabilityTests(unittest.TestCase):
    def test_trace_mode_is_not_a_supported_configuration(self):
        with mock.patch.dict(os.environ, {"FOLIUM_TRACE_MODE": "errors"}, clear=True):
            config = ObservabilityConfig.from_env()

        self.assertTrue(config.enabled)
        self.assertFalse(hasattr(config, "trace_mode"))

    def test_compact_payload_redacts_secret(self):
        payload = compact_payload(
            "OPENAI_API_KEY=sk-secretsecretsecret",
            include_full=True,
            max_preview_chars=100,
            redact=True,
        )
        self.assertNotIn("sk-secret", payload["preview"])
        self.assertNotIn("sk-secret", payload["value"])

    def test_agent_records_trace_in_sqlite(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "folium.db"
            observer = Observer(ObservabilityConfig(database_path=db_path))
            token = _observer_var.set(observer)
            try:
                agent = Agent(llm=FakeLLM(), max_rounds=3)
                agent.session_id = "session_test"
                self.assertEqual(agent.chat("hello"), "done")

                traces = list_traces(db_path)
                self.assertEqual(len(traces), 1)
                summary = read_trace_summary(traces[0]["trace_id"], db_path)
                self.assertEqual(summary["status"], "ok")
                self.assertEqual(summary["session_id"], "session_test")
                self.assertEqual(summary["llm_calls"], 2)
                self.assertEqual(summary["tool_calls"], 1)

                events = _events(db_path, traces[0]["trace_id"])
                self.assertTrue(any(event["event"] == "tool_result" for event in events))
                self.assertTrue(any(event["event"] == "llm_result" for event in events))
                request = next(event for event in events if event["event"] == "llm_request_snapshot")
                self.assertIn("preview", request["metadata"]["messages"])
                self.assertNotIn("value", request["metadata"]["messages"])
            finally:
                _observer_var.reset(token)

    def test_trace_root_records_system_prompt_hash(self):
        import tempfile
        from pathlib import Path

        from folium.observability.redaction import stable_hash

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "folium.db"
            observer = Observer(ObservabilityConfig(database_path=db_path))
            token = _observer_var.set(observer)
            try:
                agent = Agent(llm=NoToolLLM(), max_rounds=1)
                self.assertEqual(agent.chat("hello"), "done")
                trace_id = list_traces(db_path)[0]["trace_id"]
                root = next(
                    event for event in _events(db_path, trace_id)
                    if event["event"] == "span_start" and "system_prompt_hash" in event["metadata"]
                )
                self.assertEqual(root["metadata"]["system_prompt_hash"], stable_hash(agent._system))
            finally:
                _observer_var.reset(token)

    def test_full_llm_snapshot_can_include_values(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "folium.db"
            observer = Observer(ObservabilityConfig(
                database_path=db_path,
                full_llm_input=True,
                full_llm_output=True,
            ))
            token = _observer_var.set(observer)
            try:
                agent = Agent(llm=NoToolLLM(), max_rounds=1)
                self.assertEqual(agent.chat("hello"), "done")
                trace_id = list_traces(db_path)[0]["trace_id"]
                events = _events(db_path, trace_id)
                request = next(event for event in events if event["event"] == "llm_request_snapshot")
                response = next(event for event in events if event["event"] == "llm_response_snapshot")
                self.assertIn("value", request["metadata"]["messages"])
                self.assertIn("hello", request["metadata"]["messages"]["value"])
                self.assertIn("value", response["metadata"]["assistant_message"])
                self.assertIn("done", response["metadata"]["assistant_message"]["value"])
            finally:
                _observer_var.reset(token)

    def test_context_compression_trace_includes_layers(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "folium.db"
            observer = Observer(ObservabilityConfig(database_path=db_path))
            token = _observer_var.set(observer)
            try:
                agent = Agent(llm=NoToolLLM(), max_rounds=1)
                original = agent.context.maybe_compress

                def fake_compress(messages, llm=None, real_tokens=None):
                    return {"compressed": True, "layers": [{"name": "trim", "changed": True}]}

                agent.context.maybe_compress = fake_compress
                try:
                    self.assertEqual(agent.chat("hello"), "done")
                finally:
                    agent.context.maybe_compress = original

                events = _events(db_path, list_traces(db_path)[0]["trace_id"])
                compressed = next(event for event in events if event["event"] == "context_compressed")
                self.assertEqual(compressed["metadata"]["layers"][0]["name"], "trim")
                self.assertTrue(any(event["event"] == "context_snapshot" for event in events))
            finally:
                _observer_var.reset(token)

    def test_delete_traces_for_session(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "folium.db"
            with get_connection(db_path) as conn:
                conn.execute(
                    "INSERT INTO sessions VALUES (?, ?, ?, ?, ?)",
                    ("session_delete", "model", "", "now", "now"),
                )
                conn.execute(
                    "INSERT INTO traces (trace_id, session_id, status) VALUES (?, ?, ?)",
                    ("trace_delete", "session_delete", "ok"),
                )

            self.assertEqual(delete_traces_for_session("session_delete", db_path), 1)
            self.assertEqual(list_traces(db_path), [])


if __name__ == "__main__":
    unittest.main()
