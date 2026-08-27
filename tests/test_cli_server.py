import json
from io import StringIO
from types import SimpleNamespace

from folium.cli_server import JsonlServer


class FakeMaintenance:
    def __init__(self):
        self.calls = []

    def submit(self, **kwargs):
        self.calls.append(kwargs)


class FakeAgent:
    def __init__(self):
        self.llm = SimpleNamespace(
            model="test-model",
            last_prompt_tokens=12,
            last_completion_tokens=4,
            last_cached_tokens=3,
            total_prompt_tokens=12,
            total_completion_tokens=4,
            total_cached_tokens=3,
            estimated_cost=0.01,
        )
        self.mode = "build"
        self.messages = []
        self.transcript = []
        self.context = SimpleNamespace(
            max_tokens=1000,
            input_budget_tokens=900,
            reserved_output_tokens=100,
        )
        self.skills = []
        self.tools = []
        self.todo_manager = None
        self._system = "system"
        self.last_llm_request_had_visible_tools = False
        self.edit_approval_callback = None

    def chat(self, content, on_token=None, on_event=None):
        self.messages.append({"role": "user", "content": content})
        if on_event:
            on_event({"type": "agent_status", "message": "Processing request"})
        if on_token:
            on_token("answer")
        self.messages.append({"role": "assistant", "content": "answer"})
        return "answer"

    def _full_messages(self):
        return self.messages

    def _tool_schemas(self):
        return []

    def reset(self):
        self.messages = []

    def reset_todos(self):
        pass

    def set_mode(self, mode):
        if mode not in {"build", "plan"}:
            raise ValueError("invalid mode")
        self.mode = mode


def _server(input_text="", agent=None):
    output = StringIO()
    server = JsonlServer(
        agent or FakeAgent(),
        SimpleNamespace(model="test-model", max_context_tokens=1000, max_tokens=200),
        "D:/project",
        input_stream=StringIO(input_text),
        output_stream=output,
        maintenance=FakeMaintenance(),
    )
    return server, output


def _events(output):
    return [json.loads(line) for line in output.getvalue().splitlines()]


def test_jsonl_server_emits_ready_and_streamed_agent_events(monkeypatch):
    monkeypatch.setattr("folium.cli._ensure_session", lambda *args: "session-1")
    monkeypatch.setattr("folium.cli._persist_session", lambda *args: "session-1")
    server, output = _server('{"type":"message","request_id":"1","content":"hello"}\n')

    server.run()

    events = _events(output)
    assert events[0]["type"] == "ready"
    assert events[1]["type"] == "agent_event"
    assert any(event["type"] == "token" and event["content"] == "answer" for event in events)
    assert events[-1]["type"] == "done"
    assert events[-1]["session_id"] == "session-1"


def test_jsonl_server_routes_context_and_model_commands(monkeypatch):
    monkeypatch.setattr("folium.cli_server.estimate_tokens", lambda messages: 250)
    server, output = _server(
        '{"type":"command","request_id":"1","command":"/context"}\n'
        '{"type":"command","request_id":"2","command":"/model new-model"}\n'
    )

    server.run()

    results = [event for event in _events(output) if event["type"] == "command_result"]
    assert results[0]["kind"] == "context"
    assert results[0]["data"]["estimated_context_tokens"] == 250
    assert results[1]["text"] == "Switched to new-model"
    assert results[1]["session_id"] is None
    assert results[1]["model"] == "new-model"
    assert results[1]["mode"] == "build"
    assert server.agent.llm.model == "new-model"
