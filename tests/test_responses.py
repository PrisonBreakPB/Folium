from types import SimpleNamespace
from unittest import mock

import pytest

from folium.config import Config
from folium.llm import (
    LLM,
    LLMProviderError,
    ToolCall,
    _messages_to_responses_input,
    _parse_function_call_item,
    _tools_to_responses,
)


# --- translation helpers -------------------------------------------------

def test_messages_to_responses_input_translates_roles():
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "hello",
            "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "read_file", "arguments": '{"file_path":"a"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "name": "read_file", "content": "contents"},
    ]

    instructions, items = _messages_to_responses_input(messages)

    assert instructions == "You are helpful."
    assert items == [
        {"role": "user", "content": [{"type": "input_text", "text": "hi"}]},
        {"type": "message", "role": "assistant",
         "content": [{"type": "output_text", "text": "hello"}]},
        {"type": "function_call", "call_id": "c1", "name": "read_file",
         "arguments": '{"file_path":"a"}'},
        {"type": "function_call_output", "call_id": "c1", "output": "contents"},
    ]


def test_messages_to_responses_input_toolcall_only_assistant():
    messages = [
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "function": {"name": "bash", "arguments": "{}"}}]},
    ]

    instructions, items = _messages_to_responses_input(messages)

    assert instructions is None
    assert items == [
        {"type": "function_call", "call_id": "c1", "name": "bash", "arguments": "{}"},
    ]


def test_messages_to_responses_input_no_system():
    instructions, items = _messages_to_responses_input([{"role": "user", "content": "hi"}])
    assert instructions is None


def test_tools_to_responses_flattens():
    tools = [
        {"type": "function",
         "function": {"name": "read_file", "description": "d",
                      "parameters": {"type": "object"}}}
    ]
    assert _tools_to_responses(tools) == [
        {"type": "function", "name": "read_file", "description": "d",
         "parameters": {"type": "object"}},
    ]


def test_parse_function_call_item():
    item = SimpleNamespace(call_id="c1", name="bash", arguments='{"cmd": "ls"}')
    assert _parse_function_call_item(item) == ToolCall(
        id="c1", name="bash", arguments='{"cmd": "ls"}'
    )


def test_parse_function_call_item_empty_args():
    item = SimpleNamespace(call_id="c2", name="bash", arguments="")
    assert _parse_function_call_item(item).arguments == ""


def test_parse_function_call_item_preserves_raw_arguments():
    item = SimpleNamespace(call_id="c3", name="bash", arguments='{"cmd": ')
    assert _parse_function_call_item(item).arguments == '{"cmd": '


# --- responses streaming path --------------------------------------------

def _make_llm(**attrs):
    llm = LLM.__new__(LLM)
    llm.model = "test-model"
    llm.extra = {}
    llm.api_format = "responses"
    llm.client = SimpleNamespace(responses=SimpleNamespace(create=object()))
    llm.total_prompt_tokens = 0
    llm.total_completion_tokens = 0
    llm.total_cached_tokens = 0
    for k, v in attrs.items():
        setattr(llm, k, v)
    return llm


def _fake_observer():
    obs = mock.Mock()
    obs.config.full_llm_output = False
    obs.config.max_preview_chars = 1200
    obs.config.redact_secrets = True
    return obs


def test_chat_observed_responses_parses_events_and_params():
    llm = _make_llm(extra={"max_tokens": 1000, "temperature": 0.5})

    events = [
        SimpleNamespace(type="response.output_text.delta", delta="Hello"),
        SimpleNamespace(type="response.output_text.delta", delta=" world"),
        SimpleNamespace(type="response.output_item.done",
                        item=SimpleNamespace(type="function_call", call_id="c1",
                                             name="read_file",
                                             arguments='{"file_path":"a"}')),
        SimpleNamespace(type="response.completed",
                        response=SimpleNamespace(usage=SimpleNamespace(
                            input_tokens=100, output_tokens=50,
                            input_tokens_details=SimpleNamespace(cached_tokens=30)))),
    ]
    captured = {}

    def fake_call(params, **kwargs):
        captured.update(params)
        return iter(events)

    tokens = []
    with (
        mock.patch.object(llm, "_call_with_retry", side_effect=fake_call),
        mock.patch("folium.llm.active_observer", return_value=_fake_observer()),
    ):
        resp = llm._chat_observed_responses(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"type": "function",
                    "function": {"name": "read_file", "description": "d",
                                 "parameters": {"type": "object"}}}],
            on_token=tokens.append,
        )

    assert resp.content == "Hello world"
    assert tokens == ["Hello", " world"]
    assert resp.tool_calls == [ToolCall(id="c1", name="read_file",
                                        arguments='{"file_path":"a"}')]
    assert resp.prompt_tokens == 100
    assert resp.completion_tokens == 50
    assert resp.cached_tokens == 30
    assert llm.total_prompt_tokens == 100

    # request params are translated, not passed through verbatim
    assert captured["model"] == "test-model"
    assert captured["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "hi"}]}
    ]
    assert captured["tools"] == [
        {"type": "function", "name": "read_file", "description": "d",
         "parameters": {"type": "object"}}
    ]
    assert captured["max_output_tokens"] == 1000
    assert captured["temperature"] == 0.5
    assert "max_tokens" not in captured
    assert "instructions" not in captured


def test_chat_observed_responses_records_raw_tool_args():
    llm = _make_llm()
    events = [
        SimpleNamespace(type="response.output_item.done",
                        item=SimpleNamespace(type="function_call", call_id="c1",
                                             name="bash", arguments='{"cmd": ')),
    ]

    observer = _fake_observer()
    observer.config.full_tool_args = False
    recorded = []
    observer.record.side_effect = recorded.append

    def fake_call(params, **kwargs):
        return iter(events)

    with (
        mock.patch.object(llm, "_call_with_retry", side_effect=fake_call),
        mock.patch("folium.llm.active_observer", return_value=observer),
    ):
        llm._chat_observed_responses(messages=[{"role": "user", "content": "hi"}])

    result = next(e for e in recorded if e["event"] == "llm_result")
    raw = result["metadata"]["tool_calls_raw_args"]
    assert list(raw) == ["0:bash"]
    assert raw["0:bash"]["preview"] == '{"cmd": '
    assert "value" not in raw["0:bash"]


def test_chat_observed_responses_failed_raises():
    llm = _make_llm()
    secret = "AIzaSensitiveKey123456"
    llm.api_key = secret
    events = [SimpleNamespace(type="response.failed",
                              response=SimpleNamespace(error={
                                  "message": f"bad key {secret}",
                                  "status_code": 401,
                              }))]

    def fake_call(params, **kwargs):
        return iter(events)

    with (
        mock.patch.object(llm, "_call_with_retry", side_effect=fake_call),
        mock.patch("folium.llm.active_observer", return_value=_fake_observer()),
    ):
        with pytest.raises(LLMProviderError) as raised:
            llm._chat_observed_responses(messages=[], tools=None)

    assert raised.value.info.status_code == 401
    assert secret not in str(raised.value)


# --- dispatch -------------------------------------------------------------

def test_chat_dispatches_by_api_format():
    llm = _make_llm(api_format="responses")

    with (
        mock.patch.object(llm, "_chat_observed_responses", return_value="RESP") as resp_mock,
        mock.patch.object(llm, "_chat_observed", return_value="CHAT") as chat_mock,
        mock.patch("folium.llm.active_observer", return_value=_fake_observer()),
    ):
        assert llm.chat(messages=[], trace_input=False) == "RESP"

    resp_mock.assert_called_once()
    chat_mock.assert_not_called()

    llm.api_format = "chat_completions"
    with (
        mock.patch.object(llm, "_chat_observed_responses", return_value="RESP") as resp_mock2,
        mock.patch.object(llm, "_chat_observed", return_value="CHAT") as chat_mock2,
        mock.patch("folium.llm.active_observer", return_value=_fake_observer()),
    ):
        assert llm.chat(messages=[], trace_input=False) == "CHAT"

    chat_mock2.assert_called_once()
    resp_mock2.assert_not_called()


# --- config ---------------------------------------------------------------

def test_config_api_format_default(monkeypatch):
    monkeypatch.delenv("FOLIUM_API_FORMAT", raising=False)
    with mock.patch("folium.config._load_dotenv"):
        assert Config.from_env().api_format == "chat_completions"


def test_config_api_format_from_env(monkeypatch):
    monkeypatch.setenv("FOLIUM_API_FORMAT", "responses")
    with mock.patch("folium.config._load_dotenv"):
        assert Config.from_env().api_format == "responses"
