from unittest import mock

import pytest

from folium.llm import (
    LLM,
    LLMErrorInfo,
    LLMProviderError,
    LLMResponse,
    _iter_stream,
    _looks_like_unsupported_stream_options,
    _should_fallback,
)
from folium.web.server import _error_event


def test_llm_provider_error_formats_structured_info():
    err = LLMProviderError(
        LLMErrorInfo(
            message="Invalid API key",
            provider="openai",
            status_code=401,
            error_type="invalid_request_error",
            error_code="invalid_api_key",
            request_id="req_123",
        )
    )

    assert "status=401" in str(err)
    assert "invalid_api_key" in str(err)
    assert err.info.to_dict()["request_id"] == "req_123"


def test_error_event_exposes_provider_error_fields():
    exc = LLMProviderError(
        LLMErrorInfo(
            message="Context length exceeded",
            provider="openai",
            status_code=400,
            error_type="invalid_request_error",
            error_code="context_length_exceeded",
        )
    )

    event = _error_event(exc)

    assert event["type"] == "error"
    assert event["content"] == "Context length exceeded"
    assert event["status_code"] == 400
    assert event["error_code"] == "context_length_exceeded"


def test_stream_options_fallback_detector_is_specific():
    unsupported = LLMErrorInfo(
        message="Unrecognized request argument supplied: stream_options",
        status_code=400,
        error_type="invalid_request_error",
    )
    auth_error = LLMErrorInfo(
        message="Invalid API key",
        status_code=401,
        error_type="authentication_error",
        error_code="invalid_api_key",
    )

    assert _looks_like_unsupported_stream_options(unsupported) is True
    assert _looks_like_unsupported_stream_options(auth_error) is False


def test_stream_options_fallback_only_for_unsupported_parameter():
    llm = LLM.__new__(LLM)
    llm.model = "test-model"
    llm.extra = {}
    llm.total_prompt_tokens = 0
    llm.total_completion_tokens = 0
    llm.total_cached_tokens = 0
    calls = []

    def fake_call(params, **kwargs):
        calls.append(dict(params))
        if len(calls) == 1:
            raise LLMProviderError(
                LLMErrorInfo(
                    message="Unrecognized request argument supplied: stream_options",
                    status_code=400,
                    error_type="invalid_request_error",
                )
            )
        return iter([])

    recorded = []
    fake_observer = mock.Mock()
    fake_observer.config.full_tool_args = False
    fake_observer.config.max_preview_chars = 1200
    fake_observer.config.redact_secrets = True
    fake_observer.config.full_llm_output = False
    fake_observer.record.side_effect = recorded.append

    with (
        mock.patch.object(llm, "_call_with_retry", side_effect=fake_call),
        mock.patch("folium.llm.active_observer", return_value=fake_observer),
    ):
        result = llm._chat_observed(messages=[])

    events = [event["event"] for event in recorded]

    assert result.content == ""
    assert "stream_options" in calls[0]
    assert "stream_options" not in calls[1]
    assert "llm_fallback" in events
    assert "llm_error" not in events


def test_stream_options_does_not_mask_auth_errors():
    llm = LLM.__new__(LLM)
    llm.model = "test-model"
    llm.extra = {}
    llm.total_prompt_tokens = 0
    llm.total_completion_tokens = 0
    llm.total_cached_tokens = 0
    auth_error = LLMProviderError(
        LLMErrorInfo(
            message="Invalid API key",
            status_code=401,
            error_type="authentication_error",
            error_code="invalid_api_key",
        )
    )

    with mock.patch.object(llm, "_call_with_retry", side_effect=auth_error):
        with pytest.raises(LLMProviderError) as raised:
            llm._chat_observed(messages=[])

    assert raised.value.info.status_code == 401


def test_retry_success_records_llm_retry_event():
    import httpx
    from types import SimpleNamespace
    from openai import APIConnectionError

    llm = LLM.__new__(LLM)
    llm.model = "test-model"
    llm.extra = {}
    calls = []

    def flaky_create(**params):
        calls.append(params)
        if len(calls) == 1:
            raise APIConnectionError(request=httpx.Request("POST", "http://localhost"))
        return object()

    llm.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=flaky_create))
    )

    recorded = []
    fake_observer = mock.Mock()
    fake_observer.record.side_effect = recorded.append

    with mock.patch("folium.llm.active_observer", return_value=fake_observer):
        llm._call_with_retry({"model": "test-model", "messages": []})

    retries = [e for e in recorded if e["event"] == "llm_retry"]
    assert len(retries) == 1
    assert retries[0]["name"] == "chat.completions"
    assert retries[0]["metadata"]["attempt"] == 2
    assert "llm_error" not in [e["event"] for e in recorded]


def test_stream_iteration_errors_are_wrapped():
    def broken_stream():
        raise RuntimeError("stream broke")
        yield

    with pytest.raises(LLMProviderError) as raised:
        list(_iter_stream(broken_stream(), provider="openai"))

    assert raised.value.info.message == "stream broke"
    assert raised.value.info.provider == "openai"


def make_info(status_code=None, error_type=None, retryable=False):
    return LLMErrorInfo(
        message="err", status_code=status_code, error_type=error_type, retryable=retryable
    )


def test_should_fallback_returns_true_for_transient_errors():
    assert _should_fallback(make_info(status_code=429, error_type="rate_limit_error")) is True
    assert _should_fallback(make_info(status_code=503)) is True
    assert _should_fallback(make_info(status_code=500)) is True
    # timeout / connection: no HTTP status, but flagged retryable
    assert _should_fallback(make_info(status_code=None, retryable=True)) is True


def test_should_fallback_returns_false_for_request_side_errors():
    assert _should_fallback(make_info(status_code=400)) is False
    assert _should_fallback(make_info(status_code=401)) is False


def test_should_fallback_returns_false_for_context_or_safety():
    assert (
        _should_fallback(make_info(status_code=400, error_type="context_length_exceeded")) is False
    )
    assert (
        _should_fallback(make_info(status_code=400, error_type="content_policy_violation")) is False
    )


class _StubTransportLLMAllFail(LLM):
    """Real LLM whose transport always fails (tests exhausted-candidates)."""

    def __init__(self, model):
        super().__init__(model=model, api_key="test-key")
        self.tried = []

    def _chat_observed(self, messages, tools=None, on_token=None):
        self.tried.append(self.model)
        raise LLMProviderError(LLMErrorInfo(message="down", status_code=503))


def test_chat_raises_last_error_when_all_candidates_fail(monkeypatch):
    monkeypatch.setenv("FOLIUM_MODEL", "gpt-4o")
    monkeypatch.setenv("FOLIUM_MODEL_FAST", "gpt-4o-mini")
    import folium.gateway as gateway

    gateway._registry_ready = False
    gateway.MODEL_REGISTRY.clear()
    try:
        observer = mock.MagicMock()
        observer.config = _FakeConfig()
        monkeypatch.setattr("folium.llm.active_observer", lambda: observer)
        llm = _StubTransportLLMAllFail(model="custom-model")

        with pytest.raises(LLMProviderError) as raised:
            llm.chat(messages=[{"role": "user", "content": "hi"}], scene="agent_reasoning")

        # primary and fallback both tried -> no swallow, last error propagates
        assert llm.tried == ["custom-model", "gpt-4o-mini"]
        assert raised.value.info.status_code == 503
    finally:
        gateway._registry_ready = False
        gateway.MODEL_REGISTRY.clear()


class _FakeConfig:
    full_llm_input = False
    max_preview_chars = 100
    redact_secrets = True


class _StubTransportLLM(LLM):
    """Real LLM whose transport (_chat_observed) fails once then succeeds."""

    def __init__(self, model):
        super().__init__(model=model, api_key="test-key")
        self.tried = []

    def _chat_observed(self, messages, tools=None, on_token=None):
        self.tried.append(self.model)
        if len(self.tried) == 1:
            raise LLMProviderError(
                LLMErrorInfo(message="rate limited", status_code=429, error_type="rate_limit_error")
            )
        return LLMResponse(content="fallback-ok", prompt_tokens=3, completion_tokens=1)


def test_chat_falls_back_to_next_candidate_on_retryable_error(monkeypatch):
    # agent_reasoning candidates: primary = default_model, fallback = tier-fast
    monkeypatch.setenv("FOLIUM_MODEL", "gpt-4o")
    monkeypatch.setenv("FOLIUM_MODEL_FAST", "gpt-4o-mini")
    import folium.gateway as gateway

    gateway._registry_ready = False
    gateway.MODEL_REGISTRY.clear()
    try:
        obs = _FakeConfig()
        observer = mock.MagicMock()
        observer.config = obs
        monkeypatch.setattr("folium.llm.active_observer", lambda: observer)
        llm = _StubTransportLLM(model="custom-model")

        result = llm.chat(messages=[{"role": "user", "content": "hi"}], scene="agent_reasoning")

        # tried primary first, then the fallback candidate
        assert llm.tried == ["custom-model", "gpt-4o-mini"]
        # result came from the fallback
        assert result.content == "fallback-ok"
        # model pointer advanced to the fallback
        assert llm.model == "gpt-4o-mini"
        # a fallback event was recorded into the observability chain
        fallback_events = [c.args[0] for c in observer.record.call_args_list if c.args[0]["event"] == "llm_fallback"]
        assert fallback_events, "expected an llm_fallback event"
        assert fallback_events[0]["metadata"]["fallback"] == "gpt-4o-mini"
    finally:
        gateway._registry_ready = False
        gateway.MODEL_REGISTRY.clear()
