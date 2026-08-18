from unittest import mock

import pytest

from folium.llm import (
    LLM,
    LLMErrorInfo,
    LLMProviderError,
    _iter_stream,
    _looks_like_unsupported_stream_options,
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
