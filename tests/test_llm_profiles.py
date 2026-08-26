from types import SimpleNamespace
from unittest import mock

import pytest

from folium.config import Config, LLMProfile
from folium.llm import FailoverLLM, LLMErrorInfo, LLMProviderError, LLMResponse
from folium.memory_maintenance import build_memory_maintenance_runner


def _set_profile(monkeypatch, name, *, provider="openai", api_key="key", base_url=None, model="model"):
    prefix = f"FOLIUM_PROFILE_{name.upper()}_"
    monkeypatch.setenv(f"{prefix}PROVIDER", provider)
    monkeypatch.setenv(f"{prefix}API_KEY", api_key)
    monkeypatch.setenv(f"{prefix}MODEL", model)
    if base_url:
        monkeypatch.setenv(f"{prefix}BASE_URL", base_url)


def test_config_reads_active_profile_and_ordered_fallbacks(monkeypatch):
    monkeypatch.setenv("FOLIUM_ACTIVE_PROFILE", "deepseek")
    monkeypatch.setenv("FOLIUM_FALLBACK_PROFILES", "openai, backup")
    _set_profile(
        monkeypatch,
        "deepseek",
        api_key="deepseek-key",
        base_url="https://api.deepseek.com",
        model="deepseek-v4-pro",
    )
    _set_profile(monkeypatch, "openai", api_key="openai-key", model="gpt-4o-mini")
    _set_profile(monkeypatch, "backup", provider="litellm", api_key="backup-key", model="x")

    profiles = Config.from_env().endpoint_profiles()

    assert [profile.name for profile in profiles] == ["deepseek", "openai", "backup"]
    assert profiles[0].base_url == "https://api.deepseek.com"
    assert profiles[1].base_url is None
    assert profiles[2].provider == "litellm"


def test_profile_requires_key_and_model(monkeypatch):
    monkeypatch.setenv("FOLIUM_ACTIVE_PROFILE", "primary")
    monkeypatch.delenv("FOLIUM_PROFILE_PRIMARY_API_KEY", raising=False)
    monkeypatch.delenv("FOLIUM_PROFILE_PRIMARY_MODEL", raising=False)

    with pytest.raises(ValueError, match="FOLIUM_PROFILE_PRIMARY_API_KEY"):
        Config.from_env().endpoint_profiles()


class _StubTransport:
    def __init__(self, model, outcomes):
        self.model = model
        self.outcomes = list(outcomes)
        self.calls = []
        self.meter = None

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _profiles():
    return [
        LLMProfile("primary", "openai", "primary-key", "https://primary.example/v1", "primary-model"),
        LLMProfile("backup", "openai", "backup-key", "https://backup.example/v1", "backup-model"),
    ]


def test_failover_uses_backup_profile_after_auth_error(monkeypatch):
    auth_error = LLMProviderError(LLMErrorInfo(message="bad key", status_code=401))
    primary = _StubTransport("primary-model", [auth_error])
    backup = _StubTransport("backup-model", [LLMResponse(content="ok", prompt_tokens=3, completion_tokens=2)])
    transports = iter([primary, backup])
    monkeypatch.setattr(FailoverLLM, "_create_transport", lambda self, profile: next(transports))
    observer = mock.MagicMock()

    with mock.patch("folium.llm.active_observer", return_value=observer):
        llm = FailoverLLM(_profiles())
        result = llm.chat(messages=[{"role": "user", "content": "hi"}])

    assert result.content == "ok"
    assert llm.model == "backup-model"
    assert primary.calls[0]["route_models"] is True
    assert backup.calls[0]["route_models"] is False
    event = observer.record.call_args.args[0]
    assert event["event"] == "llm_endpoint_fallback"
    assert event["metadata"]["source_profile"] == "primary"
    assert event["metadata"]["fallback_profile"] == "backup"
    assert "primary-key" not in str(event)
    assert "backup-key" not in str(event)


def test_failover_does_not_switch_profiles_for_other_4xx(monkeypatch):
    request_error = LLMProviderError(LLMErrorInfo(message="bad request", status_code=400))
    primary = _StubTransport("primary-model", [request_error])
    backup = _StubTransport("backup-model", [LLMResponse(content="unexpected")])
    transports = iter([primary, backup])
    monkeypatch.setattr(FailoverLLM, "_create_transport", lambda self, profile: next(transports))

    with pytest.raises(LLMProviderError, match="bad request"):
        FailoverLLM(_profiles()).chat(messages=[])

    assert not backup.calls


def test_active_backup_keeps_its_own_model_fallback_route(monkeypatch):
    auth_error = LLMProviderError(LLMErrorInfo(message="bad key", status_code=401))
    server_error = LLMProviderError(LLMErrorInfo(message="down", status_code=503))
    primary = _StubTransport("primary-model", [auth_error])
    backup = _StubTransport(
        "backup-model",
        [LLMResponse(content="ok"), server_error],
    )
    transports = iter([primary, backup])
    monkeypatch.setattr(FailoverLLM, "_create_transport", lambda self, profile: next(transports))

    llm = FailoverLLM(_profiles())
    llm.chat(messages=[])
    with pytest.raises(LLMProviderError, match="down"):
        llm.chat(messages=[])

    assert backup.calls[0]["route_models"] is False
    assert backup.calls[1]["route_models"] is True


def test_failover_redacts_provider_error_secrets(monkeypatch):
    secret = "AIzaSensitiveKey123456"
    auth_error = LLMProviderError(
        LLMErrorInfo(
            message=f"incorrect API key {secret}",
            status_code=401,
            raw_body=f'{{"api_key":"{secret}"}}',
        )
    )
    primary = _StubTransport("primary-model", [auth_error])
    backup = _StubTransport("backup-model", [LLMResponse(content="ok")])
    transports = iter([primary, backup])
    monkeypatch.setattr(FailoverLLM, "_create_transport", lambda self, profile: next(transports))
    observer = mock.MagicMock()
    profiles = _profiles()
    profiles[0].api_key = secret

    with mock.patch("folium.llm.active_observer", return_value=observer):
        FailoverLLM(profiles).chat(messages=[])

    event = observer.record.call_args.args[0]
    assert secret not in str(event)
    assert secret not in str(auth_error)


def test_maintenance_runner_keeps_active_profile_chain(monkeypatch):
    primary = _StubTransport("primary-model", [])
    backup = _StubTransport("backup-model", [])
    backup_clone = _StubTransport("backup-model", [])
    transports = iter([primary, backup, backup_clone])
    monkeypatch.setattr(FailoverLLM, "_create_transport", lambda self, profile: next(transports))
    llm = FailoverLLM(_profiles())
    llm._active_index = 1

    runner = build_memory_maintenance_runner(
        SimpleNamespace(llm=llm, _cost_meter=object()),
        SimpleNamespace(memory_maintenance_max_tokens=99, memory_maintenance_max_steps=2),
    )

    assert isinstance(runner.llm, FailoverLLM)
    assert runner.llm.model == "backup-model"
    assert runner.llm._transport_kwargs["max_tokens"] == 99
