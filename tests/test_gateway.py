import pytest

from folium import gateway
from folium.gateway import route


@pytest.fixture(autouse=True)
def reset_registry(monkeypatch):
    """Pin env + isolate registry state so route() is deterministic."""
    monkeypatch.setenv("FOLIUM_MODEL", "gpt-4o")
    monkeypatch.setenv("FOLIUM_MODEL_FAST", "gpt-4o-mini")
    monkeypatch.setenv("FOLIUM_MODEL_FLAGSHIP", "gpt-4o")
    gateway._registry_ready = False
    gateway.MODEL_REGISTRY.clear()
    yield
    gateway._registry_ready = False
    gateway.MODEL_REGISTRY.clear()


def test_known_scene_returns_fast_tier_and_reason():
    model, reason = route("context_summarize")
    assert model == "gpt-4o-mini"  # tier-fast from env default
    assert reason == "scene=context_summarize,tier=tier-fast,rule:fixed"


def test_agent_reasoning_uses_default_model():
    # default_model overrides the balanced tier, matching the LLM's real model
    model, reason = route("agent_reasoning", default_model="custom-model")
    assert model == "custom-model"
    assert "tier=tier-balanced" in reason


def test_unknown_scene_falls_back_to_default_route():
    model, reason = route("mystery_scene")
    assert model == "gpt-4o"  # default route -> tier-balanced
    assert reason == "scene=mystery_scene,tier=tier-balanced,rule:fixed"