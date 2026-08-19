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


def test_route_returns_ordered_candidates_for_known_scene():
    # context_summarize: primary tier-fast (gpt-4o-mini), fallback tier-balanced (gpt-4o)
    candidates, reason = route("context_summarize")
    assert candidates == ["gpt-4o-mini", "gpt-4o"]
    assert reason == "scene=context_summarize,tier=tier-fast,rule:fixed"


def test_route_puts_default_model_first_and_adds_fallback_for_agent():
    # agent_reasoning: primary tier-balanced (default), fallback tier-fast
    candidates, reason = route("agent_reasoning", default_model="custom-model")
    assert candidates == ["custom-model", "gpt-4o-mini"]
    assert "tier=tier-balanced" in reason


def test_unknown_scene_falls_back_to_default_route():
    candidates, reason = route("mystery_scene")
    assert candidates == ["gpt-4o", "gpt-4o-mini"]  # default: balanced primary, fast fallback
    assert reason == "scene=mystery_scene,tier=tier-balanced,rule:fixed"