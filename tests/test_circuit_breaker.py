"""Tests for the CircuitBreaker class (process-global, per-(provider,model))."""

import pytest

from folium.circuit_breaker import CircuitBreaker, get_circuit_breaker, reset_circuit_breaker


@pytest.fixture(autouse=True)
def isolate_singleton():
    reset_circuit_breaker()
    yield
    reset_circuit_breaker()


class FakeClock:
    def __init__(self, now=0.0):
        self.now = now

    def advance(self, secs):
        self.now += secs

    def __call__(self):
        return self.now


def make_breaker(threshold=3, cooldown=10.0):
    clock = FakeClock()
    breaker = CircuitBreaker(
        failure_threshold=threshold,
        cooldown_seconds=cooldown,
        monotonic=clock,
    )
    return breaker, clock


def trip(breaker, provider="openai", model="gpt-4o", n=3):
    for _ in range(n):
        breaker.record_failure(provider, model)


def test_breaker_allows_by_default():
    breaker, _ = make_breaker()
    assert breaker.allow("openai", "gpt-4o") is True


def test_breaker_trips_after_threshold_failures():
    breaker, _ = make_breaker()
    assert breaker.record_failure("openai", "gpt-4o") is False
    assert breaker.record_failure("openai", "gpt-4o") is False
    assert breaker.record_failure("openai", "gpt-4o") is True  # third trips
    assert breaker.allow("openai", "gpt-4o") is False


def test_breaker_opens_during_cooldown_then_allows_trial():
    breaker, clock = make_breaker()
    trip(breaker)
    assert breaker.allow("openai", "gpt-4o") is False
    clock.advance(9)
    assert breaker.allow("openai", "gpt-4o") is False
    clock.advance(1)  # exactly cooldown elapsed
    assert breaker.allow("openai", "gpt-4o") is True  # trial allowed


def test_breaker_success_resets_and_closes():
    breaker, _ = make_breaker()
    trip(breaker)
    assert breaker.allow("openai", "gpt-4o") is False
    breaker.record_success("openai", "gpt-4o")
    assert breaker.allow("openai", "gpt-4o") is True


def test_breaker_states_are_per_key():
    breaker, _ = make_breaker()
    trip(breaker)
    assert breaker.allow("openai", "gpt-4o") is False
    assert breaker.allow("openai", "gpt-4o-mini") is True  # different model unaffected
    assert breaker.allow("litellm", "gpt-4o") is True  # different provider unaffected


def test_breaker_half_open_failure_reopens():
    breaker, clock = make_breaker()
    trip(breaker)
    clock.advance(11)
    assert breaker.allow("openai", "gpt-4o") is True  # trial passes
    breaker.record_failure("openai", "gpt-4o")  # trial fails
    assert breaker.allow("openai", "gpt-4o") is False  # reopened


def test_breaker_uses_configured_threshold():
    breaker, _ = make_breaker(threshold=1)
    assert breaker.record_failure("openai", "gpt-4o") is True  # one failure trips


def test_get_circuit_breaker_honors_env_thresholds(monkeypatch):
    monkeypatch.setenv("FOLIUM_CIRCUIT_FAILURE_THRESHOLD", "2")
    monkeypatch.setenv("FOLIUM_CIRCUIT_COOLDOWN_SECONDS", "5")
    monkeypatch.setattr("folium.config._load_dotenv", lambda: None)
    breaker = get_circuit_breaker()
    assert breaker.failure_threshold == 2
    assert breaker.cooldown_seconds == 5
    # singleton is cached across calls
    assert get_circuit_breaker() is breaker