"""Circuit breaker - fail-fast protection for a slow/failing LLM provider.

State is process-global and keyed by ``(provider, model)`` so each candidate
model accumulates its own failure history independently. Transitions follow the
standard closed -> open -> half-open cycle:

- **CLOSED**: calls allowed. Each fallbackable failure increments a counter;
  at ``failure_threshold`` the circuit trips open.
- **OPEN**: calls rejected up front. Stays open for ``cooldown_seconds``.
- **HALF_OPEN**: after the cooldown a single trial call is allowed; success
  resets to closed, another failure re-opens immediately.

The clock is injectable so tests can advance time without sleeping.
"""

from __future__ import annotations

import time

from .config import Config

Key = tuple[str, str]  # (provider, model)

_breaker: "CircuitBreaker | None" = None


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown_seconds: float = 10.0,
        monotonic=None,
    ):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._monotonic = monotonic or time.monotonic
        self._states: dict[Key, dict] = {}

    def _state(self, provider: str, model: str) -> dict:
        return self._states.setdefault((provider, model), {"failures": 0, "state": "closed", "opened_at": None})

    def allow(self, provider: str, model: str) -> bool:
        """True if a call to (provider, model) may proceed right now."""
        st = self._state(provider, model)
        if st["state"] != "open":
            return True
        if self._monotonic() - st["opened_at"] >= self.cooldown_seconds:
            st["state"] = "half_open"
            return True
        return False

    def record_success(self, provider: str, model: str) -> None:
        st = self._state(provider, model)
        st["failures"] = 0
        st["state"] = "closed"
        st["opened_at"] = None

    def record_failure(self, provider: str, model: str) -> bool:
        """Count a failure. Returns True if it just tripped the circuit open."""
        st = self._state(provider, model)
        # a failed half-open trial re-opens immediately
        if st["state"] == "half_open":
            st["state"] = "open"
            st["opened_at"] = self._monotonic()
            return True
        st["failures"] += 1
        if st["failures"] >= self.failure_threshold:
            st["state"] = "open"
            st["opened_at"] = self._monotonic()
            return True
        return False

    def is_open(self, provider: str, model: str) -> bool:
        """Currently rejecting calls (used for observability)."""
        st = self._state(provider, model)
        return st["state"] == "open"

    def reset(self) -> None:
        """Clear all state (mainly for tests / manual recovery)."""
        self._states.clear()


def get_circuit_breaker() -> CircuitBreaker:
    """Return the process-global breaker, built once from Config (env)."""
    global _breaker
    if _breaker is None:
        cfg = Config.from_env()
        _breaker = CircuitBreaker(
            failure_threshold=cfg.circuit_failure_threshold,
            cooldown_seconds=cfg.circuit_cooldown_seconds,
        )
    return _breaker


def reset_circuit_breaker() -> None:
    """Drop the cached singleton (for tests)."""
    global _breaker
    _breaker = None