"""Cost meter - per-session rolling budget for LLM spend.

Owned by the agent that drives a task/session; every LLM call (main loop,
context compression, background memory maintenance) reports its cost here so
the agent can watch how much it has spent. Budgets are in dollars. A
``budget_usd`` of 0 disables the meter (spend is tracked but never enforced).
"""

from __future__ import annotations


class CostMeter:
    def __init__(self, budget_usd: float = 0.0, soft_ratio: float = 0.8):
        self.budget_usd = budget_usd
        self.soft_ratio = soft_ratio
        self.spent_usd = 0.0

    @property
    def enabled(self) -> bool:
        """Whether a spend cap applies (a non-positive budget = unlimited)."""
        return self.budget_usd > 0

    def record(self, cost: float | None) -> None:
        if cost is not None and cost > 0:
            self.spent_usd += cost

    def spent(self) -> float:
        return self.spent_usd

    def ratio(self) -> float:
        """Fraction of budget spent; 0.0 when disabled."""
        if not self.enabled:
            return 0.0
        return self.spent_usd / self.budget_usd

    def soft_reached(self) -> bool:
        return self.enabled and self.ratio() >= self.soft_ratio

    def exhausted(self) -> bool:
        return self.enabled and self.ratio() >= 1.0