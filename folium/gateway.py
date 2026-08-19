"""LLM gateway - control plane for model selection (rule-based routing).

Holds everything that decides *which* model to call: scene -> tier candidates
and tier -> concrete model id. Transport (retry/streaming/errors) stays in
``llm.py``; callers only report their ``scene`` and do not pick a model.
"""

from __future__ import annotations

from .config import Config

# scene -> (primary tier, ordered fallback tiers). Define capability *levels*,
# not concrete model ids, so swapping vendors only touches the registry.
SCENE_ROUTES: dict[str, tuple[str, list[str]]] = {
    "agent_reasoning": ("tier-balanced", ["tier-fast"]),
    "context_summarize": ("tier-fast", ["tier-balanced"]),
    "memory_maintain": ("tier-fast", ["tier-balanced"]),
}
DEFAULT_ROUTE: tuple[str, list[str]] = ("tier-balanced", ["tier-fast"])

# tier -> concrete model id. Populated once from Config on first use.
MODEL_REGISTRY: dict[str, str] = {}
_registry_ready = False


def _ensure_registry() -> None:
    """Build MODEL_REGISTRY from Config (env). Lazily, on first route()."""
    global _registry_ready
    if _registry_ready:
        return
    cfg = Config.from_env()
    MODEL_REGISTRY.update(
        {
            "tier-fast": cfg.model_fast,
            "tier-balanced": cfg.model,
            "tier-flagship": cfg.model_flagship,
        }
    )
    _registry_ready = True


def route(
    scene: str, default_model: str | None = None, cheap_only: bool = False
) -> tuple[list[str], str]:
    """Pick the ordered model candidates for ``scene``.

    Returns ``([model_id, ...], route_reason)``: the primary candidate first,
    followed by fallback candidates in priority order, all resolved to concrete
    model ids. ``default_model`` (the LLM instance's own configured model) is
    used wherever a candidate resolves to ``tier-balanced``, so runtime
    overrides (CLI/model/switched models) stay as the balanced tier. Falls back
    to the default route for unknown scenes.

    ``cheap_only`` collapses the chain to the cheapest tier (tier-fast); used to
    force low-cost models once a session crosses its soft budget. Pure lookup -
    no model call, no health logic.
    """
    _ensure_registry()
    if cheap_only:
        return [MODEL_REGISTRY["tier-fast"]], f"scene={scene},tier=tier-fast,budget:cheap_only"
    primary, fallbacks = SCENE_ROUTES.get(scene, DEFAULT_ROUTE)
    tiers = [primary] + fallbacks
    candidates: list[str] = []
    for tier in tiers:
        if tier == "tier-balanced" and default_model:
            candidates.append(default_model)
        else:
            candidates.append(MODEL_REGISTRY[tier])
    reason = f"scene={scene},tier={primary},rule:fixed"
    return candidates, reason