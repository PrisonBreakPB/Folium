"""Configuration - env vars and defaults."""

import os
import re
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MAX_CONTEXT_TOKENS = 1_000_000


@dataclass
class LLMProfile:
    """A named, self-contained LLM endpoint configuration."""

    name: str
    provider: str
    api_key: str
    base_url: str | None
    model: str


def _load_dotenv():
    """Load .env from cwd, walking up to home dir. No-op if python-dotenv missing."""
    try:
        from dotenv import load_dotenv
        # search cwd first, then parent dirs up to ~
        env_path = Path(".env")
        if not env_path.exists():
            cur = Path.cwd()
            home = Path.home()
            while cur != home and cur != cur.parent:
                candidate = cur / ".env"
                if candidate.exists():
                    env_path = candidate
                    break
                cur = cur.parent
        load_dotenv(env_path, override=False)
    except ImportError:
        pass  # python-dotenv not installed, silently skip


@dataclass
class Config:
    model: str = "gpt-4o"
    api_key: str = ""
    base_url: str | None = None
    max_tokens: int = 32000
    temperature: float = 0.0
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
    provider: str = "openai"
    api_format: str = "chat_completions"
    llm_timeout: float = 30.0
    max_tool_retries: int = 10
    circuit_failure_threshold: int = 3
    circuit_cooldown_seconds: float = 10.0
    budget_usd: float = 0.0
    budget_soft_ratio: float = 0.8
    model_fast: str = "gpt-4o-mini"
    model_flagship: str = "gpt-4o"
    token_estimator: str = "approx"
    memory_maintenance_turns: int = 10
    memory_maintenance_max_steps: int = 5
    memory_maintenance_max_tokens: int = 2000

    def endpoint_profiles(self) -> list[LLMProfile]:
        """Read the active profile and its ordered fallback profiles from env."""
        active = (os.getenv("FOLIUM_ACTIVE_PROFILE") or "").strip()
        if not active:
            return []

        fallbacks = [
            name.strip()
            for name in (os.getenv("FOLIUM_FALLBACK_PROFILES") or "").split(",")
            if name.strip()
        ]
        names = [active, *fallbacks]
        if len({name.upper() for name in names}) != len(names):
            raise ValueError("FOLIUM_ACTIVE_PROFILE and FOLIUM_FALLBACK_PROFILES must not repeat profiles")
        return [_profile_from_env(name) for name in names]

    @classmethod
    def from_env(cls) -> "Config":
        # load .env if present (won't override existing env vars)
        _load_dotenv()
        # pick up common env vars automatically
        api_key = (
            os.getenv("FOLIUM_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
            or ""
        )
        return cls(
            model=os.getenv("FOLIUM_MODEL", "gpt-4o"),
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("FOLIUM_BASE_URL"),
            max_tokens=int(os.getenv("FOLIUM_MAX_TOKENS", "32000")),
            temperature=float(os.getenv("FOLIUM_TEMPERATURE", "0")),
            max_context_tokens=int(os.getenv("FOLIUM_MAX_CONTEXT", str(DEFAULT_MAX_CONTEXT_TOKENS))),
            provider=os.getenv("FOLIUM_PROVIDER", "openai"),
            api_format=os.getenv("FOLIUM_API_FORMAT", "chat_completions"),
            llm_timeout=float(os.getenv("FOLIUM_LLM_TIMEOUT", "30")),
            max_tool_retries=int(os.getenv("FOLIUM_MAX_TOOL_RETRIES", "10")),
            circuit_failure_threshold=int(
                os.getenv("FOLIUM_CIRCUIT_FAILURE_THRESHOLD", "3")
            ),
            circuit_cooldown_seconds=float(
                os.getenv("FOLIUM_CIRCUIT_COOLDOWN_SECONDS", "10")
            ),
            budget_usd=float(os.getenv("FOLIUM_BUDGET_USD", "0")),
            budget_soft_ratio=float(os.getenv("FOLIUM_BUDGET_SOFT_RATIO", "0.8")),
            model_fast=os.getenv("FOLIUM_MODEL_FAST", "gpt-4o-mini"),
            model_flagship=os.getenv("FOLIUM_MODEL_FLAGSHIP", "gpt-4o"),
            token_estimator=os.getenv("FOLIUM_TOKEN_ESTIMATOR", "approx"),
            memory_maintenance_turns=int(
                os.getenv("FOLIUM_MEMORY_MAINTENANCE_TURNS", "10")
            ),
            memory_maintenance_max_steps=int(
                os.getenv("FOLIUM_MEMORY_MAINTENANCE_MAX_STEPS", "5")
            ),
            memory_maintenance_max_tokens=int(
                os.getenv("FOLIUM_MEMORY_MAINTENANCE_MAX_TOKENS", "2000")
            ),
        )


def _profile_from_env(name: str) -> LLMProfile:
    normalized = name.upper()
    if not re.fullmatch(r"[A-Z0-9_]+", normalized):
        raise ValueError(
            f"invalid profile name {name!r}; use letters, digits, and underscores only"
        )

    prefix = f"FOLIUM_PROFILE_{normalized}_"
    api_key = os.getenv(f"{prefix}API_KEY") or ""
    model = os.getenv(f"{prefix}MODEL") or ""
    missing = [
        field for field, value in (("API_KEY", api_key), ("MODEL", model)) if not value
    ]
    if missing:
        raise ValueError(f"profile {name!r} is missing {', '.join(prefix + field for field in missing)}")

    return LLMProfile(
        name=name,
        provider=(os.getenv(f"{prefix}PROVIDER", "openai") or "openai").lower(),
        api_key=api_key,
        base_url=os.getenv(f"{prefix}BASE_URL") or None,
        model=model,
    )
