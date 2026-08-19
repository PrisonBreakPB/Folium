"""Configuration - env vars and defaults."""

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MAX_CONTEXT_TOKENS = 1_000_000


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
    circuit_failure_threshold: int = 3
    circuit_cooldown_seconds: float = 10.0
    model_fast: str = "gpt-4o-mini"
    model_flagship: str = "gpt-4o"
    token_estimator: str = "deepseek"
    memory_maintenance_turns: int = 10
    memory_maintenance_max_steps: int = 5
    memory_maintenance_max_tokens: int = 2000

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
            circuit_failure_threshold=int(
                os.getenv("FOLIUM_CIRCUIT_FAILURE_THRESHOLD", "3")
            ),
            circuit_cooldown_seconds=float(
                os.getenv("FOLIUM_CIRCUIT_COOLDOWN_SECONDS", "10")
            ),
            model_fast=os.getenv("FOLIUM_MODEL_FAST", "gpt-4o-mini"),
            model_flagship=os.getenv("FOLIUM_MODEL_FLAGSHIP", "gpt-4o"),
            token_estimator=os.getenv("FOLIUM_TOKEN_ESTIMATOR", "deepseek"),
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
