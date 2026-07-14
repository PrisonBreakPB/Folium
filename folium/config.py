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
    token_estimator: str = "deepseek"

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
            token_estimator=os.getenv("FOLIUM_TOKEN_ESTIMATOR", "deepseek"),
        )
