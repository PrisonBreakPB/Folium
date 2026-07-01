"""Observability configuration from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


@dataclass
class ObservabilityConfig:
    enabled: bool = True
    trace_mode: str = "all"
    trace_dir: Path = Path("conversations") / "traces"
    full_user_input: bool = True
    full_llm_input: bool = False
    full_llm_output: bool = False
    full_context_snapshots: bool = False
    full_tool_args: bool = True
    full_tool_output: bool = False
    redact_secrets: bool = True
    max_preview_chars: int = 1000

    @classmethod
    def from_env(cls) -> "ObservabilityConfig":
        return cls(
            enabled=_env_bool("FOLIUM_OBSERVABILITY", True),
            trace_mode=os.getenv("FOLIUM_TRACE_MODE", "all"),
            trace_dir=Path(os.getenv("FOLIUM_TRACE_DIR", "conversations/traces")),
            full_user_input=_env_bool("FOLIUM_TRACE_FULL_USER_INPUT", True),
            full_llm_input=_env_bool("FOLIUM_TRACE_FULL_LLM_INPUT", False),
            full_llm_output=_env_bool("FOLIUM_TRACE_FULL_LLM_OUTPUT", False),
            full_context_snapshots=_env_bool("FOLIUM_TRACE_FULL_CONTEXT_SNAPSHOTS", False),
            full_tool_args=_env_bool("FOLIUM_TRACE_FULL_TOOL_ARGS", True),
            full_tool_output=_env_bool("FOLIUM_TRACE_FULL_TOOL_OUTPUT", False),
            redact_secrets=_env_bool("FOLIUM_TRACE_REDACT_SECRETS", True),
            max_preview_chars=int(os.getenv("FOLIUM_TRACE_MAX_PREVIEW_CHARS", "1000")),
        )
