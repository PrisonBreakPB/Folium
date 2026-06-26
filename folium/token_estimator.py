"""Configurable token estimation for content not yet covered by API usage."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any


DEFAULT_TOKEN_ESTIMATOR = "deepseek"


def estimator_name() -> str:
    return os.getenv("FOLIUM_TOKEN_ESTIMATOR", DEFAULT_TOKEN_ESTIMATOR).strip().lower()


def estimate_text_tokens(text: str, model: str | None = None) -> int:
    name = estimator_name()
    if name == "deepseek":
        tokenizer = _deepseek_tokenizer()
        if tokenizer is not None:
            try:
                return len(tokenizer.encode(text))
            except Exception:
                pass
    return _approx_text_tokens(text)


def estimate_message_tokens(messages: list[dict], model: str | None = None) -> int:
    total = 0
    for message in messages:
        if message.get("content"):
            total += estimate_text_tokens(str(message["content"]), model=model)
        if message.get("tool_calls"):
            total += estimate_text_tokens(str(message["tool_calls"]), model=model)
    return total


def _approx_text_tokens(text: str) -> int:
    """Rough token count fallback. Kept for compatibility with existing behavior."""
    return len(text) // 3


def _deepseek_tokenizer() -> Any | None:
    path = os.getenv("FOLIUM_DEEPSEEK_TOKENIZER")
    if not path:
        return None
    return _load_deepseek_tokenizer(str(Path(path)))


@lru_cache(maxsize=4)
def _load_deepseek_tokenizer(path: str) -> Any | None:
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(Path(path), trust_remote_code=True)
    except Exception:
        return None
