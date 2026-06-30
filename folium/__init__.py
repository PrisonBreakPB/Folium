"""Folium - Minimal AI coding agent inspired by Claude Code's architecture."""

__version__ = "0.3.0"

from folium.agent import Agent
from folium.llm import LLM, LLMProviderError
from folium.config import Config
from folium.tools import ALL_TOOLS

__all__ = ["Agent", "LLM", "LLMProviderError", "Config", "ALL_TOOLS", "__version__"]
