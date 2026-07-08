"""Filesystem routing for local and sandbox-backed tools."""

import os
from pathlib import Path

from .session import SandboxPathError, get_current_session, get_host_workspace, use_copy_workspace


def resolve_tool_path(file_path: str) -> Path:
    if _use_docker_workspace() and use_copy_workspace():
        return get_current_session().resolve_path(file_path)
    raw = Path(file_path).expanduser()
    if raw.is_absolute():
        return raw.resolve()
    return (get_host_workspace() / raw).resolve()


def display_path(file_path: str) -> str:
    return file_path


def _use_docker_workspace() -> bool:
    return os.getenv("FOLIUM_BASH_BACKEND", "docker").strip().lower() == "docker"


__all__ = ["SandboxPathError", "display_path", "resolve_tool_path"]
