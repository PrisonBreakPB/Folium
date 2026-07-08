"""Shell command tool with selectable execution backends."""

import os

from .base import Tool
from ..sandbox import DockerSandboxExecutor, LocalCommandExecutor


class BashTool(Tool):
    name = "bash"
    description = (
        "Execute a shell command. Returns stdout, stderr, and exit code. "
        "Use this for running tests, installing packages, git operations, etc."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to run",
            },
            "timeout": {
                "type": "integer",
                "description": "Timeout in seconds (default 120)",
            },
        },
        "required": ["command"],
    }

    def __init__(self, executor=None):
        self._executor = executor

    @property
    def executor(self):
        if self._executor is None:
            self._executor = _executor_from_env()
        return self._executor

    def execute(self, command: str, timeout: int = 120) -> str:
        return self.executor.run(command, timeout=timeout)


def _executor_from_env():
    backend = os.getenv("FOLIUM_BASH_BACKEND", "docker").strip().lower()
    if backend == "docker":
        return DockerSandboxExecutor(
            image=os.getenv("FOLIUM_DOCKER_IMAGE", "python:3.11-slim"),
            workspace=os.getenv("FOLIUM_DOCKER_WORKSPACE") or None,
            network=os.getenv("FOLIUM_DOCKER_NETWORK", "none"),
            cpus=os.getenv("FOLIUM_DOCKER_CPUS", "1"),
            memory=os.getenv("FOLIUM_DOCKER_MEMORY", "2g"),
            pids_limit=int(os.getenv("FOLIUM_DOCKER_PIDS_LIMIT", "256")),
        )
    return LocalCommandExecutor()
