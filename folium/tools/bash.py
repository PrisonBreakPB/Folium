"""Shell command tool with selectable execution backends."""

import os

from pydantic import BaseModel, ConfigDict, Field

from .base import Tool, ToolFailure
from ..sandbox import DockerSandboxExecutor, LocalCommandExecutor
from ..sandbox.session import get_current_session, use_bash_sandbox, use_copy_workspace


class BashArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    command: str = Field(description="The shell command to run")
    timeout: int = Field(default=120, description="Timeout in seconds (default 120)")


class BashTool(Tool):
    name = "bash"
    description = (
        "Run shell commands and return stdout, stderr, and exit code. "
        "Reserve this tool for operations that require a shell, such as builds, tests, "
        "running scripts, package installation, Git, process management, and network commands. "
        "Do not use it to read files with cat, head, or tail; use read_file instead. "
        "Do not use it to search files or content with grep, rg, find, or ls; use glob or grep instead. "
        "Do not use it to edit files with sed or awk; use edit_file instead. "
        "Do not use it to create or overwrite files with echo, cat, heredocs, or shell redirection; "
        "use write_file instead."
    )
    args_model = BashArgs

    def __init__(self, executor=None):
        self._executor = executor

    @property
    def executor(self):
        if self._executor is None:
            self._executor = _executor_from_env()
        return self._executor

    def execute(self, command: str, timeout: int = 120) -> str | ToolFailure:
        return self.executor.run(command, timeout=timeout)


def _executor_from_env():
    backend = os.getenv("FOLIUM_BASH_BACKEND", "docker").strip().lower()
    if backend == "docker":
        return DockerSandboxExecutor(
            image=os.getenv("FOLIUM_DOCKER_IMAGE", "python:3.11-slim"),
            workspace=os.getenv("FOLIUM_DOCKER_WORKSPACE") or None,
            network=os.getenv("FOLIUM_DOCKER_NETWORK", "bridge"),
            cpus=os.getenv("FOLIUM_DOCKER_CPUS", "1"),
            memory=os.getenv("FOLIUM_DOCKER_MEMORY", "2g"),
            pids_limit=int(os.getenv("FOLIUM_DOCKER_PIDS_LIMIT", "256")),
        )
    cwd = (
        str(get_current_session(copy_workspace=use_copy_workspace()).workspace)
        if use_bash_sandbox()
        else None
    )
    return LocalCommandExecutor(cwd=cwd)
