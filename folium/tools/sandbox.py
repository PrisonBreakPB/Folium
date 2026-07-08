"""Sandbox inspection tools."""

from .base import Tool
from ..sandbox.diff import sandbox_diff


class SandboxDiffTool(Tool):
    name = "sandbox_diff"
    description = (
        "Show file changes made inside the Docker sandbox workspace compared "
        "with the real host workspace. This only reports changes; it does not apply them."
    )
    parameters = {
        "type": "object",
        "properties": {
            "max_chars": {
                "type": "integer",
                "description": "Maximum characters of diff output to return, default 30000",
            },
        },
        "required": [],
    }

    def execute(self, max_chars: int = 30000) -> str:
        return sandbox_diff(max_chars=max_chars)
