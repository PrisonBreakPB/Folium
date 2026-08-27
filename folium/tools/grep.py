"""Content search with regex support."""

import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .base import Tool, ToolFailure, tool_failure
from ..sandbox.filesystem import SandboxPathError, resolve_tool_path

# skip these dirs to avoid noise
_SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".tox", "dist", "build"}


class GrepArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    pattern: str = Field(description="Regex pattern to search for")
    path: str = Field(default=".", description="File or directory to search (default: cwd)")
    include: str | None = Field(default=None, description="Only search files matching this glob (e.g. '*.py')")


class GrepTool(Tool):
    name = "grep"
    description = (
        "Search file contents with regex. "
        "Returns matching lines with file path and line number."
    )
    args_model = GrepArgs

    def execute(self, pattern: str, path: str = ".", include: str | None = None) -> str | ToolFailure:
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return tool_failure("invalid_regex", "validation", f"Invalid regex: {e}")

        try:
            base = resolve_tool_path(path)
        except SandboxPathError as e:
            return tool_failure("sandbox_path_error", "permission", str(e))
        if not base.exists():
            return tool_failure("path_not_found", "resource", f"{path} not found")

        if base.is_file():
            files = [base]
        else:
            files = self._walk(base, include)

        matches = []
        for fp in files:
            try:
                text = fp.read_text(errors="ignore")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    matches.append(f"{fp}:{lineno}: {line.rstrip()}")
                    if len(matches) >= 200:
                        matches.append("... (200 match limit reached)")
                        return "\n".join(matches)

        return "\n".join(matches) if matches else "No matches found."

    @staticmethod
    def _walk(root: Path, include: str | None) -> list[Path]:
        """Walk dir tree, skipping junk dirs."""
        results = []
        for item in root.rglob(include or "*"):
            # skip hidden/junk directories
            if any(part in _SKIP_DIRS for part in item.parts):
                continue
            if item.is_file():
                results.append(item)
            if len(results) >= 5000:
                break
        return results
