"""File pattern matching."""

from pydantic import BaseModel, ConfigDict, Field

from .base import Tool
from ..sandbox.filesystem import SandboxPathError, resolve_tool_path


class GlobArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    pattern: str = Field(description="Glob pattern, e.g. '**/*.py' or 'src/**/*.ts'")
    path: str = Field(default=".", description="Directory to search in (default: cwd)")


class GlobTool(Tool):
    name = "glob"
    description = (
        "Find files matching a glob pattern. "
        "Supports ** for recursive matching (e.g. '**/*.py')."
    )
    args_model = GlobArgs

    def execute(self, pattern: str, path: str = ".") -> str:
        try:
            base = resolve_tool_path(path)
            if not base.is_dir():
                return f"Error: {path} is not a directory"

            hits = list(base.glob(pattern))
            # sort by mtime, newest first
            hits.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)

            total = len(hits)
            shown = hits[:100]
            lines = [str(h) for h in shown]
            result = "\n".join(lines)

            if total > 100:
                result += f"\n... ({total} matches, showing first 100)"
            return result or "No files matched."
        except SandboxPathError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error: {e}"
