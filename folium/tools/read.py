"""File reading with line numbers."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .base import Tool, ToolFailure, tool_failure
from ..sandbox.filesystem import SandboxPathError, resolve_tool_path


def _read_text_prefer_utf8(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        return path.read_text(errors="replace")


class ReadFileArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    file_path: str = Field(description="Path to the file")
    offset: int = Field(default=1, description="Start line (1-based). Default 1.")
    limit: int = Field(default=2000, description="Max lines to read. Default 2000.")


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read a file's contents with line numbers. "
        "Use this tool instead of shell commands such as cat, head, or tail. "
        "Always read a file before editing it."
    )
    args_model = ReadFileArgs

    def execute(self, file_path: str, offset: int = 1, limit: int = 2000) -> str | ToolFailure:
        try:
            p = resolve_tool_path(file_path)
            if not p.exists():
                return tool_failure("file_not_found", "resource", f"{file_path} not found")
            if not p.is_file():
                return tool_failure("not_a_file", "resource", f"{file_path} is a directory, not a file")

            text = _read_text_prefer_utf8(p)
            lines = text.splitlines()
            total = len(lines)

            start = max(0, offset - 1)
            chunk = lines[start : start + limit]
            numbered = [f"{start + i + 1}\t{ln}" for i, ln in enumerate(chunk)]
            result = "\n".join(numbered)

            if total > start + limit:
                result += f"\n... ({total} lines total, showing {start+1}-{start+len(chunk)})"
            return result or "(empty file)"
        except SandboxPathError as e:
            return tool_failure("sandbox_path_error", "permission", str(e))
        except Exception as e:
            return tool_failure("read_failed", "filesystem", str(e))
