"""File creation / overwrite."""

from .base import Tool, ToolOutput
from .edit import _changed_files, _unified_diff
from ..sandbox.filesystem import SandboxPathError, resolve_tool_path


class WriteFileTool(Tool):
    name = "write_file"
    description = (
        "Create a new file or write the complete contents of a file, replacing any existing contents. "
        "Use this tool to create complete LaTeX (.tex) source files. "
        "Do not create or overwrite files through bash with echo, cat, heredocs, or shell redirection. "
        "Never use this tool to create or modify memory.md; use the memory tool to manage persistent long-term memory. "
        "For targeted edits to an existing file, use edit_file instead."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path for the file",
            },
            "content": {
                "type": "string",
                "description": "Full file content to write",
            },
        },
        "required": ["file_path", "content"],
    }

    def execute(self, file_path: str, content: str) -> str | ToolOutput:
        try:
            p = resolve_tool_path(file_path)
            old_exists = p.exists()
            old_content = p.read_text(encoding="utf-8") if old_exists and p.is_file() else ""
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            _changed_files.add(str(p))
            n_lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
            summary = f"Wrote {n_lines} lines to {file_path}"
            diff = _unified_diff(old_content, content, str(p), old_exists=old_exists)
            return ToolOutput(
                content=f"{summary}\n{diff}" if diff else summary,
                preview=summary,
                diff=diff,
            )
        except SandboxPathError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error: {e}"
