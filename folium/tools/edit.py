"""Search-and-replace file editing (Claude Code's key innovation).

The core idea: instead of sending whole-file rewrites or line-number patches,
the LLM specifies an *exact* substring to find and its replacement. The
substring must appear exactly once in the file, which eliminates ambiguity
and makes edits safe and reviewable.
"""

import difflib

from .base import Tool, ToolOutput
from ..sandbox.filesystem import SandboxPathError, resolve_tool_path

# track files changed this session for /diff
_changed_files: set[str] = set()


class EditFileTool(Tool):
    name = "edit_file"
    description = (
        "Make a targeted edit to an existing file by replacing one exact string with another. "
        "old_string must appear exactly once; include enough surrounding context to make it unique. "
        "Use this tool for local changes instead of rewriting a whole file. "
        "Do not edit files through bash with sed or awk."
    )
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path to the file to edit",
            },
            "old_string": {
                "type": "string",
                "description": "Exact text to find (must be unique in file)",
            },
            "new_string": {
                "type": "string",
                "description": "Replacement text",
            },
        },
        "required": ["file_path", "old_string", "new_string"],
    }

    def execute(self, file_path: str, old_string: str, new_string: str) -> str | ToolOutput:
        try:
            p = resolve_tool_path(file_path)
            if not p.exists():
                return f"Error: {file_path} not found"

            content = p.read_text()
            occurrences = content.count(old_string)

            if occurrences == 0:
                preview = content[:500] + ("..." if len(content) > 500 else "")
                return (
                    f"Error: old_string not found in {file_path}.\n"
                    f"File starts with:\n{preview}"
                )
            if occurrences > 1:
                return (
                    f"Error: old_string appears {occurrences} times in {file_path}. "
                    f"Include more surrounding lines to make it unique."
                )

            new_content = content.replace(old_string, new_string, 1)
            p.write_text(new_content)
            _changed_files.add(str(p))

            # generate a unified diff so the user/LLM can see exactly what changed
            diff = _unified_diff(content, new_content, str(p))
            summary = f"Edited {file_path}"
            return ToolOutput(
                content=f"{summary}\n{diff}",
                preview=summary,
                diff=diff,
            )
        except SandboxPathError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error: {e}"


def _unified_diff(
    old: str,
    new: str,
    filename: str,
    context: int = 3,
    old_exists: bool = True,
) -> str:
    """Generate a compact unified diff between old and new file content."""
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=f"a/{filename}" if old_exists else "/dev/null", tofile=f"b/{filename}",
        n=context,
    )
    result = "".join(diff)
    # truncate enormous diffs
    if len(result) > 3000:
        result = result[:2500] + "\n... (diff truncated)\n"
    return result
