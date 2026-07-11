"""Write/edit approval previews."""

from __future__ import annotations

import difflib
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

from .sandbox.filesystem import resolve_tool_path


MAX_APPROVAL_DIFF_CHARS = 30_000


@dataclass
class EditApprovalProposal:
    tool_name: str
    path: str
    title: str
    diff: str
    truncated: bool = False
    diff_chars: int = 0


@dataclass
class FileChangeSnapshot:
    tool_name: str
    file_path: str
    path: Path
    before: str
    existed: bool


def capture_file_change_snapshot(tool_name: str, arguments: dict) -> FileChangeSnapshot | None:
    if tool_name not in {"write_file", "edit_file"}:
        return None

    file_path = arguments.get("file_path")
    if not isinstance(file_path, str) or not file_path:
        return None

    path = resolve_tool_path(file_path)
    if path.exists() and not path.is_file():
        return None
    return FileChangeSnapshot(
        tool_name=tool_name,
        file_path=file_path,
        path=path,
        before=_read_text(path) if path.exists() else "",
        existed=path.exists(),
    )


def build_file_change_proposal(snapshot: FileChangeSnapshot) -> EditApprovalProposal | None:
    if not snapshot.path.exists() or not snapshot.path.is_file():
        return None

    after = _read_text(snapshot.path)
    diff = _diff_text(snapshot.before, after, str(snapshot.path), old_exists=snapshot.existed)
    if not diff and not snapshot.existed:
        diff = f"--- /dev/null\n+++ b/{snapshot.path}\n"
    diff, truncated, total = _truncate_diff(diff)
    action = "创建文件" if not snapshot.existed else "修改文件"
    return EditApprovalProposal(
        tool_name=snapshot.tool_name,
        path=str(snapshot.path),
        title=f"{action}：{snapshot.file_path}",
        diff=diff,
        truncated=truncated,
        diff_chars=total,
    )


def build_edit_approval_proposal(tool_name: str, arguments: dict) -> EditApprovalProposal | None:
    if tool_name == "write_file":
        return _write_file_proposal(arguments)
    if tool_name == "edit_file":
        return _edit_file_proposal(arguments)
    if tool_name == "bash":
        return _bash_proposal(arguments)
    return None


def _write_file_proposal(arguments: dict) -> EditApprovalProposal | None:
    file_path = str(arguments.get("file_path") or "")
    content = arguments.get("content")
    if not file_path or content is None:
        return None

    path = resolve_tool_path(file_path)
    old = _read_text(path) if path.exists() and path.is_file() else ""
    new = str(content)
    diff = _diff_text(old, new, str(path), old_exists=path.exists())
    diff, truncated, total = _truncate_diff(diff)
    action = "覆盖文件" if path.exists() else "创建文件"
    return EditApprovalProposal(
        tool_name="write_file",
        path=str(path),
        title=f"{action}：{file_path}",
        diff=diff,
        truncated=truncated,
        diff_chars=total,
    )


def _edit_file_proposal(arguments: dict) -> EditApprovalProposal | None:
    file_path = str(arguments.get("file_path") or "")
    old_string = arguments.get("old_string")
    new_string = arguments.get("new_string")
    if not file_path or old_string is None or new_string is None:
        return None

    path = resolve_tool_path(file_path)
    if not path.exists() or not path.is_file():
        return None

    old = _read_text(path)
    occurrences = old.count(str(old_string))
    if occurrences != 1:
        return None

    new = old.replace(str(old_string), str(new_string), 1)
    diff = _diff_text(old, new, str(path), old_exists=True)
    diff, truncated, total = _truncate_diff(diff)
    return EditApprovalProposal(
        tool_name="edit_file",
        path=str(path),
        title=f"编辑文件：{file_path}",
        diff=diff,
        truncated=truncated,
        diff_chars=total,
    )


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(errors="replace")


def _diff_text(old: str, new: str, filename: str, old_exists: bool) -> str:
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    fromfile = f"a/{filename}" if old_exists else "/dev/null"
    tofile = f"b/{filename}"
    return "".join(difflib.unified_diff(old_lines, new_lines, fromfile=fromfile, tofile=tofile, n=3))


def _truncate_diff(diff: str) -> tuple[str, bool, int]:
    total = len(diff)
    if total <= MAX_APPROVAL_DIFF_CHARS:
        return diff, False, total
    return (
        diff[:MAX_APPROVAL_DIFF_CHARS]
        + f"\n... diff truncated ({total} chars total) ...",
        True,
        total,
    )


def _bash_proposal(arguments: dict) -> EditApprovalProposal | None:
    command = str(arguments.get("command") or "")
    if not command or not _bash_writes_workspace(command):
        return None
    preview = "这条 bash 命令可能会修改已挂载的工作区：\n\n" + command
    preview, truncated, total = _truncate_diff(preview)
    return EditApprovalProposal(
        tool_name="bash",
        path="/workspace",
        title="确认 bash 写入工作区",
        diff=preview,
        truncated=truncated,
        diff_chars=total,
    )


def _bash_writes_workspace(command: str) -> bool:
    if _has_workspace_redirection(command):
        return True
    words = _shell_words(command)
    if not words:
        return False
    return _has_workspace_write_command(words) or _has_workspace_python_write(command)


def _has_workspace_redirection(command: str) -> bool:
    for match in re.finditer(r"(?<![<])(?:^|\s)(?:\d?>{1,2})\s*(?P<target>[^\s;&|]+)", command):
        target = match.group("target").strip("'\"")
        if target.startswith("&"):
            continue
        if _targets_workspace(target):
            return True
    return False


def _has_workspace_write_command(words: list[str]) -> bool:
    for index, word in enumerate(words):
        base = Path(word).name
        remaining = words[index + 1:]
        if base in {"cp", "mv"} and remaining:
            targets = [remaining[-1]]
            if any(_targets_workspace(target) for target in targets):
                return True
        if base in {"touch", "mkdir", "rm"}:
            if any(_targets_workspace(arg) for arg in _non_option_args(remaining)):
                return True
        if base == "tee":
            if any(_targets_workspace(arg) for arg in _non_option_args(remaining)):
                return True
        if base == "sed" and any(arg == "-i" or arg.startswith("-i") for arg in remaining):
            return True
    return False


def _has_workspace_python_write(command: str) -> bool:
    if "open(" not in command and "write_text(" not in command:
        return False
    return "/workspace" in command or re.search(r"['\"][^'\"]+['\"]\s*,\s*['\"](?:w|a|x)", command) is not None


def _shell_words(command: str) -> list[str]:
    try:
        return shlex.split(command, posix=True)
    except ValueError:
        return []


def _non_option_args(args: list[str]) -> list[str]:
    return [arg for arg in args if arg and not arg.startswith("-")]


def _targets_workspace(target: str) -> bool:
    if not target:
        return False
    if target.startswith("$"):
        return True
    normalized = target.replace("\\", "/")
    if normalized.startswith("/workspace") or normalized == "." or normalized.startswith("./"):
        return True
    if normalized.startswith("/"):
        return False
    return True
