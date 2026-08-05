"""Approval proposals for workspace-changing tool calls."""

from __future__ import annotations

import difflib
import hashlib
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .sandbox.filesystem import resolve_tool_path


MAX_APPROVAL_DIFF_CHARS = 30_000
PROTECTED_FILE_SUFFIXES = frozenset({
    ".tex",
    ".bib",
    ".sty",
    ".py",
    ".m",
    ".ipynb",
    ".sh",
})
_NON_FILE_FINGERPRINT = "<non-file>"


@dataclass(frozen=True)
class ApprovalDecision:
    """A user's decision for a pending workspace change."""

    action: Literal["approved", "rejected", "revision_requested"]
    feedback: str = ""


def normalize_approval_decision(value: object) -> ApprovalDecision:
    """Keep existing boolean approval callbacks compatible."""

    if isinstance(value, ApprovalDecision):
        return value
    return ApprovalDecision("approved" if bool(value) else "rejected")


@dataclass
class EditApprovalProposal:
    """Legacy single-operation approval proposal used by Bash."""

    tool_name: str
    path: str
    title: str
    diff: str
    truncated: bool = False
    diff_chars: int = 0


@dataclass
class FileChangeProposal:
    """A reviewed file replacement that has not been applied yet."""

    tool_call_id: str
    tool_name: str
    requested_path: str
    path: str
    title: str
    diff: str
    preview_diff: str
    truncated: bool
    diff_chars: int
    additions: int
    deletions: int
    original_fingerprint: str | None
    old_exists: bool
    new_content: str = field(repr=False)


@dataclass
class ChangeSetProposal:
    """One user decision covering all protected file changes in a model round."""

    files: list[FileChangeProposal]
    title: str = "Review protected file changes"

    @property
    def additions(self) -> int:
        return sum(change.additions for change in self.files)

    @property
    def deletions(self) -> int:
        return sum(change.deletions for change in self.files)


def is_protected_file_change(tool_name: str, arguments: dict) -> bool:
    if tool_name not in {"write_file", "edit_file"}:
        return False
    file_path = arguments.get("file_path")
    if not isinstance(file_path, str) or not file_path:
        return False
    try:
        path = resolve_tool_path(file_path)
    except Exception:
        return False
    return path.suffix.lower() in PROTECTED_FILE_SUFFIXES


def build_file_change_proposal(
    tool_call_id: str,
    tool_name: str,
    arguments: dict,
) -> FileChangeProposal | None:
    """Build an unapplied protected-file change from validated tool arguments."""

    if not is_protected_file_change(tool_name, arguments):
        return None
    if tool_name == "write_file":
        return _write_file_proposal(tool_call_id, arguments)
    if tool_name == "edit_file":
        return _edit_file_proposal(tool_call_id, arguments)
    return None


def build_change_set_proposal(
    files: list[FileChangeProposal],
) -> ChangeSetProposal | None:
    if not files:
        return None
    return ChangeSetProposal(files=files)


def build_edit_approval_proposal(
    tool_name: str,
    arguments: dict,
) -> EditApprovalProposal | FileChangeProposal | None:
    """Keep the old public helper while supporting protected file proposals."""

    if tool_name in {"write_file", "edit_file"}:
        return build_file_change_proposal("", tool_name, arguments)
    if tool_name == "bash":
        return _bash_proposal(arguments)
    return None


def change_set_is_current(change_set: ChangeSetProposal) -> bool:
    """Return whether every reviewed source file still matches its baseline."""

    return all(
        _path_fingerprint(Path(change.path)) == change.original_fingerprint
        for change in change_set.files
    )


def apply_change_set(change_set: ChangeSetProposal) -> None:
    """Apply a reviewed change set after checking every source file first."""

    if not change_set_is_current(change_set):
        raise RuntimeError("reviewed file changed before approval was applied")

    for change in change_set.files:
        path = Path(change.path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(change.new_content, encoding="utf-8")


def change_result_summary(change: FileChangeProposal) -> str:
    if change.tool_name == "write_file":
        n_lines = _line_count(change.new_content)
        return f"Wrote {n_lines} lines to {change.requested_path}"
    return f"Edited {change.requested_path}"


def _write_file_proposal(
    tool_call_id: str,
    arguments: dict,
) -> FileChangeProposal | None:
    file_path = str(arguments.get("file_path") or "")
    content = arguments.get("content")
    if not file_path or content is None:
        return None

    path = resolve_tool_path(file_path)
    if path.exists() and not path.is_file():
        return None
    old_exists = path.exists()
    old = _read_text(path) if old_exists else ""
    new = str(content)
    diff = _diff_text(old, new, str(path), old_exists=old_exists)
    preview_diff, truncated, total = _truncate_diff(diff)
    action = "Overwrite file" if old_exists else "Create file"
    additions, deletions = _diff_line_counts(diff)
    return FileChangeProposal(
        tool_call_id=tool_call_id,
        tool_name="write_file",
        requested_path=file_path,
        path=str(path),
        title=f"{action}: {file_path}",
        diff=diff,
        preview_diff=preview_diff,
        truncated=truncated,
        diff_chars=total,
        additions=additions,
        deletions=deletions,
        original_fingerprint=_path_fingerprint(path),
        old_exists=old_exists,
        new_content=new,
    )


def _edit_file_proposal(
    tool_call_id: str,
    arguments: dict,
) -> FileChangeProposal | None:
    file_path = str(arguments.get("file_path") or "")
    old_string = arguments.get("old_string")
    new_string = arguments.get("new_string")
    if not file_path or old_string is None or new_string is None:
        return None

    path = resolve_tool_path(file_path)
    if not path.exists() or not path.is_file():
        return None

    old = _read_text(path)
    if old.count(str(old_string)) != 1:
        return None

    new = old.replace(str(old_string), str(new_string), 1)
    diff = _diff_text(old, new, str(path), old_exists=True)
    preview_diff, truncated, total = _truncate_diff(diff)
    additions, deletions = _diff_line_counts(diff)
    return FileChangeProposal(
        tool_call_id=tool_call_id,
        tool_name="edit_file",
        requested_path=file_path,
        path=str(path),
        title=f"Edit file: {file_path}",
        diff=diff,
        preview_diff=preview_diff,
        truncated=truncated,
        diff_chars=total,
        additions=additions,
        deletions=deletions,
        original_fingerprint=_path_fingerprint(path),
        old_exists=True,
        new_content=new,
    )


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(errors="replace")


def _path_fingerprint(path: Path) -> str | None:
    if not path.exists():
        return None
    if not path.is_file():
        return _NON_FILE_FINGERPRINT
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _diff_text(old: str, new: str, filename: str, old_exists: bool) -> str:
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    fromfile = f"a/{filename}" if old_exists else "/dev/null"
    tofile = f"b/{filename}"
    return "".join(difflib.unified_diff(old_lines, new_lines, fromfile=fromfile, tofile=tofile, n=3))


def _diff_line_counts(diff: str) -> tuple[int, int]:
    additions = 0
    deletions = 0
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            additions += 1
        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1
    return additions, deletions


def _line_count(content: str) -> int:
    return content.count("\n") + (1 if content and not content.endswith("\n") else 0)


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
    preview = "This bash command may modify the mounted workspace:\n\n" + command
    preview, truncated, total = _truncate_diff(preview)
    return EditApprovalProposal(
        tool_name="bash",
        path="/workspace",
        title="Confirm bash workspace write",
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
            if _targets_workspace(remaining[-1]):
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
