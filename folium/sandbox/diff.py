"""Diff sandbox workspace changes against the host workspace."""

import difflib
from pathlib import Path

from .session import EXCLUDE_NAMES, EXCLUDE_PARTS, SandboxSession, get_current_session, use_copy_workspace


MAX_DIFF_CHARS = 30_000


def sandbox_diff(session: SandboxSession | None = None, max_chars: int = MAX_DIFF_CHARS) -> str:
    if session is None and not use_copy_workspace():
        return (
            "sandbox_diff is only available when FOLIUM_SANDBOX_WORKSPACE_MODE=copy. "
            "Current Docker mode mounts the real workspace directly, so changes are applied "
            "to the project as tools run. Use git diff to inspect project changes."
        )
    session = session or get_current_session()
    session.prepare()
    host_files = _file_map(session.host_workspace)
    sandbox_files = _file_map(session.workspace)
    paths = sorted(set(host_files) | set(sandbox_files))

    chunks = []
    changed = 0
    for rel in paths:
        host_path = host_files.get(rel)
        sandbox_path = sandbox_files.get(rel)
        if host_path and sandbox_path and _same_file(host_path, sandbox_path):
            continue
        changed += 1
        chunks.append(_diff_one(rel, host_path, sandbox_path))

    if not chunks:
        return "No sandbox changes."

    result = f"Sandbox changes ({changed} files):\n\n" + "\n".join(chunks)
    if len(result) > max_chars:
        result = result[:max_chars] + f"\n... truncated ({len(result)} chars total) ..."
    return result


def _file_map(root: Path) -> dict[str, Path]:
    result = {}
    if not root.exists():
        return result
    for path in root.rglob("*"):
        if not path.is_file() or _is_excluded(path, root):
            continue
        result[_rel(path, root)] = path
    return result


def _diff_one(rel: str, host_path: Path | None, sandbox_path: Path | None) -> str:
    if host_path is None:
        new_lines = _read_lines(sandbox_path)
        return "".join(difflib.unified_diff([], new_lines, fromfile="/dev/null", tofile=f"b/{rel}"))
    if sandbox_path is None:
        old_lines = _read_lines(host_path)
        return "".join(difflib.unified_diff(old_lines, [], fromfile=f"a/{rel}", tofile="/dev/null"))
    old_lines = _read_lines(host_path)
    new_lines = _read_lines(sandbox_path)
    return "".join(difflib.unified_diff(old_lines, new_lines, fromfile=f"a/{rel}", tofile=f"b/{rel}"))


def _same_file(left: Path, right: Path) -> bool:
    try:
        return left.read_bytes() == right.read_bytes()
    except OSError:
        return False


def _read_lines(path: Path | None) -> list[str]:
    if path is None:
        return []
    data = path.read_bytes()
    if b"\x00" in data:
        return ["[binary file]\n"]
    return data.decode("utf-8", errors="replace").splitlines(keepends=True)


def _is_excluded(path: Path, root: Path) -> bool:
    parts = path.resolve().relative_to(root.resolve()).parts
    if any(part in EXCLUDE_NAMES for part in parts):
        return True
    return any(tuple(parts[: len(excluded)]) == excluded for excluded in EXCLUDE_PARTS)


def _rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()
