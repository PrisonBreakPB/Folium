"""Project-scoped long-term memory: git-root keyed files under a user data dir.

References the Claude Code auto-memory design (memdir/paths.ts) but simplified:

* The project boundary is the nearest enclosing git root (found by walking up the
  directory tree looking for a ``.git`` entry), falling back to the current working
  directory when not inside a git repo. The git root discovery is pure filesystem —
  no git executable is invoked.
* That boundary path is sanitized into a directory name (``sanitize_path``), the
  direct analogue of Claude Code's ``sanitizePath``.
* Memory lives in ``<base>/.folium/projects/<slug>/`` as three Markdown files, one
  per memory category.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# Memory category -> filename. Ordering is stable and meaningful for display.
MEMORY_FILES: dict[str, str] = {
    "user": "user.md",
    "feedback": "feedback.md",
    "project": "project.md",
}

MAX_SLUG_LENGTH = 200


def sanitize_path(path: Path) -> str:
    """Turn an absolute path into a safe, readable directory name.

    Mirror of Claude Code's ``sanitizePath``: replace every non-alphanumeric char
    with ``-`` (so drive separators and the leading path separator become leading
    ``-``), keep the result as-is when short, and add a short hash suffix only when
    it exceeds ``MAX_SLUG_LENGTH`` to avoid collisions.
    """
    text = str(path)
    sanitized = "".join(ch if ch.isalnum() else "-" for ch in text)
    if len(sanitized) <= MAX_SLUG_LENGTH:
        return sanitized
    digest = int(
        hashlib.sha256(text.encode("utf-8")).hexdigest(),
        16,
    )
    return f"{sanitized[:MAX_SLUG_LENGTH]}-{digest:08x}"


def find_project_boundary(cwd: Path) -> Path:
    """Return the nearest enclosing git root, or ``cwd`` when not in a git repo.

    Walks up from ``cwd`` looking for a ``.git`` file or directory. Pure filesystem
    discovery (no git invocation), matching Claude Code's ``findGitRoot``. When no
    ``.git`` is found the call is deemed a non-git directory and ``cwd`` is used as
    the boundary so every launch still gets a stable key.
    """
    current = cwd.resolve()
    for parent in (current, *current.parents):
        try:
            if (parent / ".git").exists():
                return parent
        except OSError:
            pass
        if parent == parent.parent:
            break
    return current


def slug_for(cwd: Path) -> str:
    """Stable key for a launch directory: sanitized project boundary."""
    return sanitize_path(find_project_boundary(cwd))


def memory_dir_for(cwd: Path, base_dir: Path | None = None) -> Path:
    """Directory holding this project's memory files.

    ``base_dir`` defaults to ``~/.folium``. Result is ``<base>/.folium/projects/<slug>``.
    """
    base = base_dir if base_dir is not None else Path.home() / ".folium"
    return base / "projects" / slug_for(cwd)


def memory_file_paths(cwd: Path, base_dir: Path | None = None) -> dict[str, Path]:
    """Map memory category -> path for the project rooted at ``cwd``."""
    directory = memory_dir_for(cwd, base_dir)
    return {name: directory / filename for name, filename in MEMORY_FILES.items()}


def ensure_memory_dir(cwd: Path, base_dir: Path | None = None) -> Path:
    """Create and return the project memory directory, as the harness does up front."""
    directory = memory_dir_for(cwd, base_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def current_memory_dir() -> Path:
    """Memory directory for the process's launch directory."""
    return memory_dir_for(Path.cwd())


def current_memory_file_paths() -> dict[str, Path]:
    """Per-category memory paths for the launch directory."""
    return memory_file_paths(Path.cwd())