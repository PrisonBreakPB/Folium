"""Tests for project-scoped memory storage (git-root keyed files)."""

from pathlib import Path
import pytest

from folium import memory_store as ms


def test_sanitize_replaces_non_alnum(tmp_path):
    # POSIX-style separators become '-'; letters/digits survive.
    assert ms.sanitize_path(Path("ab c.def")) == "ab-c-def"
    assert ms.sanitize_path(Path("a/b_c.0")) == "a-b-c-0"


def test_sanitize_stable():
    p = Path("some/deep x/pa-th")
    assert ms.sanitize_path(p) == ms.sanitize_path(p)


def test_sanitize_short_no_hash():
    result = ms.sanitize_path(Path("short"))
    assert "-" not in result


def test_sanitize_long_appends_hash():
    long_path = Path("x/" + "a" * 300)
    result = ms.sanitize_path(long_path)
    assert len(result) > ms.MAX_SLUG_LENGTH
    assert result.startswith(f"x-{'a' * (ms.MAX_SLUG_LENGTH - 2)}-")


def test_find_boundary_in_git_dir(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    sub = repo / "a" / "b"
    sub.mkdir(parents=True)
    assert ms.find_project_boundary(sub) == repo


def test_find_boundary_non_git_falls_back_to_cwd(tmp_path):
    sub = tmp_path / "no" / "git"
    sub.mkdir(parents=True)
    assert ms.find_project_boundary(sub) == sub


def test_slug_for_uses_git_root(tmp_path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    sub = repo / "a"
    sub.mkdir(parents=True)
    assert ms.slug_for(sub) == ms.slug_for(repo)


def test_memory_dir_layout(tmp_path):
    cwd = tmp_path / "proj"
    cwd.mkdir()
    base = tmp_path / "data"
    paths = ms.memory_file_paths(cwd, base)
    directory = ms.memory_dir_for(cwd, base)
    assert set(paths) == set(ms.MEMORY_FILES)
    assert all(str(p).startswith(str(directory)) for p in paths.values())
    # Layout: <base>/projects/<slug>/<name>.md
    assert directory == base / "projects" / ms.slug_for(cwd)


def test_ensure_memory_dir_creates(tmp_path):
    cwd = tmp_path / "proj"
    cwd.mkdir()
    base = tmp_path / "data"
    directory = ms.ensure_memory_dir(cwd, base)
    assert directory.is_dir()