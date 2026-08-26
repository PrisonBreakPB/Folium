"""Sandbox session workspace management."""

import fnmatch
import os
import shutil
import uuid
from pathlib import Path

from ..observability.context import record_sandbox_event


SANDBOX_DIR = ".folium/sandbox/sessions"
WORKSPACE_MODE_ENV = "FOLIUM_SANDBOX_WORKSPACE_MODE"
EXCLUDE_NAMES = {".git", ".venv", "__pycache__", ".folium"}
EXCLUDE_PATTERNS = {
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "id_rsa",
    "id_rsa.*",
    "id_ed25519",
    "id_ed25519.*",
    "credentials.json",
    "token.json",
    "service-account*.json",
}
EXCLUDE_PARTS = {("conversations", "traces")}

_current_session: "SandboxSession | None" = None


class SandboxPathError(Exception):
    pass


def sandbox_workspace_mode() -> str:
    return os.getenv(WORKSPACE_MODE_ENV, "host").strip().lower()


def use_copy_workspace() -> bool:
    return sandbox_workspace_mode() == "copy"


def get_host_workspace() -> Path:
    return Path(os.getenv("FOLIUM_HOST_WORKSPACE") or os.getcwd()).resolve()


def configure_host_workspace(workspace_path: str) -> Path:
    """Set the host workspace for subsequently-created sandbox sessions."""
    path = Path(workspace_path).expanduser().resolve()
    if not path.exists() or not path.is_dir():
        raise SandboxPathError(f"workspace is not a directory: {workspace_path}")
    os.environ["FOLIUM_HOST_WORKSPACE"] = str(path)
    return path


class SandboxSession:
    def __init__(self, host_workspace: str | None = None, root_dir: str | None = None):
        self.session_id = uuid.uuid4().hex[:12]
        self.host_workspace = Path(host_workspace).resolve() if host_workspace else get_host_workspace()
        base = Path(root_dir).resolve() if root_dir else self.host_workspace / SANDBOX_DIR
        self.session_dir = base / self.session_id
        self.workspace = self.session_dir / "workspace"
        self._prepared = False

    def prepare(self):
        if self._prepared:
            return
        self.workspace.mkdir(parents=True, exist_ok=True)
        _copy_workspace(self.host_workspace, self.workspace)
        self._prepared = True
        record_sandbox_event("workspace_prepared", {
            "sandbox_session_id": self.session_id,
            "host_workspace": str(self.host_workspace),
            "sandbox_workspace": str(self.workspace),
        })

    def resolve_path(self, file_path: str) -> Path:
        self.prepare()
        raw = Path(file_path).expanduser()
        if raw.is_absolute():
            try:
                relative = raw.resolve().relative_to(self.host_workspace)
            except ValueError:
                raise SandboxPathError(f"path is outside sandbox workspace: {file_path}")
        else:
            relative = raw
        resolved = (self.workspace / relative).resolve()
        try:
            resolved.relative_to(self.workspace)
        except ValueError:
            raise SandboxPathError(f"path escapes sandbox workspace: {file_path}")
        return resolved


def get_current_session() -> SandboxSession:
    global _current_session
    if _current_session is None:
        _current_session = SandboxSession()
    _current_session.prepare()
    return _current_session


def reset_current_session():
    global _current_session
    _current_session = None


def _copy_workspace(src: Path, dst: Path):
    for item in src.iterdir():
        if _should_exclude(item, src):
            continue
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, ignore=_ignore_names, dirs_exist_ok=True)
        elif item.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def _ignore_names(directory: str, names: list[str]) -> set[str]:
    base = Path(directory)
    ignored = set()
    for name in names:
        item = base / name
        if _should_exclude(item, base):
            ignored.add(name)
    return ignored


def _should_exclude(path: Path, root: Path) -> bool:
    if path.name in EXCLUDE_NAMES:
        return True
    if any(fnmatch.fnmatch(path.name, pattern) for pattern in EXCLUDE_PATTERNS):
        return True
    try:
        parts = path.resolve().relative_to(root.resolve()).parts
    except ValueError:
        return False
    return any(tuple(parts[: len(excluded)]) == excluded for excluded in EXCLUDE_PARTS)
