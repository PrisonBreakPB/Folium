"""Session persistence - save and resume conversations.

Conversations are stored in ./conversations/ under the current working directory.
Each session is a JSON file containing messages + model config.
"""

import json
import os
import re
import time
import uuid
from pathlib import Path

SESSIONS_DIR = Path(os.getcwd()) / "conversations"
_SAFE_SESSION_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _normalize_session_id(session_id: str | None) -> str:
    if not session_id:
        return _new_session_id()

    name = session_id.strip().replace("\\", "/").split("/")[-1]
    name = _SAFE_SESSION_RE.sub("-", name).strip(".-_")
    return name or _new_session_id()


def _new_session_id() -> str:
    return f"session_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _session_path(session_id: str) -> Path:
    path = (SESSIONS_DIR / f"{_normalize_session_id(session_id)}.json").resolve()
    root = SESSIONS_DIR.resolve()
    if root != path.parent:
        raise ValueError("Invalid session id")
    return path


def save_session(messages: list[dict], model: str, session_id: str | None = None) -> str:
    """Save conversation to disk. Returns the session ID."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    session_id = _normalize_session_id(session_id)

    # if file already exists, preserve created_at
    existing = _load_raw(session_id)
    created_at = existing.get("created_at", time.strftime("%Y-%m-%d %H:%M:%S")) if existing else time.strftime("%Y-%m-%d %H:%M:%S")

    data = {
        "id": session_id,
        "model": model,
        "created_at": created_at,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "messages": messages,
    }

    path = _session_path(session_id)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return session_id


def _load_raw(session_id: str) -> dict | None:
    """Load raw session data dict, or None if not found."""
    path = _session_path(session_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, KeyError):
        return None


def load_session(session_id: str) -> tuple[list[dict], str] | None:
    """Load a saved session. Returns (messages, model) or None."""
    data = _load_raw(session_id)
    if not data:
        return None
    return data["messages"], data["model"]


def delete_session(session_id: str) -> bool:
    """Delete a session file. Returns True if deleted."""
    path = _session_path(session_id)
    if path.exists():
        path.unlink()
        return True
    return False


def list_sessions() -> list[dict]:
    """List available sessions, newest first."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    sessions = []
    for f in sorted(SESSIONS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            # skip empty/corrupt files
            if not data.get("messages"):
                continue
            # grab first user message as preview
            preview = ""
            for m in data.get("messages", []):
                if m.get("role") == "user" and m.get("content"):
                    preview = m["content"][:80]
                    break
            sessions.append({
                "id": data.get("id", f.stem),
                "model": data.get("model", "?"),
                "created_at": data.get("created_at", "?"),
                "updated_at": data.get("updated_at", "?"),
                "preview": preview,
            })
        except (json.JSONDecodeError, KeyError):
            continue

    return sessions