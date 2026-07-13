"""Session persistence - save and resume conversations.

Conversations are stored in ./conversations/ under the current working directory.
Each session is a JSON file containing messages + model config.
"""

import json
import os
import re
import time
import uuid
import copy
from pathlib import Path
from .encoding import repair_mojibake_text
from .session_prompts import delete_prompt, save_prompt, load_prompt

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


def new_session_id() -> str:
    """Create a session ID without writing a session file."""
    return _new_session_id()


def _session_path(session_id: str) -> Path:
    path = (SESSIONS_DIR / f"{_normalize_session_id(session_id)}.json").resolve()
    root = SESSIONS_DIR.resolve()
    if root != path.parent:
        raise ValueError("Invalid session id")
    return path


def save_session(
    messages: list[dict],
    model: str,
    session_id: str | None = None,
    transcript: list[dict] | None = None,
    system_prompt: str | None = None,
) -> str:
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
        "transcript": transcript if transcript is not None else messages,
    }

    path = _session_path(session_id)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # Save system prompt to SQLite
    if system_prompt is not None:
        save_prompt(session_id, system_prompt)

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


def load_session(session_id: str) -> tuple[list[dict], str, list[dict], str | None] | None:
    """Load a saved session. Returns (messages, model, transcript, system_prompt) or None."""
    data = _load_raw(session_id)
    if not data:
        return None
    messages = data["messages"]
    transcript = data.get("transcript")
    # Try to load system prompt from SQLite
    system_prompt = load_prompt(session_id)
    return messages, data["model"], transcript if transcript is not None else copy.deepcopy(messages), system_prompt


def delete_session(session_id: str) -> bool:
    """Delete a session file and its stored system prompt."""
    path = _session_path(session_id)
    if path.exists():
        path.unlink()
        delete_prompt(path.stem)
        return True
    return False


def calculate_session_stats(messages: list[dict]) -> dict:
    """Aggregate token usage stats from assistant messages' _usage field."""
    total_prompt = 0
    total_cached = 0
    total_completion = 0
    total_cost = 0.0

    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        usage = msg.get("_usage")
        if not usage:
            continue
        total_prompt += usage.get("prompt_tokens", 0)
        total_cached += usage.get("cached_tokens", 0)
        total_completion += usage.get("completion_tokens", 0)
        cost = usage.get("cost")
        if cost is not None:
            total_cost += cost

    cache_miss = total_prompt - total_cached
    cache_hit_rate = total_cached / total_prompt if total_prompt > 0 else 0

    return {
        "prompt_tokens": total_prompt,
        "cache_miss_tokens": cache_miss,
        "cached_tokens": total_cached,
        "completion_tokens": total_completion,
        "total_tokens": total_prompt + total_completion,
        "cost": total_cost,
        "cache_hit_rate": cache_hit_rate,
    }


def list_sessions() -> list[dict]:
    """List available sessions, newest first."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    sessions = []
    for f in sorted(SESSIONS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            # skip empty/corrupt files
            transcript = data.get("transcript") or data.get("messages")
            if not transcript:
                continue
            # grab first user message as preview
            preview = ""
            for m in transcript:
                if m.get("role") == "user" and m.get("content"):
                    preview = repair_mojibake_text(m["content"])[:80]
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
