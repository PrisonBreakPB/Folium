from __future__ import annotations

import logging
import re
from pathlib import Path

from .types import SKILL_MD_FILE, Skill

logger = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_skill_file(skill_file: Path) -> Skill | None:
    if not skill_file.exists() or skill_file.name != SKILL_MD_FILE:
        return None

    try:
        content = skill_file.read_text(encoding="utf-8")
    except OSError:
        logger.warning("Failed to read skill file: %s", skill_file, exc_info=True)
        return None

    match = _FRONTMATTER_RE.match(content)
    if not match:
        return None

    metadata = _parse_frontmatter(match.group(1))
    name = metadata.get("name", "").strip()
    description = metadata.get("description", "").strip()
    if not name or not description:
        return None

    if name != skill_file.parent.name:
        logger.warning(
            "Skipping skill %s: frontmatter name %r does not match directory name %r",
            skill_file,
            name,
            skill_file.parent.name,
        )
        return None

    return Skill(
        name=name,
        description=description,
        skill_dir=skill_file.parent,
        skill_file=skill_file,
    )


def _parse_frontmatter(text: str) -> dict[str, str]:
    data: dict[str, str] = {}
    current_key: str | None = None
    current_lines: list[str] = []

    def flush():
        nonlocal current_key, current_lines
        if current_key:
            data[current_key] = "\n".join(current_lines).strip()
        current_key = None
        current_lines = []

    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        if raw_line.startswith((" ", "\t")) and current_key:
            current_lines.append(raw_line.strip())
            continue

        flush()
        key, sep, value = raw_line.partition(":")
        if not sep:
            continue
        current_key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        current_lines = [value] if value else []

    flush()
    return data
