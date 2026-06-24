from __future__ import annotations

from pathlib import Path

from .parser import parse_skill_file
from .types import SKILL_MD_FILE, Skill


def default_skills_root() -> Path:
    return Path.cwd() / "skills"


def load_skills(skills_root: str | Path | None = None) -> list[Skill]:
    root = Path(skills_root) if skills_root else default_skills_root()
    if not root.exists() or not root.is_dir():
        return []

    skills: list[Skill] = []
    for skill_dir in sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")):
        skill = parse_skill_file(skill_dir / SKILL_MD_FILE)
        if skill:
            skills.append(skill)
    return skills
