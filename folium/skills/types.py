from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SKILL_MD_FILE = "SKILL.md"


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    skill_dir: Path
    skill_file: Path
