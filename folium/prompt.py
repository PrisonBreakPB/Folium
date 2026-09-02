"""System prompt - the instructions that turn an LLM into a research assistant.

The prompt is assembled from modular Markdown files under ``folium/prompts/``.
Static blocks (01-04) are fixed guidance; dynamic blocks (05-07) are templates
filled with runtime data (skills, memory, environment). This keeps the prompt
readable as separate concerns instead of one large f-string.
"""

import os
import platform
from pathlib import Path

from . import memory_store

MAX_MEMORY_CHARS_PER_FILE = 2000

_ROLE_FALLBACK = """\
You are Folium, an AI research assistant designed to support the full academic research workflow.
You help researchers with literature review, mathematical derivation, simulation experiments, and paper writing.
You communicate clearly, focus on being genuinely useful, and avoid unnecessary verbosity.
When reviewing literature, you identify research gaps and suggest feasible directions.
When deriving mathematics, you show step-by-step reasoning and verify correctness.
When writing code or papers, you prioritize clarity and correctness over complexity."""

_PARALLEL_TOOLS = """\
# Multiple tool calls
Call multiple tools in one turn only when their inputs are already known
and the calls are genuinely independent, such as reading different known
files or searching separate academic sources with fixed queries.
Keep dependent work sequential: search before fetching a paper, read
before editing, and run an experiment before interpreting it.
Do not batch bash or agent calls; inspect
their result before continuing. The runtime only detects overlapping
paths among file tools, so decide other dependencies carefully and
compare, deduplicate, and cite research results after parallel calls return."""

_PROMPTS_DIR = Path(__file__).parent / "prompts"

# Runtime data fillers for the dynamic block templates.
_SKILLS_PLACEHOLDER = "{skills}"
_MEMORY_SECTIONS_PLACEHOLDER = "{sections}"


def _load_block(filename: str, fallback: str = "") -> str:
    """Read a prompt block file, falling back to a default when missing."""
    path = _PROMPTS_DIR / filename
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return fallback


def _static_sections() -> list[str]:
    """Static guidance blocks, in display order."""
    sections = [
        _load_block("01-soul.md", _ROLE_FALLBACK),
        _load_block("02-rules.md"),
        _load_block("03-parallel-tools.md", _PARALLEL_TOOLS),
        _load_block("04-scratchpad.md"),
    ]
    return [s for s in sections if s]


_MEMORY_SECTION_LABELS = {
    "user": "User profile and collaboration preferences",
    "feedback": "Methodological corrections and confirmed approaches",
    "project": "Project context, decisions, and open items",
}


def _memory_section() -> str:
    # The three user/feedback/project Markdown memory files are retired. L1 atom
    # memory is injected on-demand via the background extractor instead; a
    # dedicated user persona file is planned separately.
    return ""


def _skills_section(skills) -> str:
    if not skills:
        return ""

    skill_items = "\n".join(
        f"  - {_oneline(skill.name)}: {_oneline(skill.description)}"
        for skill in skills
    )
    template = _load_block("05-skills.md")
    if not template:
        return ""
    return template.replace(_SKILLS_PLACEHOLDER, skill_items)


def _environment_section() -> str:
    uname = platform.uname()
    template = _load_block("07-environment.md")
    if not template:
        return ""
    return template.format(
        cwd=os.getcwd(),
        os=f"{uname.system} {uname.release} ({uname.machine})",
        python=platform.python_version(),
    )


def system_prompt(tools, skills=None) -> str:
    memory_section = _memory_section()

    memory_template = _load_block("06-memory.md")
    if memory_template and memory_section:
        memory_block = memory_template.replace(_MEMORY_SECTIONS_PLACEHOLDER, memory_section)
    else:
        memory_block = ""

    dynamic = [
        _skills_section(skills or []),
        memory_block,
        _environment_section(),
    ]
    dynamic = [s for s in dynamic if s]

    sections = _static_sections() + dynamic
    return "\n\n".join(sections)


def _oneline(text: str) -> str:
    """Collapse newlines so each skill stays on one line."""
    return text.replace("\n", " ").replace("\r", " ").strip()