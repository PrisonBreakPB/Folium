import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from folium.prompt import MAX_MEMORY_CHARS, system_prompt
from folium.skills import load_skills
from folium.skills.parser import parse_skill_file


class SkillTests(unittest.TestCase):
    def test_parse_skill_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "literature-review"
            skill_dir.mkdir()
            skill_file = skill_dir / "SKILL.md"
            skill_file.write_text(
                """---
name: literature-review
description: Use for literature reviews.
---

# Literature Review
""",
                encoding="utf-8",
            )

            skill = parse_skill_file(skill_file)

            self.assertIsNotNone(skill)
            self.assertEqual(skill.name, "literature-review")
            self.assertEqual(skill.description, "Use for literature reviews.")
            self.assertEqual(skill.skill_file, skill_file)

    def test_parse_skill_file_rejects_name_directory_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "wrong-name"
            skill_dir.mkdir()
            skill_file = skill_dir / "SKILL.md"
            skill_file.write_text(
                """---
name: literature-review
description: Use for literature reviews.
---
""",
                encoding="utf-8",
            )

            with self.assertLogs("folium.skills.parser", level="WARNING"):
                self.assertIsNone(parse_skill_file(skill_file))

    def test_load_skills_scans_single_skills_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            skill_dir = root / "latex-writing"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                """---
name: latex-writing
description: Use for LaTeX writing.
---
""",
                encoding="utf-8",
            )

            skills = load_skills(root)

            self.assertEqual([skill.name for skill in skills], ["latex-writing"])

    def test_system_prompt_includes_skill_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "experiment-runner"
            skill_dir.mkdir()
            skill_file = skill_dir / "SKILL.md"
            skill_file.write_text(
                """---
name: experiment-runner
description: Use for experiments.
---
""",
                encoding="utf-8",
            )
            skill = parse_skill_file(skill_file)

            prompt = system_prompt([], [skill])

            self.assertIn("<skill_system>", prompt)
            self.assertIn("- experiment-runner: Use for experiments.", prompt)
            self.assertIn("skills/<name>/SKILL.md", prompt)
            self.assertNotIn(str(skill_file), prompt)

    def test_system_prompt_includes_memory_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_file = Path(tmp) / "memory.md"
            memory_file.write_text("# User Memory\n- Prefer Chinese.\n", encoding="utf-8")
            skill = SimpleNamespace(name="test-skill", description="Test skill.")

            with mock.patch("folium.prompt.MEMORY_FILE", memory_file):
                prompt = system_prompt([], [skill])

            self.assertIn("# Long-Term Memory", prompt)
            self.assertIn("# User Memory", prompt)
            self.assertIn("- Prefer Chinese.", prompt)
            self.assertLess(prompt.index("# Skills"), prompt.index("# Long-Term Memory"))
            self.assertLess(prompt.index("# Long-Term Memory"), prompt.index("# Environment"))

    def test_system_prompt_limits_memory_to_2000_characters(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory_file = Path(tmp) / "memory.md"
            memory_file.write_text("a" * MAX_MEMORY_CHARS + "TRUNCATED", encoding="utf-8")

            with mock.patch("folium.prompt.MEMORY_FILE", memory_file):
                prompt = system_prompt([], [])

            self.assertIn("a" * MAX_MEMORY_CHARS, prompt)
            self.assertNotIn("TRUNCATED", prompt)

    def test_missing_skills_directory_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing-skills"

            self.assertEqual(load_skills(missing), [])


if __name__ == "__main__":
    unittest.main()
