import tempfile
import unittest
from pathlib import Path

from folium.prompt import system_prompt
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
            self.assertIn("<name>experiment-runner</name>", prompt)
            self.assertIn("Use for experiments.", prompt)
            self.assertIn("skills/<name>/SKILL.md", prompt)
            self.assertNotIn(str(skill_file), prompt)

    def test_missing_skills_directory_returns_empty_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing-skills"

            self.assertEqual(load_skills(missing), [])


if __name__ == "__main__":
    unittest.main()
