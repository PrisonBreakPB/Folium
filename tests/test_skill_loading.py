import unittest
from pathlib import Path
from folium.skills.parser import parse_skill_file
from folium.skills.storage import load_skills
from folium.agent import Agent


SKILLS_DIR = Path(__file__).parent.parent / "skills"


class SkillLoadingTests(unittest.TestCase):
    def test_parse_only_reads_frontmatter(self):
        """parse_skill_file should only extract name and description from frontmatter,
        not load the entire SKILL.md content."""
        skill_file = SKILLS_DIR / "control-literature-search" / "SKILL.md"
        skill = parse_skill_file(skill_file)

        self.assertIsNotNone(skill)
        self.assertEqual(skill.name, "control-literature-search")
        self.assertIn("literature", skill.description.lower())

        # Skill object should NOT have a 'content' attribute
        self.assertFalse(hasattr(skill, 'content'))

    def test_skill_object_has_minimal_fields(self):
        """Skill dataclass should only contain name, description, and paths."""
        skill_file = SKILLS_DIR / "control-literature-search" / "SKILL.md"
        skill = parse_skill_file(skill_file)

        fields = set(vars(skill).keys())
        expected = {"name", "description", "skill_dir", "skill_file"}
        self.assertEqual(fields, expected)

    def test_load_skills_returns_list_without_content(self):
        """load_skills should return skills without loading full content."""
        skills = load_skills(SKILLS_DIR)

        self.assertGreater(len(skills), 0)
        for skill in skills:
            self.assertIsInstance(skill.name, str)
            self.assertIsInstance(skill.description, str)
            self.assertTrue(len(skill.name) > 0)
            self.assertTrue(len(skill.description) > 0)
            # No content loaded
            self.assertFalse(hasattr(skill, 'content'))

    def test_skill_file_path_is_correct(self):
        """Each skill should point to its SKILL.md file."""
        skills = load_skills(SKILLS_DIR)

        for skill in skills:
            self.assertTrue(skill.skill_file.exists())
            self.assertEqual(skill.skill_file.name, "SKILL.md")
            self.assertEqual(skill.skill_dir.name, skill.name)


class SkillActivationTests(unittest.TestCase):
    def setUp(self):
        self.agent = Agent(llm=None)

    def test_slash_command_activates_skill(self):
        """Input starting with /skill-name should inject full SKILL.md content."""
        result = self.agent._try_activate_skill("/control-literature-search find papers on LQR")
        self.assertIn("[Activated skill: control-literature-search]", result)
        self.assertIn("[User request]", result)
        self.assertIn("find papers on LQR", result)
        self.assertIn("## Workflow", result)  # SKILL.md content should be included

    def test_slash_command_without_match_returns_original(self):
        """Input with unknown skill name should return original input."""
        user_input = "/unknown-skill do something"
        result = self.agent._try_activate_skill(user_input)
        self.assertEqual(result, user_input)

    def test_no_slash_returns_original(self):
        """Input without slash should return original input."""
        user_input = "help me find papers"
        result = self.agent._try_activate_skill(user_input)
        self.assertEqual(result, user_input)

    def test_slash_only_returns_original(self):
        """Input with just / should return original input."""
        user_input = "/"
        result = self.agent._try_activate_skill(user_input)
        self.assertEqual(result, user_input)

    def test_slash_with_remaining_text(self):
        """Slash command should separate skill name from remaining text."""
        result = self.agent._try_activate_skill("/control-literature-search")
        self.assertIn("[Activated skill: control-literature-search]", result)
        self.assertIn("[User request]", result)


if __name__ == "__main__":
    unittest.main()
