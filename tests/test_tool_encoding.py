import sys
import tempfile
import unittest
from pathlib import Path

from folium.tools import get_tool


class ToolEncodingTests(unittest.TestCase):
    def test_read_file_prefers_utf8(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "中文.md"
            path.write_text("科研智能体\n第二行\n", encoding="utf-8")

            result = get_tool("read_file").execute(file_path=str(path))

            self.assertIn("科研智能体", result)
            self.assertNotIn("绉", result)
            self.assertNotIn("鏅", result)

    def test_bash_decodes_utf8_output(self):
        command = f'"{sys.executable}" -X utf8 -c "print(\'中文输出\')"'

        result = get_tool("bash").execute(command=command)

        self.assertIn("中文输出", result)


if __name__ == "__main__":
    unittest.main()
