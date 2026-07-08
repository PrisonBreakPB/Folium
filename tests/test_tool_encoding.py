import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from folium.tools.bash import BashTool
from folium.tools.read import ReadFileTool


class ToolEncodingTests(unittest.TestCase):
    def setUp(self):
        self._backend_patch = mock.patch.dict(
            "os.environ",
            {"FOLIUM_BASH_BACKEND": "local"},
            clear=False,
        )
        self._backend_patch.start()

    def tearDown(self):
        self._backend_patch.stop()

    def test_read_file_prefers_utf8(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "中文.md"
            path.write_text("科研智能体\n第二行\n", encoding="utf-8")

            result = ReadFileTool().execute(file_path=str(path))

            self.assertIn("科研智能体", result)
            self.assertNotIn("绉", result)
            self.assertNotIn("鏅", result)

    def test_bash_decodes_utf8_output(self):
        command = f'"{sys.executable}" -X utf8 -c "print(\'中文输出\')"'

        result = BashTool().execute(command=command)

        self.assertIn("中文输出", result)

    def test_bash_decodes_gbk_output(self):
        command = (
            f'"{sys.executable}" -c '
            '"import sys; sys.stdout.buffer.write(\'中文输出\'.encode(\'gbk\'))"'
        )

        result = BashTool().execute(command=command)

        self.assertIn("中文输出", result)


if __name__ == "__main__":
    unittest.main()
