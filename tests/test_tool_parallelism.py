import unittest

from folium.agent import _NEVER_PARALLEL_TOOLS, _should_parallelize_tool_batch
from folium.llm import ToolCall


def _tool_calls(*names: str) -> list[ToolCall]:
    return [
        ToolCall(id=f"call_{index}", name=name, arguments={})
        for index, name in enumerate(names, 1)
    ]


class ToolParallelismTests(unittest.TestCase):
    def test_single_tool_call_runs_sequentially(self):
        self.assertFalse(_should_parallelize_tool_batch(_tool_calls("web_search")))

    def test_write_edit_and_bash_batches_run_sequentially(self):
        for tool_name in ("write_file", "edit_file", "bash"):
            with self.subTest(tool_name=tool_name):
                self.assertFalse(
                    _should_parallelize_tool_batch(_tool_calls("web_search", tool_name))
                )

    def test_never_parallel_tool_set_is_empty_by_default(self):
        self.assertEqual(_NEVER_PARALLEL_TOOLS, set())

    def test_independent_tools_run_in_parallel(self):
        tool_calls = _tool_calls("paper_search", "arxiv_search", "web_fetch")

        self.assertTrue(_should_parallelize_tool_batch(tool_calls))


if __name__ == "__main__":
    unittest.main()
