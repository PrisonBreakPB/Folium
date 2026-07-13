import unittest

from folium.agent import _NEVER_PARALLEL_TOOLS, _should_parallelize_tool_batch
from folium.llm import ToolCall
from folium.prompt import _PARALLEL_TOOLS


def _tool_calls(*names: str) -> list[ToolCall]:
    return [
        ToolCall(id=f"call_{index}", name=name, arguments={})
        for index, name in enumerate(names, 1)
    ]


def _file_call(identifier: str, name: str, file_path: str) -> ToolCall:
    return ToolCall(
        id=identifier,
        name=name,
        arguments={"file_path": file_path},
    )


class ToolParallelismTests(unittest.TestCase):
    def test_single_tool_call_runs_sequentially(self):
        self.assertFalse(_should_parallelize_tool_batch(_tool_calls("web_search")))

    def test_bash_and_agent_batches_run_sequentially(self):
        for tool_name in ("bash", "agent"):
            with self.subTest(tool_name=tool_name):
                self.assertFalse(
                    _should_parallelize_tool_batch(_tool_calls("web_search", tool_name))
                )

    def test_independent_file_operations_run_in_parallel(self):
        tool_calls = [
            _file_call("call_1", "write_file", "src/a.py"),
            _file_call("call_2", "edit_file", "src/b.py"),
        ]

        self.assertTrue(_should_parallelize_tool_batch(tool_calls))

    def test_same_file_reads_run_in_parallel(self):
        tool_calls = [
            _file_call("call_1", "read_file", "README.md"),
            _file_call("call_2", "read_file", "./README.md"),
        ]

        self.assertTrue(_should_parallelize_tool_batch(tool_calls))

    def test_overlapping_paths_with_a_write_run_sequentially(self):
        cases = {
            "same_file": [
                _file_call("call_1", "read_file", "README.md"),
                _file_call("call_2", "write_file", "README.md"),
            ],
            "normalized_same_file": [
                _file_call("call_1", "write_file", "src/../README.md"),
                _file_call("call_2", "edit_file", "README.md"),
            ],
            "parent_child": [
                _file_call("call_1", "write_file", "generated"),
                _file_call("call_2", "edit_file", "generated/config.py"),
            ],
        }

        for name, tool_calls in cases.items():
            with self.subTest(name=name):
                self.assertFalse(_should_parallelize_tool_batch(tool_calls))

    def test_file_tool_without_a_valid_path_runs_sequentially(self):
        tool_calls = [
            _file_call("call_1", "read_file", "README.md"),
            ToolCall(id="call_2", name="write_file", arguments={}),
        ]

        self.assertFalse(_should_parallelize_tool_batch(tool_calls))

    def test_memory_tool_never_runs_in_parallel(self):
        self.assertIn("memory", _NEVER_PARALLEL_TOOLS)
        self.assertFalse(_should_parallelize_tool_batch(_tool_calls("web_search", "memory")))

    def test_prompt_matches_serial_tool_policy(self):
        self.assertIn("Do not batch bash or agent calls", _PARALLEL_TOOLS)
        self.assertNotIn("write_file, edit_file", _PARALLEL_TOOLS)

    def test_independent_tools_run_in_parallel(self):
        tool_calls = _tool_calls("paper_search", "arxiv_search", "web_fetch")

        self.assertTrue(_should_parallelize_tool_batch(tool_calls))


if __name__ == "__main__":
    unittest.main()
