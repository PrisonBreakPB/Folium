import subprocess
import tempfile
import unittest
import json
from pathlib import Path
from unittest import mock

from folium.observability.config import ObservabilityConfig
from folium.observability.context import Observer, _observer_var, observe_trace
from folium.sandbox.docker import DockerSandboxExecutor
from folium.sandbox.local import LocalCommandExecutor
from folium.sandbox.session import SandboxSession, reset_current_session
from folium.tools.bash import BashTool
from folium.tools.edit import EditFileTool
from folium.tools.glob_tool import GlobTool
from folium.tools.grep import GrepTool
from folium.tools.read import ReadFileTool
from folium.tools.write import WriteFileTool


class SandboxTests(unittest.TestCase):
    def tearDown(self):
        reset_current_session()

    def test_bash_tool_defaults_to_docker_backend(self):
        with mock.patch.dict("os.environ", {}, clear=True):
            tool = BashTool()

        self.assertIsInstance(tool.executor, DockerSandboxExecutor)

    def test_bash_tool_uses_local_backend_from_env(self):
        with mock.patch.dict("os.environ", {"FOLIUM_BASH_BACKEND": "local"}, clear=True):
            tool = BashTool()
            executor = tool.executor

        self.assertIsInstance(executor, LocalCommandExecutor)

    def test_bash_tool_defaults_to_host_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            host = Path(tmp) / "repo"
            host.mkdir()

            with (
                mock.patch.dict("os.environ", {"FOLIUM_HOST_WORKSPACE": str(host)}, clear=True),
            ):
                tool = BashTool()
                executor = tool.executor

            self.assertIsNone(executor.session)
            self.assertEqual(executor.workspace, host.resolve())

    def test_bash_tool_can_use_copy_workspace_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            host = Path(tmp) / "repo"
            root = Path(tmp) / "sessions"
            host.mkdir()
            session = SandboxSession(host_workspace=str(host), root_dir=str(root))

            with (
                mock.patch.dict("os.environ", {"FOLIUM_SANDBOX_WORKSPACE_MODE": "copy"}, clear=True),
                mock.patch("folium.sandbox.docker.get_current_session", return_value=session),
            ):
                tool = BashTool()
                executor = tool.executor

            self.assertIs(executor.session, session)
            self.assertEqual(executor.workspace, session.workspace.resolve())

    def test_docker_executor_starts_container_and_execs_command(self):
        run_calls = []
        popen_calls = []

        def fake_run(args, stdout=None, stderr=None):
            run_calls.append(args)
            return subprocess.CompletedProcess(args, 0, stdout=b"container123\n", stderr=b"")

        class FakeProc:
            returncode = 0

            def communicate(self, timeout=None):
                return b"hello\n", b""

        def fake_popen(args, stdout=None, stderr=None):
            popen_calls.append(args)
            return FakeProc()

        with tempfile.TemporaryDirectory() as tmp:
            executor = DockerSandboxExecutor(workspace=tmp, image="python:test")

            with (
                mock.patch("folium.sandbox.docker.shutil.which", return_value="docker"),
                mock.patch("folium.sandbox.docker.subprocess.run", side_effect=fake_run),
                mock.patch("folium.sandbox.docker.subprocess.Popen", side_effect=fake_popen),
            ):
                result = executor.run("echo hello")

            self.assertEqual(result, "hello")
            self.assertEqual(run_calls[0][:7], [
                "docker",
                "run",
                "-d",
                "--name",
                executor.container_name,
                "--network",
                "bridge",
            ])
            self.assertIn(f"{Path(tmp).resolve()}:/workspace", run_calls[0])
            self.assertEqual(popen_calls[0], [
                "docker",
                "exec",
                "-w",
                "/workspace",
                "container123",
                "sh",
                "-lc",
                "echo hello",
            ])

    def test_docker_executor_records_sandbox_events(self):
        def fake_run(args, stdout=None, stderr=None):
            return subprocess.CompletedProcess(args, 0, stdout=b"container123\n", stderr=b"")

        class FakeProc:
            returncode = 0

            def communicate(self, timeout=None):
                return b"hello\n", b""

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "folium.db"
            observer = Observer(ObservabilityConfig(database_path=db_path))
            token = _observer_var.set(observer)
            try:
                executor = DockerSandboxExecutor(workspace=str(Path(tmp) / "workspace"), image="python:test")
                with (
                    observe_trace("sandbox test"),
                    mock.patch("folium.sandbox.docker.shutil.which", return_value="docker"),
                    mock.patch("folium.sandbox.docker.subprocess.run", side_effect=fake_run),
                    mock.patch("folium.sandbox.docker.subprocess.Popen", return_value=FakeProc()),
                ):
                    result = executor.run("echo hello")
            finally:
                _observer_var.reset(token)

            self.assertEqual(result, "hello")
            from folium.database import get_connection

            with get_connection(db_path) as conn:
                rows = conn.execute(
                    "SELECT name, payload_json FROM trace_events WHERE event_type = 'sandbox_event'"
                ).fetchall()
            actions = [row["name"] for row in rows]
            self.assertIn("container_started", actions)
            self.assertIn("command_finished", actions)
            command_event = next(row for row in rows if row["name"] == "command_finished")
            metadata = json.loads(command_event["payload_json"])
            self.assertEqual(metadata["returncode"], 0)
            self.assertIn("command_hash", metadata)

    def test_docker_executor_reports_missing_docker(self):
        with tempfile.TemporaryDirectory() as tmp:
            executor = DockerSandboxExecutor(workspace=tmp)

            with mock.patch("folium.sandbox.docker.shutil.which", return_value=None):
                result = executor.run("echo hello")

        self.assertIn("docker executable was not found", result)

    def test_docker_executor_blocks_dangerous_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            executor = DockerSandboxExecutor(workspace=tmp)
            result = executor.run("rm -rf /")

        self.assertIn("Blocked", result)

    def test_session_copies_workspace_and_excludes_heavy_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            host = Path(tmp) / "repo"
            root = Path(tmp) / "sessions"
            host.mkdir()
            (host / "README.md").write_text("hello", encoding="utf-8")
            (host / ".git").mkdir()
            (host / ".git" / "config").write_text("secret", encoding="utf-8")
            (host / ".venv").mkdir()
            (host / ".venv" / "pyvenv.cfg").write_text("venv", encoding="utf-8")
            (host / "pkg").mkdir()
            (host / "pkg" / "mod.py").write_text("print('ok')", encoding="utf-8")

            session = SandboxSession(host_workspace=str(host), root_dir=str(root))
            session.prepare()

            self.assertEqual((session.workspace / "README.md").read_text(encoding="utf-8"), "hello")
            self.assertTrue((session.workspace / "pkg" / "mod.py").exists())
            self.assertFalse((session.workspace / ".git").exists())
            self.assertFalse((session.workspace / ".venv").exists())

    def test_session_excludes_sensitive_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            host = Path(tmp) / "repo"
            root = Path(tmp) / "sessions"
            host.mkdir()
            (host / "README.md").write_text("hello", encoding="utf-8")
            (host / ".env").write_text("OPENAI_API_KEY=secret\n", encoding="utf-8")
            (host / ".env.local").write_text("TOKEN=secret\n", encoding="utf-8")
            (host / "id_rsa").write_text("private key\n", encoding="utf-8")
            (host / "client.pem").write_text("pem\n", encoding="utf-8")
            (host / "credentials.json").write_text('{"token":"secret"}\n', encoding="utf-8")
            nested = host / "nested"
            nested.mkdir()
            (nested / "service-account-prod.json").write_text('{"key":"secret"}\n', encoding="utf-8")
            (nested / "mod.py").write_text("print('ok')\n", encoding="utf-8")

            session = SandboxSession(host_workspace=str(host), root_dir=str(root))
            session.prepare()

            self.assertTrue((session.workspace / "README.md").exists())
            self.assertTrue((session.workspace / "nested" / "mod.py").exists())
            self.assertFalse((session.workspace / ".env").exists())
            self.assertFalse((session.workspace / ".env.local").exists())
            self.assertFalse((session.workspace / "id_rsa").exists())
            self.assertFalse((session.workspace / "client.pem").exists())
            self.assertFalse((session.workspace / "credentials.json").exists())
            self.assertFalse((session.workspace / "nested" / "service-account-prod.json").exists())

    def test_file_tools_use_host_workspace_in_docker_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            host = Path(tmp) / "repo"
            host.mkdir()
            host_file = host / "note.txt"
            host_file.write_text("old\n", encoding="utf-8")

            with (
                mock.patch.dict(
                    "os.environ",
                    {"FOLIUM_BASH_BACKEND": "docker", "FOLIUM_HOST_WORKSPACE": str(host)},
                    clear=False,
                ),
            ):
                read_result = ReadFileTool().execute("note.txt")
                write_result = WriteFileTool().execute("created.txt", "new\n")
                edit_result = EditFileTool().execute("note.txt", "old", "sandbox")

            self.assertIn("old", read_result)
            self.assertIn("Wrote", write_result.content)
            self.assertIn("Edited", edit_result.content)
            self.assertEqual(host_file.read_text(encoding="utf-8"), "sandbox\n")
            self.assertEqual((host / "created.txt").read_text(), "new\n")

    def test_file_tools_use_copy_workspace_with_local_bash_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            host = Path(tmp) / "repo"
            root = Path(tmp) / "sessions"
            host.mkdir()
            host_file = host / "note.txt"
            host_file.write_text("old\n", encoding="utf-8")
            session = SandboxSession(host_workspace=str(host), root_dir=str(root))

            with (
                mock.patch.dict(
                    "os.environ",
                    {"FOLIUM_BASH_BACKEND": "local", "FOLIUM_SANDBOX_WORKSPACE_MODE": "copy"},
                    clear=False,
                ),
                mock.patch("folium.sandbox.filesystem.get_current_session", return_value=session),
            ):
                write_result = WriteFileTool().execute("created.txt", "new\n")
                edit_result = EditFileTool().execute("note.txt", "old", "sandbox")

            self.assertIn("Wrote", write_result.content)
            self.assertIn("Edited", edit_result.content)
            self.assertEqual(host_file.read_text(encoding="utf-8"), "old\n")
            self.assertEqual((session.workspace / "note.txt").read_text(), "sandbox\n")
            self.assertEqual((session.workspace / "created.txt").read_text(), "new\n")

    def test_glob_uses_sandbox_workspace_with_local_bash_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            host = Path(tmp) / "repo"
            root = Path(tmp) / "sessions"
            host.mkdir()
            (host / "host_only.py").write_text("print('host')\n", encoding="utf-8")
            session = SandboxSession(host_workspace=str(host), root_dir=str(root))
            session.prepare()
            (session.workspace / "host_only.py").unlink()
            (session.workspace / "sandbox_only.py").write_text("print('sandbox')\n", encoding="utf-8")

            with (
                mock.patch.dict(
                    "os.environ",
                    {"FOLIUM_BASH_BACKEND": "local", "FOLIUM_SANDBOX_WORKSPACE_MODE": "copy"},
                    clear=False,
                ),
                mock.patch("folium.sandbox.filesystem.get_current_session", return_value=session),
            ):
                result = GlobTool().execute("*.py")

            self.assertIn("sandbox_only.py", result)
            self.assertNotIn("host_only.py", result)

    def test_grep_uses_sandbox_workspace_with_local_bash_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            host = Path(tmp) / "repo"
            root = Path(tmp) / "sessions"
            host.mkdir()
            note = host / "note.txt"
            note.write_text("host token\n", encoding="utf-8")
            session = SandboxSession(host_workspace=str(host), root_dir=str(root))
            session.prepare()
            (session.workspace / "note.txt").write_text("sandbox token\n", encoding="utf-8")

            with (
                mock.patch.dict(
                    "os.environ",
                    {"FOLIUM_BASH_BACKEND": "local", "FOLIUM_SANDBOX_WORKSPACE_MODE": "copy"},
                    clear=False,
                ),
                mock.patch("folium.sandbox.filesystem.get_current_session", return_value=session),
            ):
                result = GrepTool().execute("sandbox token", path="note.txt")

            self.assertIn("sandbox token", result)
            self.assertNotIn("host token", result)

if __name__ == "__main__":
    unittest.main()
