import json
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from folium import session as session_module
from folium.web import server


class DummyAgent:
    def __init__(self):
        self.messages = [{"role": "user", "content": "hello"}]
        self.transcript = [{"role": "user", "content": "hello"}]
        self.session_id = None
        self._system = "dummy system prompt"
        self.llm = DummyLLM()
        self.todo_manager = DummyTodoManager()
        self.rounds_since_todo = 0
        self.skills = [
            SimpleNamespace(
                name="literature-review",
                description="Use for literature reviews.",
                skill_file=Path("skills/literature-review/SKILL.md"),
            )
        ]

    def reset(self):
        self.messages.clear()
        self.transcript.clear()
        self.reset_todos()

    def reset_todos(self):
        self.rounds_since_todo = 0
        self.todo_manager.reset()


class DummyTodoManager:
    def __init__(self):
        self.items = []

    def snapshot(self):
        return [dict(item) for item in self.items]

    def render(self):
        return "No todos." if not self.items else "todos"

    def reset(self):
        self.items.clear()


class DummyConfig:
    model = "test-model"


class DummyLLM:
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_cached_tokens = 0
    last_prompt_tokens = 0
    last_completion_tokens = 0


class WebServerSessionTests(unittest.TestCase):
    def test_auto_save_only_updates_dirty_session(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            old_dir = session_module.SESSIONS_DIR
            old_state = dict(server._state)
            try:
                session_module.SESSIONS_DIR = Path(tmp)
                server._state.update({
                    "agent": DummyAgent(),
                    "config": DummyConfig(),
                    "session_id": "session_test",
                    "dirty": True,
                })
                server._auto_save()

                path = Path(tmp) / "session_test.json"
                first_data = json.loads(path.read_text(encoding="utf-8"))
                first = first_data["updated_at"]
                self.assertEqual(first_data["transcript"], [{"role": "user", "content": "hello"}])
                time.sleep(1.1)

                server._auto_save()
                second = json.loads(path.read_text(encoding="utf-8"))["updated_at"]

                self.assertEqual(first, second)
            finally:
                session_module.SESSIONS_DIR = old_dir
                server._state.clear()
                server._state.update(old_state)

    def test_skills_command_lists_loaded_skills(self):
        from fastapi.testclient import TestClient

        old_state = dict(server._state)
        try:
            server._state.update({
                "agent": DummyAgent(),
                "config": DummyConfig(),
                "session_id": None,
                "dirty": False,
            })
            client = TestClient(server.app)
            resp = client.post("/command", json={"command": "/skills"})

            self.assertEqual(resp.status_code, 200)
            result = resp.json()["result"]
            self.assertIn("Available skills:", result)
            self.assertIn("literature-review", result)
            self.assertIn("SKILL.md", result)
        finally:
            server._state.clear()
            server._state.update(old_state)

    def test_todos_endpoint_returns_current_todos(self):
        from fastapi.testclient import TestClient

        old_state = dict(server._state)
        try:
            agent = DummyAgent()
            agent.todo_manager.items = [
                {"id": "1", "text": "Inspect", "status": "in_progress"}
            ]
            server._state.update({
                "agent": agent,
                "config": DummyConfig(),
                "session_id": None,
                "dirty": False,
            })
            client = TestClient(server.app)
            resp = client.get("/todos")

            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json()["items"][0]["text"], "Inspect")
        finally:
            server._state.clear()
            server._state.update(old_state)

    def test_switch_conversation_resets_todos(self):
        from fastapi.testclient import TestClient
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            old_dir = session_module.SESSIONS_DIR
            old_state = dict(server._state)
            try:
                session_module.SESSIONS_DIR = Path(tmp)
                session_module.save_session(
                    [{"role": "user", "content": "compressed"}],
                    "test-model",
                    "session_a",
                    transcript=[{"role": "user", "content": "saved"}],
                )
                agent = DummyAgent()
                agent.todo_manager.items = [
                    {"id": "1", "text": "Old", "status": "in_progress"}
                ]
                server._state.update({
                    "agent": agent,
                    "config": DummyConfig(),
                    "session_id": None,
                    "dirty": False,
                })
                client = TestClient(server.app)
                resp = client.post("/switch", json={"session_id": "session_a"})

                self.assertEqual(resp.status_code, 200)
                self.assertEqual(resp.json()["messages"], [{"role": "user", "content": "saved"}])
                self.assertEqual(agent.messages, [{"role": "user", "content": "compressed"}])
                self.assertEqual(agent.transcript, [{"role": "user", "content": "saved"}])
                self.assertEqual(agent.todo_manager.snapshot(), [])
                self.assertEqual(agent.rounds_since_todo, 0)
            finally:
                session_module.SESSIONS_DIR = old_dir
                server._state.clear()
                server._state.update(old_state)


if __name__ == "__main__":
    unittest.main()
