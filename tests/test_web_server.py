import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from folium import database
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


class RunnerLLM:
    def __init__(self, model, api_key, base_url=None, **kwargs):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.extra = kwargs


class WebServerSessionTests(unittest.TestCase):
    def test_memory_runner_uses_the_main_agent_model(self):
        agent = SimpleNamespace(llm=RunnerLLM("deepseek-v4-pro", "main-key"))
        config = SimpleNamespace(
            api_key="config-key",
            base_url="https://example.test/v1",
            temperature=0.2,
            memory_maintenance_max_tokens=2000,
            memory_maintenance_max_steps=5,
        )

        runner = server._new_memory_maintenance_runner(agent, config)

        self.assertEqual(runner.llm.model, "deepseek-v4-pro")
        self.assertEqual(runner.llm.api_key, "config-key")
        self.assertEqual(runner.llm.extra["max_tokens"], 2000)

    def test_background_maintenance_receives_completed_turn_usage_snapshot(self):
        class CapturingScheduler:
            def __init__(self):
                self.kwargs = None

            async def on_turn_completed(self, **kwargs):
                self.kwargs = kwargs

        old_state = dict(server._state)
        scheduler = CapturingScheduler()
        completion = {
            "session_id": "session_test",
            "messages": [{"role": "system", "content": "system"}],
            "visible_tools": [],
            "main_agent_used_memory": False,
            "main_prompt_tokens": 321,
            "main_completion_tokens": 123,
            "main_request_matches_memory_context": True,
        }
        try:
            server._state["memory_maintenance"] = scheduler
            with mock.patch("folium.web.server._auto_save"):
                import asyncio

                asyncio.run(server._after_chat_response(completion))
            self.assertEqual(scheduler.kwargs["main_prompt_tokens"], 321)
            self.assertEqual(scheduler.kwargs["main_completion_tokens"], 123)
            self.assertTrue(scheduler.kwargs["main_request_matches_memory_context"])
        finally:
            server._state.clear()
            server._state.update(old_state)

    def test_auto_save_only_updates_dirty_session(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            old_path = database.DB_PATH
            old_state = dict(server._state)
            try:
                database.DB_PATH = Path(tmp) / "folium.db"
                server._state.update({
                    "agent": DummyAgent(),
                    "config": DummyConfig(),
                    "session_id": "session_test",
                    "dirty": True,
                })
                server._auto_save()

                loaded = session_module.load_session("session_test")
                self.assertIsNotNone(loaded)
                self.assertEqual(loaded[2], [{"role": "user", "content": "hello"}])
                server._auto_save()
                self.assertFalse(server._state["dirty"])
            finally:
                database.DB_PATH = old_path
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
            old_path = database.DB_PATH
            old_state = dict(server._state)
            try:
                database.DB_PATH = Path(tmp) / "folium.db"
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
                with mock.patch("folium.web.server.reset_current_session") as reset_session:
                    resp = client.post("/switch", json={"session_id": "session_a"})

                self.assertEqual(resp.status_code, 200)
                self.assertEqual(resp.json()["messages"], [{"role": "user", "content": "saved"}])
                self.assertEqual(agent.messages, [{"role": "user", "content": "compressed"}])
                self.assertEqual(agent.transcript, [{"role": "user", "content": "saved"}])
                self.assertEqual(agent.todo_manager.snapshot(), [])
                self.assertEqual(agent.rounds_since_todo, 0)
                reset_session.assert_called_once()
            finally:
                database.DB_PATH = old_path
                server._state.clear()
                server._state.update(old_state)

    def test_new_and_reset_commands_reset_sandbox_session(self):
        from fastapi.testclient import TestClient

        old_state = dict(server._state)
        try:
            server._state.update({
                "agent": DummyAgent(),
                "config": DummyConfig(),
                "session_id": "session_test",
                "dirty": True,
            })
            client = TestClient(server.app)
            with mock.patch("folium.web.server.reset_current_session") as reset_session:
                new_response = client.post("/new")
                self.assertEqual(new_response.status_code, 200)
                reset_session.assert_called_once()

                reset_session.reset_mock()
                reset_response = client.post("/command", json={"command": "/reset"})
                self.assertEqual(reset_response.status_code, 200)
                reset_session.assert_called_once()
        finally:
            server._state.clear()
            server._state.update(old_state)

    def test_delete_current_conversation_resets_sandbox_session(self):
        from fastapi.testclient import TestClient
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            old_path = database.DB_PATH
            old_state = dict(server._state)
            try:
                database.DB_PATH = Path(tmp) / "folium.db"
                session_module.save_session([], "test-model", "session_a", transcript=[])
                server._state.update({
                    "agent": DummyAgent(),
                    "config": DummyConfig(),
                    "session_id": "session_a",
                    "dirty": False,
                })
                client = TestClient(server.app)
                with mock.patch("folium.web.server.reset_current_session") as reset_session:
                    response = client.post("/delete", json={"session_id": "session_a"})

                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.json()["deleted_current"])
                reset_session.assert_called_once()
            finally:
                database.DB_PATH = old_path
                server._state.clear()
                server._state.update(old_state)

    def test_run_server_defaults_web_to_copy_workspace(self):
        old_state = dict(server._state)
        uvicorn = mock.Mock()
        try:
            with (
                mock.patch.dict("os.environ", {}, clear=True),
                mock.patch.dict(sys.modules, {"uvicorn": uvicorn}),
                mock.patch("folium.web.server.reset_current_session"),
            ):
                server.run_server(DummyAgent(), DummyConfig())

                self.assertEqual(os.environ["FOLIUM_SANDBOX_WORKSPACE_MODE"], "copy")
                uvicorn.run.assert_called_once_with(
                    server.app, host="0.0.0.0", port=8000, log_level="info"
                )
        finally:
            server._state.clear()
            server._state.update(old_state)

    def test_run_server_respects_explicit_workspace_mode(self):
        old_state = dict(server._state)
        uvicorn = mock.Mock()
        try:
            with (
                mock.patch.dict(
                    "os.environ",
                    {"FOLIUM_SANDBOX_WORKSPACE_MODE": "host"},
                    clear=True,
                ),
                mock.patch.dict(sys.modules, {"uvicorn": uvicorn}),
                mock.patch("folium.web.server.reset_current_session"),
            ):
                server.run_server(DummyAgent(), DummyConfig())

                self.assertEqual(os.environ["FOLIUM_SANDBOX_WORKSPACE_MODE"], "host")
        finally:
            server._state.clear()
            server._state.update(old_state)


if __name__ == "__main__":
    unittest.main()
