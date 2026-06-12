import json
import time
import unittest

from folium import session as session_module
from folium.web import server


class DummyAgent:
    def __init__(self):
        self.messages = [{"role": "user", "content": "hello"}]
        self.session_id = None

    def reset(self):
        self.messages.clear()


class DummyConfig:
    model = "test-model"


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
                first = json.loads(path.read_text(encoding="utf-8"))["updated_at"]
                time.sleep(1.1)

                server._auto_save()
                second = json.loads(path.read_text(encoding="utf-8"))["updated_at"]

                self.assertEqual(first, second)
            finally:
                session_module.SESSIONS_DIR = old_dir
                server._state.clear()
                server._state.update(old_state)


if __name__ == "__main__":
    unittest.main()
