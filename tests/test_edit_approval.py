import threading
from types import SimpleNamespace

from fastapi.testclient import TestClient

from folium.agent import Agent
from folium.llm import ToolCall
from folium.tools.bash import BashTool
from folium.tools.edit import EditFileTool
from folium.tools.write import WriteFileTool
from folium.web import server


def test_write_file_approval_rejects_without_writing(tmp_path, monkeypatch):
    target = tmp_path / "note.txt"
    agent = Agent(llm=None, tools=[WriteFileTool()])
    agent.edit_approval_callback = lambda tc, proposal: False

    monkeypatch.setenv("FOLIUM_HOST_WORKSPACE", str(tmp_path))
    result = agent._exec_tool(ToolCall(id="t1", name="write_file", arguments={
        "file_path": "note.txt",
        "content": "hello\n",
    }))

    assert result.status == "error"
    assert "rejected by user" in result.content
    assert not target.exists()


def test_edit_file_approval_allows_write(tmp_path, monkeypatch):
    target = tmp_path / "note.txt"
    target.write_text("old\n", encoding="utf-8")
    seen = {}
    agent = Agent(llm=None, tools=[EditFileTool()])

    def approve(tc, proposal):
        seen["diff"] = proposal.diff
        return True

    agent.edit_approval_callback = approve
    monkeypatch.setenv("FOLIUM_HOST_WORKSPACE", str(tmp_path))

    result = agent._exec_tool(ToolCall(id="t1", name="edit_file", arguments={
        "file_path": "note.txt",
        "old_string": "old",
        "new_string": "new",
    }))

    assert result.status == "ok"
    assert target.read_text(encoding="utf-8") == "new\n"
    assert "-old" in seen["diff"]
    assert "+new" in seen["diff"]


class RecordingExecutor:
    def __init__(self):
        self.commands = []

    def run(self, command, timeout=120):
        self.commands.append(command)
        return "ran"


def test_bash_workspace_write_approval_rejects_without_running():
    executor = RecordingExecutor()
    agent = Agent(llm=None, tools=[BashTool(executor=executor)])
    agent.edit_approval_callback = lambda tc, proposal: False

    result = agent._exec_tool(ToolCall(id="t1", name="bash", arguments={
        "command": "cat > /workspace/pid_verify.py << 'EOF'\nprint('hi')\nEOF",
    }))

    assert result.status == "error"
    assert "bash command rejected by user" in result.content
    assert executor.commands == []


def test_bash_read_only_command_does_not_require_approval():
    executor = RecordingExecutor()
    agent = Agent(llm=None, tools=[BashTool(executor=executor)])
    agent.edit_approval_callback = lambda tc, proposal: False

    result = agent._exec_tool(ToolCall(id="t1", name="bash", arguments={
        "command": "pwd && ls -la",
    }))

    assert result.status == "ok"
    assert result.content == "ran"
    assert executor.commands == ["pwd && ls -la"]


def test_approval_endpoint_resolves_pending_request():
    client = TestClient(server.app)
    approval_id = "approval_test"
    pending = server.PendingApproval({"approval_id": approval_id})
    old = dict(server._pending_approvals)
    try:
        server._pending_approvals.clear()
        server._pending_approvals[approval_id] = pending

        resp = client.post("/approval", json={"approval_id": approval_id, "approved": True})

        assert resp.status_code == 200
        assert pending.event.is_set()
        assert pending.approved is True
    finally:
        server._pending_approvals.clear()
        server._pending_approvals.update(old)


def test_web_edit_approval_callback_waits_for_decision():
    queue = SimpleQueue()
    _on_token, _on_tool, _on_event, on_edit_approval = server._make_bridge(queue)
    tc = SimpleNamespace(id="tool_1", name="write_file")
    proposal = SimpleNamespace(
        path="README.md",
        title="Overwrite README.md",
        diff="--- a/README.md\n+++ b/README.md\n",
        truncated=False,
        diff_chars=36,
    )
    result = {}

    thread = threading.Thread(
        target=lambda: result.setdefault("approved", on_edit_approval(tc, proposal)),
        daemon=True,
    )
    thread.start()

    event = queue.get(timeout=1)
    assert event["type"] == "approval_request"
    assert event["tool_name"] == "write_file"

    pending = server._pending_approvals[event["approval_id"]]
    pending.approved = True
    pending.event.set()
    thread.join(timeout=1)

    assert result["approved"] is True


class SimpleQueue:
    def __init__(self):
        self.items = []
        self.event = threading.Event()

    def put_nowait(self, item):
        self.items.append(item)
        self.event.set()

    def get(self, timeout=1):
        if not self.event.wait(timeout):
            raise TimeoutError("queue empty")
        return self.items.pop(0)
