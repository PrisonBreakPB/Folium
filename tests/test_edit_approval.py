import threading
from types import SimpleNamespace

from fastapi.testclient import TestClient

from folium.agent import Agent
from folium.edit_approval import (
    ChangeSetProposal,
    MAX_APPROVAL_DIFF_CHARS,
    build_file_change_proposal,
)
from folium.llm import LLMResponse, ToolCall
from folium.tools.bash import BashTool
from folium.tools.edit import EditFileTool
from folium.tools.write import WriteFileTool
from folium.web import server


def test_write_file_runs_without_approval_and_returns_diff(tmp_path, monkeypatch):
    target = tmp_path / "note.txt"
    agent = Agent(llm=None, tools=[WriteFileTool()])
    approvals = []
    agent.edit_approval_callback = lambda tc, proposal: approvals.append((tc, proposal)) or False

    monkeypatch.setenv("FOLIUM_HOST_WORKSPACE", str(tmp_path))
    result = agent._exec_tool(ToolCall(id="t1", name="write_file", arguments={
        "file_path": "note.txt",
        "content": "hello\n",
    }))

    assert result.status == "ok"
    assert target.read_text(encoding="utf-8") == "hello\n"
    assert approvals == []
    assert result.preview == "Wrote 1 lines to note.txt"
    assert result.diff.startswith("--- /dev/null")
    assert "+hello" in result.diff


def test_edit_file_runs_without_approval_and_returns_diff(tmp_path, monkeypatch):
    target = tmp_path / "note.txt"
    target.write_text("old\n", encoding="utf-8")
    agent = Agent(llm=None, tools=[EditFileTool()])
    approvals = []
    agent.edit_approval_callback = lambda tc, proposal: approvals.append((tc, proposal)) or False
    monkeypatch.setenv("FOLIUM_HOST_WORKSPACE", str(tmp_path))

    result = agent._exec_tool(ToolCall(id="t1", name="edit_file", arguments={
        "file_path": "note.txt",
        "old_string": "old",
        "new_string": "new",
    }))

    assert result.status == "ok"
    assert target.read_text(encoding="utf-8") == "new\n"
    assert approvals == []
    assert result.preview == "Edited note.txt"
    assert "-old" in result.diff
    assert "+new" in result.diff


def test_protected_write_file_is_not_applied_when_rejected(tmp_path, monkeypatch):
    target = tmp_path / "paper.tex"
    agent = Agent(llm=None, tools=[WriteFileTool()])
    approvals = []
    agent.edit_approval_callback = lambda tc, proposal: approvals.append(proposal) or False
    monkeypatch.setenv("FOLIUM_HOST_WORKSPACE", str(tmp_path))

    result = agent._exec_tool(ToolCall(id="t1", name="write_file", arguments={
        "file_path": "paper.tex",
        "content": "\\section{Introduction}\n",
    }))

    assert result.status == "rejected"
    assert not target.exists()
    assert len(approvals) == 1
    assert isinstance(approvals[0], ChangeSetProposal)
    assert approvals[0].files[0].path == str(target)
    assert "+\\section{Introduction}" in approvals[0].files[0].diff


def test_protected_write_file_is_applied_after_approval(tmp_path, monkeypatch):
    target = tmp_path / "experiment.py"
    agent = Agent(llm=None, tools=[WriteFileTool()])
    agent.edit_approval_callback = lambda tc, proposal: True
    monkeypatch.setenv("FOLIUM_HOST_WORKSPACE", str(tmp_path))

    result = agent._exec_tool(ToolCall(id="t1", name="write_file", arguments={
        "file_path": "experiment.py",
        "content": "print('ok')\n",
    }))

    assert result.status == "ok"
    assert target.read_text(encoding="utf-8") == "print('ok')\n"
    assert result.preview == "Wrote 1 lines to experiment.py"


def test_protected_edit_file_is_not_applied_when_rejected(tmp_path, monkeypatch):
    target = tmp_path / "experiment.py"
    target.write_text("value = 1\n", encoding="utf-8")
    agent = Agent(llm=None, tools=[EditFileTool()])
    agent.edit_approval_callback = lambda tc, proposal: False
    monkeypatch.setenv("FOLIUM_HOST_WORKSPACE", str(tmp_path))

    result = agent._exec_tool(ToolCall(id="t1", name="edit_file", arguments={
        "file_path": "experiment.py",
        "old_string": "value = 1",
        "new_string": "value = 2",
    }))

    assert result.status == "rejected"
    assert target.read_text(encoding="utf-8") == "value = 1\n"


def test_protected_file_change_rechecks_the_reviewed_baseline(tmp_path, monkeypatch):
    target = tmp_path / "experiment.py"
    target.write_text("value = 1\n", encoding="utf-8")
    agent = Agent(llm=None, tools=[WriteFileTool()])
    monkeypatch.setenv("FOLIUM_HOST_WORKSPACE", str(tmp_path))

    def approve_after_external_change(tc, proposal):
        target.write_text("value = 99\n", encoding="utf-8")
        return True

    agent.edit_approval_callback = approve_after_external_change
    result = agent._exec_tool(ToolCall(id="t1", name="write_file", arguments={
        "file_path": "experiment.py",
        "content": "value = 2\n",
    }))

    assert result.status == "error"
    assert "reviewed file changed" in result.content
    assert target.read_text(encoding="utf-8") == "value = 99\n"


def test_multiple_protected_files_share_one_approval(tmp_path, monkeypatch):
    agent = Agent(llm=None, tools=[WriteFileTool()])
    approvals = []
    agent.edit_approval_callback = lambda tc, proposal: approvals.append(proposal) or True
    monkeypatch.setenv("FOLIUM_HOST_WORKSPACE", str(tmp_path))
    calls = [
        ToolCall(id="tex", name="write_file", arguments={
            "file_path": "paper.tex",
            "content": "\\title{Demo}\n",
        }),
        ToolCall(id="code", name="write_file", arguments={
            "file_path": "experiment.py",
            "content": "print('demo')\n",
        }),
    ]

    results = agent._exec_tool_batch(calls)

    assert [result.status for result in results] == ["ok", "ok"]
    assert len(approvals) == 1
    assert [change.requested_path for change in approvals[0].files] == [
        "paper.tex",
        "experiment.py",
    ]
    assert (tmp_path / "paper.tex").exists()
    assert (tmp_path / "experiment.py").exists()


def test_rejected_change_set_stops_agent_without_a_retry(tmp_path, monkeypatch):
    class RejectingWriteLLM:
        model = "fake-model"

        def __init__(self):
            self.calls = 0

        def chat(self, messages, tools=None, on_token=None):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(tool_calls=[
                    ToolCall(
                        id="write_1",
                        name="write_file",
                        arguments={"file_path": "paper.tex", "content": "draft\n"},
                    )
                ])
            return LLMResponse(content="this round should not run")

    monkeypatch.setenv("FOLIUM_HOST_WORKSPACE", str(tmp_path))
    llm = RejectingWriteLLM()
    agent = Agent(llm=llm, tools=[WriteFileTool()])
    agent.edit_approval_callback = lambda tc, proposal: False

    result = agent.chat("write a paper draft")

    assert result == "File changes were rejected by user; no files were modified."
    assert llm.calls == 1
    assert not (tmp_path / "paper.tex").exists()


def test_tool_result_event_includes_file_diff(tmp_path, monkeypatch):
    agent = Agent(llm=None, tools=[WriteFileTool()])
    monkeypatch.setenv("FOLIUM_HOST_WORKSPACE", str(tmp_path))
    tc = ToolCall(id="t1", name="write_file", arguments={
        "file_path": "note.txt",
        "content": "hello\n",
    })
    result = agent._exec_tool(tc)
    events = []

    agent._emit_tool_result(events.append, tc, result)

    assert events[0]["preview"] == "Wrote 1 lines to note.txt"
    assert events[0]["diff"] == result.diff.strip()


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


def test_blocked_bash_command_has_error_status():
    agent = Agent(llm=None, tools=[BashTool()])

    result = agent._exec_tool(ToolCall(id="t1", name="bash", arguments={
        "command": "rm -rf /",
    }))

    assert result.status == "error"
    assert result.content.startswith("[Warning] Blocked:")


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


def test_web_bash_approval_callback_waits_for_decision():
    queue = SimpleQueue()
    _on_token, _on_tool, _on_event, on_edit_approval = server._make_bridge(queue)
    tc = SimpleNamespace(id="tool_1", name="bash")
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
    assert event["tool_name"] == "bash"

    pending = server._pending_approvals[event["approval_id"]]
    pending.approved = True
    pending.event.set()
    thread.join(timeout=1)

    assert result["approved"] is True


def test_web_changeset_approval_exposes_preview_and_full_diff(tmp_path, monkeypatch):
    monkeypatch.setenv("FOLIUM_HOST_WORKSPACE", str(tmp_path))
    proposal = build_file_change_proposal(
        "tool_1",
        "write_file",
        {
            "file_path": "experiment.py",
            "content": "x = '" + ("a" * (MAX_APPROVAL_DIFF_CHARS + 100)) + "'\n",
        },
    )
    assert proposal is not None
    change_set = ChangeSetProposal([proposal])
    queue = SimpleQueue()
    _on_token, _on_tool, _on_event, on_edit_approval = server._make_bridge(queue)
    tc = SimpleNamespace(id="tool_1", name="write_file")
    result = {}

    thread = threading.Thread(
        target=lambda: result.setdefault("approved", on_edit_approval(tc, change_set)),
        daemon=True,
    )
    thread.start()
    event = queue.get(timeout=1)

    try:
        assert event["type"] == "approval_request"
        assert event["tool_name"] == "change_set"
        assert event["files"][0]["truncated"] is True
        assert len(event["files"][0]["diff"]) < proposal.diff_chars

        client = TestClient(server.app)
        first = client.get(
            f"/approval/{event['approval_id']}/diff",
            params={"file_index": 0, "offset": 0, "limit": 100},
        )
        second = client.get(
            f"/approval/{event['approval_id']}/diff",
            params={"file_index": 0, "offset": 100, "limit": 100},
        )

        assert first.status_code == 200
        assert first.json()["diff"] == proposal.diff[:100]
        assert first.json()["next_offset"] == 100
        assert second.json()["diff"] == proposal.diff[100:200]
    finally:
        pending = server._pending_approvals.get(event["approval_id"])
        if pending:
            pending.approved = False
            pending.event.set()
        thread.join(timeout=1)

    assert result["approved"] is False


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
