from types import SimpleNamespace
from io import StringIO

from folium import cli, database
from folium.config import Config
from folium.cost_meter import CostMeter
from folium.session import get_session_workspace


def test_cli_workspace_argument_is_parsed(monkeypatch):
    monkeypatch.setattr("sys.argv", ["folium", "--workspace", "D:/project"])
    assert cli._parse_args().workspace == "D:/project"


def test_cli_session_persists_workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "folium.db")
    agent = SimpleNamespace(_system="system", messages=[{"role": "user", "content": "hi"}], transcript=[])
    config = Config(model="test-model")

    session_id = cli._persist_session(agent, config, str(tmp_path), None)

    assert session_id
    assert get_session_workspace(session_id) == str(tmp_path.resolve())


def test_cli_approval_can_reject(monkeypatch):
    proposal = SimpleNamespace(title="Edit file", path="note.txt", diff="diff")
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: "n")

    assert cli._cli_edit_approval(None, proposal) is False


def test_cli_approval_rejects_without_tty(monkeypatch):
    proposal = SimpleNamespace(title="Edit file", path="note.txt", diff="diff")
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(
        cli.Prompt,
        "ask",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not prompt")),
    )

    decision = cli._cli_edit_approval(None, proposal)

    assert decision.action == "rejected"


def test_cli_resets_last_usage_when_switching_sessions():
    llm = SimpleNamespace(
        last_prompt_tokens=12,
        last_completion_tokens=7,
        last_cached_tokens=3,
    )

    cli._reset_last_llm_usage(llm)

    assert llm.last_prompt_tokens == 0
    assert llm.last_completion_tokens == 0
    assert llm.last_cached_tokens == 0


def test_cli_one_shot_submits_memory_maintenance(monkeypatch, tmp_path):
    calls = []

    class FakeMaintenance:
        def __init__(self, agent, config):
            calls.append(("init", agent, config))

        def submit(self, **kwargs):
            calls.append(("submit", kwargs))

        def wait(self, session_id):
            calls.append(("wait", session_id))

    llm = SimpleNamespace(last_prompt_tokens=12, last_completion_tokens=7)
    agent = SimpleNamespace(
        messages=[{"role": "assistant", "content": "answer"}],
        transcript=[{"role": "assistant", "content": "answer"}],
        _system="system",
        _full_messages=lambda: [{"role": "system", "content": "system"}],
        _tool_schemas=lambda: [],
        llm=llm,
        edit_approval_callback=None,
        last_llm_request_had_visible_tools=True,
        chat=lambda prompt, **kwargs: "answer",
    )
    config = Config(model="test-model")
    monkeypatch.setattr(cli, "_CliMaintenance", FakeMaintenance)
    monkeypatch.setattr(cli, "_ensure_session", lambda *args: "session-1")
    monkeypatch.setattr(cli, "save_session", lambda *args, **kwargs: None)

    cli._run_once(agent, "hello", config, str(tmp_path))

    assert calls[0][0] == "init"
    assert calls[1][0] == "submit"
    assert calls[1][1]["session_id"] == "session-1"
    assert calls[2] == ("wait", "session-1")


def test_cli_one_shot_ignores_maintenance_timeout(monkeypatch, tmp_path):
    output = StringIO()
    from rich.console import Console

    monkeypatch.setattr(cli, "console", Console(file=output, width=100))

    class TimedOutFuture:
        def result(self, timeout):
            raise TimeoutError

    class TimedOutMaintenance:
        def __init__(self, agent, config):
            pass

        def submit(self, **kwargs):
            return TimedOutFuture()

        def wait(self, session_id):
            raise TimeoutError

    llm = SimpleNamespace(last_prompt_tokens=0, last_completion_tokens=0)
    agent = SimpleNamespace(
        messages=[],
        transcript=[],
        _system="system",
        _full_messages=lambda: [],
        _tool_schemas=lambda: [],
        llm=llm,
        edit_approval_callback=None,
        last_llm_request_had_visible_tools=False,
        chat=lambda prompt, **kwargs: "answer",
    )
    monkeypatch.setattr(cli, "_CliMaintenance", TimedOutMaintenance)
    monkeypatch.setattr(cli, "_ensure_session", lambda *args: "session-1")
    monkeypatch.setattr(cli, "save_session", lambda *args, **kwargs: None)

    cli._run_once(agent, "hello", Config(model="test-model"), str(tmp_path))

    assert "Background memory maintenance did not finish" in output.getvalue()


def test_cli_banner_contains_identity_and_input_guidance(monkeypatch):
    output = StringIO()
    from rich.console import Console

    monkeypatch.setattr(cli, "console", Console(file=output, width=100))
    cli._show_banner(
        SimpleNamespace(mode="build"),
        SimpleNamespace(model="deepseek-v4-pro", base_url=None),
        "D:/project",
    )

    rendered = output.getvalue()
    assert "FOLIUM / RESEARCH AGENT / v0.3.0" in rendered
    assert "######" in rendered
    assert "v0.3.0" in rendered
    assert "Your turn" in rendered


def test_cli_skills_command_renders_loaded_skills(monkeypatch):
    output = StringIO()
    from rich.console import Console

    monkeypatch.setattr(cli, "console", Console(file=output, width=100))
    agent = SimpleNamespace(
        skills=[SimpleNamespace(name="literature-review", description="Review papers.")]
    )

    cli._show_skills(agent)

    rendered = output.getvalue()
    assert "Available skills" in rendered
    assert "/literature-review" in rendered
    assert "Review papers." in rendered


def test_cli_status_renders_runtime_usage_and_budget(monkeypatch, tmp_path):
    output = StringIO()
    from rich.console import Console

    monkeypatch.setattr(cli, "console", Console(file=output, width=120))
    monkeypatch.setenv("FOLIUM_SANDBOX_WORKSPACE_MODE", "host")
    llm = SimpleNamespace(
        last_prompt_tokens=12,
        last_completion_tokens=7,
        last_cached_tokens=3,
        total_prompt_tokens=100,
        total_completion_tokens=40,
        total_cached_tokens=20,
    )
    agent = SimpleNamespace(
        mode="build",
        messages=[{"role": "user", "content": "hello"}],
        context=SimpleNamespace(
            max_tokens=1000,
            input_budget_tokens=900,
            reserved_output_tokens=100,
        ),
        llm=llm,
        _cost_meter=CostMeter(budget_usd=1.0),
    )
    agent._cost_meter.record(0.25)

    cli._show_status(agent, Config(model="test-model"), str(tmp_path), "session-1")

    rendered = output.getvalue()
    assert "Runtime status" in rendered
    assert "test-model" in rendered
    assert "session-1" in rendered
    assert "CONTEXT" in rendered
    assert "COST" in rendered
    assert "$0.250000 / $1.000000" in rendered


def test_cli_renders_agent_progress_events(monkeypatch):
    output = StringIO()
    from rich.console import Console

    monkeypatch.setattr(cli, "console", Console(file=output, width=120))
    cli._render_agent_event({
        "type": "tool_start",
        "name": "read_file",
        "arguments_preview": "file_path='note.txt'",
    })
    cli._render_agent_event({
        "type": "tool_result",
        "name": "read_file",
        "status": "ok",
        "preview": "content [safe]",
    })
    cli._render_agent_event({
        "type": "context_compress",
        "before_tokens": 1000,
        "after_tokens": 600,
    })
    cli._render_agent_event({
        "type": "context_update",
        "estimated_context_tokens": 600,
        "max_context_tokens": 1000,
    })
    cli._render_agent_event({"type": "todo_reminder"})
    cli._render_agent_event({
        "type": "todo_update",
        "items": [
            {"status": "completed"},
            {"status": "in_progress"},
        ],
    })
    cli._render_agent_event({
        "type": "usage_update",
        "prompt_tokens": 12,
        "completion_tokens": 7,
        "cached_tokens": 3,
        "cost": 0.001,
    })

    rendered = output.getvalue()
    assert "> read_file(file_path='note.txt')" in rendered
    assert "< read_file status=ok content [safe]" in rendered
    assert "context compressed: 1,000 -> 600 tokens" in rendered
    assert "context: 600 / 1,000 tokens" in rendered
    assert "todo reminder: update the task progress" in rendered
    assert "todo updated: 1/2 completed" in rendered
    assert "usage: 12 in + 7 out + 3 cached, cost $0.001000" in rendered


def test_cli_diff_uses_red_and_green_backgrounds(monkeypatch):
    rendered = []

    class CaptureConsole:
        def print(self, value):
            rendered.append(value)

    monkeypatch.setattr(cli, "console", CaptureConsole())
    cli._render_diff("--- old\n+++ new\n-old line\n+new line\n context")

    assert rendered[0].style == "bold cyan"
    assert rendered[1].style == "bold cyan"
    assert rendered[2].style == "white on red"
    assert rendered[3].style == "black on green"
    assert rendered[4].style == "dim"
