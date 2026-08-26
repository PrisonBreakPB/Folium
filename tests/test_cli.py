from types import SimpleNamespace
from io import StringIO

from folium import cli, database
from folium.config import Config
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
    monkeypatch.setattr(cli.Prompt, "ask", lambda *args, **kwargs: "n")

    assert cli._cli_edit_approval(None, proposal) is False


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
    assert "v0.3.0" in rendered
    assert "Your turn" in rendered
