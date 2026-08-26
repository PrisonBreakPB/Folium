"""Interactive REPL - the user-facing terminal interface."""

import sys
import os
import argparse
import asyncio
import threading
import copy
from pathlib import Path

from rich.console import Console
from rich.markup import escape
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text
from rich.align import Align
from rich.table import Table
from rich.box import ROUNDED
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings

from .agent import Agent
from .llm import LLM, LiteLLM
from .config import Config
from .context import estimate_tokens
from .session import (
    calculate_session_stats,
    delete_session,
    ensure_session,
    get_session_workspace,
    list_sessions,
    load_session,
    new_session_id,
    normalize_workspace_path,
    save_session,
)
from .observability import list_traces, read_trace_summary
from .sandbox.session import configure_host_workspace, reset_current_session, get_current_session
from .edit_approval import ApprovalDecision
from .memory_maintenance import MemoryMaintenanceScheduler, build_memory_maintenance_runner
from . import __version__

console = Console()


def _parse_args():
    p = argparse.ArgumentParser(
        prog="folium",
        description="Minimal AI coding agent. Works with any OpenAI-compatible LLM.",
    )
    p.add_argument("-m", "--model", help="Model name (default: $FOLIUM_MODEL or gpt-4o)")
    p.add_argument("--base-url", help="API base URL (default: $OPENAI_BASE_URL)")
    p.add_argument("--api-key", help="API key (default: $OPENAI_API_KEY)")
    p.add_argument("-p", "--prompt", help="One-shot prompt (non-interactive mode)")
    p.add_argument("-r", "--resume", metavar="ID", help="Resume a saved session")
    p.add_argument("--workspace", help="Project workspace directory")
    p.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")
    return p.parse_args()


def main():
    args = _parse_args()
    os.environ.setdefault("FOLIUM_SANDBOX_WORKSPACE_MODE", "copy")
    saved_workspace = get_session_workspace(args.resume) if args.resume else None
    workspace_input = args.workspace or os.getenv("FOLIUM_HOST_WORKSPACE")
    if not workspace_input and not args.resume and not args.prompt and sys.stdin.isatty():
        workspace_input = Prompt.ask("Project workspace", default=os.getcwd())
    try:
        workspace = normalize_workspace_path(workspace_input or os.getcwd())
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        sys.exit(2)
    if args.resume and saved_workspace:
        try:
            workspace = normalize_workspace_path(saved_workspace)
        except ValueError:
            console.print(
                f"[yellow]Saved workspace is unavailable; using {workspace}[/yellow]"
            )
    configure_host_workspace(workspace)
    reset_current_session()

    # Load project-local settings after the selected workspace is known.
    try:
        from dotenv import load_dotenv
        load_dotenv(Path(workspace) / ".env", override=False)
    except ImportError:
        pass
    config = Config.from_env()

    # CLI args override env vars
    if args.model:
        config.model = args.model
    if args.base_url:
        config.base_url = args.base_url
    if args.api_key:
        config.api_key = args.api_key

    if not config.api_key:
        console.print("[red bold]No API key found.[/]")
        console.print(
            "Set one of: OPENAI_API_KEY, DEEPSEEK_API_KEY, or FOLIUM_API_KEY\n"
            "\nExamples:\n"
            "  # OpenAI\n"
            "  export OPENAI_API_KEY=sk-...\n"
            "\n"
            "  # DeepSeek\n"
            "  export OPENAI_API_KEY=sk-... OPENAI_BASE_URL=https://api.deepseek.com\n"
            "\n"
            "  # Ollama (local)\n"
            "  export OPENAI_API_KEY=ollama OPENAI_BASE_URL=http://localhost:11434/v1 FOLIUM_MODEL=qwen2.5-coder\n"
        )
        sys.exit(1)

    llm_cls = LiteLLM if config.provider == "litellm" else LLM
    llm = llm_cls(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        api_format=config.api_format,
    )
    agent = Agent(llm=llm, max_context_tokens=config.max_context_tokens)

    # resume saved session
    if args.resume:
        loaded = load_session(args.resume)
        if loaded:
            agent.messages, loaded_model, agent.transcript, system_prompt = loaded
            agent.session_id = args.resume
            # restore the model from the saved session unless overridden by CLI
            if not args.model:
                agent.llm.model = loaded_model
                config.model = loaded_model
            if system_prompt:
                agent._system = system_prompt
            stats = calculate_session_stats(args.resume)
            _reset_last_llm_usage(agent.llm)
            agent.llm.total_prompt_tokens = stats["prompt_tokens"]
            agent.llm.total_completion_tokens = stats["completion_tokens"]
            agent.llm.total_cached_tokens = stats["cached_tokens"]
            console.print(
                f"[green]Resumed session: {args.resume} (model: {agent.llm.model})[/green]"
            )
        else:
            console.print(f"[red]Session '{args.resume}' not found.[/red]")
            sys.exit(1)

    # one-shot mode
    if args.prompt:
        _run_once(agent, args.prompt, config, workspace, args.resume)
        return

    # interactive REPL
    _repl(agent, config, workspace, args.resume)


def _run_once(agent: Agent, prompt: str, config: Config, workspace: str, session_id: str | None = None):
    """Non-interactive: run one prompt and exit."""
    def on_token(tok):
        print(tok, end="", flush=True)

    def on_event(event):
        _render_agent_event(event)

    maintenance = _CliMaintenance(agent, config)
    transcript_start = len(agent.transcript)
    session_id = _ensure_session(agent, config, workspace, session_id)
    agent.edit_approval_callback = _cli_edit_approval
    agent.chat(prompt, on_token=on_token, on_event=on_event)
    save_session(
        agent.messages,
        config.model,
        session_id,
        transcript=agent.transcript,
        system_prompt=agent._system,
        workspace_path=workspace,
    )
    scheduled = maintenance.submit(
        session_id=session_id,
        messages=copy.deepcopy(agent._full_messages()),
        visible_tools=copy.deepcopy(agent._tool_schemas()),
        main_agent_used_memory=any(
            message.get("role") == "tool" and message.get("name") == "memory"
            for message in agent.transcript[transcript_start:]
        ),
        main_prompt_tokens=getattr(agent.llm, "last_prompt_tokens", 0),
        main_completion_tokens=getattr(agent.llm, "last_completion_tokens", 0),
        main_request_matches_memory_context=getattr(
            agent, "last_llm_request_had_visible_tools", False
        ),
    )
    try:
        if scheduled is not None:
            scheduled.result(timeout=5)
        maintenance.wait(session_id)
    except TimeoutError:
        console.print("\n[yellow]Background memory maintenance did not finish before exit.[/yellow]")
    print()


def _repl(agent: Agent, config: Config, workspace: str, session_id: str | None = None):
    """Interactive read-eval-print loop."""
    _show_banner(agent, config, workspace)

    hist_path = os.path.expanduser("~/.folium_history")
    history = FileHistory(hist_path)
    current_session_id = session_id
    agent.edit_approval_callback = _cli_edit_approval
    maintenance = _CliMaintenance(agent, config)

    # Enter submits, Escape+Enter inserts a newline (for pasting code blocks etc.)
    kb = KeyBindings()

    @kb.add("enter")
    def _submit(event):
        event.current_buffer.validate_and_handle()

    @kb.add("escape", "enter")
    def _newline(event):
        event.current_buffer.insert_text("\n")

    while True:
        try:
            user_input = pt_prompt(
                HTML("<ansibrightcyan><b> YOU </b></ansibrightcyan><ansiyellow> &gt;&gt; </ansiyellow>"),
                history=history,
                multiline=True,
                key_bindings=kb,
                prompt_continuation="...  ",
            ).strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye!")
            break

        if not user_input:
            continue

        # built-in commands
        if user_input.lower() in ("quit", "exit", "/quit", "/exit"):
            break
        if user_input in ("/help", "help"):
            _show_help()
            continue
        if user_input == "/reset":
            current_session_id = _persist_session(agent, config, workspace, current_session_id)
            agent.reset()
            _reset_last_llm_usage(agent.llm)
            current_session_id = None
            agent.session_id = None
            reset_current_session()
            configure_host_workspace(workspace)
            console.print("[yellow]Conversation reset.[/yellow]")
            continue
        if user_input == "/new":
            current_session_id = _persist_session(agent, config, workspace, current_session_id)
            agent.reset()
            _reset_last_llm_usage(agent.llm)
            current_session_id = None
            agent.session_id = None
            reset_current_session()
            configure_host_workspace(workspace)
            console.print("[yellow]New conversation started.[/yellow]")
            continue
        if user_input == "/tokens":
            p = agent.llm.total_prompt_tokens
            c = agent.llm.total_completion_tokens
            line = f"Tokens: [cyan]{p}[/cyan] prompt + [cyan]{c}[/cyan] completion = [bold]{p+c}[/bold] total"
            cost = agent.llm.estimated_cost
            if cost is not None:
                line += f"  (~${cost:.4f})"
            console.print(line)
            continue
        if user_input == "/model" or user_input.startswith("/model "):
            new_model = user_input[7:].strip() if user_input.startswith("/model ") else ""
            if new_model:
                agent.llm.model = new_model
                config.model = new_model
                console.print(f"Switched to [cyan]{new_model}[/cyan]")
            else:
                console.print(f"Current model: [cyan]{config.model}[/cyan]")
            continue
        if user_input in ("/skills", "skills"):
            _show_skills(agent)
            continue
        if user_input in ("/status", "status", "/usage", "usage"):
            _show_status(agent, config, workspace, current_session_id)
            continue
        if user_input == "/mode" or user_input.startswith("/mode "):
            requested = user_input[6:].strip() if user_input.startswith("/mode ") else ""
            if not requested:
                console.print(f"Current mode: [cyan]{agent.mode}[/cyan]")
            else:
                try:
                    agent.set_mode(requested)
                    console.print(f"Switched to [cyan]{agent.mode}[/cyan]")
                except ValueError as exc:
                    console.print(f"[red]{exc}[/red]")
            continue
        if user_input == "/workspace":
            sandbox = get_current_session() if os.getenv("FOLIUM_SANDBOX_WORKSPACE_MODE") == "copy" else None
            console.print(f"Host workspace: [cyan]{workspace}[/cyan]")
            if sandbox:
                console.print(f"Sandbox workspace: [cyan]{sandbox.workspace}[/cyan]")
            continue
        if user_input == "/todos":
            manager = agent.todo_manager
            console.print(manager.render() if manager else "No todos.")
            continue
        if user_input == "/compact":
            from .context import estimate_tokens
            before = estimate_tokens(agent.messages)
            report = agent.context.maybe_compress(agent.messages, agent.llm)
            after = estimate_tokens(agent.messages)
            if report["compressed"]:
                console.print(f"[green]Compressed: {before} → {after} tokens ({len(agent.messages)} messages)[/green]")
            else:
                console.print(f"[dim]Nothing to compress ({before} tokens, {len(agent.messages)} messages)[/dim]")
            continue
        if user_input == "/save":
            current_session_id = _persist_session(agent, config, workspace, current_session_id)
            sid = current_session_id
            agent.session_id = sid
            console.print(f"[green]Session saved: {sid}[/green]")
            console.print(f"Resume with: folium -r {sid}")
            continue
        if user_input == "/diff":
            from .tools.edit import _changed_files
            if not _changed_files:
                console.print("[dim]No files modified this session.[/dim]")
            else:
                console.print(f"[bold]Files modified this session ({len(_changed_files)}):[/bold]")
                for f in sorted(_changed_files):
                    console.print(f"  [cyan]{f}[/cyan]")
            continue
        if user_input == "/sessions":
            sessions = list_sessions()
            if not sessions:
                console.print("[dim]No saved sessions.[/dim]")
            else:
                for s in sessions:
                    workspace_info = f", {s['workspace_path']}" if s.get("workspace_path") else ""
                    console.print(f"  [cyan]{s['id']}[/cyan] ({s['model']}, {s['updated_at']}{workspace_info}) {s['preview']}")
            continue
        if user_input.startswith("/switch "):
            target = user_input.split(" ", 1)[1].strip()
            current_session_id = _persist_session(agent, config, workspace, current_session_id)
            loaded = load_session(target)
            if not loaded:
                console.print("[red]Session not found.[/red]")
                continue
            messages, loaded_model, transcript, saved_prompt = loaded
            saved_workspace = get_session_workspace(target)
            if saved_workspace:
                try:
                    workspace = normalize_workspace_path(saved_workspace)
                except ValueError as exc:
                    console.print(f"[red]{exc}[/red]")
                    continue
                configure_host_workspace(workspace)
            reset_current_session()
            agent.messages = messages
            agent.transcript = transcript
            agent.reset_todos()
            agent.session_id = target
            current_session_id = target
            agent.llm.model = loaded_model
            config.model = loaded_model
            if saved_prompt is not None:
                agent._system = saved_prompt
            stats = calculate_session_stats(target)
            _reset_last_llm_usage(agent.llm)
            agent.llm.total_prompt_tokens = stats["prompt_tokens"]
            agent.llm.total_completion_tokens = stats["completion_tokens"]
            agent.llm.total_cached_tokens = stats["cached_tokens"]
            console.print(f"[green]Switched to {target}[/green]")
            continue
        if user_input.startswith("/delete "):
            target = user_input.split(" ", 1)[1].strip()
            current_session_id = _persist_session(agent, config, workspace, current_session_id)
            if delete_session(target):
                if target == current_session_id:
                    agent.reset()
                    _reset_last_llm_usage(agent.llm)
                    agent.session_id = None
                    current_session_id = None
                    reset_current_session()
                console.print(f"[green]Deleted {target}[/green]")
            else:
                console.print("[red]Session not found.[/red]")
            continue
        if user_input == "/traces":
            traces = list_traces()
            if not traces:
                console.print("[dim]No traces found.[/dim]")
            else:
                for t in traces:
                    console.print(
                        f"  [cyan]{t['trace_id']}[/cyan] status={t['status']} "
                        f"duration={t['duration_ms']}ms llm={t['llm_calls']} "
                        f"tools={t['tool_calls']} errors={t['errors']}"
                    )
            continue
        if user_input.startswith("/trace "):
            trace_id = user_input.split(" ", 1)[1].strip()
            summary = read_trace_summary(trace_id)
            if not summary:
                console.print("[red]Trace not found.[/red]")
            else:
                console.print(f"[bold]Trace:[/bold] {summary['trace_id']}")
                console.print(f"Status: {summary['status']}")
                console.print(f"Duration: {summary['duration_ms']}ms")
                console.print(f"LLM calls: {summary['llm_calls']}")
                console.print(f"Tool calls: {summary['tool_calls']}")
                console.print(f"Errors: {summary['errors']}")
                for s in summary.get("spans", [])[:20]:
                    console.print(
                        f"  {s['type']}:{s['name']} {s['status']} {s['duration_ms']}ms"
                    )
            continue

        # call the agent
        streamed: list[str] = []

        def on_token(tok):
            streamed.append(tok)
            print(tok, end="", flush=True)

        def on_event(event):
            _render_agent_event(event)

        try:
            current_session_id = _ensure_session(agent, config, workspace, current_session_id)
            transcript_start = len(agent.transcript)
            response = agent.chat(
                user_input,
                on_token=on_token,
                on_event=on_event,
            )
            agent.session_id = current_session_id
            current_session_id = _persist_session(agent, config, workspace, current_session_id)
            maintenance.submit(
                session_id=current_session_id,
                messages=copy.deepcopy(agent._full_messages()),
                visible_tools=copy.deepcopy(agent._tool_schemas()),
                main_agent_used_memory=any(
                    message.get("role") == "tool" and message.get("name") == "memory"
                    for message in agent.transcript[transcript_start:]
                ),
                main_prompt_tokens=getattr(agent.llm, "last_prompt_tokens", 0),
                main_completion_tokens=getattr(agent.llm, "last_completion_tokens", 0),
                main_request_matches_memory_context=getattr(
                    agent, "last_llm_request_had_visible_tools", False
                ),
            )
            if streamed:
                print()  # newline after streamed tokens
            else:
                # response wasn't streamed (came after tool calls)
                console.print(Markdown(response))
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")


def _show_skills(agent: Agent) -> None:
    skills = getattr(agent, "skills", [])
    if not skills:
        console.print("[dim]No skills found. Add skills under skills/<name>/SKILL.md.[/dim]")
        return
    console.print("[bold]Available skills[/bold]")
    for skill in skills:
        console.print(f"  [cyan]/{skill.name}[/cyan]  {skill.description}")


def _show_status(
    agent: Agent,
    config: Config,
    workspace: str,
    session_id: str | None,
) -> None:
    context = getattr(agent, "context", None)
    meter = getattr(agent, "_cost_meter", None)
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim")
    table.add_column(style="white")
    table.add_row("SESSION", session_id or "(unsaved)")
    table.add_row("MODEL", config.model)
    table.add_row("MODE", agent.mode)
    table.add_row("WORKSPACE", workspace)
    if os.getenv("FOLIUM_SANDBOX_WORKSPACE_MODE") == "copy":
        table.add_row("SANDBOX", str(get_current_session().workspace))
    if context:
        estimated = estimate_tokens(agent.messages)
        table.add_row(
            "CONTEXT",
            f"{estimated:,} / {context.max_tokens:,} tokens "
            f"(input budget {context.input_budget_tokens:,}, reserve {context.reserved_output_tokens:,})",
        )
    prompt_tokens = getattr(agent.llm, "last_prompt_tokens", 0)
    completion_tokens = getattr(agent.llm, "last_completion_tokens", 0)
    cached_tokens = getattr(agent.llm, "last_cached_tokens", 0)
    table.add_row(
        "LAST TURN",
        f"{prompt_tokens:,} in + {completion_tokens:,} out + {cached_tokens:,} cached",
    )
    total_prompt = getattr(agent.llm, "total_prompt_tokens", 0)
    total_completion = getattr(agent.llm, "total_completion_tokens", 0)
    total_cached = getattr(agent.llm, "total_cached_tokens", 0)
    hit_rate = total_cached / total_prompt if total_prompt else 0
    table.add_row(
        "SESSION TOTAL",
        f"{total_prompt + total_completion:,} tokens, cache hit {hit_rate:.1%}",
    )
    if meter is not None:
        budget = meter.budget_usd
        spent = meter.spent()
        budget_text = f"${spent:.6f}"
        if budget and budget > 0:
            budget_text += f" / ${budget:.6f}"
        table.add_row("COST", budget_text)
        if budget and budget > 0:
            table.add_row(
                "BUDGET",
                "exhausted" if meter.exhausted() else (
                    "soft threshold reached" if meter.soft_reached() else "within limit"
                ),
            )
    console.print(Panel(table, title="[bold]Runtime status[/bold]", border_style="cyan"))


def _render_agent_event(event: dict) -> None:
    event_type = event.get("type")
    if event_type == "tool_start":
        console.print(
            f"\n[dim]> {event.get('name', 'tool')}({event.get('arguments_preview', '')})[/dim]"
        )
    elif event_type in {"tool_result", "tool_error"}:
        preview = event.get("preview") or event.get("content") or ""
        safe_preview = str(preview)[:120].replace("\n", " ")
        suffix = f" {escape(safe_preview)}" if preview else ""
        console.print(
            f"\n[dim]< {event.get('name', 'tool')} status={event.get('status', 'ok')}"
            f"{suffix}[/dim]"
        )
    elif event_type == "context_compress":
        console.print(
            f"\n[dim]context compressed: {event.get('before_tokens', 0):,} -> "
            f"{event.get('after_tokens', 0):,} tokens[/dim]"
        )
    elif event_type == "context_update":
        estimated = event.get("estimated_context_tokens", 0)
        maximum = event.get("max_context_tokens", 0)
        if maximum:
            console.print(f"\n[dim]context: {estimated:,} / {maximum:,} tokens[/dim]")
    elif event_type == "todo_reminder":
        console.print("\n[dim]todo reminder: update the task progress[/dim]")
    elif event_type == "todo_update":
        items = event.get("items") or []
        done = sum(1 for item in items if item.get("status") == "completed")
        console.print(f"\n[dim]todo updated: {done}/{len(items)} completed[/dim]")
    elif event_type == "budget_exhausted":
        spent = event.get("spent")
        budget = event.get("budget")
        if spent is not None and budget:
            message = f"Budget exhausted (${spent:.6f} / ${budget:.6f}); task stopped."
        else:
            message = "Budget exhausted; task stopped."
        console.print(f"\n[bold yellow]{message}[/bold yellow]")
    elif event_type == "usage_update":
        prompt = event.get("prompt_tokens", 0)
        completion = event.get("completion_tokens", 0)
        cached = event.get("cached_tokens", 0)
        cost = event.get("cost")
        suffix = f", cost ${cost:.6f}" if cost is not None else ""
        console.print(
            f"\n[dim]usage: {prompt:,} in + {completion:,} out + "
            f"{cached:,} cached{suffix}[/dim]"
        )
    elif event_type == "agent_status" and event.get("status") in {"error", "rejected"}:
        console.print(f"\n[bold yellow]{event.get('message', 'Agent status changed')}[/bold yellow]")
    elif event_type == "error":
        console.print(f"\n[bold red]{event.get('content', 'Agent error')}[/bold red]")


def _show_help():
    console.print(Panel(
        "[bold]Commands:[/bold]\n"
        "  /help          Show this help\n"
        "  /new           Start a new conversation\n"
        "  /reset         Clear conversation history\n"
        "  /model         Show current model\n"
        "  /model <name>  Switch model mid-conversation\n"
        "  /skills        List available skills\n"
        "  /status        Show model, context, budget, and workspace\n"
        "  /usage         Alias for /status\n"
        "  /mode          Show current mode\n"
        "  /mode <name>   Switch between build and plan\n"
        "  /workspace     Show host and sandbox workspace\n"
        "  /todos         Show todo list\n"
        "  /tokens        Show token usage\n"
        "  /compact       Compress conversation context\n"
        "  /diff          Show files modified this session\n"
        "  /save          Save session to disk\n"
        "  /sessions      List saved sessions\n"
        "  /switch <id>   Switch to a saved session\n"
        "  /delete <id>   Delete a saved session\n"
        "  /traces        List recent execution traces\n"
        "  /trace <id>    Show a trace summary\n"
        "  quit           Exit Folium\n"
        "\n"
        "[bold]Input:[/bold]\n"
        "  Enter          Submit message\n"
        "  Esc+Enter      Insert newline (for pasting code)",
        title="Folium Help",
        border_style="dim",
    ))


def _show_banner(agent: Agent, config: Config, workspace: str) -> None:
    """Render the compact CLI identity and runtime status."""
    logo = Text(_block_wordmark("FOLIUM"), style="bold bright_cyan", justify="center")
    tagline = Text(
        f"RESEARCH AGENT  /  v{__version__}",
        style="bold white",
        justify="center",
    )
    status = Table.grid(padding=(0, 2))
    status.add_column(style="dim")
    status.add_column(style="white")
    status.add_row("VERSION", f"v{__version__}")
    status.add_row("MODEL", config.model)
    status.add_row("MODE", agent.mode)
    status.add_row("WORKSPACE", workspace)
    if config.base_url:
        status.add_row("API", config.base_url)

    content = Table.grid(padding=(0, 1), expand=True)
    content.add_column()
    content.add_row(logo)
    content.add_row(tagline)
    content.add_row(status)
    console.print(Panel(
        Align.left(content),
        title=f"[bold white]FOLIUM / RESEARCH AGENT / v{__version__}[/bold white]",
        subtitle="[dim]Type /help for commands[/dim]",
        border_style="bright_cyan",
        box=ROUNDED,
        padding=(1, 2),
    ))
    console.print("[dim]─[/dim]" * max(24, min(console.width, 96)))
    console.print("[bold bright_cyan]Your turn[/bold bright_cyan]  [dim]Enter a request or use /help[/dim]")


def _block_wordmark(word: str) -> str:
    """Build a terminal-safe large wordmark from ASCII strokes."""
    glyphs = {
        "F": ("######", "#     ", "#     ", "####  ", "#     ", "#     ", "#     "),
        "O": (" ##### ", "#     #", "#     #", "#     #", "#     #", "#     #", " ##### "),
        "L": ("#     ", "#     ", "#     ", "#     ", "#     ", "#     ", "######"),
        "I": ("#####", "  #  ", "  #  ", "  #  ", "  #  ", "  #  ", "#####"),
        "U": ("#     #", "#     #", "#     #", "#     #", "#     #", "#     #", " ##### "),
        "M": ("#     #", "##   ##", "##   ##", "# # # #", "# # # #", "#  #  #", "#     #"),
    }
    rows = []
    for row in range(7):
        rows.append("  ".join(glyphs[char][row] for char in word))
    return "\n".join(rows)


def _brief(kwargs: dict, maxlen: int = 80) -> str:
    s = ", ".join(f"{k}={repr(v)[:40]}" for k, v in kwargs.items())
    return s[:maxlen] + ("..." if len(s) > maxlen else "")


def _ensure_session(
    agent: Agent,
    config: Config,
    workspace: str,
    session_id: str | None,
) -> str:
    session_id = session_id or new_session_id()
    ensure_session(session_id, config.model, agent._system, workspace_path=workspace)
    agent.session_id = session_id
    return session_id


def _persist_session(
    agent: Agent,
    config: Config,
    workspace: str,
    session_id: str | None,
) -> str | None:
    if not agent.messages:
        return session_id
    session_id = _ensure_session(agent, config, workspace, session_id)
    return save_session(
        agent.messages,
        config.model,
        session_id,
        transcript=agent.transcript,
        system_prompt=agent._system,
        workspace_path=workspace,
    )


def _cli_edit_approval(_tool_call, proposal):
    """Prompt for the same workspace-changing approvals exposed by the Web UI."""
    console.print(f"\n[bold yellow]{proposal.title}[/bold yellow]")
    files = getattr(proposal, "files", None)
    if files:
        for change in files:
            console.print(f"[cyan]{change.path}[/cyan]")
            console.print(change.preview_diff or change.diff)
    else:
        console.print(f"[cyan]{getattr(proposal, 'path', '')}[/cyan]")
        console.print(getattr(proposal, "diff", ""))
    if not sys.stdin.isatty():
        console.print("[yellow]Interactive approval is required; change rejected.[/yellow]")
        return ApprovalDecision("rejected", "Interactive approval is required in CLI one-shot mode.")
    choice = Prompt.ask(
        "Apply this change? [y]es / [n]o / [e]dit",
        choices=["y", "n", "e"],
        default="n",
    )
    if choice == "e":
        feedback = Prompt.ask("Change request")
        return ApprovalDecision("revision_requested", feedback)
    return choice == "y"


class _CliMaintenance:
    """Keep the Web scheduler's background behavior without blocking the REPL."""

    def __init__(self, agent: Agent, config: Config):
        self.loop = asyncio.new_event_loop()
        self.scheduler = None
        self._ready = threading.Event()
        self.thread = threading.Thread(
            target=self._run,
            args=(agent, config),
            name="folium-memory-maintenance",
            daemon=True,
        )
        self.thread.start()
        self._ready.wait(timeout=5)

    def _run(self, agent: Agent, config: Config):
        asyncio.set_event_loop(self.loop)
        self.scheduler = MemoryMaintenanceScheduler(
            lambda: build_memory_maintenance_runner(agent, config),
            threshold=getattr(config, "memory_maintenance_turns", 10),
            max_context_tokens=getattr(
                getattr(agent, "context", None),
                "max_tokens",
                getattr(config, "max_context_tokens", 1_000_000),
            ),
            max_output_tokens=getattr(config, "memory_maintenance_max_tokens", 2_000),
        )
        self._ready.set()
        self.loop.run_forever()

    def submit(self, **kwargs):
        if self.scheduler is None:
            return None
        return asyncio.run_coroutine_threadsafe(
            self.scheduler.on_turn_completed(**kwargs), self.loop
        )

    def wait(self, session_id: str, timeout: float = 30) -> None:
        """Wait for a scheduled pass when a short-lived CLI process needs it."""
        if self.scheduler is None:
            return

        async def wait_for_task():
            async with self.scheduler._lock:
                state = self.scheduler._states.get(session_id)
                task = state.task if state else None
            if task is not None:
                await task

        future = asyncio.run_coroutine_threadsafe(wait_for_task(), self.loop)
        future.result(timeout=timeout)


def _reset_last_llm_usage(llm) -> None:
    for name in ("last_prompt_tokens", "last_completion_tokens", "last_cached_tokens"):
        if hasattr(llm, name):
            setattr(llm, name, 0)
