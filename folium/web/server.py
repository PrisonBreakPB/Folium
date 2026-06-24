"""FastAPI server with SSE streaming for Folium."""

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from ..agent import Agent
from ..config import Config
from ..session import save_session, load_session, list_sessions, delete_session, new_session_id
from ..context import estimate_tokens
from ..encoding import repair_mojibake_payload
from ..tools.edit import _changed_files
from ..observability import delete_traces_for_session, list_traces, read_trace_summary

STATIC_DIR = Path(__file__).parent / "static"

# mutable state dict — avoids global declaration ordering issues
_state = {
    "agent": None,
    "config": None,
    "session_id": None,
    "dirty": False,
}
_chat_lock = asyncio.Lock()

app = FastAPI(title="Folium")


# ── request models ──────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str

class CommandRequest(BaseModel):
    command: str

class SwitchRequest(BaseModel):
    session_id: str


# ── SSE bridge: sync callbacks → async queue ────────────────

def _make_bridge(queue: asyncio.Queue):
    """Create callbacks that push agent events into an asyncio.Queue."""
    def on_token(tok: str):
        queue.put_nowait({"type": "token", "content": tok})

    def on_tool(name: str, kwargs: dict, status: str | None = None):
        # Web uses structured on_event updates; keep this callback for Agent
        # compatibility without duplicating tool rows in the UI.
        return None

    def on_event(event: dict):
        queue.put_nowait(event)

    return on_token, on_tool, on_event


# ── auto-save helper ────────────────────────────────────────

def _auto_save():
    """Save current conversation to disk."""
    agent = _state["agent"]
    config = _state["config"]
    if agent and agent.messages and _state["dirty"]:
        sid = save_session(agent.messages, config.model, _state["session_id"])
        _state["session_id"] = sid
        agent.session_id = sid
        _state["dirty"] = False


# ── endpoints ───────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html = (STATIC_DIR / "index.html").read_text("utf-8")
    return HTMLResponse(html)


@app.post("/chat")
async def chat(req: ChatRequest):
    if _chat_lock.locked():
        return JSONResponse({"error": "A chat is already in progress"}, status_code=409)

    queue: asyncio.Queue = asyncio.Queue()
    on_token, on_tool, on_event = _make_bridge(queue)

    async with _chat_lock:
        if _state["session_id"] is None:
            _state["session_id"] = new_session_id()
        _state["agent"].session_id = _state["session_id"]
        task = asyncio.create_task(
            asyncio.to_thread(
                _state["agent"].chat,
                req.message,
                on_token=on_token,
                on_tool=on_tool,
                on_event=on_event,
            )
        )

        def _on_complete(t):
            if t.exception():
                queue.put_nowait({"type": "error", "content": str(t.exception())})
            else:
                _state["dirty"] = True
                queue.put_nowait({"type": "done", "content": ""})
        task.add_done_callback(_on_complete)

        async def event_stream():
            while True:
                event = await queue.get()
                event = repair_mojibake_payload(event)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event["type"] in ("done", "error"):
                    break

        # use BackgroundTask to auto-save after response is sent
        from starlette.background import BackgroundTask
        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            background=BackgroundTask(_auto_save),
        )


@app.post("/new")
async def new_conversation():
    if _chat_lock.locked():
        return JSONResponse({"error": "Chat in progress, cannot switch"}, status_code=409)

    _auto_save()
    _state["agent"].reset()
    _state["session_id"] = None
    _state["agent"].session_id = None
    _state["dirty"] = False
    return {"result": "New conversation started.", "session_id": _state["session_id"]}


@app.get("/conversations")
async def get_conversations():
    sessions = list_sessions()
    return {"sessions": sessions, "current": _state["session_id"]}


@app.post("/switch")
async def switch_conversation(req: SwitchRequest):
    if _chat_lock.locked():
        return JSONResponse({"error": "Chat in progress, cannot switch"}, status_code=409)

    _auto_save()

    loaded = load_session(req.session_id)
    if not loaded:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    messages, model = loaded
    _state["agent"].messages = messages
    _state["session_id"] = req.session_id
    _state["agent"].session_id = req.session_id
    _state["dirty"] = False
    # return messages so frontend can render them
    return {
        "result": f"Switched to {req.session_id}",
        "session_id": _state["session_id"],
        "messages": repair_mojibake_payload(messages),
    }


@app.post("/delete")
async def delete_conversation(req: SwitchRequest):
    deleted_current = req.session_id == _state["session_id"]
    deleted = delete_session(req.session_id)
    if deleted:
        deleted_traces = delete_traces_for_session(req.session_id)
        if deleted_current:
            _state["agent"].reset()
            _state["session_id"] = None
            _state["agent"].session_id = None
            _state["dirty"] = False
        return {
            "result": f"Deleted {req.session_id}",
            "deleted_traces": deleted_traces,
            "deleted_current": deleted_current,
        }
    return JSONResponse({"error": "Session not found"}, status_code=404)


@app.post("/command")
async def command(req: CommandRequest):
    agent = _state["agent"]
    config = _state["config"]
    cmd = req.command.strip()

    if cmd in ("/reset", "reset"):
        agent.reset()
        _state["session_id"] = None
        agent.session_id = None
        _state["dirty"] = False
        return {"result": "Conversation reset."}

    if cmd in ("/tokens", "tokens"):
        p = agent.llm.total_prompt_tokens
        c = agent.llm.total_completion_tokens
        total = p + c
        line = f"Tokens: {p} prompt + {c} completion = {total} total"
        cost = agent.llm.estimated_cost
        if cost is not None:
            line += f"  (~${cost:.4f})"
        return {"result": line}

    if cmd in ("/compact", "compact"):
        before = estimate_tokens(agent.messages)
        compressed = agent.context.maybe_compress(agent.messages, agent.llm)
        after = estimate_tokens(agent.messages)
        if compressed:
            return {"result": f"Compressed: {before} -> {after} tokens ({len(agent.messages)} messages)"}
        return {"result": f"Nothing to compress ({before} tokens, {len(agent.messages)} messages)"}

    if cmd in ("/skills", "skills"):
        skills = getattr(agent, "skills", [])
        if not skills:
            return {"result": "No skills found. Add skills under skills/<name>/SKILL.md."}
        lines = ["Available skills:"]
        for skill in skills:
            lines.append(f"- {skill.name}: {skill.description}\n  {skill.skill_file}")
        return {"result": "\n".join(lines)}

    if cmd in ("/diff", "diff"):
        if not _changed_files:
            return {"result": "No files modified this session."}
        files = sorted(_changed_files)
        return {"result": f"Files modified ({len(files)}):\n" + "\n".join(files)}

    if cmd in ("/save", "save"):
        _state["dirty"] = True
        _auto_save()
        return {"result": f"Session saved: {_state['session_id']}"}

    if cmd in ("/sessions", "sessions"):
        sessions = list_sessions()
        if not sessions:
            return {"result": "No saved sessions."}
        lines = [f"  {s['id']} ({s['model']}, {s['updated_at']}) {s['preview']}" for s in sessions]
        return {"result": "\n".join(lines)}

    if cmd in ("/traces", "traces"):
        traces = list_traces()
        if not traces:
            return {"result": "No traces found."}
        lines = [
            (
                f"  {t['trace_id']} status={t['status']} "
                f"duration={t['duration_ms']}ms llm={t['llm_calls']} "
                f"tools={t['tool_calls']} errors={t['errors']}"
            )
            for t in traces
        ]
        return {"result": "\n".join(lines)}

    if cmd.startswith("/trace ") or cmd.startswith("trace "):
        trace_id = cmd.split(" ", 1)[1].strip()
        summary = read_trace_summary(trace_id)
        if not summary:
            return JSONResponse({"error": "Trace not found"}, status_code=404)
        lines = [
            f"Trace: {summary['trace_id']}",
            f"Status: {summary['status']}",
            f"Duration: {summary['duration_ms']}ms",
            f"LLM calls: {summary['llm_calls']}",
            f"Tool calls: {summary['tool_calls']}",
            f"Errors: {summary['errors']}",
        ]
        for s in summary.get("spans", [])[:20]:
            lines.append(
                f"- {s['type']}:{s['name']} {s['status']} {s['duration_ms']}ms"
            )
        return {"result": "\n".join(lines)}

    if cmd.startswith("/model ") or cmd.startswith("model "):
        new_model = cmd.split(" ", 1)[1].strip()
        agent.llm.model = new_model
        config.model = new_model
        return {"result": f"Switched to {new_model}"}

    if cmd in ("/model", "model"):
        return {"result": f"Current model: {config.model}"}

    if cmd in ("/help", "help"):
        return {"result": "Commands: /help /reset /skills /tokens /compact /diff /save /sessions /traces /trace <id> /model <name>"}

    return {"result": f"Unknown command: {cmd}"}


# ── server runner ───────────────────────────────────────────

def run_server(agent: Agent, config: Config, host: str = "0.0.0.0", port: int = 8000):
    _state["agent"] = agent
    _state["config"] = config
    _state["session_id"] = None
    _state["dirty"] = False
    agent.session_id = None

    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")
