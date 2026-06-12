"""FastAPI server with SSE streaming for CoreCoder."""

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from ..agent import Agent
from ..config import Config
from ..session import save_session, load_session, list_sessions, delete_session
from ..context import estimate_tokens
from ..tools.edit import _changed_files

STATIC_DIR = Path(__file__).parent / "static"

# mutable state dict — avoids global declaration ordering issues
_state = {
    "agent": None,
    "config": None,
    "session_id": None,
}
_chat_lock = asyncio.Lock()

app = FastAPI(title="CoreCoder")


# ── request models ──────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str

class CommandRequest(BaseModel):
    command: str

class SwitchRequest(BaseModel):
    session_id: str


# ── SSE bridge: sync callbacks → async queue ────────────────

def _make_bridge(queue: asyncio.Queue):
    """Create on_token/on_tool callbacks that push events into an asyncio.Queue."""
    def on_token(tok: str):
        queue.put_nowait({"type": "token", "content": tok})

    def on_tool(name: str, kwargs: dict):
        queue.put_nowait({"type": "tool", "name": name, "kwargs": kwargs})

    return on_token, on_tool


# ── auto-save helper ────────────────────────────────────────

def _auto_save():
    """Save current conversation to disk."""
    agent = _state["agent"]
    config = _state["config"]
    if agent and agent.messages:
        sid = save_session(agent.messages, config.model, _state["session_id"])
        _state["session_id"] = sid


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
    on_token, on_tool = _make_bridge(queue)

    async with _chat_lock:
        task = asyncio.create_task(
            asyncio.to_thread(_state["agent"].chat, req.message, on_token=on_token, on_tool=on_tool)
        )

        def _on_complete(t):
            if t.exception():
                queue.put_nowait({"type": "error", "content": str(t.exception())})
            else:
                queue.put_nowait({"type": "done", "content": ""})
        task.add_done_callback(_on_complete)

        async def event_stream():
            while True:
                event = await queue.get()
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
    # return messages so frontend can render them
    return {
        "result": f"Switched to {req.session_id}",
        "session_id": _state["session_id"],
        "messages": messages,
    }


@app.post("/delete")
async def delete_conversation(req: SwitchRequest):
    if req.session_id == _state["session_id"]:
        return JSONResponse({"error": "Cannot delete current conversation"}, status_code=400)
    deleted = delete_session(req.session_id)
    if deleted:
        return {"result": f"Deleted {req.session_id}"}
    return JSONResponse({"error": "Session not found"}, status_code=404)


@app.post("/command")
async def command(req: CommandRequest):
    agent = _state["agent"]
    config = _state["config"]
    cmd = req.command.strip()

    if cmd in ("/reset", "reset"):
        agent.reset()
        _state["session_id"] = None
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

    if cmd in ("/diff", "diff"):
        if not _changed_files:
            return {"result": "No files modified this session."}
        files = sorted(_changed_files)
        return {"result": f"Files modified ({len(files)}):\n" + "\n".join(files)}

    if cmd in ("/save", "save"):
        _auto_save()
        return {"result": f"Session saved: {_state['session_id']}"}

    if cmd in ("/sessions", "sessions"):
        sessions = list_sessions()
        if not sessions:
            return {"result": "No saved sessions."}
        lines = [f"  {s['id']} ({s['model']}, {s['updated_at']}) {s['preview']}" for s in sessions]
        return {"result": "\n".join(lines)}

    if cmd.startswith("/model ") or cmd.startswith("model "):
        new_model = cmd.split(" ", 1)[1].strip()
        agent.llm.model = new_model
        config.model = new_model
        return {"result": f"Switched to {new_model}"}

    if cmd in ("/model", "model"):
        return {"result": f"Current model: {config.model}"}

    if cmd in ("/help", "help"):
        return {"result": "Commands: /help /reset /tokens /compact /diff /save /sessions /model <name>"}

    return {"result": f"Unknown command: {cmd}"}


# ── server runner ───────────────────────────────────────────

def run_server(agent: Agent, config: Config, host: str = "0.0.0.0", port: int = 8000):
    _state["agent"] = agent
    _state["config"] = config
    _state["session_id"] = None

    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")