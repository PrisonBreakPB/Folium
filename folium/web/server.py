"""FastAPI server with SSE streaming for Folium."""

import asyncio
import copy
import json
import os
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from ..agent import Agent
from ..config import Config
from ..llm import LLMProviderError
from ..memory_maintenance import (
    MemoryAgent,
    MemoryMaintenanceScheduler,
)
from ..session import (
    calculate_session_stats,
    delete_session,
    ensure_session,
    list_sessions,
    load_session,
    new_session_id,
    save_session,
)
from ..context import estimate_tokens
from ..encoding import repair_mojibake_payload
from ..tools.edit import _changed_files
from ..observability import delete_traces_for_session, list_traces, read_trace_summary
from ..sandbox.session import reset_current_session

STATIC_DIR = Path(__file__).parent / "static"

# mutable state dict — avoids global declaration ordering issues
_state = {
    "agent": None,
    "config": None,
    "session_id": None,
    "dirty": False,
    "memory_maintenance": None,
}
_chat_lock = asyncio.Lock()
_pending_approvals = {}

app = FastAPI(title="Folium")


# ── request models ──────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str

class CommandRequest(BaseModel):
    command: str

class SwitchRequest(BaseModel):
    session_id: str


class ApprovalDecisionRequest(BaseModel):
    approval_id: str
    approved: bool


class PendingApproval:
    def __init__(self, payload: dict, proposal=None):
        self.payload = payload
        self.proposal = proposal
        self.event = threading.Event()
        self.approved = False


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

    def on_edit_approval(tc, proposal):
        approval_id = uuid.uuid4().hex
        if hasattr(proposal, "files"):
            files = [
                {
                    "index": index,
                    "tool_call_id": change.tool_call_id,
                    "tool_name": change.tool_name,
                    "path": change.path,
                    "title": change.title,
                    "diff": change.preview_diff,
                    "truncated": change.truncated,
                    "diff_chars": change.diff_chars,
                    "additions": change.additions,
                    "deletions": change.deletions,
                }
                for index, change in enumerate(proposal.files)
            ]
            payload = {
                "type": "approval_request",
                "approval_id": approval_id,
                "tool_call_id": tc.id,
                "tool_name": "change_set",
                "title": proposal.title,
                "files": files,
                "additions": proposal.additions,
                "deletions": proposal.deletions,
            }
        else:
            payload = {
                "type": "approval_request",
                "approval_id": approval_id,
                "tool_call_id": tc.id,
                "tool_name": tc.name,
                "path": proposal.path,
                "title": proposal.title,
                "diff": proposal.diff,
                "truncated": proposal.truncated,
                "diff_chars": proposal.diff_chars,
            }
        pending = PendingApproval(payload, proposal)
        _pending_approvals[approval_id] = pending
        queue.put_nowait(payload)
        pending.event.wait()
        _pending_approvals.pop(approval_id, None)
        return pending.approved

    return on_token, on_tool, on_event, on_edit_approval


# ── auto-save helper ────────────────────────────────────────

def _auto_save():
    """Save current conversation to disk."""
    agent = _state["agent"]
    config = _state["config"]
    if agent and agent.messages and _state["dirty"]:
        sid = save_session(
            agent.messages,
            config.model,
            _state["session_id"],
            transcript=getattr(agent, "transcript", agent.messages),
            system_prompt=agent._system,
        )
        _state["session_id"] = sid
        agent.session_id = sid
        _state["dirty"] = False


async def _after_chat_response(completion: dict | None = None):
    """Persist the completed turn, then schedule maintenance without waiting for it."""
    _auto_save()
    scheduler = _state.get("memory_maintenance")
    if not scheduler or not completion:
        return
    await scheduler.on_turn_completed(
        session_id=completion["session_id"],
        messages=completion["messages"],
        visible_tools=completion["visible_tools"],
        main_agent_used_memory=completion["main_agent_used_memory"],
        main_prompt_tokens=completion["main_prompt_tokens"],
        main_completion_tokens=completion["main_completion_tokens"],
        main_request_matches_memory_context=completion["main_request_matches_memory_context"],
    )


def _new_memory_maintenance_runner(agent: Agent, config: Config) -> MemoryAgent:
    llm_cls = type(agent.llm)
    llm = llm_cls(
        model=agent.llm.model,
        api_key=getattr(config, "api_key", ""),
        base_url=getattr(config, "base_url", None),
        temperature=getattr(config, "temperature", 0.0),
        max_tokens=getattr(config, "memory_maintenance_max_tokens", 2000),
    )
    return MemoryAgent(
        llm,
        max_steps=getattr(config, "memory_maintenance_max_steps", 5),
    )


def _context_budget_payload() -> dict:
    agent = _state.get("agent")
    context = getattr(agent, "context", None)
    if not context:
        return {}
    return {
        "max_context_tokens": getattr(context, "max_tokens", 0),
        "reserved_output_tokens": getattr(context, "reserved_output_tokens", 0),
        "input_budget_tokens": getattr(context, "input_budget_tokens", 0),
    }


def _todo_payload() -> dict:
    agent = _state.get("agent")
    manager = getattr(agent, "todo_manager", None)
    if not manager:
        return {"items": [], "rendered": "No todos."}
    return {"items": manager.snapshot(), "rendered": manager.render()}


def _error_event(exc: BaseException) -> dict:
    if isinstance(exc, LLMProviderError):
        info = exc.info.to_dict()
        return {
            "type": "error",
            "content": info["message"],
            **info,
        }
    return {"type": "error", "content": str(exc)}


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
    on_token, on_tool, on_event, on_edit_approval = _make_bridge(queue)

    async with _chat_lock:
        if _state["session_id"] is None:
            _state["session_id"] = new_session_id()
            ensure_session(
                _state["session_id"],
                _state["config"].model,
                _state["agent"]._system,
            )
        _state["agent"].session_id = _state["session_id"]
        _state["agent"].edit_approval_callback = on_edit_approval
        chat_session_id = _state["session_id"]
        transcript_start = len(_state["agent"].transcript)
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
            _state["agent"].edit_approval_callback = None
            if t.exception():
                queue.put_nowait(_error_event(t.exception()))
            else:
                _state["dirty"] = True
                turn_transcript = _state["agent"].transcript[transcript_start:]
                completion["messages"] = copy.deepcopy(_state["agent"]._full_messages())
                completion["visible_tools"] = copy.deepcopy(_state["agent"]._tool_schemas())
                completion["session_id"] = chat_session_id
                completion["main_prompt_tokens"] = getattr(
                    _state["agent"].llm,
                    "last_prompt_tokens",
                    0,
                )
                completion["main_completion_tokens"] = getattr(
                    _state["agent"].llm,
                    "last_completion_tokens",
                    0,
                )
                completion["main_request_matches_memory_context"] = getattr(
                    _state["agent"],
                    "last_llm_request_had_visible_tools",
                    False,
                )
                completion["main_agent_used_memory"] = any(
                    message.get("role") == "tool" and message.get("name") == "memory"
                    for message in turn_transcript
                )
                queue.put_nowait({"type": "done", "content": ""})
        completion: dict = {}
        task.add_done_callback(_on_complete)

        async def event_stream():
            while True:
                event = await queue.get()
                event = repair_mojibake_payload(event)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
                if event["type"] in ("done", "error"):
                    break

        # Run after the final SSE event has been sent; this does not await maintenance.
        from starlette.background import BackgroundTask
        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            background=BackgroundTask(_after_chat_response, completion),
        )


@app.post("/approval")
async def approval(req: ApprovalDecisionRequest):
    pending = _pending_approvals.get(req.approval_id)
    if pending is None:
        return JSONResponse({"error": "审批请求不存在或已失效"}, status_code=404)
    pending.approved = bool(req.approved)
    pending.event.set()
    return {"result": "ok"}


@app.get("/approval/{approval_id}/diff")
async def approval_diff(
    approval_id: str,
    file_index: int = 0,
    offset: int = 0,
    limit: int = 30_000,
):
    pending = _pending_approvals.get(approval_id)
    if pending is None:
        return JSONResponse({"error": "Approval request not found"}, status_code=404)

    files = getattr(pending.proposal, "files", None)
    if files is None or file_index < 0 or file_index >= len(files):
        return JSONResponse({"error": "Approval file not found"}, status_code=404)

    diff = files[file_index].diff
    start = max(offset, 0)
    size = max(limit, 1)
    chunk = diff[start:start + size]
    next_offset = start + len(chunk)
    return {
        "file_index": file_index,
        "offset": start,
        "diff": chunk,
        "next_offset": next_offset if next_offset < len(diff) else None,
        "total_chars": len(diff),
    }


@app.post("/new")
async def new_conversation():
    if _chat_lock.locked():
        return JSONResponse({"error": "Chat in progress, cannot switch"}, status_code=409)

    _auto_save()
    _state["agent"].reset()
    _state["session_id"] = None
    _state["agent"].session_id = None
    _state["dirty"] = False
    reset_current_session()
    return {"result": "New conversation started.", "session_id": _state["session_id"]}


@app.get("/conversations")
async def get_conversations():
    sessions = list_sessions()
    return {"sessions": sessions, "current": _state["session_id"], "context_budget": _context_budget_payload()}


@app.get("/todos")
async def get_todos():
    return _todo_payload()


@app.get("/skills")
async def get_skills():
    agent = _state.get("agent")
    if not agent:
        return {"skills": []}
    skills = getattr(agent, "skills", [])
    return {"skills": [{"name": s.name, "description": s.description} for s in skills]}


@app.post("/switch")
async def switch_conversation(req: SwitchRequest):
    if _chat_lock.locked():
        return JSONResponse({"error": "Chat in progress, cannot switch"}, status_code=409)

    _auto_save()

    loaded = load_session(req.session_id)
    if not loaded:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    messages, model, transcript, system_prompt = loaded
    _state["agent"].messages = messages
    _state["agent"].transcript = transcript
    _state["agent"].reset_todos()
    _state["session_id"] = req.session_id
    _state["agent"].session_id = req.session_id
    _state["dirty"] = False
    reset_current_session()

    # Restore system prompt from DB if available
    if system_prompt is not None:
        _state["agent"]._system = system_prompt

    # Restore cumulative token counts from persisted LLM trace events.
    llm = _state["agent"].llm
    llm.total_prompt_tokens = 0
    llm.total_completion_tokens = 0
    llm.total_cached_tokens = 0
    llm.last_prompt_tokens = 0
    llm.last_completion_tokens = 0
    stats = calculate_session_stats(req.session_id)
    llm.total_prompt_tokens = stats["prompt_tokens"]
    llm.total_completion_tokens = stats["completion_tokens"]
    llm.total_cached_tokens = stats["cached_tokens"]
    return {
        "result": f"Switched to {req.session_id}",
        "session_id": _state["session_id"],
        "messages": repair_mojibake_payload(transcript),
        "stats": stats,
        "context_budget": _context_budget_payload(),
    }


@app.post("/delete")
async def delete_conversation(req: SwitchRequest):
    deleted_current = req.session_id == _state["session_id"]
    deleted = delete_session(req.session_id)
    if deleted:
        # Session foreign-key cascade already removed related traces.
        deleted_traces = 0
        if deleted_current:
            _state["agent"].reset()
            _state["session_id"] = None
            _state["agent"].session_id = None
            _state["dirty"] = False
            reset_current_session()
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
        reset_current_session()
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
        report = agent.context.maybe_compress(agent.messages, agent.llm)
        after = estimate_tokens(agent.messages)
        if report["compressed"]:
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
    os.environ.setdefault("FOLIUM_SANDBOX_WORKSPACE_MODE", "copy")
    reset_current_session()
    _state["agent"] = agent
    _state["config"] = config
    _state["session_id"] = None
    _state["dirty"] = False
    _state["memory_maintenance"] = MemoryMaintenanceScheduler(
        lambda: _new_memory_maintenance_runner(agent, config),
        threshold=getattr(config, "memory_maintenance_turns", 10),
        max_context_tokens=getattr(
            getattr(agent, "context", None),
            "max_tokens",
            getattr(config, "max_context_tokens", 1_000_000),
        ),
        max_output_tokens=getattr(config, "memory_maintenance_max_tokens", 2_000),
    )
    agent.session_id = None

    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")
