"""JSON Lines bridge used by the TypeScript/Ink CLI UI."""

from __future__ import annotations

import copy
import json
import queue
import sys
import threading
import uuid
from typing import TextIO

from .context import estimate_tokens
from .edit_approval import ApprovalDecision
from .memory_maintenance import main_agent_wrote_to_memory
from .sandbox.session import (
    configure_host_workspace,
    get_current_session,
    reset_current_session,
    use_bash_sandbox,
    use_copy_workspace,
)
from .session import (
    calculate_session_stats,
    delete_session,
    get_session_workspace,
    list_sessions,
    load_session,
    normalize_workspace_path,
)
from .observability import list_traces, read_trace_summary


class JsonlServer:
    """Keep Agent execution in Python while exposing UI-safe JSON events."""

    def __init__(
        self,
        agent,
        config,
        workspace: str,
        session_id: str | None = None,
        input_stream: TextIO | None = None,
        output_stream: TextIO | None = None,
        maintenance=None,
    ):
        self.agent = agent
        self.config = config
        self.workspace = workspace
        self.session_id = session_id
        self.input_stream = input_stream or sys.stdin
        self.output_stream = output_stream or sys.stdout
        self._write_lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._approval_queues: dict[str, queue.Queue] = {}
        self._approval_lock = threading.Lock()
        self._stop = threading.Event()
        if maintenance is None:
            from .cli import _CliMaintenance

            maintenance = _CliMaintenance(agent, config)
        self.maintenance = maintenance
        self.agent.edit_approval_callback = self._request_approval

    def emit(self, event_type: str, **payload) -> None:
        event = {"type": event_type, **payload}
        line = json.dumps(event, ensure_ascii=False, default=str)
        with self._write_lock:
            self.output_stream.write(line + "\n")
            self.output_stream.flush()

    def run(self) -> None:
        self.emit(
            "ready",
            model=self.agent.llm.model,
            mode=self.agent.mode,
            workspace=self.workspace,
            session_id=self.session_id,
            skills=[getattr(skill, "name", "") for skill in getattr(self.agent, "skills", [])],
        )
        try:
            for raw_line in self.input_stream:
                if self._stop.is_set():
                    break
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    request = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    self.emit("error", message=f"Invalid JSON request: {exc}")
                    continue
                self._handle_request(request)
        finally:
            self._stop.set()
            self._release_approvals()
            if self._worker and self._worker.is_alive():
                self._worker.join(timeout=2)
            self._cleanup()

    def _handle_request(self, request: dict) -> None:
        request_type = request.get("type")
        request_id = request.get("request_id")
        if request_type == "message":
            self._start_message(str(request.get("content", "")), request_id)
        elif request_type == "command":
            self._handle_command(str(request.get("command", "")), request_id)
        elif request_type == "approve":
            self._resolve_approval(request)
        elif request_type == "shutdown":
            self.emit("bye")
            self._stop.set()
        else:
            self.emit("error", request_id=request_id, message="Unknown request type")

    def _start_message(self, content: str, request_id: str | None) -> None:
        if not content.strip():
            return
        if self._worker and self._worker.is_alive():
            self.emit("error", request_id=request_id, message="Agent is still processing the previous request.")
            return
        self._worker = threading.Thread(
            target=self._run_message,
            args=(content, request_id),
            name="folium-agent-turn",
            daemon=True,
        )
        self._worker.start()

    def _run_message(self, content: str, request_id: str | None) -> None:
        from .cli import _ensure_session, _persist_session

        streamed = False
        try:
            self.session_id = _ensure_session(
                self.agent, self.config, self.workspace, self.session_id
            )
            transcript_start = len(self.agent.transcript)

            def on_token(token: str) -> None:
                nonlocal streamed
                streamed = True
                self.emit("token", request_id=request_id, content=token)

            response = self.agent.chat(
                content,
                on_token=on_token,
                on_event=lambda event: self._forward_agent_event(event, request_id),
            )
            self.agent.session_id = self.session_id
            self.session_id = _persist_session(
                self.agent, self.config, self.workspace, self.session_id
            )
            self._submit_maintenance(transcript_start)
            self.emit(
                "done",
                request_id=request_id,
                response=response,
                streamed=streamed,
                session_id=self.session_id,
                model=self.agent.llm.model,
            )
        except Exception as exc:
            self.emit("error", request_id=request_id, message=str(exc))

    def _forward_agent_event(self, event: dict, request_id: str | None) -> None:
        self.emit("agent_event", request_id=request_id, event=event)

    def _submit_maintenance(self, transcript_start: int) -> None:
        if self.maintenance is None:
            return
        self.maintenance.submit(
            session_id=self.session_id,
            messages=copy.deepcopy(self.agent._full_messages()),
            visible_tools=copy.deepcopy(self.agent._tool_schemas()),
            main_agent_used_memory=main_agent_wrote_to_memory(
                self.agent.transcript[transcript_start:]
            ),
            main_prompt_tokens=getattr(self.agent.llm, "last_prompt_tokens", 0),
            main_completion_tokens=getattr(self.agent.llm, "last_completion_tokens", 0),
            main_request_matches_memory_context=getattr(
                self.agent, "last_llm_request_had_visible_tools", False
            ),
        )

    def _request_approval(self, _tool_call, proposal):
        request_id = uuid.uuid4().hex
        pending = queue.Queue(maxsize=1)
        with self._approval_lock:
            self._approval_queues[request_id] = pending
        self.emit("approval_required", request_id=request_id, proposal=_proposal_payload(proposal))
        try:
            action, feedback = pending.get()
            return ApprovalDecision(action, feedback)
        finally:
            with self._approval_lock:
                self._approval_queues.pop(request_id, None)

    def _resolve_approval(self, request: dict) -> None:
        request_id = str(request.get("request_id", ""))
        action = str(request.get("decision", "rejected"))
        if action not in {"approved", "rejected", "revision_requested"}:
            self.emit("error", message="Invalid approval decision")
            return
        with self._approval_lock:
            pending = self._approval_queues.get(request_id)
        if pending is None:
            self.emit("error", message="Approval request not found")
            return
        pending.put((action, str(request.get("feedback", ""))))

    def _release_approvals(self) -> None:
        with self._approval_lock:
            pending = list(self._approval_queues.values())
        for item in pending:
            try:
                item.put_nowait(("rejected", "CLI UI disconnected"))
            except queue.Full:
                pass

    def _handle_command(self, raw: str, request_id: str | None) -> None:
        command = raw.strip()
        lowered = command.lower()
        if lowered in {"exit", "/exit"}:
            self.emit("command_result", request_id=request_id, command=command, text="Bye!")
            self.emit("bye")
            self._stop.set()
            return
        if self._worker and self._worker.is_alive():
            self.emit("error", request_id=request_id, message="Wait for the current request to finish.")
            return

        if lowered in {"/help", "help"}:
            self._result(request_id, command, _help_text())
        elif lowered in {"/new", "new", "/reset", "reset"}:
            self._reset_conversation()
            self._result(request_id, command, "Conversation reset.")
        elif lowered == "/model" or lowered.startswith("/model "):
            value = command[6:].strip()
            if value:
                self.agent.llm.model = value
                self.config.model = value
                self._result(request_id, command, f"Switched to {value}")
            else:
                self._result(request_id, command, f"Current model: {self.config.model}")
        elif lowered in {"/mode", "mode"} or lowered.startswith("/mode "):
            value = command[5:].strip()
            if not value:
                self._result(request_id, command, f"Current mode: {self.agent.mode}")
            else:
                try:
                    self.agent.set_mode(value)
                    self._result(request_id, command, f"Switched to {self.agent.mode}")
                except ValueError as exc:
                    self._result(request_id, command, str(exc), level="error")
        elif lowered in {"/status", "status", "/usage", "usage"}:
            self._result(request_id, command, data=self._status_payload(), kind="status")
        elif lowered in {"/context", "context"}:
            self._result(request_id, command, data=self._context_payload(), kind="context")
        elif lowered in {"/tokens", "tokens"}:
            self._result(request_id, command, _tokens_text(self.agent.llm))
        elif lowered in {"/skills", "skills"}:
            self._result(request_id, command, _skills_text(self.agent))
        elif lowered == "/workspace":
            self._result(request_id, command, self._workspace_text())
        elif lowered == "/todos":
            manager = getattr(self.agent, "todo_manager", None)
            self._result(request_id, command, manager.render() if manager else "No todos.")
        elif lowered == "/compact":
            before = estimate_tokens(self.agent.messages)
            report = self.agent.context.maybe_compress(self.agent.messages, self.agent.llm)
            after = estimate_tokens(self.agent.messages)
            text = (
                f"Compressed: {before} -> {after} tokens ({len(self.agent.messages)} messages)"
                if report["compressed"]
                else f"Nothing to compress ({before} tokens, {len(self.agent.messages)} messages)"
            )
            self._result(request_id, command, text)
        elif lowered == "/save":
            self.session_id = self._persist()
            self._result(request_id, command, f"Session saved: {self.session_id}")
        elif lowered == "/diff":
            from .tools.edit import _changed_files

            text = "No files modified this session." if not _changed_files else (
                f"Files modified this session ({len(_changed_files)}):\n"
                + "\n".join(sorted(_changed_files))
            )
            self._result(request_id, command, text)
        elif lowered == "/sessions":
            sessions = list_sessions()
            text = "No saved sessions." if not sessions else "\n".join(
                f"{s['id']} ({s['model']}, {s['updated_at']}) {s['preview']}"
                for s in sessions
            )
            self._result(request_id, command, text)
        elif lowered.startswith("/switch "):
            self._switch_session(command[8:].strip(), request_id, command)
        elif lowered.startswith("/delete "):
            self._delete_session(command[8:].strip(), request_id, command)
        elif lowered == "/traces":
            traces = list_traces()
            text = "No traces found." if not traces else "\n".join(
                f"{t['trace_id']} status={t['status']} duration={t['duration_ms']}ms "
                f"llm={t['llm_calls']} tools={t['tool_calls']} errors={t['errors']}"
                for t in traces
            )
            self._result(request_id, command, text)
        elif lowered.startswith("/trace "):
            self._trace(command[7:].strip(), request_id, command)
        else:
            self._start_message(command, request_id)

    def _result(self, request_id, command, text=None, *, data=None, kind=None, level="ok"):
        payload = {
            "request_id": request_id,
            "command": command,
            "level": level,
            "session_id": self.session_id,
            "model": self.config.model,
            "mode": self.agent.mode,
            "workspace": self.workspace,
        }
        if text is not None:
            payload["text"] = text
        if data is not None:
            payload["data"] = data
        if kind is not None:
            payload["kind"] = kind
        self.emit("command_result", **payload)

    def _reset_conversation(self) -> None:
        from .cli import _cleanup_bash_executors, _reset_last_llm_usage

        self.agent.reset()
        _reset_last_llm_usage(self.agent.llm)
        _cleanup_bash_executors(self.agent)
        self.session_id = None
        self.agent.session_id = None
        reset_current_session()
        configure_host_workspace(self.workspace)

    def _persist(self):
        from .cli import _persist_session

        return _persist_session(self.agent, self.config, self.workspace, self.session_id)

    def _switch_session(self, target, request_id, command):
        from .cli import _cleanup_bash_executors, _reset_last_llm_usage

        loaded = load_session(target)
        if not loaded:
            self._result(request_id, command, "Session not found.", level="error")
            return
        self._persist()
        messages, model, transcript, saved_prompt = loaded
        saved_workspace = get_session_workspace(target)
        if saved_workspace:
            try:
                self.workspace = normalize_workspace_path(saved_workspace)
            except ValueError as exc:
                self._result(request_id, command, str(exc), level="error")
                return
            configure_host_workspace(self.workspace)
        _cleanup_bash_executors(self.agent)
        reset_current_session()
        self.agent.messages = messages
        self.agent.transcript = transcript
        self.agent.reset_todos()
        self.agent.session_id = target
        self.session_id = target
        self.agent.llm.model = model
        self.config.model = model
        if saved_prompt is not None:
            self.agent._system = saved_prompt
        stats = calculate_session_stats(target)
        _reset_last_llm_usage(self.agent.llm)
        self.agent.llm.total_prompt_tokens = stats["prompt_tokens"]
        self.agent.llm.total_completion_tokens = stats["completion_tokens"]
        self.agent.llm.total_cached_tokens = stats["cached_tokens"]
        self._result(request_id, command, f"Switched to {target}")

    def _delete_session(self, target, request_id, command):
        self._persist()
        if not delete_session(target):
            self._result(request_id, command, "Session not found.", level="error")
            return
        if target == self.session_id:
            self._reset_conversation()
        self._result(request_id, command, f"Deleted {target}")

    def _trace(self, trace_id, request_id, command):
        summary = read_trace_summary(trace_id)
        if not summary:
            self._result(request_id, command, "Trace not found.", level="error")
            return
        lines = [
            f"Trace: {summary['trace_id']}",
            f"Status: {summary['status']}",
            f"Duration: {summary['duration_ms']}ms",
            f"LLM calls: {summary['llm_calls']}",
            f"Tool calls: {summary['tool_calls']}",
            f"Errors: {summary['errors']}",
        ]
        lines.extend(
            f"{span['type']}:{span['name']} {span['status']} {span['duration_ms']}ms"
            for span in summary.get("spans", [])[:20]
        )
        self._result(request_id, command, "\n".join(lines))

    def _status_payload(self) -> dict:
        return {
            "session_id": self.session_id,
            "model": self.config.model,
            "mode": self.agent.mode,
            "workspace": self.workspace,
            **self._context_payload(),
        }

    def _context_payload(self) -> dict:
        context = getattr(self.agent, "context", None)
        max_context = getattr(context, "max_tokens", getattr(self.config, "max_context_tokens", 0))
        estimated = estimate_tokens(self.agent.messages)
        input_budget = getattr(context, "input_budget_tokens", max_context)
        reserved_output = getattr(context, "reserved_output_tokens", 0)
        llm = self.agent.llm
        total_prompt = getattr(llm, "total_prompt_tokens", 0)
        total_completion = getattr(llm, "total_completion_tokens", 0)
        total_cached = getattr(llm, "total_cached_tokens", 0)
        meter = getattr(self.agent, "_cost_meter", None)
        return {
            "estimated_context_tokens": estimated,
            "max_context_tokens": max_context,
            "context_usage_ratio": estimated / max_context if max_context else 0,
            "input_budget_tokens": input_budget,
            "input_budget_usage_ratio": estimated / input_budget if input_budget else 0,
            "reserved_output_tokens": reserved_output,
            "api_max_output_tokens": getattr(self.config, "max_tokens", 0),
            "last_prompt_tokens": getattr(llm, "last_prompt_tokens", 0),
            "last_completion_tokens": getattr(llm, "last_completion_tokens", 0),
            "last_cached_tokens": getattr(llm, "last_cached_tokens", 0),
            "last_total_tokens": getattr(llm, "last_prompt_tokens", 0) + getattr(llm, "last_completion_tokens", 0),
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_cached_tokens": total_cached,
            "total_tokens": total_prompt + total_completion,
            "cache_hit_rate": total_cached / total_prompt if total_prompt else 0,
            "estimated_cost": getattr(llm, "estimated_cost", None),
            "budget_usd": meter.budget_usd if meter is not None else 0,
            "budget_spent": meter.spent() if meter is not None else 0,
        }

    def _workspace_text(self) -> str:
        sandbox = (
            get_current_session(copy_workspace=use_copy_workspace())
            if use_bash_sandbox()
            else None
        )
        text = f"Host workspace: {self.workspace}"
        if sandbox:
            text += f"\nSandbox workspace: {sandbox.workspace}"
        return text

    def _cleanup(self) -> None:
        from .cli import _cleanup_bash_executors

        _cleanup_bash_executors(self.agent)


def _proposal_payload(proposal) -> dict:
    if hasattr(proposal, "files"):
        return {
            "title": proposal.title,
            "files": [
                {
                    "path": change.path,
                    "title": change.title,
                    "diff": change.preview_diff or change.diff,
                    "additions": change.additions,
                    "deletions": change.deletions,
                }
                for change in proposal.files
            ],
        }
    return {
        "title": getattr(proposal, "title", "Review change"),
        "path": getattr(proposal, "path", ""),
        "diff": getattr(proposal, "diff", ""),
    }


def _tokens_text(llm) -> str:
    prompt = getattr(llm, "total_prompt_tokens", 0)
    completion = getattr(llm, "total_completion_tokens", 0)
    text = f"Tokens: {prompt} prompt + {completion} completion = {prompt + completion} total"
    cost = getattr(llm, "estimated_cost", None)
    if cost is not None:
        text += f"  (~${cost:.4f})"
    return text


def _help_text() -> str:
    return (
        "Commands:\n"
        "/help /new /reset /model [name] /mode [name] /skills /status /context\n"
        "/workspace /todos /tokens /compact /diff /save /sessions /switch <id>\n"
        "/delete <id> /traces /trace <id> /exit"
    )


def _skills_text(agent) -> str:
    skills = getattr(agent, "skills", [])
    if not skills:
        return "No skills found."
    return "Available skills:\n" + "\n".join(
        f"/{skill.name}  {skill.description}" for skill in skills
    )


def run_jsonl(agent, config, workspace: str, session_id: str | None = None) -> None:
    JsonlServer(agent, config, workspace, session_id).run()
