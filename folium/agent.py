"""Core agent loop.

This is the heart of Folium.  The pattern is simple:

    user message -> LLM (with tools) -> tool calls? -> execute -> loop
                                      -> text reply? -> return to user

It keeps looping until the LLM responds with plain text (no tool calls),
which means it's done working and ready to report back.
"""

import concurrent.futures
import contextvars
import copy
import re
import threading
import time
from dataclasses import dataclass
from .skills.types import Skill
from .llm import LLM, LLMResponse, estimate_cost, parse_arguments_lenient
from .tools import create_tools
from .tools.base import Tool, ToolOutput, ToolValidationError
from .tools.agent import AgentTool
from .tools.session_history import SessionHistoryTool
from .tools.todo import TODO_REMINDER, TodoTool
from .prompt import system_prompt
from .context import ContextManager, estimate_tokens, _approx_tokens
from .config import DEFAULT_MAX_CONTEXT_TOKENS
from .skills import load_skills
from .skills.parser import parse_skill_file
from .observability import mark_current_span_status, observe_trace, span
from .observability.context import active_observer, current_span_id, current_trace_id
from .observability.redaction import compact_payload
from .encoding import repair_mojibake_text
from .edit_approval import (
    FileChangeProposal,
    apply_change_set,
    build_change_set_proposal,
    build_edit_approval_proposal,
    build_file_change_proposal,
    change_result_summary,
    is_protected_file_change,
    normalize_approval_decision,
)
from .sandbox.filesystem import resolve_tool_path
from .tools.edit import _changed_files


FINAL_ROUND_REMINDER = (
    "<reminder>\n"
    "This is the final allowed model round for this task. Do not call tools.\n\n"
    "Use only the information already gathered in the conversation and tool results.\n"
    "If that information is sufficient, provide the best possible answer to the user now.\n"
    "If it is not sufficient, do not guess, invent evidence, or force a conclusion. "
    "Clearly explain what is still missing, why it matters, and what the recommended next step should be.\n"
    "</reminder>"
)

TOOL_PARAMETER_WARN = (
    "Several of your recent tool calls had invalid arguments and were rejected. "
    "Stop retrying them unchanged; fix the arguments, change strategy, or explain the blocker."
)
TOOL_EXECUTION_WARN = (
    "Several of your recent tool calls failed during execution. "
    "Stop retrying them unchanged; change strategy or explain the blocker."
)

_PLAN_REMINDER = (
    "<system-reminder>\n"
    "# Plan Mode - System Reminder\n\n"
    "CRITICAL: Plan mode ACTIVE - you are in READ-ONLY phase. STRICTLY FORBIDDEN: "
    "ANY file edits, modifications, or system changes. Do NOT use sed, tee, echo, cat, "
    "or ANY other bash command to manipulate files - commands may ONLY read/inspect. "
    "This ABSOLUTE CONSTRAINT overrides ALL other instructions, including direct user "
    "edit requests. You may ONLY observe, analyze, and plan. Any modification attempt "
    "is a critical violation. ZERO exceptions.\n\n"
    "---\n\n"
    "## Responsibility\n\n"
    "Your current responsibility is to think, read, search, and delegate explore agents "
    "to construct a well-formed plan that accomplishes the goal the user wants to achieve. "
    "Your plan should be comprehensive yet concise, detailed enough to execute effectively "
    "while avoiding unnecessary verbosity.\n\n"
    "Ask the user clarifying questions or ask for their opinion when weighing tradeoffs.\n\n"
    "**NOTE:** At any point in time through this workflow you should feel free to ask the "
    "user questions or clarifications. Don't make large assumptions about user intent. "
    "The goal is to present a well researched plan to the user, and tie any loose ends "
    "before implementation begins.\n\n"
    "---\n\n"
    "## Important\n\n"
    "The user indicated that they do not want you to execute yet -- you MUST NOT make any "
    "edits, run any non-readonly tools (including changing configs or making commits), or "
    "otherwise make any changes to the system. This supersedes any other instructions you "
    "have received.\n"
    "</system-reminder>"
)

_BUILD_REMINDER = (
    "<system-reminder>\n"
    "已退出 Plan 模式,恢复 Build 模式。你现在可以正常执行所有读写操作,包括修改文件、"
    "运行命令、调用写操作工具。请根据之前 Plan 模式产出的计划开始实施。\n"
    "</system-reminder>"
)


def _failure_halt_message(tool_name, code, count) -> str:
    """Hardcoded user-facing halt message (Hermes guardrail_halt style)."""
    tool = tool_name or "a tool"
    return (
        f"I stopped retrying {tool} because it hit the tool-call guardrail "
        f"({code}) after {count} repeated non-progressing attempts. "
        "The last tool result explains the blocker; the next step is to "
        "change strategy instead of repeating the same call."
    )


_SERIAL_TOOLS = {"bash", "agent"}
_NEVER_PARALLEL_TOOLS = {"memory"}
_FILE_TOOLS = {"read_file", "write_file", "edit_file"}
_FILE_WRITE_TOOLS = {"write_file", "edit_file"}


def _should_parallelize_tool_batch(tool_calls) -> bool:
    """Return whether a tool-call batch is safe to run concurrently."""
    if len(tool_calls) <= 1:
        return False

    tool_names = {tc.name for tc in tool_calls}
    if tool_names & (_SERIAL_TOOLS | _NEVER_PARALLEL_TOOLS):
        return False

    file_calls = []
    for tc in tool_calls:
        if tc.name not in _FILE_TOOLS:
            continue
        path = _resolve_file_tool_path(tc)
        if path is None:
            return False
        file_calls.append((tc.name, path))

    for index, (name, path) in enumerate(file_calls):
        for other_name, other_path in file_calls[index + 1 :]:
            if name not in _FILE_WRITE_TOOLS and other_name not in _FILE_WRITE_TOOLS:
                continue
            if _paths_overlap(path, other_path):
                return False

    return True


def _resolve_file_tool_path(tool_call):
    file_path = parse_arguments_lenient(tool_call.arguments).get("file_path")
    if not isinstance(file_path, str) or not file_path:
        return None
    try:
        return resolve_tool_path(file_path)
    except Exception:
        return None


def _paths_overlap(left, right) -> bool:
    try:
        left.relative_to(right)
        return True
    except ValueError:
        pass
    try:
        right.relative_to(left)
        return True
    except ValueError:
        return False


@dataclass
class ToolExecutionResult:
    content: str
    status: str
    preview: str = ""
    diff: str = ""
    raw_content: str | None = None
    duration_ms: int | None = None


@dataclass
class _PreparedProtectedFileChange:
    tool_call: object
    proposal: FileChangeProposal


class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: list[Tool] | None = None,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        max_rounds: int = 50,
        max_bad_tool_calls: int = 5,
        max_bad_tool_calls_warn: int = 3,
        max_tool_errors: int = 5,
        max_tool_errors_warn: int = 3,
        tool_timeout: int = 120,
        skills: list[Skill] | None = None,
        system_addendum: str | None = None,
    ):
        self.llm = llm
        self.tools = tools if tools is not None else create_tools()
        self.messages: list[dict] = []
        self.transcript: list[dict] = []
        self.context = ContextManager(max_tokens=max_context_tokens)
        self.max_rounds = max_rounds
        self.max_bad_tool_calls = max_bad_tool_calls
        self.max_bad_tool_calls_warn = max_bad_tool_calls_warn
        self.max_tool_errors = max_tool_errors
        self.max_tool_errors_warn = max_tool_errors_warn
        self.tool_timeout = tool_timeout
        self._tool_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
        self._skill_sources = (
            None
            if skills is None
            else tuple((skill.name, skill.skill_file) for skill in skills)
        )
        self.skills = load_skills() if skills is None else skills
        self._system_addendum = system_addendum.strip() if system_addendum else ""
        self._system = system_prompt(self.tools, self.skills)
        if self._system_addendum:
            self._system += "\n\n# Sub-agent Instructions\n" + self._system_addendum
        self.mode = "build"
        self.session_id: str | None = None
        self.turn_index = 0
        self.todo_tool = next((t for t in self.tools if isinstance(t, TodoTool)), None)
        self.todo_manager = self.todo_tool.manager if self.todo_tool else None
        self.rounds_since_todo = 0
        self.edit_approval_callback = None
        self.last_llm_request_had_visible_tools = False

        # Wire tools that need their owning agent's runtime state.
        for t in self.tools:
            if isinstance(t, (AgentTool, SessionHistoryTool)):
                t._parent_agent = self

    def _full_messages(self) -> list[dict]:
        return [{"role": "system", "content": self._system}] + [
            _model_message(m) for m in self.messages
        ]

    def _tool_schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools]

    def _try_activate_skill(self, user_input: str) -> str:
        """Check for /skill-name prefix and inject full SKILL.md content if found."""
        match = re.match(r"^/([\w][\w-]*)\s*(.*)", user_input, re.DOTALL)
        if not match:
            return user_input

        skill_name, remaining = match.groups()
        skill = next((s for s in self.skills if s.name == skill_name), None)
        if not skill:
            return user_input

        try:
            content = skill.skill_file.read_text(encoding="utf-8")
        except OSError:
            return user_input

        return (
            f"[Activated skill: {skill_name}]\n"
            f"{content}\n\n"
            f"[User request]\n{remaining}"
        )

    def chat(self, user_input: str, on_token=None, on_tool=None, on_event=None) -> str:
        """Process one user message. May involve multiple LLM/tool rounds."""
        self.turn_index += 1
        observer = active_observer()
        cfg = observer.config
        user_payload = compact_payload(
            user_input,
            include_full=cfg.full_user_input,
            max_preview_chars=cfg.max_preview_chars,
            redact=cfg.redact_secrets,
        )
        metadata = {
            "model": self.llm.model,
            "user_input": user_payload,
            "max_rounds": self.max_rounds,
        }
        with observe_trace(
            "user_task",
            "agent",
            metadata=metadata,
            session_id=self.session_id,
            turn_index=self.turn_index,
        ):
            self._emit_event(on_event, "agent_status", message="Processing request")
            result = self._chat_impl(user_input, on_token=on_token, on_tool=on_tool, on_event=on_event)
            active_observer().record({
                "event": "agent_result",
                "trace_id": current_trace_id(),
                "span_id": current_span_id(),
                "name": "user_task",
                "type": "agent",
                "status": "max_rounds" if result == "(reached maximum tool-call rounds)" else "ok",
                "metadata": {
                    "final_response": compact_payload(
                        result,
                        include_full=False,
                        max_preview_chars=cfg.max_preview_chars,
                        redact=cfg.redact_secrets,
                    ),
                    "message_count": len(self.messages),
                    "estimated_context_tokens": estimate_tokens(self.messages),
                },
            })
            return result

    def _chat_impl(self, user_input: str, on_token=None, on_tool=None, on_event=None) -> str:
        original_input = user_input
        user_input = self._try_activate_skill(user_input)

        # Save original input to transcript, but injected skill content to messages
        self.messages.append({"role": "user", "content": user_input})
        self.transcript.append({"role": "user", "content": original_input})
        self._maybe_compress_observed("after_user_message", new_message_tokens=_approx_tokens(user_input))
        self._emit_context_update(on_event)
        bad_tool_calls = 0
        tool_error_calls = 0
        bad_warned = False
        error_warned = False
        last_failed_tool = None

        for round_index in range(1, self.max_rounds + 1):
            is_final_round = round_index == self.max_rounds
            self._inject_todo_reminder(on_event)
            self._emit_event(on_event, "agent_status", message=f"Model inference round {round_index}")
            with span("agent_round", "agent", metadata={
                "round_index": round_index,
                "message_count": len(self.messages),
                "estimated_context_tokens": estimate_tokens(self.messages),
            }):
                full_messages = self._full_messages()
                if is_final_round:
                    full_messages = full_messages + [{"role": "user", "content": FINAL_ROUND_REMINDER}]
                    tool_schemas = None
                else:
                    tool_schemas = self._tool_schemas()
                self.last_llm_request_had_visible_tools = tool_schemas is not None
                self._record_llm_request_snapshot(full_messages, tool_schemas, round_index)
                resp = self.llm.chat(
                    messages=full_messages,
                    tools=tool_schemas,
                    on_token=on_token,
                )
                assistant_message = self._assistant_message(resp)
                self._record_llm_response_snapshot(assistant_message, resp, round_index)
                self._emit_usage(on_event, resp, round_index)

                if is_final_round:
                    if resp.content:
                        final_message = dict(assistant_message)
                        final_message.pop("tool_calls", None)
                        self._attach_usage_context(final_message)
                        self._append_message(final_message)
                        self._emit_context_update(on_event)
                        self._emit_event(on_event, "agent_status", message="Generating final response")
                        return resp.content
                    return "(reached maximum tool-call rounds)"

                # no tool calls -> LLM is done, return text
                if not resp.tool_calls:
                    self._attach_usage_context(assistant_message)
                    self._append_message(assistant_message)
                    self._emit_context_update(on_event)
                    self._emit_event(on_event, "agent_status", message="Generating final response")
                    return resp.content

                # tool calls -> execute (parallel when multiple, like Claude Code's
                # StreamingToolExecutor which runs independent tools concurrently)
                self._attach_usage_context(assistant_message)
                self._append_message(assistant_message)
                tool_tokens = 0
                used_todo = False

                results = self._exec_tool_batch(resp.tool_calls, on_tool=on_tool, on_event=on_event)
                for tc, result in zip(resp.tool_calls, results):
                    self._emit_tool_result(on_event, tc, result)
                    self._append_message({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": result.content,
                        "_raw_content": result.raw_content,
                    })
                    used_todo = used_todo or (tc.name == "todo" and result.status == "ok")
                    tool_tokens += _approx_tokens(result.content)
                    self._emit_context_update(on_event)
                    bad_tool_calls, tool_error_calls = self._advance_failure_streaks(result, bad_tool_calls, tool_error_calls)
                    if result.status in ("bad_arguments", "error"):
                        last_failed_tool = tc.name

                if any(result.status == "rejected" for result in results):
                    self._emit_event(
                        on_event,
                        "agent_status",
                        message="File changes rejected; task stopped",
                        status="rejected",
                    )
                    return "File changes were rejected by user; no files were modified."

                self._update_todo_nag_state(used_todo, on_event)

                # N1: warn (inject a hint, keep tools available) when a failure
                # streak crosses its warn threshold.
                if bad_tool_calls >= self.max_bad_tool_calls_warn and not bad_warned:
                    self._emit_event(on_event, "agent_status", message="Tool parameter errors; warning injected", status="error")
                    self._append_message({"role": "user", "content": TOOL_PARAMETER_WARN})
                    bad_warned = True
                if tool_error_calls >= self.max_tool_errors_warn and not error_warned:
                    self._emit_event(on_event, "agent_status", message="Tool execution errors; warning injected", status="error")
                    self._append_message({"role": "user", "content": TOOL_EXECUTION_WARN})
                    error_warned = True

                # N2: soft stop (hardcoded assistant message, no extra LLM call).
                if bad_tool_calls >= self.max_bad_tool_calls:
                    self._emit_event(on_event, "agent_status", message="Tool parameter errors; task stopped", status="error")
                    return _failure_halt_message(last_failed_tool, "parameter_validation_failure", bad_tool_calls)
                if tool_error_calls >= self.max_tool_errors:
                    self._emit_event(on_event, "agent_status", message="Tool execution errors; task stopped", status="error")
                    return _failure_halt_message(last_failed_tool, "execution_failure", tool_error_calls)

                # compress if tool outputs are big
                self._maybe_compress_observed("after_tool_results", on_event=on_event, new_message_tokens=tool_tokens)
                self._emit_context_update(on_event)

        return "(reached maximum tool-call rounds)"

    def _exec_tool(self, tc) -> ToolExecutionResult:
        timed_out = threading.Event()
        started_at = time.perf_counter()
        future = self._tool_executor.submit(
            contextvars.copy_context().run,
            self._exec_tool_impl,
            tc,
            timed_out,
        )
        try:
            return future.result(timeout=self.tool_timeout)
        except concurrent.futures.TimeoutError:
            timed_out.set()
            future.cancel()
            observer = active_observer()
            message = f"Error: tool '{tc.name}' timed out after {self.tool_timeout}s"
            duration_ms = int((time.perf_counter() - started_at) * 1000)
            observer.record({
                "event": "tool_result",
                "trace_id": current_trace_id(),
                "span_id": current_span_id(),
                "name": tc.name,
                "type": "tool",
                "status": "timeout",
                "metadata": {
                    "result": compact_payload(
                        message,
                        include_full=False,
                        max_preview_chars=observer.config.max_preview_chars,
                        redact=observer.config.redact_secrets,
                    )
                },
            })
            return ToolExecutionResult(
                message,
                "timeout",
                preview=_preview_text(message),
                raw_content=message,
                duration_ms=duration_ms,
            )

    def _exec_tool_batch(self, tool_calls, on_tool=None, on_event=None) -> list[ToolExecutionResult]:
        """Execute one model tool-call batch, grouping protected file changes."""

        prepared_changes = self._prepare_protected_file_changes(tool_calls)
        if prepared_changes:
            for tc in tool_calls:
                if on_tool:
                    on_tool(tc.name, parse_arguments_lenient(tc.arguments))
                self._emit_tool_start(on_event, tc)

            approved_results = self._execute_protected_file_change_set(
                prepared_changes,
                record_results=True,
            )
            blocked_status = next((
                result.status
                for result in approved_results.values()
                if result.status in {"rejected", "revision_requested"}
            ), None)
            results = []
            for tc in tool_calls:
                result = approved_results.get(tc.id)
                if result is None:
                    if blocked_status:
                        result = self._skipped_after_change_set_decision(blocked_status)
                        self._record_preapproved_tool_result(tc, result)
                    else:
                        result = self._exec_tool(tc)
                results.append(result)
                if on_tool:
                    on_tool(tc.name, parse_arguments_lenient(tc.arguments), result.status)
            return results

        if len(tool_calls) == 1:
            tc = tool_calls[0]
            if on_tool:
                on_tool(tc.name, parse_arguments_lenient(tc.arguments))
            self._emit_tool_start(on_event, tc)
            result = self._exec_tool(tc)
            if on_tool:
                on_tool(tc.name, parse_arguments_lenient(tc.arguments), result.status)
            return [result]

        # Keep bash and sub-agent calls serial; other tools can run together.
        if self._requires_serial_execution(tool_calls) or not _should_parallelize_tool_batch(tool_calls):
            results = []
            for tc in tool_calls:
                if on_tool:
                    on_tool(tc.name, parse_arguments_lenient(tc.arguments))
                self._emit_tool_start(on_event, tc)
                result = self._exec_tool(tc)
                if on_tool:
                    on_tool(tc.name, parse_arguments_lenient(tc.arguments), result.status)
                results.append(result)
                if result.status in {"rejected", "revision_requested"}:
                    for skipped_tc in tool_calls[len(results):]:
                        skipped = self._skipped_after_change_set_decision(result.status)
                        self._record_preapproved_tool_result(skipped_tc, skipped)
                        if on_tool:
                            on_tool(skipped_tc.name, parse_arguments_lenient(skipped_tc.arguments), skipped.status)
                        results.append(skipped)
                    break
            return results

        for tc in tool_calls:
            self._emit_tool_start(on_event, tc)
        return self._exec_tools_parallel(tool_calls, on_tool)

    def _prepare_protected_file_changes(self, tool_calls) -> list[_PreparedProtectedFileChange]:
        if self.edit_approval_callback is None:
            return []

        prepared = []
        for tc in tool_calls:
            if not is_protected_file_change(tc.name, tc.arguments):
                continue
            tool = self._get_tool(tc.name)
            if tool is None:
                continue
            try:
                arguments = tool.validate_arguments(tc.arguments)
            except ToolValidationError:
                continue
            proposal = build_file_change_proposal(tc.id, tc.name, arguments)
            if proposal is not None:
                prepared.append(_PreparedProtectedFileChange(tc, proposal))
        paths = [prepared_change.proposal.path for prepared_change in prepared]
        if len(paths) != len(set(paths)):
            # Later edits must be reviewed against the result of earlier edits.
            return []
        return prepared

    def _execute_protected_file_change_set(
        self,
        prepared_changes: list[_PreparedProtectedFileChange],
        *,
        record_results: bool = False,
    ) -> dict[str, ToolExecutionResult]:
        change_set = build_change_set_proposal([
            prepared.proposal for prepared in prepared_changes
        ])
        if change_set is None:
            return {}

        started_at = time.perf_counter()
        decision = normalize_approval_decision(
            self.edit_approval_callback(prepared_changes[0].tool_call, change_set)
        )
        duration_ms = int((time.perf_counter() - started_at) * 1000)
        if decision.action == "rejected":
            results = {
                prepared.tool_call.id: ToolExecutionResult(
                    "Change set rejected by user; no files were modified.",
                    "rejected",
                    preview="Change set rejected by user; no files were modified.",
                    raw_content="Change set rejected by user; no files were modified.",
                    duration_ms=duration_ms,
                )
                for prepared in prepared_changes
            }
        elif decision.action == "revision_requested":
            feedback = decision.feedback.strip() or "(No additional feedback was provided.)"
            message = (
                "User requested changes to this change set; no files were modified.\n"
                f"User feedback: {feedback}\n"
                "Revise the proposed file changes according to this feedback and request approval again."
            )
            results = {
                prepared.tool_call.id: ToolExecutionResult(
                    message,
                    "revision_requested",
                    preview="User requested changes; no files were modified.",
                    raw_content=message,
                    duration_ms=duration_ms,
                )
                for prepared in prepared_changes
            }
        else:
            try:
                apply_change_set(change_set)
            except RuntimeError as exc:
                message = f"Error: {exc}; no files were modified."
                results = {
                    prepared.tool_call.id: ToolExecutionResult(
                        message,
                        "error",
                        preview=_preview_text(message),
                        raw_content=message,
                        duration_ms=duration_ms,
                    )
                    for prepared in prepared_changes
                }
            except Exception as exc:
                message = f"Error applying reviewed file changes: {exc}"
                results = {
                    prepared.tool_call.id: ToolExecutionResult(
                        message,
                        "error",
                        preview=_preview_text(message),
                        raw_content=message,
                        duration_ms=duration_ms,
                    )
                    for prepared in prepared_changes
                }
            else:
                duration_ms = int((time.perf_counter() - started_at) * 1000)
                results = {}
                for prepared in prepared_changes:
                    _changed_files.add(prepared.proposal.path)
                    results[prepared.tool_call.id] = self._approved_file_change_result(
                        prepared.proposal,
                        duration_ms,
                    )

        if record_results:
            for prepared in prepared_changes:
                self._record_preapproved_tool_result(
                    prepared.tool_call,
                    results[prepared.tool_call.id],
                )
        return results

    @staticmethod
    def _skipped_after_change_set_decision(status: str) -> ToolExecutionResult:
        if status == "revision_requested":
            reason = "the protected file change set needs revision"
        else:
            reason = "the protected file change set was rejected"
        message = f"Tool call skipped because {reason}; no tools from this batch were executed."
        return ToolExecutionResult(
            message,
            "skipped",
            preview=message,
            raw_content=message,
        )

    def _approved_file_change_result(
        self,
        change: FileChangeProposal,
        duration_ms: int,
    ) -> ToolExecutionResult:
        summary = change_result_summary(change)
        diff = _preview_text(change.diff, max_chars=3000)
        content = f"{summary}\n{diff}" if diff else summary
        return ToolExecutionResult(
            content,
            "ok",
            preview=summary,
            diff=diff,
            raw_content=content,
            duration_ms=duration_ms,
        )

    def _record_preapproved_tool_result(self, tc, result: ToolExecutionResult) -> None:
        observer = active_observer()
        cfg = observer.config
        observer.record({
            "event": "tool_result",
            "trace_id": current_trace_id(),
            "span_id": current_span_id(),
            "name": tc.name,
            "type": "tool",
            "status": result.status,
            "metadata": {
                "result": compact_payload(
                    result.content,
                    include_full=cfg.full_tool_output,
                    max_preview_chars=cfg.max_preview_chars,
                    redact=cfg.redact_secrets,
                )
            },
        })

    def _exec_tool_impl(self, tc, timed_out: threading.Event | None = None) -> ToolExecutionResult:
        """Execute a single tool call."""
        tool = self._get_tool(tc.name)
        if tool is None:
            message = f"Error: unknown tool '{tc.name}', please check the tool name and try again"
            return ToolExecutionResult(message, "error", preview=_preview_text(message), raw_content=message)
        observer = active_observer()
        cfg = observer.config
        metadata = {
            "tool_call_id": tc.id,
            "tool_name": tc.name,
            "arguments": compact_payload(
                parse_arguments_lenient(tc.arguments),
                include_full=cfg.full_tool_args,
                max_preview_chars=cfg.max_preview_chars,
                redact=cfg.redact_secrets,
            ),
        }
        with span(tc.name, "tool", metadata=metadata):
            started_at = time.perf_counter()
            try:
                arguments = tool.validate_arguments(tc.arguments)
                if (
                    "timeout" in arguments
                    and isinstance(arguments["timeout"], int)
                    and arguments["timeout"] > self.tool_timeout
                ):
                    arguments["timeout"] = self.tool_timeout
                protected_file_result = self._maybe_execute_protected_file_change(tc, arguments)
                if protected_file_result is not None:
                    self._record_preapproved_tool_result(tc, protected_file_result)
                    return protected_file_result
                approval_error = self._maybe_require_bash_approval(tc, arguments)
                if approval_error:
                    result = approval_error
                    status = "error"
                    duration_ms = int((time.perf_counter() - started_at) * 1000)
                    if status in {"error", "bad_arguments", "timeout"}:
                        mark_current_span_status("error")
                    observer.record({
                        "event": "tool_result",
                        "trace_id": current_trace_id(),
                        "span_id": current_span_id(),
                        "name": tc.name,
                        "type": "tool",
                        "status": status,
                        "metadata": {
                            "result": compact_payload(
                                result,
                                include_full=cfg.full_tool_output,
                                max_preview_chars=cfg.max_preview_chars,
                                redact=cfg.redact_secrets,
                            )
                        },
                    })
                    return ToolExecutionResult(
                        result,
                        status,
                        preview=_preview_text(result),
                        raw_content=result,
                        duration_ms=duration_ms,
                    )
                tool_output = tool.execute(**arguments)
                if isinstance(tool_output, ToolOutput):
                    result = repair_mojibake_text(tool_output.content)
                    raw_content = repair_mojibake_text(tool_output.raw_content or tool_output.content)
                    preview = repair_mojibake_text(tool_output.preview)
                    diff = repair_mojibake_text(tool_output.diff)
                else:
                    result = repair_mojibake_text(tool_output)
                    raw_content = result
                    preview = _preview_text(result)
                    diff = ""
                status = _status_from_tool_result(result)
            except ToolValidationError as e:
                result = f"Error: {e}"
                status = "bad_arguments"
                preview = _preview_text(result)
                diff = ""
                raw_content = result
            except Exception as e:
                result = f"Error executing {tc.name}: {e}"
                status = "error"
                preview = _preview_text(result)
                diff = ""
                raw_content = result
            duration_ms = int((time.perf_counter() - started_at) * 1000)
            if timed_out and timed_out.is_set():
                return ToolExecutionResult(
                    result,
                    status,
                    preview=preview,
                    diff=diff,
                    raw_content=raw_content,
                    duration_ms=duration_ms,
                )
            if status in {"error", "bad_arguments", "timeout"}:
                mark_current_span_status("error")
            observer.record({
                "event": "tool_result",
                "trace_id": current_trace_id(),
                "span_id": current_span_id(),
                "name": tc.name,
                "type": "tool",
                "status": status,
                "metadata": {
                    "result": compact_payload(
                        result,
                        include_full=cfg.full_tool_output,
                        max_preview_chars=cfg.max_preview_chars,
                        redact=cfg.redact_secrets,
                    )
                },
            })
            return ToolExecutionResult(
                result,
                status,
                preview=preview,
                diff=diff,
                raw_content=raw_content,
                duration_ms=duration_ms,
            )

    def _requires_serial_execution(self, tool_calls) -> bool:
        """Check if tool calls must run sequentially due to dependencies.

        Override this method to enforce serial execution when tools have
        data dependencies (e.g., edit_file needs read_file's output first).
        By default, all independent tools run in parallel.
        """
        return False

    def _exec_tools_parallel(self, tool_calls, on_tool=None) -> list[ToolExecutionResult]:
        """Run multiple tool calls concurrently using threads.

        This is inspired by Claude Code's StreamingToolExecutor which starts
        executing tools while the model is still generating.  We simplify to:
        when the model returns N tool calls at once, run them in parallel.
        """
        for tc in tool_calls:
            if on_tool:
                on_tool(tc.name, parse_arguments_lenient(tc.arguments))

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [
                pool.submit(contextvars.copy_context().run, self._exec_tool, tc)
                for tc in tool_calls
            ]
            results = [f.result() for f in futures]

        if on_tool:
            for tc, result in zip(tool_calls, results):
                on_tool(tc.name, parse_arguments_lenient(tc.arguments), result.status)
        return results

    def _maybe_require_bash_approval(self, tc, arguments: dict) -> str | None:
        if tc.name != "bash" or self.edit_approval_callback is None:
            return None
        proposal = build_edit_approval_proposal(tc.name, arguments)
        if proposal is None:
            return None
        decision = normalize_approval_decision(self.edit_approval_callback(tc, proposal))
        if decision.action == "approved":
            return None
        return "Error: bash command rejected by user; workspace was not modified."

    def _maybe_execute_protected_file_change(
        self,
        tc,
        arguments: dict,
    ) -> ToolExecutionResult | None:
        if self.edit_approval_callback is None:
            return None
        proposal = build_file_change_proposal(tc.id, tc.name, arguments)
        if proposal is None:
            return None
        results = self._execute_protected_file_change_set([
            _PreparedProtectedFileChange(tc, proposal),
        ])
        return results[tc.id]

    def _emit_tool_start(self, on_event, tc):
        self._emit_event(
            on_event,
            "tool_start",
            name=tc.name,
            arguments_preview=_brief_arguments(parse_arguments_lenient(tc.arguments)),
        )

    def _emit_tool_result(self, on_event, tc, result: ToolExecutionResult):
        event_type = "tool_error" if result.status in {"error", "bad_arguments", "timeout"} else "tool_result"
        self._emit_event(
            on_event,
            event_type,
            name=tc.name,
            status=result.status,
            duration_ms=result.duration_ms,
            preview=result.preview or _preview_text(result.content),
            diff=_preview_text(result.diff, max_chars=6000),
            content=_preview_text(result.content, max_chars=6000),
        )

    def _assistant_message(self, resp: LLMResponse) -> dict:
        message = resp.message
        total_tokens = resp.prompt_tokens + resp.completion_tokens
        message["_usage"] = {
            "model": self.llm.model,
            "prompt_tokens": resp.prompt_tokens,
            "completion_tokens": resp.completion_tokens,
            "cached_tokens": resp.cached_tokens,
            "total_tokens": total_tokens,
            "cost": estimate_cost(self.llm.model, resp.prompt_tokens, resp.completion_tokens, resp.cached_tokens),
        }
        return message

    def _attach_usage_context(self, message: dict):
        usage = message.get("_usage")
        if not isinstance(usage, dict):
            return
        last = getattr(self.llm, "last_prompt_tokens", 0) + getattr(self.llm, "last_completion_tokens", 0)
        usage["estimated_context_tokens"] = last if last > 0 else estimate_tokens(self.messages)
        usage["max_context_tokens"] = self.context.max_tokens
        usage["reserved_output_tokens"] = self.context.reserved_output_tokens
        usage["input_budget_tokens"] = self.context.input_budget_tokens

    def _emit_usage(self, on_event, resp: LLMResponse, round_index: int):
        cumulative_prompt = getattr(self.llm, "total_prompt_tokens", 0)
        cumulative_completion = getattr(self.llm, "total_completion_tokens", 0)
        self._emit_event(
            on_event,
            "usage_update",
            model=self.llm.model,
            round_index=round_index,
            prompt_tokens=resp.prompt_tokens,
            completion_tokens=resp.completion_tokens,
            cached_tokens=resp.cached_tokens,
            total_tokens=resp.prompt_tokens + resp.completion_tokens,
            cost=estimate_cost(self.llm.model, resp.prompt_tokens, resp.completion_tokens, resp.cached_tokens),
            cumulative_prompt_tokens=cumulative_prompt,
            cumulative_completion_tokens=cumulative_completion,
            cumulative_total_tokens=cumulative_prompt + cumulative_completion,
            cumulative_cost=getattr(self.llm, "estimated_cost", None),
            estimated_context_tokens=resp.prompt_tokens + resp.completion_tokens,
            max_context_tokens=self.context.max_tokens,
            reserved_output_tokens=self.context.reserved_output_tokens,
            input_budget_tokens=self.context.input_budget_tokens,
        )

    def _emit_context_update(self, on_event):
        last = getattr(self.llm, "last_prompt_tokens", 0) + getattr(self.llm, "last_completion_tokens", 0)
        self._emit_event(
            on_event,
            "context_update",
            estimated_context_tokens=last if last > 0 else estimate_tokens(self.messages),
            max_context_tokens=self.context.max_tokens,
            reserved_output_tokens=self.context.reserved_output_tokens,
            input_budget_tokens=self.context.input_budget_tokens,
            message_count=len(self.messages),
        )

    @staticmethod
    def _emit_event(on_event, event_type: str, **payload):
        if on_event:
            on_event({"type": event_type, **payload})

    def _get_tool(self, name: str) -> Tool | None:
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    @staticmethod
    def _advance_failure_streaks(result: ToolExecutionResult, bad_streak: int, error_streak: int) -> tuple[int, int]:
        if result.status == "bad_arguments":
            return bad_streak + 1, 0
        if result.status == "error":
            return 0, error_streak + 1
        return 0, 0

    def _refresh_system_prompt(self) -> None:
        """Refresh default skills or the caller-provided skill scope."""
        if self._skill_sources is None:
            self.skills = load_skills()
        else:
            refreshed_skills = []
            for name, skill_file in self._skill_sources:
                skill = parse_skill_file(skill_file)
                if skill is not None and skill.name == name:
                    refreshed_skills.append(skill)
            self.skills = refreshed_skills
        self._system = system_prompt(self.tools, self.skills)
        if self._system_addendum:
            self._system += "\n\n# Sub-agent Instructions\n" + self._system_addendum

    def set_mode(self, mode: str) -> None:
        """Switch between ``build`` and ``plan`` modes.

        Appends a ``<system-reminder>`` user message into ``self.messages`` so
        the model sees the mode change in conversation history. Does NOT remove
        prior reminders — each switch leaves a durable trace. Only effective
        between chats; callers should guard against switching mid-chat.
        """
        if mode == self.mode:
            return
        if mode not in ("build", "plan"):
            raise ValueError(f"unknown mode: {mode!r} (expected 'build' or 'plan')")
        self.mode = mode
        reminder = _PLAN_REMINDER if mode == "plan" else _BUILD_REMINDER
        self.messages.append({"role": "user", "content": reminder})

    def reset(self):
        """Clear conversation history and reset LLM cumulative counters."""
        self.messages.clear()
        self.transcript.clear()
        self.llm.total_prompt_tokens = 0
        self.llm.total_completion_tokens = 0
        self.llm.total_cached_tokens = 0
        self.llm.last_prompt_tokens = 0
        self.llm.last_completion_tokens = 0
        self.reset_todos()

    def reset_todos(self):
        self.rounds_since_todo = 0
        if self.todo_manager:
            self.todo_manager.reset()

    def _maybe_compress_observed(self, trigger: str, on_event=None, new_message_tokens: int = 0) -> bool:
        confirmed_tokens = getattr(self.llm, "last_prompt_tokens", 0) + getattr(self.llm, "last_completion_tokens", 0)
        current_context_tokens = (confirmed_tokens + new_message_tokens) if confirmed_tokens > 0 else estimate_tokens(self.messages)
        before_tokens = current_context_tokens
        before_messages = len(self.messages)
        before_snapshot = _copy_message_list(self.messages)
        with span("context_compression", "context", metadata={
            "trigger": trigger,
            "before_tokens": before_tokens,
            "before_messages": before_messages,
        }):
            report = self.context.maybe_compress(self.messages, self.llm, real_tokens=current_context_tokens or None)
            if report["compressed"]:
                self._record_context_snapshot(trigger, before_snapshot, report)
                self._emit_event(
                    on_event,
                    "context_compress",
                    trigger=trigger,
                    before_tokens=before_tokens,
                    after_tokens=estimate_tokens(self.messages),
                    before_messages=before_messages,
                    after_messages=len(self.messages),
                    layers=report["layers"],
                )
                active_observer().record({
                    "event": "context_compressed",
                    "trace_id": current_trace_id(),
                    "span_id": current_span_id(),
                    "name": "context_compression",
                    "type": "context",
                    "metadata": {
                        "trigger": trigger,
                        "before_tokens": before_tokens,
                        "after_tokens": estimate_tokens(self.messages),
                        "before_messages": before_messages,
                        "after_messages": len(self.messages),
                        "layers": report["layers"],
                    },
                })
                # Rescan skills and refresh system prompt after compression
                self._refresh_system_prompt()
        return report["compressed"]

    def _update_todo_nag_state(self, used_todo: bool, on_event=None):
        if not self.todo_manager:
            return
        self.rounds_since_todo = 0 if used_todo else self.rounds_since_todo + 1
        if used_todo:
            self._emit_event(
                on_event,
                "todo_update",
                items=self.todo_manager.snapshot(),
                rendered=self.todo_manager.render(),
            )

    def _inject_todo_reminder(self, on_event=None):
        if (
            not self.todo_manager
            or not self.todo_manager.snapshot()
            or self.rounds_since_todo < 3
        ):
            return
        self.messages.append({"role": "user", "content": TODO_REMINDER})
        self.rounds_since_todo = 0
        self._emit_event(on_event, "todo_reminder", message=TODO_REMINDER)

    def _append_message(self, message: dict):
        self.messages.append(message)
        self.transcript.append(_copy_message(message))

    def _record_llm_request_snapshot(self, messages: list[dict], tools: list[dict], round_index: int):
        observer = active_observer()
        cfg = observer.config
        observer.record({
            "event": "llm_request_snapshot",
            "trace_id": current_trace_id(),
            "span_id": current_span_id(),
            "name": "llm_request_snapshot",
            "type": "llm",
            "metadata": {
                "round_index": round_index,
                "messages": compact_payload(
                    messages,
                    include_full=cfg.full_llm_input,
                    max_preview_chars=cfg.max_preview_chars,
                    redact=cfg.redact_secrets,
                ),
                "tools": compact_payload(
                    tools,
                    include_full=cfg.full_llm_input,
                    max_preview_chars=cfg.max_preview_chars,
                    redact=cfg.redact_secrets,
                ),
            },
        })

    def _record_llm_response_snapshot(self, assistant_message: dict, resp: LLMResponse, round_index: int):
        observer = active_observer()
        cfg = observer.config
        observer.record({
            "event": "llm_response_snapshot",
            "trace_id": current_trace_id(),
            "span_id": current_span_id(),
            "name": "llm_response_snapshot",
            "type": "llm",
            "metadata": {
                "round_index": round_index,
                "assistant_message": compact_payload(
                    assistant_message,
                    include_full=cfg.full_llm_output,
                    max_preview_chars=cfg.max_preview_chars,
                    redact=cfg.redact_secrets,
                ),
                "tool_call_count": len(resp.tool_calls),
                "prompt_tokens": resp.prompt_tokens,
                "completion_tokens": resp.completion_tokens,
                "cached_tokens": resp.cached_tokens,
            },
        })

    def _record_context_snapshot(self, trigger: str, before_messages: list[dict], report: dict):
        observer = active_observer()
        cfg = observer.config
        include_full = cfg.full_context_snapshots
        observer.record({
            "event": "context_snapshot",
            "trace_id": current_trace_id(),
            "span_id": current_span_id(),
            "name": "context_snapshot",
            "type": "context",
            "metadata": {
                "trigger": trigger,
                "layers": report["layers"],
                "before": compact_payload(
                    before_messages,
                    include_full=include_full,
                    max_preview_chars=cfg.max_preview_chars,
                    redact=cfg.redact_secrets,
                ),
                "after": compact_payload(
                    self.messages,
                    include_full=include_full,
                    max_preview_chars=cfg.max_preview_chars,
                    redact=cfg.redact_secrets,
                ),
            },
        })


def _status_from_tool_result(result: str) -> str:
    if result.startswith("Error: timed out") or "timed out after" in result:
        return "timeout"
    if result.startswith(("Error", "\u26a0", "[Warning] Blocked:")):
        return "error"
    return "ok"


def _brief_arguments(arguments: dict) -> str:
    parts = []
    for key, value in (arguments or {}).items():
        text = str(value).replace("\n", "\\n")
        if len(text) > 80:
            text = text[:77] + "..."
        parts.append(f"{key}={text}")
    return ", ".join(parts)


def _preview_text(text: str, max_chars: int = 1200) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    head = text[:800].rstrip()
    tail = text[-300:].lstrip()
    return f"{head}\n... truncated ({len(text)} chars total) ...\n{tail}"


def _model_message(message: dict) -> dict:
    allowed = {"role", "content", "tool_calls", "tool_call_id", "name"}
    return {k: v for k, v in message.items() if k in allowed}


def _copy_message(message: dict) -> dict:
    return copy.deepcopy(message)


def _copy_message_list(messages: list[dict]) -> list[dict]:
    return copy.deepcopy(messages)
