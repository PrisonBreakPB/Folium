"""Core agent loop.

This is the heart of Folium.  The pattern is simple:

    user message -> LLM (with tools) -> tool calls? -> execute -> loop
                                      -> text reply? -> return to user

It keeps looping until the LLM responds with plain text (no tool calls),
which means it's done working and ready to report back.
"""

import concurrent.futures
import contextvars
import threading
import time
from dataclasses import dataclass
from .llm import LLM, LLMResponse, estimate_cost
from .tools import ALL_TOOLS
from .tools.base import Tool, ToolValidationError
from .tools.agent import AgentTool
from .prompt import system_prompt
from .context import ContextManager, estimate_tokens, _approx_tokens
from .config import DEFAULT_MAX_CONTEXT_TOKENS
from .skills import load_skills
from .observability import mark_current_span_status, observe_trace, span
from .observability.context import active_observer, current_span_id, current_trace_id
from .observability.redaction import compact_payload
from .encoding import repair_mojibake_text


@dataclass
class ToolExecutionResult:
    content: str
    status: str
    preview: str = ""
    duration_ms: int | None = None


class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: list[Tool] | None = None,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        max_rounds: int = 50,
        max_bad_tool_calls: int = 5,
        tool_timeout: int = 120,
    ):
        self.llm = llm
        self.tools = tools if tools is not None else ALL_TOOLS
        self.messages: list[dict] = []
        self.context = ContextManager(max_tokens=max_context_tokens)
        self.max_rounds = max_rounds
        self.max_bad_tool_calls = max_bad_tool_calls
        self.tool_timeout = tool_timeout
        self._tool_executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
        self.skills = load_skills()
        self._system = system_prompt(self.tools, self.skills)
        self.session_id: str | None = None
        self.turn_index = 0

        # wire up sub-agent capability
        for t in self.tools:
            if isinstance(t, AgentTool):
                t._parent_agent = self

    def _full_messages(self) -> list[dict]:
        return [{"role": "system", "content": self._system}] + [
            _model_message(m) for m in self.messages
        ]

    def _tool_schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools]

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
            self._emit_event(on_event, "agent_status", message="开始处理请求")
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
        self.messages.append({"role": "user", "content": user_input})
        self._maybe_compress_observed("after_user_message", new_message_tokens=_approx_tokens(user_input))
        self._emit_context_update(on_event)
        bad_tool_calls = 0

        for round_index in range(1, self.max_rounds + 1):
            self._emit_event(on_event, "agent_status", message=f"模型推理第 {round_index} 轮")
            with span("agent_round", "agent", metadata={
                "round_index": round_index,
                "message_count": len(self.messages),
                "estimated_context_tokens": estimate_tokens(self.messages),
            }):
                resp = self.llm.chat(
                    messages=self._full_messages(),
                    tools=self._tool_schemas(),
                    on_token=on_token,
                )
                assistant_message = self._assistant_message(resp)
                self._emit_usage(on_event, resp, round_index)

                # no tool calls -> LLM is done, return text
                if not resp.tool_calls:
                    self.messages.append(assistant_message)
                    self._attach_usage_context(assistant_message)
                    self._emit_context_update(on_event)
                    self._emit_event(on_event, "agent_status", message="生成最终回复")
                    return resp.content

                # tool calls -> execute (parallel when multiple, like Claude Code's
                # StreamingToolExecutor which runs independent tools concurrently)
                self.messages.append(assistant_message)
                self._attach_usage_context(assistant_message)
                tool_tokens = 0

                if len(resp.tool_calls) == 1:
                    tc = resp.tool_calls[0]
                    if on_tool:
                        on_tool(tc.name, tc.arguments)
                    self._emit_tool_start(on_event, tc)
                    result = self._exec_tool(tc)
                    if on_tool:
                        on_tool(tc.name, tc.arguments, result.status)
                    self._emit_tool_result(on_event, tc, result)
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tc.name,
                        "content": result.content,
                    })
                    tool_tokens += _approx_tokens(result.content)
                    self._emit_context_update(on_event)
                    bad_tool_calls = self._count_bad_tool_call_streak(result, bad_tool_calls)
                else:
                    # parallel execution for multiple tool calls
                    for tc in resp.tool_calls:
                        self._emit_tool_start(on_event, tc)
                    results = self._exec_tools_parallel(resp.tool_calls, on_tool)
                    for tc, result in zip(resp.tool_calls, results):
                        self._emit_tool_result(on_event, tc, result)
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tc.name,
                            "content": result.content,
                        })
                        tool_tokens += _approx_tokens(result.content)
                        self._emit_context_update(on_event)
                        bad_tool_calls = self._count_bad_tool_call_streak(result, bad_tool_calls)

                if bad_tool_calls >= self.max_bad_tool_calls:
                    self._emit_event(on_event, "agent_status", message="连续工具参数错误，已停止任务", status="error")
                    return f"连续 {bad_tool_calls} 次工具调用失败，已停止当前任务。"

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
            return ToolExecutionResult(message, "timeout", preview=_preview_text(message), duration_ms=duration_ms)

    def _exec_tool_impl(self, tc, timed_out: threading.Event | None = None) -> ToolExecutionResult:
        """Execute a single tool call."""
        tool = self._get_tool(tc.name)
        if tool is None:
            message = f"Error: unknown tool '{tc.name}'"
            return ToolExecutionResult(message, "error", preview=_preview_text(message))
        observer = active_observer()
        cfg = observer.config
        metadata = {
            "tool_call_id": tc.id,
            "tool_name": tc.name,
            "arguments": compact_payload(
                tc.arguments,
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
                result = tool.execute(**arguments)
                result = repair_mojibake_text(result)
                status = _status_from_tool_result(result)
            except ToolValidationError as e:
                result = f"Error: {e}"
                status = "bad_arguments"
            except Exception as e:
                result = f"Error executing {tc.name}: {e}"
                status = "error"
            duration_ms = int((time.perf_counter() - started_at) * 1000)
            if timed_out and timed_out.is_set():
                return ToolExecutionResult(result, status, preview=_preview_text(result), duration_ms=duration_ms)
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
            return ToolExecutionResult(result, status, preview=_preview_text(result), duration_ms=duration_ms)

    def _exec_tools_parallel(self, tool_calls, on_tool=None) -> list[ToolExecutionResult]:
        """Run multiple tool calls concurrently using threads.

        This is inspired by Claude Code's StreamingToolExecutor which starts
        executing tools while the model is still generating.  We simplify to:
        when the model returns N tool calls at once, run them in parallel.
        """
        for tc in tool_calls:
            if on_tool:
                on_tool(tc.name, tc.arguments)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [
                pool.submit(contextvars.copy_context().run, self._exec_tool, tc)
                for tc in tool_calls
            ]
            results = [f.result() for f in futures]

        if on_tool:
            for tc, result in zip(tool_calls, results):
                on_tool(tc.name, tc.arguments, result.status)
        return results

    def _emit_tool_start(self, on_event, tc):
        self._emit_event(
            on_event,
            "tool_start",
            name=tc.name,
            arguments_preview=_brief_arguments(tc.arguments),
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
    def _count_bad_tool_call_streak(result: ToolExecutionResult, current: int) -> int:
        if result.status == "bad_arguments":
            return current + 1
        return 0

    def reset(self):
        """Clear conversation history and reset LLM cumulative counters."""
        self.messages.clear()
        self.llm.total_prompt_tokens = 0
        self.llm.total_completion_tokens = 0
        self.llm.total_cached_tokens = 0
        self.llm.last_prompt_tokens = 0
        self.llm.last_completion_tokens = 0

    def _maybe_compress_observed(self, trigger: str, on_event=None, new_message_tokens: int = 0) -> bool:
        confirmed_tokens = getattr(self.llm, "last_prompt_tokens", 0) + getattr(self.llm, "last_completion_tokens", 0)
        current_context_tokens = (confirmed_tokens + new_message_tokens) if confirmed_tokens > 0 else estimate_tokens(self.messages)
        before_tokens = current_context_tokens
        before_messages = len(self.messages)
        with span("context_compression", "context", metadata={
            "trigger": trigger,
            "before_tokens": before_tokens,
            "before_messages": before_messages,
        }):
            report = self.context.maybe_compress(self.messages, self.llm, real_tokens=current_context_tokens or None)
            if report["compressed"]:
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
        return report["compressed"]


def _status_from_tool_result(result: str) -> str:
    if result.startswith("Error: timed out") or "timed out after" in result:
        return "timeout"
    if result.startswith("Error") or result.startswith("\u26a0"):
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
