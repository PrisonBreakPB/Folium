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
from dataclasses import dataclass
from .llm import LLM
from .tools import ALL_TOOLS
from .tools.base import Tool, ToolValidationError
from .tools.agent import AgentTool
from .prompt import system_prompt
from .context import ContextManager, estimate_tokens
from .observability import mark_current_span_status, observe_trace, span
from .observability.context import active_observer, current_span_id, current_trace_id
from .observability.redaction import compact_payload


@dataclass
class ToolExecutionResult:
    content: str
    status: str


class Agent:
    def __init__(
        self,
        llm: LLM,
        tools: list[Tool] | None = None,
        max_context_tokens: int = 128_000,
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
        self._system = system_prompt(self.tools)
        self.session_id: str | None = None
        self.turn_index = 0

        # wire up sub-agent capability
        for t in self.tools:
            if isinstance(t, AgentTool):
                t._parent_agent = self

    def _full_messages(self) -> list[dict]:
        return [{"role": "system", "content": self._system}] + self.messages

    def _tool_schemas(self) -> list[dict]:
        return [t.schema() for t in self.tools]

    def chat(self, user_input: str, on_token=None, on_tool=None) -> str:
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
            result = self._chat_impl(user_input, on_token=on_token, on_tool=on_tool)
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

    def _chat_impl(self, user_input: str, on_token=None, on_tool=None) -> str:
        self.messages.append({"role": "user", "content": user_input})
        self._maybe_compress_observed("after_user_message")
        bad_tool_calls = 0

        for round_index in range(1, self.max_rounds + 1):
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

                # no tool calls -> LLM is done, return text
                if not resp.tool_calls:
                    self.messages.append(resp.message)
                    return resp.content

                # tool calls -> execute (parallel when multiple, like Claude Code's
                # StreamingToolExecutor which runs independent tools concurrently)
                self.messages.append(resp.message)

                if len(resp.tool_calls) == 1:
                    tc = resp.tool_calls[0]
                    if on_tool:
                        on_tool(tc.name, tc.arguments)
                    result = self._exec_tool(tc)
                    if on_tool:
                        on_tool(tc.name, tc.arguments, result.status)
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result.content,
                    })
                    bad_tool_calls = self._count_bad_tool_call_streak(result, bad_tool_calls)
                else:
                    # parallel execution for multiple tool calls
                    results = self._exec_tools_parallel(resp.tool_calls, on_tool)
                    for tc, result in zip(resp.tool_calls, results):
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result.content,
                        })
                        bad_tool_calls = self._count_bad_tool_call_streak(result, bad_tool_calls)

                if bad_tool_calls >= self.max_bad_tool_calls:
                    return f"连续 {bad_tool_calls} 次工具调用失败，已停止当前任务。"

                # compress if tool outputs are big
                self._maybe_compress_observed("after_tool_results")

        return "(reached maximum tool-call rounds)"

    def _exec_tool(self, tc) -> ToolExecutionResult:
        timed_out = threading.Event()
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
            return ToolExecutionResult(message, "timeout")

    def _exec_tool_impl(self, tc, timed_out: threading.Event | None = None) -> ToolExecutionResult:
        """Execute a single tool call."""
        tool = self._get_tool(tc.name)
        if tool is None:
            return ToolExecutionResult(f"Error: unknown tool '{tc.name}'", "error")
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
            try:
                arguments = tool.validate_arguments(tc.arguments)
                if (
                    "timeout" in arguments
                    and isinstance(arguments["timeout"], int)
                    and arguments["timeout"] > self.tool_timeout
                ):
                    arguments["timeout"] = self.tool_timeout
                result = tool.execute(**arguments)
                status = _status_from_tool_result(result)
            except ToolValidationError as e:
                result = f"Error: {e}"
                status = "bad_arguments"
            except Exception as e:
                result = f"Error executing {tc.name}: {e}"
                status = "error"
            if timed_out and timed_out.is_set():
                return ToolExecutionResult(result, status)
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
            return ToolExecutionResult(result, status)

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
        """Clear conversation history."""
        self.messages.clear()

    def _maybe_compress_observed(self, trigger: str) -> bool:
        before_tokens = estimate_tokens(self.messages)
        before_messages = len(self.messages)
        with span("context_compression", "context", metadata={
            "trigger": trigger,
            "before_tokens": before_tokens,
            "before_messages": before_messages,
        }):
            compressed = self.context.maybe_compress(self.messages, self.llm)
            if compressed:
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
                    },
                })
        return compressed


def _status_from_tool_result(result: str) -> str:
    if result.startswith("Error: timed out") or "timed out after" in result:
        return "timeout"
    if result.startswith("Error") or result.startswith("\u26a0"):
        return "error"
    return "ok"
