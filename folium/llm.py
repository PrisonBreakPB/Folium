"""LLM provider layer - thin wrapper over OpenAI-compatible APIs.

Since most providers (DeepSeek, Qwen, Kimi, GLM, Ollama, etc.) expose an
OpenAI-compatible endpoint, we just use the openai SDK directly.  Switch
provider by changing OPENAI_BASE_URL + OPENAI_API_KEY. That's it.

For providers that are NOT OpenAI-compatible (AWS Bedrock, Google Vertex,
etc.), use the LiteLLM backend which routes to 100+ providers through a
single unified interface. Set FOLIUM_PROVIDER=litellm.
"""

import json
import time
from dataclasses import dataclass, field

from openai import OpenAI, APIError, RateLimitError, APITimeoutError, APIConnectionError
from .observability import span
from .observability.context import active_observer, current_span_id, current_trace_id
from .observability.redaction import compact_payload


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0

    @property
    def message(self) -> dict:
        """Convert to OpenAI message format for appending to history."""
        msg: dict = {"role": "assistant", "content": self.content or None}
        if self.tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in self.tool_calls
            ]
        return msg


# pricing per million tokens: (input, output)
# sources: openai.com/api/pricing, api-docs.deepseek.com, platform.claude.com,
#          platform.moonshot.ai, alibabacloud.com/help/en/model-studio
_PRICING = {
    # OpenAI - current flagships
    "gpt-5.4": (2.5, 15),
    "gpt-5.4-mini": (0.75, 4.5),
    "gpt-5.4-nano": (0.2, 1.25),
    "o4-mini": (1.1, 4.4),
    # OpenAI - previous gen (still widely used)
    "gpt-4.1": (2, 8),
    "gpt-4.1-mini": (0.4, 1.6),
    "gpt-4.1-nano": (0.1, 0.4),
    "gpt-4o": (2.5, 10),
    "gpt-4o-mini": (0.15, 0.6),
    # DeepSeek
    "deepseek-chat": (0.27, 1.10),
    "deepseek-reasoner": (0.55, 2.19),
    # Anthropic Claude
    "claude-opus-4-6": (5, 25),
    "claude-sonnet-4-6": (3, 15),
    "claude-haiku-4-5": (1, 5),
    # Alibaba Qwen
    "qwen3-max": (0.78, 3.9),
    "qwen3-plus": (0.26, 0.78),
    "qwen-max": (0.78, 3.9),
    # Moonshot Kimi
    "kimi-k2.5": (0.6, 3),
    # DeepSeek V4 Pro (人民币/百万 token)
    "deepseek-v4-pro": (3, 6, 0.025),
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> float | None:
    """Estimate cost for a token pair using the local pricing table.

    Pricing unit is per million tokens. For models with cache pricing (3-tuple),
    cached_tokens are charged at the reduced cache rate.
    """
    pricing = _PRICING.get(model)
    if not pricing:
        return None
    if len(pricing) == 3:
        input_rate, output_rate, cache_rate = pricing
        regular_input = prompt_tokens - cached_tokens
        return (
            regular_input * input_rate / 1_000_000
            + cached_tokens * cache_rate / 1_000_000
            + completion_tokens * output_rate / 1_000_000
        )
    input_rate, output_rate = pricing
    return (
        prompt_tokens * input_rate / 1_000_000
        + completion_tokens * output_rate / 1_000_000
    )


class LLM:
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str | None = None,
        **kwargs,
    ):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.extra = kwargs  # temperature, max_tokens, etc.
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cached_tokens = 0
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0

    @property
    def estimated_cost(self) -> float | None:
        """Rough cost estimate in USD. Returns None if model not in pricing table."""
        return estimate_cost(self.model, self.total_prompt_tokens, self.total_completion_tokens, self.total_cached_tokens)

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_token=None,
    ) -> LLMResponse:
        """Send messages, stream back response, handle tool calls."""
        observer = active_observer()
        cfg = observer.config
        llm_metadata = {
            "model": self.model,
            "message_count": len(messages),
            "tools_count": len(tools or []),
            "parameters": {
                k: v for k, v in self.extra.items()
                if k in {"temperature", "max_tokens", "top_p"}
            },
            "input": compact_payload(
                messages,
                include_full=cfg.full_llm_input,
                max_preview_chars=cfg.max_preview_chars,
                redact=cfg.redact_secrets,
            ),
        }
        with span("chat.completions", "llm", metadata=llm_metadata):
            return self._chat_observed(messages, tools, on_token)

    def _chat_observed(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_token=None,
    ) -> LLMResponse:
        params: dict = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            **self.extra,
        }
        if tools:
            params["tools"] = tools

        # stream_options is an OpenAI extension; not all providers support it
        try:
            params["stream_options"] = {"include_usage": True}
            stream = self._call_with_retry(params)
        except Exception:
            params.pop("stream_options", None)
            stream = self._call_with_retry(params)

        content_parts: list[str] = []
        tc_map: dict[int, dict] = {}  # index -> {id, name, arguments_str}
        prompt_tok = 0
        completion_tok = 0
        cached_tok = 0
        first_token_at = None
        stream_started_at = time.time()

        for chunk in stream:
            # usage info comes in the final chunk
            if chunk.usage:
                prompt_tok = chunk.usage.prompt_tokens
                completion_tok = chunk.usage.completion_tokens
                cached_tok = getattr(chunk.usage, 'prompt_cache_hit_tokens', 0) or 0

            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # accumulate text
            if delta.content:
                if first_token_at is None:
                    first_token_at = time.time()
                content_parts.append(delta.content)
                if on_token:
                    on_token(delta.content)

            # accumulate tool calls across chunks
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tc_map:
                        tc_map[idx] = {"id": "", "name": "", "args": ""}
                    if tc_delta.id:
                        tc_map[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tc_map[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tc_map[idx]["args"] += tc_delta.function.arguments

        # parse accumulated tool calls
        parsed: list[ToolCall] = []
        for idx in sorted(tc_map):
            raw = tc_map[idx]
            try:
                args = json.loads(raw["args"])
            except json.JSONDecodeError as e:
                # Don't silently drop arguments — return the raw string as a
                # special "malformed_arguments" tool call so the agent loop can
                # feed the parse error back to the LLM instead of cycling on
                # empty args forever.
                args = {
                    "__malformed_arguments__": raw["args"],
                    "__parse_error__": str(e),
                }
            parsed.append(ToolCall(id=raw["id"], name=raw["name"], arguments=args))

        self.total_prompt_tokens += prompt_tok
        self.total_completion_tokens += completion_tok
        self.total_cached_tokens += cached_tok
        self.last_prompt_tokens = prompt_tok
        self.last_completion_tokens = completion_tok

        response = LLMResponse(
            content="".join(content_parts),
            tool_calls=parsed,
            prompt_tokens=prompt_tok,
            completion_tokens=completion_tok,
            cached_tokens=cached_tok,
        )
        cfg = active_observer().config
        # Record raw tool-call arguments BEFORE json.loads so we can
        # distinguish "model sent {}" from "model sent malformed JSON
        # that was silently dropped".  Only log non-empty ones.
        raw_tc_args = {}
        for idx in sorted(tc_map):
            raw = tc_map[idx]
            name = raw.get("name", "?")
            raw_args = raw.get("args", "")
            raw_tc_args[f"{idx}:{name}"] = compact_payload(
                raw_args,
                include_full=cfg.full_tool_args,
                max_preview_chars=cfg.max_preview_chars,
                redact=cfg.redact_secrets,
            )
        active_observer().record({
            "event": "llm_result",
            "trace_id": current_trace_id(),
            "span_id": current_span_id(),
            "name": "chat.completions",
            "type": "llm",
            "metadata": {
                "model": self.model,
                "prompt_tokens": prompt_tok,
                "completion_tokens": completion_tok,
                "tool_call_count": len(parsed),
                "time_to_first_token_ms": (
                    int((first_token_at - stream_started_at) * 1000)
                    if first_token_at is not None else None
                ),
                "tool_calls_raw_args": raw_tc_args,
                "output": compact_payload(
                    response.content,
                    include_full=cfg.full_llm_output,
                    max_preview_chars=cfg.max_preview_chars,
                    redact=cfg.redact_secrets,
                ),
            },
        })
        return response

    def _call_with_retry(self, params: dict, max_retries: int = 3):
        """Retry on transient errors with exponential backoff."""
        for attempt in range(max_retries):
            try:
                return self.client.chat.completions.create(**params)
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                if attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                time.sleep(wait)
            except APIError as e:
                # 5xx = server error, retry; 4xx = client error, don't
                if e.status_code and e.status_code >= 500 and attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise


class LiteLLM(LLM):
    """LLM backend via LiteLLM, supporting 100+ providers.

    Use this when your target provider is NOT OpenAI-compatible
    (AWS Bedrock, Google Vertex, Cohere, etc.) or when you want
    a single interface to switch between any provider by changing
    the model string.

    Set FOLIUM_PROVIDER=litellm and use LiteLLM model strings
    like ``anthropic/claude-3-haiku``, ``bedrock/anthropic.claude-v2``,
    ``vertex_ai/gemini-pro``, etc.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ):
        # skip LLM.__init__ which creates an OpenAI client
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.extra = kwargs
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cached_tokens = 0

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_token=None,
    ) -> LLMResponse:
        """Send messages via litellm, stream back response, handle tool calls."""
        observer = active_observer()
        cfg = observer.config
        llm_metadata = {
            "model": self.model,
            "provider": "litellm",
            "message_count": len(messages),
            "tools_count": len(tools or []),
            "parameters": {
                k: v for k, v in self.extra.items()
                if k in {"temperature", "max_tokens", "top_p"}
            },
            "input": compact_payload(
                messages,
                include_full=cfg.full_llm_input,
                max_preview_chars=cfg.max_preview_chars,
                redact=cfg.redact_secrets,
            ),
        }
        with span("litellm.completion", "llm", metadata=llm_metadata):
            return self._chat_observed(messages, tools, on_token)

    def _chat_observed(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        on_token=None,
    ) -> LLMResponse:
        params: dict = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            **self.extra,
        }
        if tools:
            params["tools"] = tools

        stream = self._call_with_retry(params)

        content_parts: list[str] = []
        tc_map: dict[int, dict] = {}
        prompt_tok = 0
        completion_tok = 0
        cached_tok = 0
        first_token_at = None
        stream_started_at = time.time()

        for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage:
                prompt_tok = getattr(usage, "prompt_tokens", 0) or 0
                completion_tok = getattr(usage, "completion_tokens", 0) or 0
                cached_tok = getattr(usage, "prompt_cache_hit_tokens", 0) or 0

            if not getattr(chunk, "choices", None):
                continue
            delta = chunk.choices[0].delta

            if getattr(delta, "content", None):
                if first_token_at is None:
                    first_token_at = time.time()
                content_parts.append(delta.content)
                if on_token:
                    on_token(delta.content)

            if getattr(delta, "tool_calls", None):
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tc_map:
                        tc_map[idx] = {"id": "", "name": "", "args": ""}
                    if tc_delta.id:
                        tc_map[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tc_map[idx]["name"] = tc_delta.function.name
                        if tc_delta.function.arguments:
                            tc_map[idx]["args"] += tc_delta.function.arguments

        parsed: list[ToolCall] = []
        for idx in sorted(tc_map):
            raw = tc_map[idx]
            try:
                args = json.loads(raw["args"])
            except json.JSONDecodeError as e:
                args = {
                    "__malformed_arguments__": raw["args"],
                    "__parse_error__": str(e),
                }
            parsed.append(ToolCall(id=raw["id"], name=raw["name"], arguments=args))

        self.total_prompt_tokens += prompt_tok
        self.total_completion_tokens += completion_tok
        self.total_cached_tokens += cached_tok
        self.last_prompt_tokens = prompt_tok
        self.last_completion_tokens = completion_tok

        response = LLMResponse(
            content="".join(content_parts),
            tool_calls=parsed,
            prompt_tokens=prompt_tok,
            completion_tokens=completion_tok,
            cached_tokens=cached_tok,
        )
        cfg = active_observer().config
        raw_tc_args = {}
        for idx in sorted(tc_map):
            raw = tc_map[idx]
            name = raw.get("name", "?")
            raw_args = raw.get("args", "")
            raw_tc_args[f"{idx}:{name}"] = compact_payload(
                raw_args,
                include_full=cfg.full_tool_args,
                max_preview_chars=cfg.max_preview_chars,
                redact=cfg.redact_secrets,
            )
        active_observer().record({
            "event": "llm_result",
            "trace_id": current_trace_id(),
            "span_id": current_span_id(),
            "name": "litellm.completion",
            "type": "llm",
            "metadata": {
                "model": self.model,
                "prompt_tokens": prompt_tok,
                "completion_tokens": completion_tok,
                "tool_call_count": len(parsed),
                "time_to_first_token_ms": (
                    int((first_token_at - stream_started_at) * 1000)
                    if first_token_at is not None else None
                ),
                "tool_calls_raw_args": raw_tc_args,
                "output": compact_payload(
                    response.content,
                    include_full=cfg.full_llm_output,
                    max_preview_chars=cfg.max_preview_chars,
                    redact=cfg.redact_secrets,
                ),
            },
        })
        return response

    def _call_with_retry(self, params: dict, max_retries: int = 3):
        """Retry on transient errors with exponential backoff via litellm."""
        import litellm

        params["drop_params"] = True
        if self.api_key:
            params["api_key"] = self.api_key
        if self.base_url:
            params["api_base"] = self.base_url

        for attempt in range(max_retries):
            try:
                return litellm.completion(**params)
            except Exception as e:
                err = str(e).lower()
                is_transient = any(
                    kw in err
                    for kw in ["rate_limit", "timeout", "connection", "502", "503", "529"]
                )
                is_server = any(kw in err for kw in ["500", "502", "503", "504"])
                if (is_transient or is_server) and attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    raise
