"""Multi-layer context compression.

Claude Code uses a 4-layer strategy:
  1. HISTORY_SNIP   - trim old tool outputs to a one-line summary
  2. Microcompact   - LLM-powered summary of old turns (cached)
  3. CONTEXT_COLLAPSE - aggressive compression when nearing hard limit
  4. Autocompact    - periodic background compaction

Folium implements a 3-layer progressive strategy:
  Layer 1 (snip)      - 60%: truncate tool outputs, keep first/last lines
  Layer 2 (prune)     - 80%: replace snipped tool outputs with placeholders
  Layer 3 (summarize) - 90%: incremental LLM summary + protected user messages
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from .config import DEFAULT_MAX_CONTEXT_TOKENS
from .token_estimator import estimate_message_tokens, estimate_text_tokens

if TYPE_CHECKING:
    from .llm import LLM

DEFAULT_RESERVED_OUTPUT_TOKENS = 20_000
DEFAULT_PROTECTED_USER_TOKENS = 20_000
SUMMARY_PREFIX = "[Context compressed - incremental summary]"
THINKING_PREFIX = (
    "另一个语言模型已经开始解决这个问题，并生成了其推理过程的摘要。"
    "你还可以访问该语言模型使用过的工具状态。"
    "请基于已完成的工作继续推进，避免重复劳动。"
    "以下是另一个语言模型生成的摘要，请利用其中的信息辅助你的分析：\n"
)


def _approx_tokens(text: str) -> int:
    """Estimate token count for content not covered by API usage."""
    return estimate_text_tokens(text)


def estimate_tokens(messages: list[dict]) -> int:
    return estimate_message_tokens(messages)


class ContextManager:
    def __init__(self, max_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
                 reserved_output_tokens: int = DEFAULT_RESERVED_OUTPUT_TOKENS,
                 protected_user_tokens: int = DEFAULT_PROTECTED_USER_TOKENS):
        self.max_tokens = max_tokens
        self.reserved_output_tokens = max(0, reserved_output_tokens)
        self.protected_user_tokens = max(1, protected_user_tokens)
        self.input_budget_tokens = max(1, max_tokens - self.reserved_output_tokens)
        # layer thresholds (fraction of input budget after reserving output tokens)
        self._snip_at = int(self.input_budget_tokens * 0.60)    # 60% -> snip tool outputs
        self._prune_at = int(self.input_budget_tokens * 0.80)    # 80% -> prune to placeholders
        self._summarize_at = int(self.input_budget_tokens * 0.90)  # 90% -> LLM summarize

    def maybe_compress(self, messages: list[dict], llm: LLM | None = None,
                       real_tokens: int | None = None) -> bool:
        """Apply compression layers as needed. Returns True if any compression happened.

        Args:
            real_tokens: Actual token count from LLM API (prompt_tokens + completion_tokens).
                         Falls back to estimate_tokens() if not provided.
        """
        current = real_tokens if real_tokens is not None else estimate_tokens(messages)
        compressed = False

        # Layer 1: snip verbose tool outputs
        if current > self._snip_at:
            if self._snip_tool_outputs(messages):
                compressed = True
                current = estimate_tokens(messages)

        # Layer 2: prune snipped tool outputs to placeholders
        if current > self._prune_at:
            if self._prune_tool_outputs(messages):
                compressed = True
                current = estimate_tokens(messages)

        # Layer 3: LLM-powered summarization of old turns
        if current > self._summarize_at:
            if self._summarize_old(messages, llm):
                compressed = True
                current = estimate_tokens(messages)

        return compressed

    @staticmethod
    def _snip_tool_outputs(messages: list[dict]) -> bool:
        """Layer 1: Truncate tool results over 1500 chars to their first/last lines.

        This mirrors Claude Code's HISTORY_SNIP which replaces old tool outputs
        with a one-line summary to reclaim context space.
        """
        changed = False
        for m in messages:
            if m.get("role") != "tool":
                continue
            content = m.get("content", "")
            if len(content) <= 1500:
                continue
            lines = content.splitlines()
            if len(lines) <= 6:
                continue
            # keep first 3 + last 3 lines
            snipped = (
                "\n".join(lines[:3])
                + f"\n... ({len(lines)} lines, snipped to save context) ...\n"
                + "\n".join(lines[-3:])
            )
            m["content"] = snipped
            changed = True
        return changed

    @staticmethod
    def _prune_tool_outputs(messages: list[dict]) -> bool:
        """Layer 2: Replace already-snipped tool outputs with compact placeholders.

        Tool outputs that were truncated by Layer 1 still carry several lines of
        content. This layer replaces them with a single-line placeholder to
        release more space at zero LLM cost.
        """
        PRUNED = "[Content compacted to save context]"
        changed = False
        for m in messages:
            if m.get("role") != "tool":
                continue
            content = m.get("content", "")
            if not content:
                continue
            # only prune outputs that were already snipped by Layer 1
            if "snipped to save context" not in content:
                continue
            m["content"] = PRUNED
            changed = True
        return changed

    def _summarize_old(self, messages: list[dict], llm: LLM | None) -> bool:
        """Layer 3: Incrementally summarize history and protect user messages."""
        if not messages:
            return False

        existing_summary = self._extract_existing_summary(messages)
        delta = self._delta_messages(messages)
        if not delta and existing_summary:
            return False

        summary = self._get_incremental_summary(existing_summary, delta, llm)
        protected_users = self._collect_protected_user_messages(messages)
        messages.clear()
        messages.extend(protected_users)
        messages.append(self._summary_message(summary))
        return True

    def _summary_message(self, summary: str) -> dict:
        return {
            "role": "user",
            "content": f"{THINKING_PREFIX}{SUMMARY_PREFIX}\n{summary}",
        }

    @staticmethod
    def _is_summary_message(message: dict) -> bool:
        if message.get("role") != "user":
            return False
        content = message.get("content") or ""
        return f"{SUMMARY_PREFIX}\n" in content

    @staticmethod
    def _summary_text(message: dict) -> str:
        content = message.get("content") or ""
        marker = f"{SUMMARY_PREFIX}\n"
        idx = content.find(marker)
        if idx >= 0:
            return content[idx + len(marker):]
        return ""

    def _extract_existing_summary(self, messages: list[dict]) -> str:
        summaries = [
            self._summary_text(message)
            for message in messages
            if self._is_summary_message(message)
        ]
        return "\n\n".join(summary for summary in summaries if summary)

    def _delta_messages(self, messages: list[dict]) -> list[dict]:
        return [
            message
            for message in messages
            if not self._is_summary_message(message) and not message.get("_protected")
        ]

    def _collect_protected_user_messages(self, messages: list[dict]) -> list[dict]:
        selected: list[dict] = []
        used = 0

        for message in reversed(messages):
            if message.get("role") != "user" or self._is_summary_message(message):
                continue
            content = message.get("content") or ""
            tokens = estimate_text_tokens(str(content))
            if used + tokens <= self.protected_user_tokens:
                selected.append({"role": "user", "content": content, "_protected": True})
                used += tokens
                continue
            if not selected:
                selected.append({"role": "user", "content": content, "_protected": True})
            break

        selected.reverse()
        return selected

    def _get_incremental_summary(self, existing_summary: str, delta_messages: list[dict],
                                 llm: LLM | None) -> str:
        if not existing_summary:
            return self._get_summary(delta_messages, llm)

        delta = self._flatten(delta_messages)
        if not delta:
            return existing_summary

        if llm:
            try:
                resp = llm.chat(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "你正在执行上下文检查点压缩。为接续任务的下一个 LLM 创建一份交接摘要。"
                                "请包含："
                                "- 当前进展和已做出的关键决策"
                                "- 重要的上下文、约束条件或用户偏好"
                                "- 待完成的工作（清晰的下一步）"
                                "- 继续工作所需的任何关键数据、示例或参考"
                                "保持简洁、结构化，专注于帮助下一个 LLM 无缝接续工作。"
                                "始终使用中文输出。"
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Existing summary:\n{existing_summary}\n\n"
                                f"New messages to incorporate:\n{delta[:15000]}"
                            ),
                        },
                    ],
                )
                return resp.content
            except Exception:
                pass

        extracted = self._extract_key_info(delta_messages)
        if extracted and extracted != "(no extractable context)":
            return f"{existing_summary}\n\nRecent update: {extracted}"
        return existing_summary

    def _get_summary(self, messages: list[dict], llm: LLM | None) -> str:
        """Generate summary via LLM or fallback to extraction."""
        flat = self._flatten(messages)

        if llm:
            try:
                resp = llm.chat(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "你正在执行上下文检查点压缩。为接续任务的下一个 LLM 创建一份交接摘要。"
                                "请包含："
                                "- 当前进展和已做出的关键决策"
                                "- 重要的上下文、约束条件或用户偏好"
                                "- 待完成的工作（清晰的下一步）"
                                "- 继续工作所需的任何关键数据、示例或参考"
                                "保持简洁、结构化，专注于帮助下一个 LLM 无缝接续工作。"
                                "始终使用中文输出。"
                            ),
                        },
                        {"role": "user", "content": flat[:15000]},
                    ],
                )
                return resp.content
            except Exception:
                pass

        # fallback: extract key lines
        return self._extract_key_info(messages)

    @staticmethod
    def _flatten(messages: list[dict]) -> str:
        parts = []
        for m in messages:
            role = m.get("role", "?")
            text = m.get("content", "") or ""
            if text:
                parts.append(f"[{role}] {text[:400]}")
        return "\n".join(parts)

    @staticmethod
    def _extract_key_info(messages: list[dict]) -> str:
        """Fallback: extract file paths, errors, and decisions without LLM."""
        import re
        files_seen = set()
        errors = []
        decisions = []

        for m in messages:
            text = m.get("content", "") or ""
            # extract file paths
            for match in re.finditer(r'[\w./\-]+\.\w{1,5}', text):
                files_seen.add(match.group())
            # extract error lines
            for line in text.splitlines():
                if 'error' in line.lower() or 'Error' in line:
                    errors.append(line.strip()[:150])

        parts = []
        if files_seen:
            parts.append(f"Files touched: {', '.join(sorted(files_seen)[:20])}")
        if errors:
            parts.append(f"Errors seen: {'; '.join(errors[:5])}")
        return "\n".join(parts) or "(no extractable context)"
