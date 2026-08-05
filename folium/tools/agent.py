"""Sub-agent spawning (inspired by Claude Code's AgentTool, 1397 lines).

The idea: for complex sub-tasks, spawn an independent agent with its own
conversation history and tool access. This lets the main agent delegate
work like "go research this codebase and report back" without polluting
its own context window.

The sub-agent runs to completion and returns a text summary.
"""

from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from typing import Literal

from .base import Tool, ToolOutput
from .todo import TodoTool


OutputFormat = Literal["text", "papers", "paper_note", "review"]


@dataclass(frozen=True)
class SubAgentSpec:
    prompt: str
    tools: tuple[str, ...] | Literal["inherit"]
    skills: tuple[str, ...] | Literal["inherit"]
    max_rounds: int
    default_timeout: int


LITERATURE_SEARCHER_PROMPT = """\
You are the literature-searcher sub-agent for Folium.

Scope:
- Search academic papers and return traceable candidate metadata.
- Prefer `paper_search` for structured literature search.
- Use `arxiv_search` to supplement preprints, abstracts, and direct PDFs.
- Use `web_search` / `web_fetch` only for specific source verification or when structured tools are insufficient.
- Do initial DOI/title deduplication and metadata cleanup.
- Use `paper_validate` before presenting final candidate papers as verified.

Boundaries:
- Do not write files.
- Do not edit code.
- Do not perform final research-gap judgment.
- Do not write a broad literature review unless the task explicitly asks for it.

Skill loading:
- You only receive skill metadata in the system prompt.
- If a task matches an available skill, first call `read_file` on that skill's SKILL.md path before following its workflow.
- Do not rely on skill descriptions alone for execution details.
"""


SUBAGENT_SPECS: dict[str, SubAgentSpec] = {
    "general": SubAgentSpec(
        prompt="",
        tools="inherit",
        skills="inherit",
        max_rounds=20,
        default_timeout=300,
    ),
    "literature-searcher": SubAgentSpec(
        prompt=LITERATURE_SEARCHER_PROMPT,
        tools=("read_file", "paper_search", "paper_validate", "arxiv_search", "web_search", "web_fetch"),
        skills=("control-literature-search",),
        max_rounds=12,
        default_timeout=180,
    ),
}


OUTPUT_FORMAT_INSTRUCTIONS: dict[str, str] = {
    "text": "",
    "papers": """\
Return the final answer as JSON only, with this shape:
{
  "papers": [
    {
      "title": "...",
      "authors": ["..."],
      "year": 2024,
      "venue": "...",
      "doi": "...",
      "url": "...",
      "abstract": "...",
      "source": ["openalex"]
    }
  ],
  "notes": []
}
Use null or an empty string for unavailable scalar fields. Do not include markdown fences.
""",
    "paper_note": "Return a structured paper note. Include paper metadata, problem, model, assumptions, method, theorem/proof sketch, experiments, limitations, relevance, and page evidence when available.",
    "review": "Return a structured literature review with method categories, paper matrix, candidate gaps, supporting evidence, and risks.",
}


class AgentTool(Tool):
    name = "agent"
    description = (
        "Delegate a self-contained, multi-step task to a separate sub-agent "
        "that works independently and reports back. Use this when the task "
        "needs its own context, focused investigation, or specialized handling. "
        "Do not use it for simple questions, single tool calls, quick lookups, "
        "or work the main agent can complete directly without losing focus."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "What the sub-agent should accomplish",
            },
            "agent_type": {
                "type": "string",
                "description": "Sub-agent type: general or literature-searcher",
            },
            "output_format": {
                "type": "string",
                "description": "Expected final output style: text, papers, paper_note, or review",
            },
            "context": {
                "type": "string",
                "description": "Context inheritance mode. Only 'none' is supported in this first version.",
            },
            "timeout": {
                "type": "integer",
                "description": "Maximum seconds to wait for the sub-agent. Clamped to the parent tool timeout.",
            },
        },
        "required": ["task"],
    }

    # set by Agent.__init__ after construction
    _parent_agent = None

    def execute(
        self,
        task: str,
        agent_type: str = "general",
        output_format: str = "text",
        context: str = "none",
        timeout: int | None = None,
    ) -> str | ToolOutput:
        if self._parent_agent is None:
            return "Error: agent tool not initialized (no parent agent)"
        if agent_type not in SUBAGENT_SPECS:
            return f"Error: unknown agent_type '{agent_type}'. Available: {', '.join(SUBAGENT_SPECS)}"
        if output_format not in OUTPUT_FORMAT_INSTRUCTIONS:
            return f"Error: unknown output_format '{output_format}'. Available: {', '.join(OUTPUT_FORMAT_INSTRUCTIONS)}"
        if context != "none":
            return "Error: only context='none' is supported for Folium sub-agents right now"

        # import here to avoid circular dep
        from ..agent import Agent

        parent = self._parent_agent
        spec = SUBAGENT_SPECS[agent_type]
        timeout_seconds = _clamp_timeout(timeout, spec.default_timeout, parent.tool_timeout)
        sub = Agent(
            llm=parent.llm,
            tools=_sub_agent_tools(parent.tools, spec),
            max_context_tokens=parent.context.max_tokens,
            max_rounds=spec.max_rounds,
            tool_timeout=min(parent.tool_timeout, timeout_seconds),
            skills=_sub_agent_skills(parent.skills, spec),
            system_addendum=spec.prompt,
        )
        sub.edit_approval_callback = parent.edit_approval_callback

        try:
            result = _run_with_timeout(sub.chat, _format_task(task, output_format), timeout_seconds)
            raw_result = result
            # trim long results to avoid blowing up parent's context
            if len(result) > 5000:
                result = result[:4500] + "\n... (sub-agent output truncated)"
                return ToolOutput(
                    content=f"[Sub-agent completed: {agent_type}]\n{result}",
                    raw_content=f"[Sub-agent completed: {agent_type}]\n{raw_result}",
                )
            return f"[Sub-agent completed: {agent_type}]\n{result}"
        except TimeoutError:
            return f"Sub-agent error: timed out after {timeout_seconds}s"
        except Exception as e:
            return f"Sub-agent error: {e}"


def _clamp_timeout(timeout: int | None, default: int, parent_timeout: int) -> int:
    requested = timeout if timeout is not None else default
    return max(10, min(requested, parent_timeout))


def _sub_agent_tools(tools: list[Tool], spec: SubAgentSpec) -> list[Tool]:
    allowed = None if spec.tools == "inherit" else set(spec.tools)
    return [
        _sub_agent_tool(t)
        for t in tools
        if t.name not in {"agent", "session_history"} and (allowed is None or t.name in allowed)
    ]


def _sub_agent_skills(skills, spec: SubAgentSpec):
    if spec.skills == "inherit":
        return skills
    allowed = set(spec.skills)
    return [skill for skill in skills if skill.name in allowed]


def _format_task(task: str, output_format: str) -> str:
    instruction = OUTPUT_FORMAT_INSTRUCTIONS[output_format].strip()
    if not instruction:
        return task
    return f"{task.strip()}\n\n[Required output format]\n{instruction}"


def _run_with_timeout(fn, task: str, timeout: int) -> str:
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(fn, task)
    try:
        return future.result(timeout=timeout)
    except TimeoutError:
        future.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        if future.done():
            pool.shutdown(wait=True)


def _sub_agent_tool(tool: Tool) -> Tool:
    if isinstance(tool, TodoTool):
        return TodoTool()
    return tool
