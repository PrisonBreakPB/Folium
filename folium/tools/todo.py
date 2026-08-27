"""Todo list tool for long-running agent tasks."""

from pydantic import BaseModel, ConfigDict, Field

from .base import Tool, ToolFailure, tool_failure


TODO_REMINDER = "<reminder>Update your todos.</reminder>"


class TodoManager:
    def __init__(self):
        self.items: list[dict] = []

    def update(self, items: list) -> str:
        if len(items) > 20:
            raise ValueError("Max 20 todos allowed")

        validated = []
        in_progress_count = 0
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                raise ValueError(f"Item {i + 1}: must be an object")
            item_id = str(item.get("id", str(i + 1))).strip()
            text = str(item.get("text", "")).strip()
            status = str(item.get("status", "pending")).strip().lower()
            if not item_id:
                raise ValueError(f"Item {i + 1}: id required")
            if not text:
                raise ValueError(f"Item {item_id}: text required")
            if status not in {"pending", "in_progress", "completed"}:
                raise ValueError(f"Item {item_id}: invalid status '{status}'")
            if status == "in_progress":
                in_progress_count += 1
            validated.append({"id": item_id, "text": text, "status": status})

        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")

        self.items = validated
        return self.render()

    def render(self) -> str:
        if not self.items:
            return "No todos."

        lines = []
        for item in self.items:
            marker = {
                "pending": "[ ]",
                "in_progress": "[>]",
                "completed": "[x]",
            }[item["status"]]
            lines.append(f"{marker} #{item['id']}: {item['text']}")
        done = sum(1 for item in self.items if item["status"] == "completed")
        lines.append(f"\n({done}/{len(self.items)} completed)")
        return "\n".join(lines)

    def snapshot(self) -> list[dict]:
        return [dict(item) for item in self.items]

    def reset(self):
        self.items.clear()


class TodoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    items: list[dict] = Field(description="Full replacement todo list. Each item: {id, text, status} where status is pending|in_progress|completed")


class TodoTool(Tool):
    name = "todo"
    description = (
        "Update the structured task list for multi-step work. "
        "Use this to create a plan, mark exactly one task in_progress, "
        "and mark completed tasks as work finishes."
    )
    args_model = TodoArgs

    def __init__(self, manager: TodoManager | None = None):
        self.manager = manager or TodoManager()

    def execute(self, items: list) -> str | ToolFailure:
        try:
            return self.manager.update(items)
        except ValueError as e:
            return tool_failure("invalid_todo", "validation", str(e))
