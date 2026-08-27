"""Base class for all tools."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

from pydantic import BaseModel, ValidationError


@dataclass
class ToolValidationError(Exception):
    """Raised when a tool call does not match the tool's JSON schema."""

    tool_name: str
    errors: list[str]

    def __str__(self) -> str:
        return f"bad arguments for {self.tool_name}: " + "; ".join(self.errors)


@dataclass
class ToolOutput:
    """Structured tool output with optional full text for persistence."""

    content: str
    preview: str = ""
    diff: str = ""
    raw_content: str | None = None


@dataclass(frozen=True)
class ToolError:
    """Structured information about a tool execution failure."""

    code: str
    category: str
    message: str
    retryable: bool | None = None
    details: dict | None = None


class ToolFailure(str):
    """Explicit, string-compatible failure result returned by a tool."""

    def __new__(
        cls,
        error: ToolError,
        content: str | None = None,
        preview: str = "",
        diff: str = "",
        raw_content: str | None = None,
    ):
        instance = super().__new__(cls, content or error.message)
        instance.error = error
        instance.content = content
        instance.preview = preview
        instance.diff = diff
        instance.raw_content = raw_content
        return instance

    @property
    def message(self) -> str:
        return self.content or self.error.message


def tool_failure(
    code: str,
    category: str,
    message: str,
    *,
    retryable: bool | None = None,
    details: dict | None = None,
    content: str | None = None,
    preview: str = "",
    diff: str = "",
    raw_content: str | None = None,
) -> ToolFailure:
    """Create a string-compatible structured tool failure."""

    display_content = content or (message if message.startswith(("Error:", "[Warning]")) else f"Error: {message}")
    return ToolFailure(
        ToolError(code, category, message, retryable=retryable, details=details),
        content=display_content,
        preview=preview,
        diff=diff,
        raw_content=raw_content,
    )


class ToolExecutionError(Exception):
    """Raised by a tool when it cannot return a ``ToolFailure`` directly."""

    def __init__(self, error: ToolError):
        super().__init__(error.message)
        self.error = error

class Tool(ABC):
    """Minimal tool interface. Subclass this to add new capabilities."""

    name: str
    description: str
    args_model: ClassVar[type[BaseModel]]

    # Whether a failed execution may be automatically retried by the system.
    # Default True is suitable for read/query tools (idempotent, no side
    # effects); tools that mutate state or have side effects must set False.
    retry_safe: ClassVar[bool] = True

    @abstractmethod
    def execute(self, **kwargs) -> str | ToolOutput | ToolFailure:
        """Run the tool and return a text, success, or structured failure result."""
        ...

    def validate_arguments(self, arguments: str | dict) -> dict:
        """Validate model-provided arguments against this tool's Pydantic model.

        Accepts either the raw JSON string from the model (parsed + validated in
        one step via ``model_validate_json``) or an already-parsed dict.
        """
        try:
            if isinstance(arguments, str):
                validated = self.args_model.model_validate_json(arguments or "{}")
            else:
                validated = self.args_model.model_validate(arguments)
        except ValidationError as exc:
            raise ToolValidationError(self.name, _format_pydantic_errors(exc)) from exc
        return validated.model_dump(exclude_unset=True)

    def schema(self) -> dict:
        """OpenAI function-calling schema, generated from the Pydantic model."""
        params = self.args_model.model_json_schema()
        params.setdefault("required", [])
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": params,
            },
        }


def _format_pydantic_errors(exc: ValidationError) -> list[str]:
    errors = []
    for error in exc.errors():
        error_type = error.get("type")
        if error_type == "json_invalid":
            errors.append("arguments JSON could not be parsed")
            continue
        location = ".".join(str(part) for part in error.get("loc", ())) or "<root>"
        if error_type == "missing":
            errors.append(f"missing required field '{location}'")
        elif error_type == "extra_forbidden":
            errors.append(f"unknown field '{location}'")
        else:
            errors.append(f"field '{location}': {error.get('msg', 'invalid value')}")
    return errors or ["invalid arguments"]
