"""Base class for all tools."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolValidationError(Exception):
    """Raised when a tool call does not match the tool's JSON schema."""

    tool_name: str
    errors: list[str]

    def __str__(self) -> str:
        return f"bad arguments for {self.tool_name}: " + "; ".join(self.errors)


@dataclass
class ToolOutput:
    """Structured tool output for the UI while preserving model-visible text."""

    content: str
    preview: str = ""
    diff: str = ""

class Tool(ABC):
    """Minimal tool interface. Subclass this to add new capabilities."""

    name: str
    description: str
    parameters: dict  # JSON Schema for the function args

    @abstractmethod
    def execute(self, **kwargs) -> str | ToolOutput:
        """Run the tool and return a text result."""
        ...

    def validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate raw model-provided arguments against this tool's schema."""
        if not isinstance(arguments, dict):
            raise ToolValidationError(self.name, ["arguments must be an object"])

        # When the LLM returns arguments that fail JSON parsing, the llm layer
        # wraps the raw string in a __malformed_arguments__ key.  Surface the
        # real error so the LLM can adjust, instead of just saying "missing field".
        if "__malformed_arguments__" in arguments:
            raise ToolValidationError(self.name, [
                f"arguments JSON could not be parsed: {arguments['__parse_error__']}",
                f"raw argument text was: {arguments['__malformed_arguments__'][:500]}",
            ])

        schema = self.parameters
        if schema.get("type") != "object":
            raise ToolValidationError(self.name, ["tool parameters schema must be an object"])

        properties = schema.get("properties", {})
        required = set(schema.get("required", []))
        errors: list[str] = []

        for field in required:
            if field not in arguments:
                errors.append(f"missing required field '{field}'")

        for field in arguments:
            if field not in properties:
                errors.append(f"unknown field '{field}'")

        validated: dict[str, Any] = {}
        for field, value in arguments.items():
            field_schema = properties.get(field)
            if not field_schema:
                continue
            ok, expected = _matches_schema_type(value, field_schema.get("type"))
            if not ok:
                errors.append(f"field '{field}' must be {expected}, got {type(value).__name__}")
                continue
            validated[field] = value

        if errors:
            raise ToolValidationError(self.name, errors)
        return validated

    def schema(self) -> dict:
        """OpenAI function-calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


def _matches_schema_type(value: Any, schema_type: Any) -> tuple[bool, str]:
    if schema_type is None:
        return True, "any"

    types = schema_type if isinstance(schema_type, list) else [schema_type]
    if value is None:
        return (("null" in types), _format_expected(types))

    checks = {
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: (isinstance(v, int | float) and not isinstance(v, bool)),
        "boolean": lambda v: isinstance(v, bool),
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
    }
    for t in types:
        checker = checks.get(t)
        if checker is None or checker(value):
            return True, _format_expected(types)
    return False, _format_expected(types)


def _format_expected(types: list[str]) -> str:
    return " or ".join(types)
