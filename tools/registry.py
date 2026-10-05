from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ValidationError

from agent.models import ToolCall


class ToolError(RuntimeError):
    """Raised when tool lookup, validation, or execution fails."""


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, tuple[type[BaseModel], Callable[..., object]]] = {}

    def register(
        self,
        name: str,
        schema: type[BaseModel],
        handler: Callable[[Any], object],
    ) -> None:
        if not name or not name.strip():
            raise ToolError("Tool name must not be empty")
        if name in self._tools:
            raise ToolError(f"Tool {name!r} is already registered")
        self._tools[name] = (schema, handler)

    def execute(self, call: ToolCall) -> dict[str, object]:
        registration = self._tools.get(call.name)
        if registration is None:
            raise ToolError(f"Unknown tool: {call.name}")
        schema, handler = registration
        try:
            arguments = schema.model_validate(call.arguments)
        except ValidationError as error:
            raise ToolError(f"Invalid arguments for tool {call.name!r}: {error}") from error
        try:
            result = handler(arguments)
        except Exception as error:
            raise ToolError(f"Tool {call.name!r} failed: {error}") from error
        if not isinstance(result, dict):
            raise ToolError(f"Tool {call.name!r} must return a JSON object")
        return result
