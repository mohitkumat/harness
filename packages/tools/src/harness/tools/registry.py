"""Tool registry: validates arguments against each tool's JSON schema and runs its handler.

Implements ``harness.core.ToolExecutor``. Every failure (unknown tool, bad arguments, timeout,
handler error) comes back as an ``is_error`` result for the model; nothing is raised to the loop.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError
from pydantic import JsonValue

from harness.core import Tool, ToolCall, ToolError, ToolOutcome, ToolResult

DEFAULT_MAX_OUTPUT_CHARS = 20_000


@dataclass(frozen=True)
class ToolContext:
    """What a handler may use besides its arguments."""

    workspace: Path


Handler = Callable[[dict[str, JsonValue], ToolContext], Awaitable[str]]
"""Returns the text sent to the model. Raise ``ToolError`` for expected failures."""


@dataclass(frozen=True)
class ToolSpec:
    """A tool definition plus the handler that runs it."""

    tool: Tool
    handler: Handler


def truncate(text: str, max_chars: int) -> tuple[str, bool]:
    """Keep the head and tail of ``text`` with a marker in between if it exceeds ``max_chars``."""
    if len(text) <= max_chars:
        return text, False
    half = max_chars // 2
    omitted = len(text) - 2 * half
    marker = f"\n\n[... {omitted} characters truncated ...]\n\n"
    return text[:half] + marker + text[-half:], True


class ToolRegistry:
    """Holds the tools available to a session, all bound to one workspace."""

    def __init__(
        self,
        workspace: Path,
        specs: list[ToolSpec] | None = None,
        *,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    ) -> None:
        self.context = ToolContext(workspace=workspace.resolve())
        self.max_output_chars = max_output_chars
        self._specs: dict[str, ToolSpec] = {}
        self._validators: dict[str, Draft202012Validator] = {}
        for spec in specs or []:
            self.register(spec)

    def register(self, spec: ToolSpec) -> None:
        """Add a tool. Raises ``ValueError`` on duplicate names or an invalid schema."""
        name = spec.tool.name
        if name in self._specs:
            raise ValueError(f"tool {name!r} is already registered")
        Draft202012Validator.check_schema(spec.tool.input_schema)
        self._specs[name] = spec
        self._validators[name] = Draft202012Validator(spec.tool.input_schema)

    @property
    def tools(self) -> list[Tool]:
        """Definitions in registration order."""
        return [s.tool for s in self._specs.values()]

    async def execute(self, call: ToolCall) -> ToolOutcome:
        """Validate and run ``call`` within the tool's timeout. Never raises for tool failures."""
        spec = self._specs.get(call.name)
        if spec is None:
            known = ", ".join(self._specs) or "none"
            return _error(call, f"Unknown tool {call.name!r}. Available tools: {known}.")

        validator = self._validators[call.name]
        errors: list[ValidationError] = list(
            validator.iter_errors(call.arguments)  # pyright: ignore[reportUnknownMemberType]  (stubs are untyped here)
        )
        problems = [
            f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}" for e in errors
        ]
        if problems:
            return _error(call, "Invalid arguments:\n" + "\n".join(problems))

        try:
            output = await asyncio.wait_for(
                spec.handler(call.arguments, self.context), spec.tool.timeout_s
            )
        except TimeoutError:
            return _error(call, f"Tool {call.name!r} timed out after {spec.tool.timeout_s:g}s.")
        except ToolError as e:
            return _error(call, e.message)
        except Exception as e:  # handler bug: report it to the model and in the ToolEnd event
            return _error(call, f"Tool {call.name!r} failed: {type(e).__name__}: {e}")

        content, truncated = truncate(output, self.max_output_chars)
        return ToolOutcome(ToolResult(tool_call_id=call.id, content=content), truncated=truncated)


def _error(call: ToolCall, message: str) -> ToolOutcome:
    return ToolOutcome(ToolResult(tool_call_id=call.id, content=message, is_error=True))
