"""Extension points the agent loop depends on. Other packages implement them; the loop stays small.

- ``ToolExecutor``: implemented by ``harness.tools`` (registry + built-ins).
- ``ApprovalPolicy``: implemented by ``harness.guardrails``.
- ``ContextBuilder``: implemented by ``harness.context`` (Phase 3); ``PassthroughContext`` until then.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from harness.core.types import Message, ModelRequest, Tool, ToolCall, ToolResult


@dataclass(frozen=True)
class ToolOutcome:
    """A tool result plus execution metadata that is reported in events but not sent to the model."""

    result: ToolResult
    truncated: bool = False


class ToolExecutor(Protocol):
    """Runs tool calls. Tool failures come back as ``is_error`` results; ``execute`` never raises them."""

    @property
    def tools(self) -> list[Tool]:
        """Definitions advertised to the model."""
        ...

    async def execute(self, call: ToolCall) -> ToolOutcome:
        """Validate the arguments and run the call, enforcing the tool's timeout."""
        ...


PolicyAction = Literal["allow", "ask", "deny"]


@dataclass(frozen=True)
class PolicyDecision:
    """What to do with a tool call before running it, and why (shown to the user and the model)."""

    action: PolicyAction
    reason: str = ""


class ApprovalPolicy(Protocol):
    """Decides whether a tool call runs, needs user approval, or is refused."""

    def check(self, tool: Tool | None, call: ToolCall) -> PolicyDecision:
        """``tool`` is ``None`` when the model called a tool that does not exist."""
        ...


class ContextBuilder(Protocol):
    """Builds the model request for one call from the full session history."""

    def build(
        self, *, system: str | None, history: list[Message], tools: list[Tool]
    ) -> ModelRequest:
        """Return a budgeted view of ``history``. Must never split a tool call from its result."""
        ...


@dataclass(frozen=True)
class PassthroughContext:
    """Sends the full history every call. Stable prefix (system + tools) comes first by construction."""

    # TODO(phase-3): replace with harness.context's budgeted, compacting context manager.

    max_output_tokens: int | None = None

    def build(
        self, *, system: str | None, history: list[Message], tools: list[Tool]
    ) -> ModelRequest:
        """Return all of ``history`` unchanged."""
        return ModelRequest(
            system=system,
            messages=history,
            tools=tools,
            max_output_tokens=self.max_output_tokens,
        )
