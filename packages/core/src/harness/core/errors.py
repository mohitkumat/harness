"""Typed exceptions raised across the harness.

Every error has a stable ``code`` that matches the ``code`` field of the protocol ``Error`` event,
so interfaces can react to failures without parsing messages.
"""

from __future__ import annotations

from typing import ClassVar, Literal

LimitKind = Literal["steps", "tokens", "cost", "wall_clock"]


class HarnessError(Exception):
    """Base class for all harness errors."""

    code: ClassVar[str] = "internal"

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.retryable = retryable


class ModelError(HarnessError):
    """A model provider call failed. Adapters mark 429 / 5xx / timeouts as retryable."""

    code: ClassVar[str] = "model_error"

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
        provider: str | None = None,
    ) -> None:
        super().__init__(message, retryable=retryable)
        self.status_code = status_code
        self.provider = provider


class ToolError(HarnessError):
    """A tool failed. The loop converts these into error results for the model; they never end a run."""

    code: ClassVar[str] = "tool_error"

    def __init__(self, message: str, *, tool_name: str, retryable: bool = False) -> None:
        super().__init__(message, retryable=retryable)
        self.tool_name = tool_name


class LimitExceeded(HarnessError):
    """A run hit one of its budgets (steps, tokens, cost, wall-clock)."""

    code: ClassVar[str] = "limit_exceeded"

    def __init__(self, limit: LimitKind, *, limit_value: float, actual: float) -> None:
        super().__init__(f"{limit} limit exceeded: {actual} > {limit_value}")
        self.limit: LimitKind = limit
        self.limit_value = limit_value
        self.actual = actual


class PermissionDenied(HarnessError):
    """A guardrail policy or the user refused an action."""

    code: ClassVar[str] = "permission_denied"


class InvalidCommand(HarnessError):
    """A client command is malformed or does not apply (unknown session, no pending approval, ...)."""

    code: ClassVar[str] = "invalid_command"
