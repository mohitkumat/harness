"""The ``ModelAdapter`` contract and the canonical streaming deltas.

Every provider adapter in ``harness.adapters`` implements ``ModelAdapter``. Nothing outside
``harness.adapters`` may know which provider sits behind it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from harness.core.types import (
    Message,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    StopReason,
    Usage,
)


class _Delta(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TextDelta(_Delta):
    """A chunk of assistant text."""

    type: Literal["text"] = "text"
    text: str


class ToolCallStartDelta(_Delta):
    """A new tool call begins. ``index`` identifies it among parallel calls in this response."""

    type: Literal["tool_call_start"] = "tool_call_start"
    index: int
    id: str
    name: str


class ToolCallArgsDelta(_Delta):
    """A fragment of a tool call's JSON arguments. Concatenate fragments per ``index``."""

    type: Literal["tool_call_args"] = "tool_call_args"
    index: int
    json_fragment: str


class StopDelta(_Delta):
    """The final delta of every stream.

    If ``message`` is set, it is the authoritative assistant message and consumers use it instead
    of reassembling the earlier deltas. That lets an adapter stream text live but build structural
    content (reasoning, tool calls) from the provider's final response.
    """

    type: Literal["stop"] = "stop"
    stop_reason: StopReason
    usage: Usage
    model: str
    response_id: str | None = None
    message: Message | None = None


Delta = Annotated[
    TextDelta | ToolCallStartDelta | ToolCallArgsDelta | StopDelta, Field(discriminator="type")
]


@runtime_checkable
class ModelAdapter(Protocol):
    """A swappable model endpoint speaking canonical types.

    Implementations must map provider failures to ``harness.core.errors.ModelError`` and mark
    429 / 5xx / timeouts as retryable.
    """

    @property
    def capabilities(self) -> ModelCapabilities:
        """Static capabilities of the configured model."""
        ...

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Run one non-streaming model call."""
        ...

    def stream(self, request: ModelRequest) -> AsyncIterator[Delta]:
        """Run one streaming model call. The last delta is always a ``StopDelta``."""
        ...
