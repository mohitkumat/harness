"""Canonical, provider-independent types. The agent loop only ever sees these.

Adapters translate between these and each provider's wire format (both directions).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# ----------------------------------------------------------------------------- content parts


class TextPart(_Model):
    """Plain text content."""

    type: Literal["text"] = "text"
    text: str


class ImagePart(_Model):
    """An image, either inline (base64 ``data``) or by ``url``. Only sent to vision-capable models."""

    type: Literal["image"] = "image"
    media_type: str
    data: str | None = None
    url: str | None = None

    @model_validator(mode="after")
    def _exactly_one_source(self) -> Self:
        if (self.data is None) == (self.url is None):
            raise ValueError("ImagePart needs exactly one of `data` or `url`")
        return self


class ReasoningPart(_Model):
    """Model reasoning that must be replayed to the model that produced it, byte for byte.

    ``data`` is opaque: only the adapter named by ``source`` interprets it (e.g. signed thinking
    blocks or encrypted reasoning items). Other adapters drop parts whose ``source`` they don't own.
    ``text`` is a human-readable summary, possibly empty; it is never sent back to a model.
    """

    type: Literal["reasoning"] = "reasoning"
    source: str
    data: dict[str, JsonValue]
    text: str = ""


class ToolCall(_Model):
    """A model's request to invoke a tool. ``arguments`` is always a parsed object, never a JSON string."""

    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict[str, JsonValue]


class ToolResult(_Model):
    """The outcome of a tool call, sent back to the model. Tool failures are results with ``is_error``."""

    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    content: str
    is_error: bool = False


ContentPart = Annotated[
    TextPart | ImagePart | ReasoningPart | ToolCall | ToolResult, Field(discriminator="type")
]

Role = Literal["user", "assistant", "tool"]

_ALLOWED_PARTS: dict[Role, tuple[type[_Model], ...]] = {
    "user": (TextPart, ImagePart),
    "assistant": (TextPart, ReasoningPart, ToolCall),
    "tool": (ToolResult,),
}


class Message(_Model):
    """One turn of conversation history.

    The system prompt is not a message; it lives on ``ModelRequest.system``. Tool results travel
    in ``role="tool"`` messages that directly follow the assistant message that made the calls.
    """

    role: Role
    content: list[ContentPart]

    @model_validator(mode="after")
    def _parts_match_role(self) -> Self:
        allowed = _ALLOWED_PARTS[self.role]
        for part in self.content:
            if not isinstance(part, allowed):
                raise ValueError(f"{part.type!r} content is not allowed in a {self.role!r} message")
        return self

    @classmethod
    def user(cls, text: str) -> Message:
        """Build a plain-text user message."""
        return cls(role="user", content=[TextPart(text=text)])

    @classmethod
    def assistant(cls, text: str) -> Message:
        """Build a plain-text assistant message."""
        return cls(role="assistant", content=[TextPart(text=text)])

    @property
    def text(self) -> str:
        """All text parts concatenated."""
        return "".join(p.text for p in self.content if isinstance(p, TextPart))

    @property
    def tool_calls(self) -> list[ToolCall]:
        """Tool calls in this message, in order."""
        return [p for p in self.content if isinstance(p, ToolCall)]


# ----------------------------------------------------------------------------- tools


class Tool(_Model):
    """A tool definition as advertised to the model. Handlers are attached by the tool registry."""

    name: Annotated[str, Field(pattern=r"^[a-zA-Z0-9_-]{1,64}$")]
    description: str
    input_schema: dict[str, JsonValue]
    risky: bool = False
    timeout_s: Annotated[float, Field(gt=0)] = 30.0


# ----------------------------------------------------------------------------- model I/O


class Usage(_Model):
    """Token usage (and cost, when known) for one or more model calls."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float | None = None

    def __add__(self, other: Usage) -> Usage:
        cost = (
            None
            if self.cost_usd is None and other.cost_usd is None
            else (self.cost_usd or 0.0) + (other.cost_usd or 0.0)
        )
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cost_usd=cost,
        )


class StopReason(StrEnum):
    """Why the model stopped generating, normalised across providers."""

    END_TURN = "end_turn"
    TOOL_USE = "tool_use"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    CONTENT_FILTER = "content_filter"
    OTHER = "other"


class ModelRequest(_Model):
    """Everything an adapter needs for one model call. Built fresh per call by the context manager."""

    system: str | None = None
    messages: list[Message]
    tools: list[Tool] = []
    tool_choice: Literal["auto", "none", "required"] = "auto"
    max_output_tokens: int | None = None
    temperature: float | None = None
    stop_sequences: list[str] = []


class ModelResponse(_Model):
    """A complete model reply in canonical form."""

    message: Message
    stop_reason: StopReason
    usage: Usage
    model: str
    response_id: str | None = None

    @model_validator(mode="after")
    def _assistant_only(self) -> Self:
        if self.message.role != "assistant":
            raise ValueError("ModelResponse.message must be an assistant message")
        return self


class ModelCapabilities(_Model):
    """What a model endpoint supports. The harness adapts behaviour to these, never to provider names."""

    context_window: int
    max_output_tokens: int
    native_tools: bool
    parallel_tool_calls: bool
    vision: bool
    streaming: bool = True
    prompt_caching: bool = False
    tokenizer: str = Field(
        description="Tokenizer id used for budgeting, e.g. 'tiktoken:o200k_base' or 'approx'."
    )
