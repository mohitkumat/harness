"""Canonical ↔ Anthropic Messages API translation. Pure functions, no I/O."""

from __future__ import annotations

from typing import Literal, cast

import anthropic
from anthropic.types.beta import (
    BetaContentBlockParam,
    BetaFallbackMessageIterationUsage,
    BetaMessage,
    BetaMessageIterationUsage,
    BetaMessageParam,
    BetaToolChoiceParam,
    BetaToolParam,
    BetaToolResultBlockParam,
    BetaToolUseBlockParam,
    BetaUsage,
)
from anthropic.types.beta.beta_tool_param import InputSchema
from pydantic import JsonValue, TypeAdapter

from harness.adapters.anthropic.models import cost_usd
from harness.core import (
    ImagePart,
    Message,
    ModelError,
    ModelResponse,
    ReasoningPart,
    StopReason,
    TextPart,
    Tool,
    ToolCall,
    ToolResult,
    Usage,
)

SOURCE = "anthropic"
"""``ReasoningPart.source`` for blocks this adapter produced and may replay."""

_DROPPED_BEFORE_FALLBACK = {"thinking", "redacted_thinking", "tool_use", "server_tool_use"}

_JSON_OBJECT: TypeAdapter[dict[str, JsonValue]] = TypeAdapter(dict[str, JsonValue])

_STOP_REASONS: dict[str, StopReason] = {
    "end_turn": StopReason.END_TURN,
    "tool_use": StopReason.TOOL_USE,
    "max_tokens": StopReason.MAX_TOKENS,
    "stop_sequence": StopReason.STOP_SEQUENCE,
    "refusal": StopReason.CONTENT_FILTER,
}


# ----------------------------------------------------------------------------- request


def _user_block(part: TextPart | ImagePart) -> BetaContentBlockParam | None:
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text} if part.text else None
    if part.data is not None:
        media = cast(Literal["image/jpeg", "image/png", "image/gif", "image/webp"], part.media_type)
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": media, "data": part.data},
        }
    assert part.url is not None  # ImagePart guarantees exactly one source
    return {"type": "image", "source": {"type": "url", "url": part.url}}


def _assistant_block(part: TextPart | ReasoningPart | ToolCall) -> BetaContentBlockParam | None:
    match part:
        case TextPart(text=text):
            return {"type": "text", "text": text} if text else None
        case ReasoningPart(source=source, data=data):
            # Replayed verbatim (preserved thinking); other providers' reasoning is dropped.
            return cast(BetaContentBlockParam, data) if source == SOURCE else None
        case ToolCall(id=call_id, name=name, arguments=arguments):
            tool_use: BetaToolUseBlockParam = {
                "type": "tool_use",
                "id": call_id,
                "name": name,
                "input": cast(dict[str, object], arguments),
            }
            return tool_use


def _tool_result_block(result: ToolResult) -> BetaToolResultBlockParam:
    return {
        "type": "tool_result",
        "tool_use_id": result.tool_call_id,
        "content": result.content or "(no output)",
        "is_error": result.is_error,
    }


def to_messages(messages: list[Message]) -> list[BetaMessageParam]:
    """Translate canonical history. ``role="tool"`` becomes a user turn of ``tool_result`` blocks.

    The output is a pure function of the input, so an append-only history yields an append-only
    request prefix (required for prompt caching and preserved thinking).
    """
    out: list[BetaMessageParam] = []
    for msg in messages:
        blocks: list[BetaContentBlockParam] = []
        for part in msg.content:
            block: BetaContentBlockParam | None
            if isinstance(part, ToolResult):
                block = _tool_result_block(part)
            elif isinstance(part, TextPart | ImagePart) and msg.role == "user":
                block = _user_block(part)
            elif isinstance(part, TextPart | ReasoningPart | ToolCall):
                block = _assistant_block(part)
            else:
                raise ModelError(f"cannot send {part.type!r} content in a {msg.role!r} message")
            if block is not None:
                blocks.append(block)
        if blocks:  # the API rejects empty turns
            out.append(
                {"role": "assistant" if msg.role == "assistant" else "user", "content": blocks}
            )
    return out


def to_tools(tools: list[Tool]) -> list[BetaToolParam]:
    """Translate tool definitions."""
    return [
        {
            "name": t.name,
            "description": t.description,
            "input_schema": cast(InputSchema, t.input_schema),
        }
        for t in tools
    ]


def to_tool_choice(choice: Literal["auto", "none", "required"]) -> BetaToolChoiceParam:
    """Map canonical tool choice. ``required`` → ``any``, which newer models reject with a 400."""
    match choice:
        case "required":
            return {"type": "any"}
        case "none":
            return {"type": "none"}
        case "auto":
            return {"type": "auto"}


# ----------------------------------------------------------------------------- response


def to_usage(usage: BetaUsage, model: str) -> Usage:
    """Canonical usage. Cost is summed per iteration so a server-side fallback bills each model."""
    pieces: list[tuple[str, int, int, int, int]] = [
        (
            it.model or model,
            it.input_tokens,
            it.output_tokens,
            it.cache_read_input_tokens,
            it.cache_creation_input_tokens,
        )
        for it in usage.iterations or []
        if isinstance(it, BetaMessageIterationUsage | BetaFallbackMessageIterationUsage)
    ] or [
        (
            model,
            usage.input_tokens,
            usage.output_tokens,
            usage.cache_read_input_tokens or 0,
            usage.cache_creation_input_tokens or 0,
        )
    ]
    costs = [
        cost_usd(m, input_tokens=i, output_tokens=o, cache_read=r, cache_write=w)
        for m, i, o, r, w in pieces
    ]
    known = [c for c in costs if c is not None]
    return Usage(
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_input_tokens or 0,
        cache_write_tokens=usage.cache_creation_input_tokens or 0,
        cost_usd=sum(known) if len(known) == len(costs) else None,
    )


def _json_object(value: object, what: str) -> dict[str, JsonValue]:
    """Validate that SDK-decoded JSON is an object (it always is JSON-compatible)."""
    if not isinstance(value, dict):
        raise ModelError(f"{what} is not a JSON object")
    return _JSON_OBJECT.validate_python(value)


def from_message(message: BetaMessage) -> ModelResponse:
    """Translate a complete response.

    After a mid-output server-side fallback, reasoning and tool-use blocks before the last
    ``fallback`` marker belong to the declined attempt and must not be echoed back, so they are
    dropped here. The marker itself is an audit record and is dropped too.
    """
    blocks = list(message.content)
    boundary = max((i for i, b in enumerate(blocks) if b.type == "fallback"), default=-1)

    content: list[TextPart | ReasoningPart | ToolCall] = []
    for i, block in enumerate(blocks):
        if i < boundary and block.type in _DROPPED_BEFORE_FALLBACK:
            continue
        match block.type:
            case "text":
                if block.text:
                    content.append(TextPart(text=block.text))
            case "thinking" | "redacted_thinking":
                summary = block.thinking if block.type == "thinking" else ""
                data = _json_object(block.to_dict(mode="json"), "thinking block")
                content.append(ReasoningPart(source=SOURCE, data=data, text=summary))
            case "tool_use":
                arguments = _json_object(block.input, f"tool_use {block.name!r} input")
                content.append(ToolCall(id=block.id, name=block.name, arguments=arguments))
            case _:
                pass  # fallback markers; server-tool blocks never appear (no server tools sent)

    reason = message.stop_reason or ""
    return ModelResponse(
        message=Message(role="assistant", content=list(content)),
        stop_reason=_STOP_REASONS.get(reason, StopReason.OTHER),
        usage=to_usage(message.usage, message.model),
        model=message.model,
        response_id=message.id,
    )


# ----------------------------------------------------------------------------- errors


def to_model_error(error: anthropic.APIError) -> ModelError:
    """Map SDK errors. Timeouts, connection errors, 408/409/429 and 5xx are retryable."""
    if isinstance(error, anthropic.APIConnectionError):  # includes APITimeoutError
        return ModelError(f"connection error: {error.message}", retryable=True, provider=SOURCE)
    if isinstance(error, anthropic.APIStatusError):
        status = error.status_code
        retryable = status in (408, 409, 429) or status >= 500
        return ModelError(
            f"HTTP {status}: {error.message}",
            retryable=retryable,
            status_code=status,
            provider=SOURCE,
        )
    return ModelError(f"{type(error).__name__}: {error.message}", provider=SOURCE)
