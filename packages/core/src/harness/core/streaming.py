"""Turns a canonical ``Delta`` stream into a ``ModelResponse``."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field

from pydantic import JsonValue

from harness.core.adapter import Delta, StopDelta, TextDelta, ToolCallArgsDelta, ToolCallStartDelta
from harness.core.errors import ModelError
from harness.core.types import ContentPart, Message, ModelResponse, TextPart, ToolCall


@dataclass
class _PendingCall:
    id: str
    name: str
    fragments: list[str] = field(default_factory=list[str])


def _parse_arguments(call: _PendingCall) -> dict[str, JsonValue]:
    raw = "".join(call.fragments).strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ModelError(f"tool call {call.name!r} has malformed JSON arguments: {e}") from e
    if not isinstance(value, dict):
        raise ModelError(f"tool call {call.name!r} arguments are not a JSON object")
    return value  # pyright: ignore[reportUnknownVariableType]  (json.loads is untyped)


async def collect_stream(
    deltas: AsyncIterator[Delta], on_text: Callable[[str], Awaitable[None]] | None = None
) -> ModelResponse:
    """Consume ``deltas`` into one response, calling ``on_text`` for every text chunk as it arrives.

    Text and tool calls keep their stream order. Raises ``ModelError`` if the stream ends without
    a ``StopDelta`` or a tool call's arguments are not a JSON object.
    """
    parts: list[TextPart | _PendingCall] = []
    calls: dict[int, _PendingCall] = {}
    stop: StopDelta | None = None

    async for delta in deltas:
        match delta:
            case TextDelta(text=text):
                if on_text is not None and text:
                    await on_text(text)
                last = parts[-1] if parts else None
                if isinstance(last, TextPart):
                    parts[-1] = TextPart(text=last.text + text)
                else:
                    parts.append(TextPart(text=text))
            case ToolCallStartDelta(index=index, id=call_id, name=name):
                calls[index] = _PendingCall(id=call_id, name=name)
                parts.append(calls[index])
            case ToolCallArgsDelta(index=index, json_fragment=fragment):
                if index not in calls:
                    raise ModelError(f"tool call arguments for unknown index {index}")
                calls[index].fragments.append(fragment)
            case StopDelta():
                stop = delta

    if stop is None:
        raise ModelError("model stream ended without a stop delta", retryable=True)

    if stop.message is not None:
        return ModelResponse(
            message=stop.message,
            stop_reason=stop.stop_reason,
            usage=stop.usage,
            model=stop.model,
            response_id=stop.response_id,
        )

    content: list[ContentPart] = []
    for part in parts:
        if isinstance(part, TextPart):
            if part.text:
                content.append(part)
        else:
            content.append(ToolCall(id=part.id, name=part.name, arguments=_parse_arguments(part)))
    return ModelResponse(
        message=Message(role="assistant", content=content),
        stop_reason=stop.stop_reason,
        usage=stop.usage,
        model=stop.model,
        response_id=stop.response_id,
    )
