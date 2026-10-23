"""Canonical ↔ OpenAI Chat Completions translation. Pure functions, no I/O.

Quirks of local OpenAI-compatible servers (Ollama, vLLM, LM Studio) handled here:

- Some servers report ``finish_reason="stop"`` for a reply that contains tool calls. A reply
  with tool calls is always ``TOOL_USE``.
- Some servers omit tool-call ids. Missing ids are derived from the response id and the call's
  index, so they are deterministic (cassettes replay).
- Small models emit arguments that are not valid JSON objects. Those become
  ``{"_raw": "<the string>"}``: the tool registry's schema check then rejects the call and the
  model gets a tool error back, instead of the run crashing.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal

import openai
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionContentPartParam,
    ChatCompletionMessageParam,
    ChatCompletionMessageToolCallParam,
    ChatCompletionToolChoiceOptionParam,
    ChatCompletionToolParam,
)
from openai.types.completion_usage import CompletionUsage
from pydantic import JsonValue

from harness.core import (
    RAW_ARGUMENTS_KEY,
    ImagePart,
    Message,
    ModelError,
    ModelResponse,
    StopReason,
    TextPart,
    Tool,
    ToolCall,
    ToolResult,
    Usage,
)

SOURCE = "openai_compat"

_STOP_REASONS: dict[str, StopReason] = {
    "stop": StopReason.END_TURN,
    "tool_calls": StopReason.TOOL_USE,
    "function_call": StopReason.TOOL_USE,
    "length": StopReason.MAX_TOKENS,
    "content_filter": StopReason.CONTENT_FILTER,
}


# ----------------------------------------------------------------------------- request


def _user_part(part: TextPart | ImagePart) -> ChatCompletionContentPartParam:
    if isinstance(part, TextPart):
        return {"type": "text", "text": part.text}
    url = part.url if part.url is not None else f"data:{part.media_type};base64,{part.data}"
    return {"type": "image_url", "image_url": {"url": url}}


def _tool_message(result: ToolResult) -> ChatCompletionMessageParam:
    # Chat Completions has no error flag on tool results; say it in the content.
    content = result.content or "(no output)"
    return {
        "role": "tool",
        "tool_call_id": result.tool_call_id,
        "content": f"Error: {content}" if result.is_error else content,
    }


def _assistant_message(msg: Message) -> ChatCompletionMessageParam | None:
    text = msg.text  # ReasoningPart is dropped: these servers have no reasoning round trip.
    calls: list[ChatCompletionMessageToolCallParam] = [
        {
            "id": c.id,
            "type": "function",
            "function": {"name": c.name, "arguments": _arguments_json(c.arguments)},
        }
        for c in msg.tool_calls
    ]
    if not text and not calls:
        return None
    if calls:
        return {"role": "assistant", "content": text or None, "tool_calls": calls}
    return {"role": "assistant", "content": text}


def _arguments_json(arguments: dict[str, JsonValue]) -> str:
    raw = arguments.get(RAW_ARGUMENTS_KEY)
    if len(arguments) == 1 and isinstance(raw, str):
        return raw  # echo what the model actually sent
    return json.dumps(arguments)


def to_messages(system: str | None, messages: list[Message]) -> list[ChatCompletionMessageParam]:
    """Translate the system prompt and history. Each ``ToolResult`` becomes its own tool message."""
    out: list[ChatCompletionMessageParam] = []
    if system:
        out.append({"role": "system", "content": system})
    for msg in messages:
        match msg.role:
            case "user":
                parts = [p for p in msg.content if isinstance(p, TextPart | ImagePart)]
                if all(isinstance(p, TextPart) for p in parts):
                    if msg.text:
                        out.append({"role": "user", "content": msg.text})
                elif parts:
                    out.append({"role": "user", "content": [_user_part(p) for p in parts]})
            case "assistant":
                if (assistant := _assistant_message(msg)) is not None:
                    out.append(assistant)
            case "tool":
                out.extend(_tool_message(p) for p in msg.content if isinstance(p, ToolResult))
    return out


def to_tools(tools: list[Tool]) -> list[ChatCompletionToolParam]:
    """Translate tool definitions to function tools."""
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": dict(t.input_schema),
            },
        }
        for t in tools
    ]


def to_tool_choice(
    choice: Literal["auto", "none", "required"],
) -> ChatCompletionToolChoiceOptionParam:
    """Canonical tool choice maps one to one."""
    return choice


# ----------------------------------------------------------------------------- response


def parse_arguments(raw: str) -> dict[str, JsonValue]:
    """Parse a JSON-string argument object; anything else is wrapped under ``RAW_ARGUMENTS_KEY``."""
    if not raw.strip():
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {RAW_ARGUMENTS_KEY: raw}
    if not isinstance(value, dict):
        return {RAW_ARGUMENTS_KEY: raw}
    return value  # pyright: ignore[reportUnknownVariableType]  (json.loads is untyped)


def call_id(response_id: str, index: int, given: str | None) -> str:
    """Return the provider's tool-call id, or a deterministic one if the server sent none."""
    return given or f"call_{response_id}_{index}"


def stop_reason(finish_reason: str | None, *, has_tool_calls: bool) -> StopReason:
    """Map ``finish_reason``. A reply with tool calls is always ``TOOL_USE``."""
    if has_tool_calls:
        return StopReason.TOOL_USE
    return _STOP_REASONS.get(finish_reason or "", StopReason.OTHER)


def to_usage(
    usage: CompletionUsage | None, *, input_per_mtok: float, output_per_mtok: float
) -> Usage:
    """Canonical usage, with cost from the configured per-million-token prices."""
    if usage is None:
        return Usage(cost_usd=0.0 if input_per_mtok == output_per_mtok == 0 else None)
    cached = usage.prompt_tokens_details.cached_tokens if usage.prompt_tokens_details else None
    return Usage(
        input_tokens=usage.prompt_tokens,
        output_tokens=usage.completion_tokens,
        cache_read_tokens=cached or 0,
        cost_usd=(usage.prompt_tokens * input_per_mtok + usage.completion_tokens * output_per_mtok)
        / 1_000_000,
    )


@dataclass
class PendingCall:
    """A tool call being assembled from stream chunks."""

    id: str | None = None
    name: str = ""
    fragments: list[str] = field(default_factory=list[str])


def build_message(text: str, calls: Iterable[tuple[int, PendingCall]], response_id: str) -> Message:
    """Assemble the assistant message from final text and (index, call) pairs, in index order."""
    content: list[TextPart | ToolCall] = [TextPart(text=text)] if text else []
    for index, call in sorted(calls, key=lambda item: item[0]):
        content.append(
            ToolCall(
                id=call_id(response_id, index, call.id),
                name=call.name,
                arguments=parse_arguments("".join(call.fragments)),
            )
        )
    return Message(role="assistant", content=list(content))


def from_completion(
    completion: ChatCompletion, *, input_per_mtok: float, output_per_mtok: float
) -> ModelResponse:
    """Translate a complete (non-streaming) response."""
    if not completion.choices:
        raise ModelError("response has no choices", provider=SOURCE)
    choice = completion.choices[0]
    calls = [
        (i, PendingCall(id=c.id, name=c.function.name, fragments=[c.function.arguments or ""]))
        for i, c in enumerate(choice.message.tool_calls or [])
        if c.type == "function"
    ]
    message = build_message(choice.message.content or "", calls, completion.id)
    return ModelResponse(
        message=message,
        stop_reason=stop_reason(choice.finish_reason, has_tool_calls=bool(calls)),
        usage=to_usage(
            completion.usage, input_per_mtok=input_per_mtok, output_per_mtok=output_per_mtok
        ),
        model=completion.model,
        response_id=completion.id,
    )


# ----------------------------------------------------------------------------- errors


def to_model_error(error: openai.APIError) -> ModelError:
    """Map SDK errors. Timeouts, connection errors, 408/409/429 and 5xx are retryable."""
    if isinstance(error, openai.APIConnectionError):  # includes APITimeoutError
        return ModelError(f"connection error: {error.message}", retryable=True, provider=SOURCE)
    if isinstance(error, openai.APIStatusError):
        status = error.status_code
        retryable = status in (408, 409, 429) or status >= 500
        return ModelError(
            f"HTTP {status}: {error.message}",
            retryable=retryable,
            status_code=status,
            provider=SOURCE,
        )
    return ModelError(f"{type(error).__name__}: {error.message}", provider=SOURCE)
