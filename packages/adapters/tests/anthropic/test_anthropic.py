"""Unit tests for the Anthropic adapter's translation, streaming and error handling.

These use hand-written API payloads served by ``httpx2.MockTransport``: they pin down parsing and
mapping logic. Behaviour against the real API is covered by the recorded contract suite.
"""

import json
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic.types.beta import BetaMessage
from pydantic import JsonValue

from harness.adapters.anthropic import AnthropicAdapter, AnthropicConfig
from harness.adapters.anthropic.models import capabilities, cost_usd
from harness.adapters.anthropic.translate import from_message, to_messages, to_tool_choice
from harness.core import (
    ImagePart,
    Message,
    ModelError,
    ModelRequest,
    ReasoningPart,
    StopReason,
    TextPart,
    Tool,
    ToolCall,
    ToolResult,
    collect_stream,
)

THINKING: dict[str, JsonValue] = {"type": "thinking", "thinking": "", "signature": "sig-abc"}


def message_payload(
    content: list[dict[str, Any]], **overrides: Any
) -> dict[str, Any]:  # Any: raw API JSON
    payload: dict[str, Any] = {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-opus-5-5",
        "content": content,
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1_000_000, "output_tokens": 100_000},
    }
    payload.update(overrides)
    return payload


def sse(payload: dict[str, Any]) -> str:  # Any: raw API JSON
    """Encode a final message as the event stream the API would send for it."""
    events: list[tuple[str, dict[str, Any]]] = [
        (
            "message_start",
            {"type": "message_start", "message": {**payload, "content": [], "stop_reason": None}},
        )
    ]
    for i, block in enumerate(payload["content"]):
        if block["type"] == "text":
            events.append(
                (
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": i,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
            )
            for word in block["text"].split(" "):
                events.append(
                    (
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": i,
                            "delta": {"type": "text_delta", "text": word + " "},
                        },
                    )
                )
        elif block["type"] == "tool_use":
            events.append(
                (
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": i,
                        "content_block": {**block, "input": {}},
                    },
                )
            )
            events.append(
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": i,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(block["input"]),
                        },
                    },
                )
            )
        else:
            events.append(
                (
                    "content_block_start",
                    {"type": "content_block_start", "index": i, "content_block": block},
                )
            )
        events.append(("content_block_stop", {"type": "content_block_stop", "index": i}))
    events.append(
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": payload["stop_reason"], "stop_sequence": None},
                "usage": payload["usage"],
            },
        )
    )
    events.append(("message_stop", {"type": "message_stop"}))
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)


def adapter_for(handler: Any, **config: Any) -> AnthropicAdapter:  # Any: test plumbing
    transport = httpx2.MockTransport(handler)
    return AnthropicAdapter(
        AnthropicConfig(api_key="test-key", max_retries=0, **config),  # pyright: ignore[reportArgumentType]
        http_client=anthropic.DefaultAsyncHttpxClient(transport=transport),
    )


TOOLS = [Tool(name="get_weather", description="weather", input_schema={"type": "object"})]


# ----------------------------------------------------------------------------- request translation


def test_to_messages_maps_roles_and_parts() -> None:
    history = [
        Message(
            role="user",
            content=[TextPart(text="look"), ImagePart(media_type="image/png", data="AAA")],
        ),
        Message(
            role="assistant",
            content=[
                ReasoningPart(source="anthropic", data=THINKING),
                ReasoningPart(source="someone-else", data={"x": 1}),
                TextPart(text=""),
                ToolCall(id="t1", name="get_weather", arguments={"city": "Paris"}),
            ],
        ),
        Message(role="tool", content=[ToolResult(tool_call_id="t1", content="", is_error=True)]),
        Message(role="assistant", content=[]),
    ]

    out = to_messages(history)

    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert list(out[0]["content"])[1] == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "AAA"},
    }
    # Foreign reasoning and empty text are dropped; ours is replayed verbatim.
    assert list(out[1]["content"]) == [
        THINKING,
        {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Paris"}},
    ]
    assert list(out[2]["content"]) == [
        {"type": "tool_result", "tool_use_id": "t1", "content": "(no output)", "is_error": True}
    ]


def test_tool_choice() -> None:
    assert to_tool_choice("auto") == {"type": "auto"}
    assert to_tool_choice("none") == {"type": "none"}
    assert to_tool_choice("required") == {"type": "any"}


# ----------------------------------------------------------------------------- response translation


def test_from_message_keeps_order_and_reasoning() -> None:
    msg = BetaMessage.model_validate(
        message_payload(
            [
                THINKING,
                {"type": "text", "text": "checking"},
                {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Paris"}},
            ],
            stop_reason="tool_use",
        )
    )

    response = from_message(msg)

    parts = response.message.content
    assert isinstance(parts[0], ReasoningPart)
    assert parts[0].source == "anthropic"
    assert parts[0].data == THINKING
    assert parts[1] == TextPart(text="checking")
    assert parts[2] == ToolCall(id="t1", name="get_weather", arguments={"city": "Paris"})
    assert response.stop_reason == StopReason.TOOL_USE
    assert response.response_id == "msg_1"


def test_from_message_drops_declined_blocks_before_fallback() -> None:
    fallback = {
        "type": "fallback",
        "from": {"model": "claude-opus-5-5"},
        "to": {"model": "claude-opus-4-8"},
        "trigger": {"type": "refusal", "category": "cyber"},
    }
    msg = BetaMessage.model_validate(
        message_payload(
            [
                THINKING,
                {"type": "text", "text": "partial"},
                {"type": "tool_use", "id": "t0", "name": "get_weather", "input": {}},
                fallback,
                {"type": "text", "text": "rescued"},
            ],
            model="claude-opus-4-8",
        )
    )

    parts = from_message(msg).message.content

    assert parts == [TextPart(text="partial"), TextPart(text="rescued")]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("end_turn", StopReason.END_TURN),
        ("max_tokens", StopReason.MAX_TOKENS),
        ("refusal", StopReason.CONTENT_FILTER),
        ("pause_turn", StopReason.OTHER),
    ],
)
def test_stop_reasons(raw: str, expected: StopReason) -> None:
    msg = BetaMessage.model_validate(message_payload([], stop_reason=raw))
    assert from_message(msg).stop_reason == expected


def test_usage_and_cost() -> None:
    msg = BetaMessage.model_validate(
        message_payload(
            [],
            usage={
                "input_tokens": 1_000_000,
                "output_tokens": 100_000,
                "cache_read_input_tokens": 1_000_000,
                "cache_creation_input_tokens": 0,
            },
        )
    )
    usage = from_message(msg).usage
    assert usage.cache_read_tokens == 1_000_000
    assert usage.cost_usd == pytest.approx(4.0 + 2.0 + 0.2)


def test_cost_sums_fallback_iterations() -> None:
    msg = BetaMessage.model_validate(
        message_payload(
            [],
            model="claude-opus-4-8",
            usage={
                "input_tokens": 2_000_000,
                "output_tokens": 0,
                "iterations": [
                    {
                        "type": "message",
                        "model": "claude-opus-5-5",
                        "input_tokens": 1_000_000,
                        "output_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                    {
                        "type": "fallback_message",
                        "model": "claude-opus-4-8",
                        "input_tokens": 1_000_000,
                        "output_tokens": 0,
                        "cache_read_input_tokens": 0,
                        "cache_creation_input_tokens": 0,
                    },
                ],
            },
        )
    )
    assert from_message(msg).usage.cost_usd == pytest.approx(4.0 + 5.0)


def test_unknown_model_has_no_cost_and_safe_capabilities() -> None:
    assert cost_usd("mystery", input_tokens=1, output_tokens=1, cache_read=0, cache_write=0) is None
    caps = capabilities("mystery")
    assert caps.context_window == 200_000
    assert capabilities("claude-opus-5-5").context_window == 1_000_000


# ----------------------------------------------------------------------------- HTTP round trips


async def test_stream_sends_expected_request_and_parses_events() -> None:
    seen: list[dict[str, Any]] = []
    payload = message_payload(
        [
            THINKING,
            {"type": "text", "text": "let me check"},
            {"type": "tool_use", "id": "t1", "name": "get_weather", "input": {"city": "Paris"}},
        ],
        stop_reason="tool_use",
    )

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.read()))
        assert "test-key" in request.headers["x-api-key"]
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, text=sse(payload)
        )

    chunks: list[str] = []

    async def on_text(text: str) -> None:
        chunks.append(text)

    adapter = adapter_for(handler)
    response = await collect_stream(
        adapter.stream(ModelRequest(system="sys", messages=[Message.user("hi")], tools=TOOLS)),
        on_text,
    )

    body = seen[0]
    assert body["model"] == "claude-opus-5-5"
    assert body["system"] == "sys"
    assert body["stream"] is True
    assert body["cache_control"] == {"type": "ephemeral"}
    assert body["output_config"] == {"effort": "high"}
    assert body["fallbacks"] == "default"
    assert body["tool_choice"] == {"type": "auto"}
    assert "temperature" not in body
    assert "".join(chunks).strip() == "let me check"
    assert isinstance(response.message.content[0], ReasoningPart)
    assert response.message.tool_calls == [
        ToolCall(id="t1", name="get_weather", arguments={"city": "Paris"})
    ]


async def test_haiku_gets_no_effort_or_fallbacks() -> None:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.read()))
        payload = message_payload([{"type": "text", "text": "ok"}], model="claude-haiku-4-5")
        return httpx2.Response(
            200, headers={"content-type": "text/event-stream"}, text=sse(payload)
        )

    await adapter_for(handler, model="claude-haiku-4-5").complete(
        ModelRequest(messages=[Message.user("hi")])
    )

    assert "output_config" not in seen[0]
    assert "fallbacks" not in seen[0]
    assert "tools" not in seen[0]


@pytest.mark.parametrize(
    ("status", "retryable"),
    [(400, False), (401, False), (404, False), (408, True), (429, True), (500, True), (529, True)],
)
async def test_http_errors_map_to_model_error(status: int, retryable: bool) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            status, json={"type": "error", "error": {"type": "x", "message": "boom"}}
        )

    with pytest.raises(ModelError) as info:
        await adapter_for(handler).complete(ModelRequest(messages=[Message.user("hi")]))

    assert info.value.status_code == status
    assert info.value.retryable is retryable
    assert info.value.provider == "anthropic"
    assert "test-key" not in str(info.value)


async def test_connection_error_is_retryable() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("down", request=request)

    with pytest.raises(ModelError) as info:
        await adapter_for(handler).complete(ModelRequest(messages=[Message.user("hi")]))

    assert info.value.retryable
    assert info.value.status_code is None
