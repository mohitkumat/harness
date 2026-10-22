"""Unit tests for the OpenAI-compatible adapter's translation, streaming and error handling.

These use hand-written API payloads served by ``httpx2.MockTransport``: they pin down parsing and
mapping logic. Behaviour against a real server is covered by the recorded contract suite.
"""

import json
from typing import Any

import httpx2
import openai
import pytest

from harness.adapters import AdapterSpec, build_adapter
from harness.adapters.anthropic import AnthropicAdapter
from harness.adapters.openai_compat import OpenAICompatAdapter, OpenAICompatConfig
from harness.adapters.openai_compat.translate import (
    parse_arguments,
    stop_reason,
    to_messages,
)
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

TOOLS = [Tool(name="get_weather", description="weather", input_schema={"type": "object"})]
Handler = Any  # Any: test plumbing (httpx2 handler signature)


def completion(
    message: dict[str, Any], finish: str = "stop", **overrides: Any
) -> dict[str, Any]:  # Any: raw API JSON
    payload: dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "qwen2.5:7b",
        "choices": [
            {"index": 0, "message": {"role": "assistant", **message}, "finish_reason": finish}
        ],
        "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 100_000, "total_tokens": 0},
    }
    payload.update(overrides)
    return payload


def sse(chunks: list[dict[str, Any]]) -> str:  # Any: raw API JSON
    base = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 0, "model": "m"}
    body = "".join(f"data: {json.dumps({**base, **c})}\n\n" for c in chunks)
    return body + "data: [DONE]\n\n"


def delta(d: dict[str, Any], finish: str | None = None) -> dict[str, Any]:  # Any: raw API JSON
    return {"choices": [{"index": 0, "delta": d, "finish_reason": finish}]}


def adapter_for(handler: Handler, **config: Any) -> OpenAICompatAdapter:  # Any: test plumbing
    return OpenAICompatAdapter(
        OpenAICompatConfig(model="qwen2.5:7b", max_retries=0, **config),
        http_client=openai.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
    )


def json_handler(
    payload: dict[str, Any], seen: list[dict[str, Any]] | None = None
) -> Handler:  # Any: raw API JSON
    def handle(request: httpx2.Request) -> httpx2.Response:
        if seen is not None:
            seen.append(json.loads(request.read()))
        return httpx2.Response(200, json=payload)

    return handle


# ----------------------------------------------------------------------------- request translation


def test_to_messages_maps_roles_and_parts() -> None:
    history = [
        Message.user("hi"),
        Message(
            role="user",
            content=[TextPart(text="look"), ImagePart(media_type="image/png", data="AAA")],
        ),
        Message(
            role="assistant",
            content=[
                ReasoningPart(source="anthropic", data={"x": 1}),
                ToolCall(id="t1", name="get_weather", arguments={"city": "Paris"}),
                ToolCall(id="t2", name="get_weather", arguments={"_raw": "{bad"}),
            ],
        ),
        Message(
            role="tool",
            content=[
                ToolResult(tool_call_id="t1", content="down", is_error=True),
                ToolResult(tool_call_id="t2", content=""),
            ],
        ),
        Message(role="assistant", content=[]),
    ]

    out = to_messages("be terse", history)

    assert out[0] == {"role": "system", "content": "be terse"}
    assert out[1] == {"role": "user", "content": "hi"}
    assert out[2] == {
        "role": "user",
        "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
        ],
    }
    assert out[3] == {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "t1",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
            },
            # Unparseable arguments are echoed exactly as the model sent them.
            {
                "id": "t2",
                "type": "function",
                "function": {"name": "get_weather", "arguments": "{bad"},
            },
        ],
    }
    assert out[4] == {"role": "tool", "tool_call_id": "t1", "content": "Error: down"}
    assert out[5] == {"role": "tool", "tool_call_id": "t2", "content": "(no output)"}
    assert len(out) == 6  # empty assistant turn dropped


def test_parse_arguments() -> None:
    assert parse_arguments('{"a": 1}') == {"a": 1}
    assert parse_arguments("  ") == {}
    assert parse_arguments("{bad") == {"_raw": "{bad"}
    assert parse_arguments("[1]") == {"_raw": "[1]"}


def test_stop_reason() -> None:
    assert stop_reason("stop", has_tool_calls=False) == StopReason.END_TURN
    assert stop_reason("length", has_tool_calls=False) == StopReason.MAX_TOKENS
    assert stop_reason("content_filter", has_tool_calls=False) == StopReason.CONTENT_FILTER
    assert stop_reason(None, has_tool_calls=False) == StopReason.OTHER
    # Ollama quirk: "stop" with tool calls.
    assert stop_reason("stop", has_tool_calls=True) == StopReason.TOOL_USE


async def test_request_params() -> None:
    seen: list[dict[str, Any]] = []
    adapter = adapter_for(json_handler(completion({"content": "ok"}), seen))

    await adapter.complete(ModelRequest(system="s", messages=[Message.user("hi")], tools=TOOLS))
    await adapter.complete(
        ModelRequest(messages=[Message.user("hi")], tools=TOOLS, tool_choice="required")
    )

    first, second = seen
    assert first["model"] == "qwen2.5:7b"
    assert first["max_tokens"] == 8_192
    assert first["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "weather",
                "parameters": {"type": "object"},
            },
        }
    ]
    assert "tool_choice" not in first  # "auto" is the default; not every server accepts it
    assert "temperature" not in first
    assert second["tool_choice"] == "required"


# ----------------------------------------------------------------------------- complete


async def test_complete_text_usage_and_cost() -> None:
    adapter = adapter_for(
        json_handler(completion({"content": "pong"})), input_per_mtok=1.0, output_per_mtok=10.0
    )

    response = await adapter.complete(ModelRequest(messages=[Message.user("ping")]))

    assert response.message.text == "pong"
    assert response.stop_reason == StopReason.END_TURN
    assert response.usage.input_tokens == 1_000_000
    assert response.usage.output_tokens == 100_000
    assert response.usage.cost_usd == pytest.approx(2.0)
    assert response.response_id == "chatcmpl-1"


async def test_complete_local_cost_is_zero() -> None:
    response = await adapter_for(json_handler(completion({"content": "x"}))).complete(
        ModelRequest(messages=[Message.user("hi")])
    )
    assert response.usage.cost_usd == 0.0


async def test_complete_tool_calls_with_quirks() -> None:
    payload = completion(
        {
            "content": "",
            "tool_calls": [
                {
                    "id": "",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                },
                {
                    "id": "c2",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{oops"},
                },
            ],
        },
        finish="stop",
    )

    response = await adapter_for(json_handler(payload)).complete(
        ModelRequest(messages=[Message.user("hi")], tools=TOOLS)
    )

    assert response.stop_reason == StopReason.TOOL_USE
    first, second = response.message.tool_calls
    assert first == ToolCall(
        id="call_chatcmpl-1_0", name="get_weather", arguments={"city": "Paris"}
    )
    assert second == ToolCall(id="c2", name="get_weather", arguments={"_raw": "{oops"})
    assert response.message.text == ""


async def test_complete_no_choices() -> None:
    with pytest.raises(ModelError, match="no choices"):
        await adapter_for(json_handler(completion({}, choices=[]))).complete(
            ModelRequest(messages=[Message.user("hi")])
        )


# ----------------------------------------------------------------------------- stream


async def test_stream_text_and_parallel_tool_calls() -> None:
    body = sse(
        [
            delta({"role": "assistant", "content": "Check"}),
            delta({"content": "ing"}),
            delta(
                {
                    "tool_calls": [
                        {
                            "index": 1,
                            "id": "b",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": ""},
                        }
                    ]
                }
            ),
            delta(
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "a",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"ci'},
                        }
                    ]
                }
            ),
            delta({"tool_calls": [{"index": 0, "function": {"arguments": 'ty":"Paris"}'}}]}),
            delta({"tool_calls": [{"index": 1, "function": {"arguments": '{"city":"Tokyo"}'}}]}),
            delta({}, finish="tool_calls"),
            {
                "choices": [],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        ]
    )
    seen: list[dict[str, Any]] = []

    def handle(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.read()))
        return httpx2.Response(200, text=body, headers={"content-type": "text/event-stream"})

    chunks: list[str] = []

    async def on_text(text: str) -> None:
        chunks.append(text)

    response = await collect_stream(
        adapter_for(handle).stream(ModelRequest(messages=[Message.user("hi")], tools=TOOLS)),
        on_text,
    )

    assert seen[0]["stream"] is True
    assert seen[0]["stream_options"] == {"include_usage": True}
    assert chunks == ["Check", "ing"]
    assert response.message.text == "Checking"
    assert [(c.id, c.arguments["city"]) for c in response.message.tool_calls] == [
        ("a", "Paris"),
        ("b", "Tokyo"),
    ]
    assert response.stop_reason == StopReason.TOOL_USE
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 5


async def test_stream_without_usage_chunk() -> None:
    body = sse([delta({"content": "hi"}), delta({}, finish="stop")])

    def handle(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, text=body, headers={"content-type": "text/event-stream"})

    response = await collect_stream(
        adapter_for(handle).stream(ModelRequest(messages=[Message.user("hi")]))
    )

    assert response.message.text == "hi"
    assert response.stop_reason == StopReason.END_TURN
    assert response.usage.input_tokens == 0
    assert response.usage.cost_usd == 0.0


# ----------------------------------------------------------------------------- errors


@pytest.mark.parametrize(
    ("status", "retryable"), [(400, False), (404, False), (429, True), (500, True), (503, True)]
)
async def test_status_errors(status: int, retryable: bool) -> None:
    def handle(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(status, json={"error": {"message": "nope", "type": "x"}})

    with pytest.raises(ModelError) as info:
        await adapter_for(handle).complete(ModelRequest(messages=[Message.user("hi")]))

    assert info.value.status_code == status
    assert info.value.retryable is retryable
    assert info.value.provider == "openai_compat"


async def test_connection_error_is_retryable() -> None:
    def handle(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("refused", request=request)

    with pytest.raises(ModelError) as info:
        await adapter_for(handle).complete(ModelRequest(messages=[Message.user("hi")]))

    assert info.value.retryable
    assert info.value.status_code is None


async def test_stream_error_mapping() -> None:
    def handle(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(404, json={"error": {"message": "model not found"}})

    with pytest.raises(ModelError) as info:
        await collect_stream(
            adapter_for(handle).stream(ModelRequest(messages=[Message.user("hi")]))
        )

    assert info.value.status_code == 404


# ----------------------------------------------------------------------------- capabilities + registry


def test_capabilities_come_from_config() -> None:
    caps = adapter_for(
        json_handler({}), context_window=8_192, parallel_tool_calls=True
    ).capabilities
    assert caps.context_window == 8_192
    assert caps.parallel_tool_calls
    assert caps.native_tools
    assert not caps.vision
    assert caps.tokenizer == "approx"


def test_registry_builds_each_provider() -> None:
    local = build_adapter(
        AdapterSpec(provider="openai_compat", model="m", anthropic_api_key="sk-ant-secret")  # pyright: ignore[reportArgumentType]
    )
    assert isinstance(local, OpenAICompatAdapter)
    assert local.config.base_url == "http://localhost:11434/v1"
    assert local.config.api_key is None  # a Claude key is never sent to another endpoint

    claude = build_adapter(AdapterSpec(model="claude-opus-5-5", effort="low"))
    assert isinstance(claude, AnthropicAdapter)
    assert claude.config.effort == "low"
