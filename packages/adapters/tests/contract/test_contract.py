"""The adapter contract (CLAUDE.md §9): every adapter must pass these, in canonical types only."""

from collections.abc import Callable

import pytest
from pydantic import JsonValue

from harness.core import (
    Message,
    ModelAdapter,
    ModelError,
    ModelRequest,
    StopReason,
    Tool,
    ToolResult,
    collect_stream,
)

Make = Callable[..., ModelAdapter]
INVALID_MODEL = "harness-contract-no-such-model"

WEATHER: dict[str, JsonValue] = {
    "type": "object",
    "properties": {"city": {"type": "string", "description": "City name, e.g. Paris"}},
    "required": ["city"],
    "additionalProperties": False,
}
TOOLS = [
    Tool(
        name="get_weather",
        description="Get the current weather for one city.",
        input_schema=WEATHER,
    )
]
SYSTEM = "You are a terse test assistant. Follow instructions exactly."


def request(*messages: Message, tools: list[Tool] | None = None) -> ModelRequest:
    return ModelRequest(system=SYSTEM, messages=list(messages), tools=tools or [])


async def test_text_reply(make: Make) -> None:
    response = await make().complete(request(Message.user("Reply with exactly the word: pong")))

    assert "pong" in response.message.text.lower()
    assert response.stop_reason == StopReason.END_TURN
    assert response.message.tool_calls == []


async def test_single_tool_call(make: Make) -> None:
    response = await make().complete(
        request(Message.user("What's the weather in Paris? Use the tool."), tools=TOOLS)
    )

    assert response.stop_reason == StopReason.TOOL_USE
    (call,) = response.message.tool_calls
    assert call.name == "get_weather"
    assert "paris" in str(call.arguments["city"]).lower()
    assert call.id


async def test_parallel_tool_calls(make: Make) -> None:
    adapter = make()
    if not adapter.capabilities.parallel_tool_calls:
        pytest.skip("model does not support parallel tool calls")
    response = await adapter.complete(
        request(
            Message.user(
                "Get the weather for Paris and for Tokyo. Call the tool for both cities at once, "
                "in a single response."
            ),
            tools=TOOLS,
        )
    )

    calls = response.message.tool_calls
    assert len(calls) >= 2
    assert len({c.id for c in calls}) == len(calls)
    cities = " ".join(str(c.arguments["city"]).lower() for c in calls)
    assert "paris" in cities
    assert "tokyo" in cities


async def test_tool_error_round_trip(make: Make) -> None:
    adapter = make()
    user = Message.user(
        "What's the weather in Paris? Use the tool. If it fails, say 'unavailable'."
    )
    first = await adapter.complete(request(user, tools=TOOLS))
    (call,) = first.message.tool_calls
    error = ToolResult(tool_call_id=call.id, content="weather service is down", is_error=True)

    second = await adapter.complete(
        request(user, first.message, Message(role="tool", content=[error]), tools=TOOLS)
    )

    assert second.stop_reason == StopReason.END_TURN
    assert "unavailable" in second.message.text.lower()


async def test_streaming(make: Make) -> None:
    chunks: list[str] = []

    async def on_text(text: str) -> None:
        chunks.append(text)

    response = await collect_stream(
        make().stream(request(Message.user("Count from 1 to 5, separated by spaces."))), on_text
    )

    assert chunks
    assert "".join(chunks).strip() == response.message.text.strip()
    assert "1 2 3 4 5" in response.message.text


async def test_usage_reporting(make: Make) -> None:
    response = await make().complete(request(Message.user("Say hi.")))

    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0
    assert response.usage.cost_usd is not None
    assert response.usage.cost_usd > 0
    assert response.model


async def test_error_mapping(make: Make) -> None:
    with pytest.raises(ModelError) as info:
        await make(INVALID_MODEL).complete(request(Message.user("hi")))

    assert info.value.status_code == 404
    assert not info.value.retryable
