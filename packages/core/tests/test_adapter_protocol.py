from collections.abc import AsyncIterator

from harness.core import (
    Delta,
    Message,
    ModelAdapter,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    StopDelta,
    StopReason,
    TextDelta,
    Usage,
)

CAPS = ModelCapabilities(
    context_window=8000,
    max_output_tokens=1000,
    native_tools=False,
    parallel_tool_calls=False,
    vision=False,
    tokenizer="approx",
)


class EchoAdapter:
    """Minimal in-memory adapter used to check the Protocol is implementable."""

    @property
    def capabilities(self) -> ModelCapabilities:
        return CAPS

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            message=Message.assistant(request.messages[-1].text),
            stop_reason=StopReason.END_TURN,
            usage=Usage(input_tokens=1, output_tokens=1),
            model="echo",
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[Delta]:
        yield TextDelta(text=request.messages[-1].text)
        yield StopDelta(stop_reason=StopReason.END_TURN, usage=Usage(), model="echo")


async def test_echo_adapter_satisfies_protocol() -> None:
    adapter: ModelAdapter = EchoAdapter()
    assert isinstance(adapter, ModelAdapter)
    req = ModelRequest(messages=[Message.user("ping")])
    assert (await adapter.complete(req)).message.text == "ping"
    deltas = [d async for d in adapter.stream(req)]
    assert isinstance(deltas[-1], StopDelta)
