"""A stalled model call fails fast with a retryable ``ModelError`` (issue #6), for every adapter.

The server accepts the connection, reads the request, then sends nothing: the stall seen in the
Phase 1 eval. Without an idle (read) timeout the call would hang for the SDK's 600 s default.
"""

import asyncio
import time
from collections.abc import AsyncIterator, Callable

import pytest

from harness.adapters import AdapterSpec, build_adapter
from harness.adapters.anthropic import AnthropicAdapter, AnthropicConfig
from harness.adapters.openai_compat import OpenAICompatAdapter, OpenAICompatConfig
from harness.core import Message, ModelAdapter, ModelError, ModelRequest, collect_stream

IDLE = 0.3


@pytest.fixture
async def silent_server() -> AsyncIterator[str]:
    """Base URL of a server that never answers."""

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.read(65536)
        await asyncio.sleep(30)
        writer.close()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    server.close()


def anthropic_adapter(base_url: str) -> ModelAdapter:
    return AnthropicAdapter(
        AnthropicConfig(api_key="k", base_url=base_url, max_retries=0, idle_timeout_s=IDLE)  # pyright: ignore[reportArgumentType]
    )


def openai_compat_adapter(base_url: str) -> ModelAdapter:
    return OpenAICompatAdapter(
        OpenAICompatConfig(model="m", base_url=f"{base_url}/v1", max_retries=0, idle_timeout_s=IDLE)
    )


ADAPTERS: dict[str, Callable[[str], ModelAdapter]] = {
    "anthropic": anthropic_adapter,
    "openai_compat": openai_compat_adapter,
}


@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.parametrize("provider", sorted(ADAPTERS))
async def test_stalled_call_fails_fast_and_retryable(
    silent_server: str, provider: str, streaming: bool
) -> None:
    adapter = ADAPTERS[provider](silent_server)
    request = ModelRequest(messages=[Message.user("hi")])

    async def call() -> None:
        if streaming:
            await collect_stream(adapter.stream(request))
        else:
            await adapter.complete(request)

    started = time.monotonic()
    with pytest.raises(ModelError) as info:
        await call()

    assert info.value.retryable
    assert time.monotonic() - started < IDLE + 2


@pytest.mark.parametrize("provider", ["anthropic", "openai_compat"])
def test_spec_idle_timeout_reaches_the_config(provider: str) -> None:
    spec = AdapterSpec(provider=provider, model="m", idle_timeout_s=12.5)  # pyright: ignore[reportArgumentType]
    adapter = build_adapter(spec)
    assert isinstance(adapter, AnthropicAdapter | OpenAICompatAdapter)
    assert adapter.config.idle_timeout_s == 12.5

    default = build_adapter(spec.model_copy(update={"idle_timeout_s": None}))
    assert isinstance(default, AnthropicAdapter | OpenAICompatAdapter)
    assert (
        default.config.idle_timeout_s == type(default.config).model_fields["idle_timeout_s"].default
    )
