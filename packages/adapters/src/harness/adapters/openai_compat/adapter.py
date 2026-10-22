"""``ModelAdapter`` for OpenAI-compatible Chat Completions servers (Ollama, vLLM, LM Studio)."""

from __future__ import annotations

from collections.abc import AsyncIterator

import openai
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.completion_create_params import CompletionCreateParamsBase
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from harness.adapters.openai_compat import translate
from harness.core import (
    Delta,
    ModelCapabilities,
    ModelError,
    ModelRequest,
    ModelResponse,
    StopDelta,
    TextDelta,
)

DEFAULT_BASE_URL = "http://localhost:11434/v1"
"""Ollama's OpenAI-compatible endpoint."""

_PLACEHOLDER_KEY = "not-needed"  # local servers ignore the key, but the SDK requires one


class OpenAICompatConfig(BaseModel):
    """Adapter settings. Capabilities are declared here because local models vary widely.

    Prices default to 0 (a local model costs nothing per token), so ``cost_usd`` is 0.0.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    base_url: str = DEFAULT_BASE_URL
    api_key: SecretStr | None = None
    max_output_tokens: int = Field(default=8_192, gt=0)
    timeout_s: float = Field(default=600.0, gt=0)
    max_retries: int = Field(default=2, ge=0)

    context_window: int = Field(default=32_768, gt=0)
    parallel_tool_calls: bool = False
    vision: bool = False
    input_per_mtok: float = Field(default=0.0, ge=0)
    output_per_mtok: float = Field(default=0.0, ge=0)


class OpenAICompatAdapter:
    """Any Chat Completions endpoint via the official ``openai`` SDK.

    ``complete`` makes a non-streaming call (local servers report usage reliably there).
    ``stream`` streams text live and puts the assembled message on the final ``StopDelta``.
    """

    # TODO(phase-2): real OpenAI specifics (max_completion_tokens, reasoning models, price table).

    def __init__(
        self,
        config: OpenAICompatConfig,
        *,
        http_client: openai.DefaultAsyncHttpxClient | None = None,
    ) -> None:
        """``http_client`` is for tests (cassette replay); production uses the SDK default."""
        self.config = config
        self._client = openai.AsyncOpenAI(
            api_key=config.api_key.get_secret_value() if config.api_key else _PLACEHOLDER_KEY,
            base_url=config.base_url,
            timeout=config.timeout_s,
            max_retries=config.max_retries,
            http_client=http_client,
        )
        self._capabilities = ModelCapabilities(
            context_window=config.context_window,
            max_output_tokens=config.max_output_tokens,
            native_tools=True,
            parallel_tool_calls=config.parallel_tool_calls,
            vision=config.vision,
            streaming=True,
            prompt_caching=False,
            tokenizer="approx",
        )

    @property
    def capabilities(self) -> ModelCapabilities:
        """Capabilities declared in the config."""
        return self._capabilities

    def _params(self, request: ModelRequest) -> CompletionCreateParamsBase:
        """Request params; optional ones are left out rather than sent as null."""
        params: CompletionCreateParamsBase = {
            "model": self.config.model,
            "messages": translate.to_messages(request.system, request.messages),
            "max_tokens": request.max_output_tokens or self.config.max_output_tokens,
        }
        if request.tools:
            params["tools"] = translate.to_tools(request.tools)
            if request.tool_choice != "auto":  # not every server accepts tool_choice
                params["tool_choice"] = translate.to_tool_choice(request.tool_choice)
        if request.stop_sequences:
            params["stop"] = request.stop_sequences
        if request.temperature is not None:
            params["temperature"] = request.temperature
        return params

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """One non-streaming model call."""
        try:
            completion = await self._client.chat.completions.create(
                **self._params(request),
                stream=False,
            )
        except openai.APIError as e:
            raise translate.to_model_error(e) from e
        return translate.from_completion(
            completion,
            input_per_mtok=self.config.input_per_mtok,
            output_per_mtok=self.config.output_per_mtok,
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[Delta]:
        """Stream text live; the final ``StopDelta`` carries the assembled message."""
        text: list[str] = []
        calls: dict[int, translate.PendingCall] = {}
        finish: str | None = None
        last: ChatCompletionChunk | None = None
        params = self._params(request)
        params["stream_options"] = {"include_usage": True}
        try:
            chunks = await self._client.chat.completions.create(**params, stream=True)
            async for chunk in chunks:
                last = chunk
                for choice in chunk.choices[:1]:
                    delta = choice.delta
                    if delta.content:
                        text.append(delta.content)
                        yield TextDelta(text=delta.content)
                    for tc in delta.tool_calls or []:
                        call = calls.setdefault(tc.index, translate.PendingCall())
                        call.id = call.id or tc.id
                        if tc.function is not None:
                            call.name += tc.function.name or ""
                            call.fragments.append(tc.function.arguments or "")
                    finish = choice.finish_reason or finish
        except openai.APIError as e:
            raise translate.to_model_error(e) from e

        if last is None:
            raise ModelError(
                "model stream ended without any chunk", retryable=True, provider=translate.SOURCE
            )
        message = translate.build_message("".join(text), calls.items(), last.id)
        yield StopDelta(
            stop_reason=translate.stop_reason(finish, has_tool_calls=bool(calls)),
            usage=translate.to_usage(
                last.usage,
                input_per_mtok=self.config.input_per_mtok,
                output_per_mtok=self.config.output_per_mtok,
            ),
            model=last.model,
            response_id=last.id,
            message=message,
        )
