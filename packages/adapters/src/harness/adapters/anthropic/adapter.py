"""``ModelAdapter`` for the Anthropic Messages API (Claude models)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal

import anthropic
from anthropic import omit
from anthropic.types.beta import BetaMessage
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from harness.adapters.anthropic import models, translate
from harness.core import (
    Delta,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    StopDelta,
    TextDelta,
)

Effort = Literal["low", "medium", "high", "xhigh", "max"]

_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicConfig(BaseModel):
    """Adapter settings. ``api_key=None`` lets the SDK resolve credentials (env, ``ant`` profile)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = models.DEFAULT_MODEL
    api_key: SecretStr | None = None
    base_url: str | None = None
    max_output_tokens: int = Field(default=64_000, gt=0)
    effort: Effort | None = Field(
        default="high",
        description="Thinking/effort level. Claude Opus 5.5 defaults to 'medium' if unset.",
    )
    refusal_fallback: bool = Field(
        default=True,
        description="Opt into server-side fallbacks ('default' routing) when a classifier declines.",
    )
    timeout_s: float = Field(default=600.0, gt=0)
    max_retries: int = Field(default=2, ge=0)


class AnthropicAdapter:
    """Claude via the official SDK. Always streams, so large ``max_tokens`` never times out.

    Thinking blocks come back as ``ReasoningPart`` (source ``"anthropic"``) and are replayed
    unchanged, as preserved thinking requires. Automatic prompt caching is on for every request.
    """

    def __init__(
        self,
        config: AnthropicConfig | None = None,
        *,
        http_client: anthropic.DefaultAsyncHttpxClient | None = None,
    ) -> None:
        """``http_client`` is for tests (cassette replay); production uses the SDK default."""
        self.config = config or AnthropicConfig()
        self._client = anthropic.AsyncAnthropic(
            api_key=self.config.api_key.get_secret_value() if self.config.api_key else None,
            base_url=self.config.base_url,
            timeout=self.config.timeout_s,
            max_retries=self.config.max_retries,
            http_client=http_client,
        )
        self._capabilities = models.capabilities(self.config.model)
        self._info = models.model_info(self.config.model)

    @property
    def capabilities(self) -> ModelCapabilities:
        """Capabilities of the configured model."""
        return self._capabilities

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """One model call, streamed under the hood and returned whole."""
        final: BetaMessage | None = None
        async for item in self._iterate(request):
            if not isinstance(item, str):
                final = item
        assert final is not None  # _iterate always ends with the final message
        return translate.from_message(final)

    async def stream(self, request: ModelRequest) -> AsyncIterator[Delta]:
        """Stream text live; the final ``StopDelta`` carries the authoritative message."""
        async for item in self._iterate(request):
            if isinstance(item, str):
                yield TextDelta(text=item)
                continue
            response = translate.from_message(item)
            yield StopDelta(
                stop_reason=response.stop_reason,
                usage=response.usage,
                model=response.model,
                response_id=response.response_id,
                message=response.message,
            )

    async def _iterate(self, request: ModelRequest) -> AsyncIterator[str | BetaMessage]:
        """Yield text chunks as they arrive, then the SDK's final message."""
        cfg, info = self.config, self._info
        use_fallback = cfg.refusal_fallback and info is not None and info.supports_server_fallback
        try:
            async with self._client.beta.messages.stream(
                model=cfg.model,
                max_tokens=request.max_output_tokens or cfg.max_output_tokens,
                system=request.system if request.system else omit,
                messages=translate.to_messages(request.messages),
                tools=translate.to_tools(request.tools) if request.tools else omit,
                tool_choice=translate.to_tool_choice(request.tool_choice)
                if request.tools
                else omit,
                stop_sequences=request.stop_sequences or omit,
                cache_control={"type": "ephemeral"},
                output_config=(
                    {"effort": cfg.effort}
                    if cfg.effort is not None and info is not None and info.supports_effort
                    else omit
                ),
                fallbacks="default" if use_fallback else omit,
                betas=[_FALLBACK_BETA] if use_fallback else omit,
                # Sampling params are rejected by current Claude models; sent only if asked for.
                extra_body=(
                    {"temperature": request.temperature}
                    if request.temperature is not None
                    else None
                ),
            ) as stream:
                async for event in stream:
                    if event.type == "content_block_delta" and event.delta.type == "text_delta":
                        yield event.delta.text
                yield await stream.get_final_message()
        except anthropic.APIError as e:
            raise translate.to_model_error(e) from e
