"""Builds a ``ModelAdapter`` from a provider-neutral spec.

The only place a provider name is looked at, so apps choose a provider by config alone.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from harness.adapters.anthropic import AnthropicAdapter, AnthropicConfig, Effort
from harness.adapters.openai_compat import (
    DEFAULT_BASE_URL,
    OpenAICompatAdapter,
    OpenAICompatConfig,
)
from harness.adapters.prompt_tools import PromptToolsAdapter
from harness.core import ModelAdapter

Provider = Literal["anthropic", "openai_compat"]


class AdapterSpec(BaseModel):
    """What an app knows about the model it wants. Fields a provider doesn't use are ignored."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: Provider = "anthropic"
    model: str
    base_url: str | None = None
    api_key: SecretStr | None = Field(default=None, description="Key for any provider; wins.")
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        description="Used for Anthropic only, so a Claude key is never sent to another endpoint.",
    )
    max_output_tokens: int = Field(default=64_000, gt=0)
    effort: Effort | None = Field(default=None, description="Anthropic only.")
    idle_timeout_s: float | None = Field(
        default=None, gt=0, description="Max silence on a model stream; None = adapter default."
    )
    native_tools: bool = Field(
        default=True,
        description="False forces prompt-based tool calling (ADR 0006), for models without native tools.",
    )


def build_adapter(spec: AdapterSpec) -> ModelAdapter:
    """Build the adapter for ``spec.provider``, wrapped in the prompt fallback if it has no native tools."""
    adapter = _build_provider(spec)
    if not spec.native_tools or not adapter.capabilities.native_tools:
        return PromptToolsAdapter(adapter)
    return adapter


def _build_provider(spec: AdapterSpec) -> ModelAdapter:
    idle = {"idle_timeout_s": spec.idle_timeout_s} if spec.idle_timeout_s is not None else {}
    match spec.provider:
        case "anthropic":
            config = AnthropicConfig(
                model=spec.model,
                api_key=spec.api_key or spec.anthropic_api_key,
                base_url=spec.base_url,
                effort=spec.effort,
                max_output_tokens=spec.max_output_tokens,
            )
            return AnthropicAdapter(config.model_copy(update=idle))
        case "openai_compat":
            config = OpenAICompatConfig(
                model=spec.model,
                base_url=spec.base_url or DEFAULT_BASE_URL,
                api_key=spec.api_key,
                max_output_tokens=spec.max_output_tokens,
                native_tools=spec.native_tools,
            )
            return OpenAICompatAdapter(config.model_copy(update=idle))
