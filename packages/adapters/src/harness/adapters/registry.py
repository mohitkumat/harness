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


def build_adapter(spec: AdapterSpec) -> ModelAdapter:
    """Build the adapter for ``spec.provider``."""
    match spec.provider:
        case "anthropic":
            return AnthropicAdapter(
                AnthropicConfig(
                    model=spec.model,
                    api_key=spec.api_key or spec.anthropic_api_key,
                    base_url=spec.base_url,
                    effort=spec.effort,
                    max_output_tokens=spec.max_output_tokens,
                )
            )
        case "openai_compat":
            return OpenAICompatAdapter(
                OpenAICompatConfig(
                    model=spec.model,
                    base_url=spec.base_url or DEFAULT_BASE_URL,
                    api_key=spec.api_key,
                    max_output_tokens=spec.max_output_tokens,
                )
            )
