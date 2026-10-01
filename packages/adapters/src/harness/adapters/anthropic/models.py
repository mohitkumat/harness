"""Static facts about Claude models: capabilities and list prices.

Source: Anthropic model table, cached 2026-09-25. Unknown model ids get conservative defaults and
no cost (``cost_usd=None``), so cost limits are not enforced for them.
"""

from __future__ import annotations

from dataclasses import dataclass

from harness.core import ModelCapabilities


@dataclass(frozen=True)
class ModelInfo:
    """Capabilities, per-million-token prices (USD) and request features for one model."""

    context_window: int
    max_output_tokens: int
    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float
    supports_effort: bool = True
    supports_server_fallback: bool = False

    @property
    def cache_write_per_mtok(self) -> float:
        """5-minute-TTL cache writes bill at 1.25x the input price."""
        return self.input_per_mtok * 1.25


MODELS: dict[str, ModelInfo] = {
    "claude-fable-5-1": ModelInfo(
        1_000_000, 128_000, 10.0, 50.0, 0.25, supports_server_fallback=True
    ),
    "claude-opus-5-5": ModelInfo(
        1_000_000, 128_000, 4.0, 20.0, 0.20, supports_server_fallback=True
    ),
    "claude-opus-5": ModelInfo(1_000_000, 128_000, 5.0, 25.0, 0.50, supports_server_fallback=True),
    "claude-opus-4-8": ModelInfo(1_000_000, 128_000, 5.0, 25.0, 0.50),
    "claude-sonnet-5-5": ModelInfo(
        1_000_000, 128_000, 2.0, 10.0, 0.20, supports_server_fallback=True
    ),
    "claude-sonnet-5": ModelInfo(1_000_000, 128_000, 2.0, 10.0, 0.20),
    "claude-haiku-4-5": ModelInfo(200_000, 64_000, 1.0, 5.0, 0.10, supports_effort=False),
}

DEFAULT_MODEL = "claude-opus-5-5"

_UNKNOWN = ModelInfo(200_000, 32_000, 0.0, 0.0, 0.0, supports_effort=False)


def model_info(model: str) -> ModelInfo | None:
    """Known facts about ``model``, or ``None`` if it is not in the table."""
    return MODELS.get(model)


def capabilities(model: str) -> ModelCapabilities:
    """Canonical capabilities for ``model`` (conservative defaults for unknown ids)."""
    info = MODELS.get(model, _UNKNOWN)
    return ModelCapabilities(
        context_window=info.context_window,
        max_output_tokens=info.max_output_tokens,
        native_tools=True,
        parallel_tool_calls=True,
        vision=True,
        streaming=True,
        prompt_caching=True,
        tokenizer="anthropic",
    )


def cost_usd(
    model: str, *, input_tokens: int, output_tokens: int, cache_read: int, cache_write: int
) -> float | None:
    """List-price cost of one model call, or ``None`` for unknown models."""
    info = MODELS.get(model)
    if info is None:
        return None
    return (
        input_tokens * info.input_per_mtok
        + output_tokens * info.output_per_mtok
        + cache_read * info.cache_read_per_mtok
        + cache_write * info.cache_write_per_mtok
    ) / 1_000_000
