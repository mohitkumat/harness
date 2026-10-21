"""OpenAI-compatible Chat Completions adapter (Ollama, vLLM, LM Studio)."""

from harness.adapters.openai_compat.adapter import (
    DEFAULT_BASE_URL,
    OpenAICompatAdapter,
    OpenAICompatConfig,
)

__all__ = ["DEFAULT_BASE_URL", "OpenAICompatAdapter", "OpenAICompatConfig"]
