"""Shared adapter contract suite wiring.

Every adapter registers a factory in ``FACTORIES``; every contract test runs once per adapter
against its recorded cassette ``tests/cassettes/<adapter>/<test>.json``.

- Replay (default, CI): no network. A missing cassette skips the test with a hint.
- Record: ``just record TEST=packages/adapters/tests/contract`` (sets ``HARNESS_RECORD=1``, needs
  real credentials, e.g. ``ANTHROPIC_API_KEY`` in ``.env``). ``openai_compat`` records against a
  local server: ``HARNESS_CONTRACT_OPENAI_BASE_URL`` (default Ollama) and
  ``HARNESS_CONTRACT_OPENAI_MODEL`` (default ``qwen2.5:7b``). ``prompt_tools`` is the prompt-based
fallback (ADR 0006) over that same local model.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import cast

import anthropic
import openai
import pytest

from harness.adapters.anthropic import AnthropicAdapter, AnthropicConfig
from harness.adapters.openai_compat import DEFAULT_BASE_URL, OpenAICompatAdapter, OpenAICompatConfig
from harness.adapters.prompt_tools import PromptToolsAdapter
from harness.adapters.testing import CassetteTransport
from harness.core import ModelAdapter

CASSETTES = Path(__file__).resolve().parents[1] / "cassettes"

AdapterFactory = Callable[[CassetteTransport, str | None], ModelAdapter]
"""``(transport, model_override) -> adapter``. Retries are off so recordings stay deterministic."""


def _anthropic(transport: CassetteTransport, model: str | None) -> ModelAdapter:
    config = AnthropicConfig(max_retries=0, max_output_tokens=4_000, effort="low")
    if model is not None:
        config = config.model_copy(update={"model": model})
    return AnthropicAdapter(
        config, http_client=anthropic.DefaultAsyncHttpxClient(transport=transport)
    )


def _openai_compat(transport: CassetteTransport, model: str | None) -> ModelAdapter:
    config = OpenAICompatConfig(
        model=model or os.environ.get("HARNESS_CONTRACT_OPENAI_MODEL", "qwen2.5:7b"),
        base_url=os.environ.get("HARNESS_CONTRACT_OPENAI_BASE_URL", DEFAULT_BASE_URL),
        max_retries=0,
        max_output_tokens=1_000,
    )
    return OpenAICompatAdapter(
        config, http_client=openai.DefaultAsyncHttpxClient(transport=transport)
    )


def _prompt_tools(transport: CassetteTransport, model: str | None) -> ModelAdapter:
    """The prompt-based fallback over the local model, with its native tools switched off."""
    inner = _openai_compat(transport, model)
    assert isinstance(inner, OpenAICompatAdapter)
    config = inner.config.model_copy(update={"native_tools": False})
    return PromptToolsAdapter(
        OpenAICompatAdapter(config, http_client=openai.DefaultAsyncHttpxClient(transport=transport))
    )


FACTORIES: dict[str, AdapterFactory] = {
    "anthropic": _anthropic,
    "openai_compat": _openai_compat,
    "prompt_tools": _prompt_tools,
}

# TODO(phase-2): register the gemini factory.


@pytest.fixture(params=sorted(FACTORIES))
def provider(request: pytest.FixtureRequest) -> str:
    return str(request.param)


@pytest.fixture
async def transport(
    provider: str, request: pytest.FixtureRequest
) -> AsyncIterator[CassetteTransport]:
    node = cast(pytest.Item, request.node)  # pyright: ignore[reportUnknownMemberType]
    name = node.name.split("[")[0]  # test function name without the provider param
    path = CASSETTES / provider / f"{name}.json"
    recording = os.environ.get("HARNESS_RECORD") == "1"
    if not recording and not path.exists():
        pytest.skip(
            f"no cassette {path.relative_to(CASSETTES.parent)}; record it with `just record`"
        )
    t = CassetteTransport(path, "record" if recording else "replay")
    yield t
    t.save()
    await t.aclose()


@pytest.fixture
def make(provider: str, transport: CassetteTransport) -> Callable[[str | None], ModelAdapter]:
    """Build the adapter under test, optionally with a different model id."""
    return lambda model=None: FACTORIES[provider](transport, model)
