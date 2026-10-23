# harness-adapters

Model adapters: translate canonical types to/from each provider. The **only** place
provider-specific code may live. May import `harness.core` only (CLAUDE.md §3).

| Adapter | Module | Docs |
|---|---|---|
| Anthropic (Claude) | `harness.adapters.anthropic` | [docs/providers/anthropic.md](../../docs/providers/anthropic.md) |
| OpenAI-compatible (Ollama, vLLM, LM Studio) | `harness.adapters.openai_compat` | [docs/providers/openai_compat.md](../../docs/providers/openai_compat.md) |
| Prompt-based tool calling (wraps any adapter) | `harness.adapters.prompt_tools` | [ADR 0006](../../docs/adr/0006-prompt-based-tool-calling-fallback.md) |

Apps pick one by config through `harness.adapters.build_adapter(AdapterSpec(provider=...))`, the
only code that looks at a provider name. `build_adapter` wraps the result in `PromptToolsAdapter` when
`AdapterSpec.native_tools` is false or the model reports no native tools.

Shared test infrastructure: `harness.adapters.testing.CassetteTransport` (HTTP record/replay, see
[ADR 0004](../../docs/adr/0004-reasoning-parts-and-cassettes.md)).

Contract suite: `tests/contract/`. Every adapter registers a factory in `FACTORIES` and must pass
text reply, single and parallel tool calls, tool-error round trip, streaming, usage reporting
and error mapping. Record cassettes with `just record TEST=packages/adapters/tests/contract`.

`openai_compat/` was brought forward from Phase 2 ([ADR 0005](../../docs/adr/0005-openai-compat-adapter-in-phase-1.md)). Coming in Phase 2: `gemini/`.
