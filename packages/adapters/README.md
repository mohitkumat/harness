# harness-adapters

Model adapters: translate canonical types to/from each provider. The **only** place
provider-specific code may live. May import `harness.core` only (CLAUDE.md §3).

| Adapter | Module | Docs |
|---|---|---|
| Anthropic (Claude) | `harness.adapters.anthropic` | [docs/providers/anthropic.md](../../docs/providers/anthropic.md) |

Shared test infrastructure: `harness.adapters.testing.CassetteTransport` (HTTP record/replay, see
[ADR 0004](../../docs/adr/0004-reasoning-parts-and-cassettes.md)).

Contract suite: `tests/contract/`. Every adapter registers a factory in `FACTORIES` and must pass
text reply, single and parallel tool calls, tool-error round trip, streaming, usage reporting
and error mapping. Record cassettes with `just record TEST=packages/adapters/tests/contract`.

Coming in Phase 2: `openai_compat/` (OpenAI, vLLM, Ollama), `gemini/`.
