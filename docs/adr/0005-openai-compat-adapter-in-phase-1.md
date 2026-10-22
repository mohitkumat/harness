# 0005 — OpenAI-compatible adapter brought forward into Phase 1

- Status: accepted
- Date: 2026-10-21
- Phase: 1 (item taken from Phase 2)

## Context

Day-to-day development and eval runs against Claude cost money on every iteration. Local model
servers (Ollama, vLLM, LM Studio) are free to run and all expose the OpenAI Chat Completions API.
The OpenAI-compatible adapter is planned for Phase 2 (CLAUDE.md §6), and §5 says later-phase work
waits. The owner decided to make one exception so the harness can be developed against a local
model now.

## Decision

1. **Only `adapters/openai_compat/` moves forward.** Gemini, the prompt-based tool-calling
   fallback, the eval matrix and the comparison report stay in Phase 2. The Phase 1 exit check
   (~50% eval pass rate) is still measured with Claude, because small local models measure the
   model as much as the harness.
2. **Use the official `openai` SDK.** It is built on `httpx2`, like `anthropic`, so
   `CassetteTransport` ([ADR 0004](0004-reasoning-parts-and-cassettes.md)) records and replays it
   unchanged. It also brings SSE parsing, retries and typed errors. The import-linter rule already
   kept `openai` inside `adapters/`.
3. **Provider selection is config.** `harness.adapters.build_adapter(AdapterSpec)` is the only code
   that looks at a provider name. Apps set `HARNESS_PROVIDER` and never branch on it. A Claude key
   (`anthropic_api_key`) is only ever passed to the Anthropic adapter.
4. **Capabilities come from config, not a model table.** Local models vary too much to list.
   Defaults are conservative: 32k context, no parallel tool calls, no vision, `tokenizer="approx"`.
5. **Cost is 0 for local models.** Prices default to 0 per Mtok, so `cost_usd=0.0` (known and
   free, not unknown). The shared contract test now asserts `cost_usd >= 0` instead of `> 0`.
6. **Unparseable tool arguments become `{"_raw": "<string>"}`** instead of raising. The tool
   registry's schema validation rejects the call, and the model gets the error back as a tool
   result (CLAUDE.md §8: tool errors are results). On replay the adapter sends the original string
   back as it was.

## Consequences

- `HARNESS_PROVIDER=openai_compat` runs the CLI and evals against any OpenAI-compatible endpoint.
- Real OpenAI (hosted) works only roughly: no price table, `max_tokens` instead of
  `max_completion_tokens`, no reasoning items. These are `TODO(phase-2)` in the adapter.
- `_raw` is a reserved argument name. A tool must not declare a parameter called `_raw`.
