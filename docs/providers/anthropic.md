# Anthropic (Claude)

Adapter: `harness.adapters.anthropic.AnthropicAdapter` (official `anthropic` Python SDK, 1.x).

## Setup

Credentials come from the SDK's usual chain: `ANTHROPIC_API_KEY`, then `ANTHROPIC_AUTH_TOKEN`, then
an `ant auth login` profile. For local work, put `ANTHROPIC_API_KEY=...` in `.env` (git-ignored).
The CLI and `just record` read it from there. Never commit keys.

## Configuration (`AnthropicConfig`)

| Field | Default | Notes |
|---|---|---|
| `model` | `claude-opus-5-5` | See `harness/adapters/anthropic/models.py` for known ids, prices and windows |
| `max_output_tokens` | 64000 | Requests always stream, so large values don't hit HTTP timeouts |
| `effort` | `high` | `output_config.effort`. Claude Opus 5.5's own default is `medium`. Not sent to Haiku 4.5 |
| `refusal_fallback` | `true` | `fallbacks: "default"` (beta `server-side-fallback-2026-07-01`): a classifier decline is re-run server-side on Anthropic's recommended model |
| `max_retries` / `timeout_s` | 2 / 600 | SDK retries 408/409/429/5xx and connection errors |

## Behaviour

- **Prompt caching:** automatic (`cache_control: {type: "ephemeral"}` at top level). The request
  prefix stays stable because translation is deterministic and history is append-only.
- **Thinking:** Claude Opus 5.5 always thinks. Thinking blocks come back as
  `ReasoningPart(source="anthropic")` and are replayed verbatim, as preserved thinking requires
  ([ADR 0004](../adr/0004-reasoning-parts-and-cassettes.md)). Their text is empty by default
  (`display: "omitted"`).
- **Refusals:** `stop_reason: "refusal"` maps to `StopReason.CONTENT_FILTER`. After a mid-output
  fallback, thinking/tool-use blocks before the `fallback` marker are not echoed back.
- **Tool choice:** `required` maps to `any`, which Claude Opus 5.5 / Sonnet 5.5 / Fable 5.1 reject
  with a 400. The loop only uses `auto`.
- **Sampling:** `temperature` is passed only when set. Current models reject it.
- **Errors:** SDK errors become `ModelError(provider="anthropic", status_code=…)`. Timeouts,
  connection errors, 408/409/429 and 5xx are `retryable`.
- **Cost:** computed from list prices, summed per usage iteration so a fallback bills each model.
  Unknown model ids report `cost_usd=None`.

## Tests

- Unit: `packages/adapters/tests/anthropic/` (hand-written payloads via `httpx2.MockTransport`).
- Contract: `packages/adapters/tests/contract/` replays `tests/cassettes/anthropic/*.json`.
  Re-record with `just record TEST=packages/adapters/tests/contract` (real API calls, small cost).
