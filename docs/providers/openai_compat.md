# OpenAI-compatible servers (Ollama, vLLM, LM Studio)

Adapter: `harness.adapters.openai_compat.OpenAICompatAdapter` (official `openai` Python SDK, 3.x,
Chat Completions API). Brought forward from Phase 2 for local development
([ADR 0005](../adr/0005-openai-compat-adapter-in-phase-1.md)).

## Setup (Ollama on macOS)

```bash
brew install ollama
ollama pull qwen2.5:7b          # ~4.7 GB; fits a 16 GB machine
OLLAMA_CONTEXT_LENGTH=32768 ollama serve
```

**Set the context length.** Ollama's default context is small, and it silently truncates the
prompt when the prompt is longer. The system prompt and tool definitions are cut first, and the
agent then behaves as if it had no tools. 32k matches the adapter's declared `context_window`.

Then point the harness at it (env or `.env`):

```bash
HARNESS_PROVIDER=openai_compat
HARNESS_MODEL=qwen2.5:7b
HARNESS_MAX_OUTPUT_TOKENS=8192
# HARNESS_BASE_URL=http://localhost:11434/v1   # the default; vLLM is usually :8000/v1, LM Studio :1234/v1
# HARNESS_API_KEY=...                          # only if the server needs one
```

```bash
just dev-cli run "list the files here"
uv run harness-eval --agent harness
```

Models with reliable native tool calling in Ollama at 7–8B: `qwen2.5:7b`, `llama3.1:8b`. Expect a
much lower eval pass rate than Claude.

**Avoid `qwen2.5-coder`.** Under Ollama it writes tool calls as JSON in the reply text
(`finish_reason: stop`) instead of native `tool_calls`. The harness then sees a plain answer. Text
tool-call parsing is the Phase 2 prompt-based fallback.

## Configuration (`OpenAICompatConfig`)

| Field | Default | Notes |
|---|---|---|
| `model` | (required) | The server's model name, e.g. `qwen2.5:7b` |
| `base_url` | `http://localhost:11434/v1` | Any Chat Completions endpoint |
| `api_key` | none | A placeholder is sent when unset; local servers ignore it |
| `max_output_tokens` | 8192 | Sent as `max_tokens` |
| `context_window` | 32768 | Declared capability. Keep it in line with the server's context length |
| `parallel_tool_calls` / `vision` | `false` / `false` | Declared capabilities |
| `input_per_mtok` / `output_per_mtok` | 0 / 0 | USD per million tokens; 0 means `cost_usd=0.0` |
| `max_retries` / `timeout_s` | 2 / 600 | SDK retries 408/409/429/5xx and connection errors |

The CLI sets `provider`, `model`, `base_url`, `api_key` and `max_output_tokens`. The other fields
keep their defaults until per-model config arrives in Phase 2.

## Behaviour

- **`complete`** makes a non-streaming call, because local servers report usage reliably there.
  **`stream`** streams text live with `stream_options.include_usage`. The final `StopDelta`
  carries the assembled message.
- **Tool results:** Chat Completions has no error flag, so error results are sent as
  `"Error: <content>"`.
- **Quirks handled:** a reply with tool calls is always `TOOL_USE`, even if the server says
  `stop`. Missing tool-call ids get deterministic ids (`call_<response id>_<index>`). Arguments
  that aren't a JSON object become `{"_raw": "<string>"}`, so the registry returns a schema error
  to the model.
- **Reasoning:** `ReasoningPart`s are never sent. Reasoning that a model writes into `content`
  (e.g. `<think>` tags) is kept as text.
- **Errors:** SDK errors become `ModelError(provider="openai_compat", status_code=…)`. Timeouts,
  connection errors, 408/409/429 and 5xx are `retryable`.

## Tests

- Unit: `packages/adapters/tests/openai_compat/` (hand-written payloads via `httpx2.MockTransport`).
- Contract: `packages/adapters/tests/contract/` replays `tests/cassettes/openai_compat/*.json`.
  Re-record against a running server with `just record TEST=packages/adapters/tests/contract`.
  Override the target with `HARNESS_CONTRACT_OPENAI_BASE_URL` and `HARNESS_CONTRACT_OPENAI_MODEL`.
