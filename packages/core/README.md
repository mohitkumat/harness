# harness-core

Provider-independent foundation. Imports nothing else from this repo.

| Module | Contents |
|---|---|
| `harness.core.types` | `Message` + content parts (`TextPart`, `ImagePart`, `ToolCall`, `ToolResult`), `Tool`, `ModelRequest`, `ModelResponse`, `Usage`, `StopReason`, `ModelCapabilities` |
| `harness.core.adapter` | `ModelAdapter` protocol and streaming `Delta`s |
| `harness.core.errors` | `HarnessError`, `ModelError`, `ToolError`, `LimitExceeded`, `PermissionDenied`, each with a stable `code` |
| `harness.core.protocol` | Pydantic models **generated** from `protocol/` plus `parse_command` / `parse_event` |
| `harness.core.events` | `EventBus`: stamps `seq`/`ts` on protocol events, fans them out to sinks, replays for `resume` |
| `harness.core.hooks` | Protocols the loop plugs into: `ToolExecutor`, `ApprovalPolicy`, `ContextBuilder` (+ `PassthroughContext`) |
| `harness.core.loop` | `AgentLoop` and `Limits` (steps, tokens, cost, wall-clock) — cancel the task to cancel the run |
| `harness.core.approvals` | `ApprovalGate`: pending approvals the loop awaits and `approve` commands resolve |
| `harness.core.session` | `Harness` / `Session`: applies protocol commands (`create_session`, `send`, `approve`, `cancel`, `resume`) |
| `harness.core.streaming` | `collect_stream`: canonical `Delta` stream → `ModelResponse` |
| `harness.core.testing` | Test doubles: `ScriptedAdapter`, `EchoTools`, `RiskyAsks` |

Conventions: canonical types are frozen Pydantic models with `extra="forbid"`. The system prompt
lives on `ModelRequest.system`, not in `messages`. Tool results go in `role="tool"` messages.

Loop invariants: every tool call in history is followed by a `role="tool"` message with its
results, however the run ends (completed, limit, cancel, timeout, error). Each run ends with exactly
one `Done` event followed by `StatusUpdate(idle)`. Refused tool calls (policy or user) emit only
`ToolEnd(is_error=true)`, never `ToolStart`.
