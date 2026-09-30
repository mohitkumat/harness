# harness-core

Provider-independent foundation. Imports nothing else from this repo.

| Module | Contents |
|---|---|
| `harness.core.types` | `Message` + content parts (`TextPart`, `ImagePart`, `ToolCall`, `ToolResult`), `Tool`, `ModelRequest`, `ModelResponse`, `Usage`, `StopReason`, `ModelCapabilities` |
| `harness.core.adapter` | `ModelAdapter` protocol and streaming `Delta`s |
| `harness.core.errors` | `HarnessError`, `ModelError`, `ToolError`, `LimitExceeded`, `PermissionDenied`, each with a stable `code` |
| `harness.core.protocol` | Pydantic models **generated** from `protocol/` plus `parse_command` / `parse_event` |

Conventions: canonical types are frozen Pydantic models with `extra="forbid"`. The system prompt
lives on `ModelRequest.system`, not in `messages`. Tool results go in `role="tool"` messages.

Coming in Phase 1: agent loop, event bus.
