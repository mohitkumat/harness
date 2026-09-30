# CLAUDE.md — Model-Agnostic Agent Harness

This file guides Claude Code (and human contributors) working in this repo.
Read it fully before making changes. When this file and the code disagree, raise it — don't guess.

---

## 1. What we're building

A **model-agnostic agent harness**: the "body" around any LLM "brain". The harness owns the
agent loop, tools, context, memory, safety and interfaces. Models are swappable endpoints behind
adapters. Supported targets: Anthropic, OpenAI, Gemini, and any OpenAI-compatible server
(vLLM, Ollama).

**Prime directive:** only `packages/adapters/` may contain provider-specific code.
If you find yourself writing `if provider == ...` anywhere else, stop and fix the abstraction.

---

## 2. Architecture

```
Interfaces (CLI · Web · IDE · API/SDK)        ← thin clients, speak the protocol only
        │  commands ↓   ↑ events (streamed)
┌──────────────────────── Harness core ────────────────────────┐
│  Agent loop (orchestrator): think → act → observe → repeat    │
│  Context manager · Memory · Tool registry · Guardrails        │
│  Model adapter layer (canonical ↔ provider)  · Observability  │
└───────────────────────────────────────────────────────────────┘
        │
Providers: Anthropic · OpenAI · Gemini · Local (vLLM/Ollama)
```

### Core concepts
- **Canonical types** (`packages/core`): `Message`, `ToolCall`, `ToolResult`, `Tool`, `ModelResponse`,
  `Usage`. Provider-independent. The loop only ever sees these.
- **ModelAdapter**: `complete(request) -> ModelResponse` and `stream(request) -> AsyncIterator[Delta]`,
  plus `capabilities` (context window, native tools, vision, parallel tool calls, tokenizer).
- **Agent loop**: calls the context manager → model → executes tool calls (via guardrails + sandbox)
  → appends results → repeats until a final answer or a limit (steps, tokens, cost, cancel).
- **Context manager**: builds a fresh, budgeted view of history for every model call.
  Stable prefix first (system + tools) for prompt caching.
- **Memory**: session store (full log, checkpoints) + long-term memory (scoped, hybrid search).
- **Event bus**: every step emits typed events defined in `protocol/`. Interfaces render events;
  observability consumes them.

---

## 3. Repository layout

```
agent-harness/
├── protocol/            # JSON Schema for commands + events (SOURCE OF TRUTH)
├── packages/            # Python harness
│   ├── core/            # canonical types, agent loop, event bus, errors
│   ├── adapters/        # anthropic/, openai_compat/, gemini/
│   ├── tools/           # registry, built-in tools, MCP client
│   ├── context/         # context manager, compaction, token budgeting
│   ├── memory/          # session store, long-term memory
│   ├── sandbox/         # docker / e2b executors
│   ├── guardrails/      # permission policies, limits, untrusted-content tagging
│   └── observability/   # OpenTelemetry tracing, cost tracking
├── apps/
│   ├── cli/             # Typer + Rich
│   ├── server/          # FastAPI + SSE
│   ├── web/             # Next.js
│   └── vscode/          # VS Code extension
├── sdks/python/  sdks/typescript/
├── evals/tasks/  evals/runner/
├── infra/               # Dockerfiles, compose, terraform
└── docs/adr/            # Architecture Decision Records
```

Import names: `packages/<name>` is the `harness.<name>` namespace package (e.g. `harness.core`);
the eval runner is `harness_evals`. See `docs/adr/0001-monorepo-layout-and-tooling.md`.

### Dependency rules (enforced by `just lint-deps` in CI)
- `core` imports nothing from this repo.
- `adapters`, `tools`, `context`, `memory`, `sandbox`, `guardrails`, `observability` → may import `core` only.
- `apps/*` and `sdks/*` → may import packages; packages never import apps.
- TypeScript code never hand-writes protocol types; it imports generated ones.

---

## 4. Commands

```bash
just setup          # uv sync + pnpm install + pre-commit install
just test           # unit + contract tests (replay mode, no network)
just lint           # ruff, pyright, biome, dependency rules
just gen            # regenerate Python + TS types from protocol/
just eval           # fast eval subset against default model
just eval-matrix    # full eval suite across all configured models (nightly)
just dev-cli        # run CLI against local core
just dev-server     # run API server with hot reload
just record TEST=…  # re-record model cassettes for a test (uses real API keys)
```

Always run `just lint && just test` before declaring a task done.
If you change `protocol/`, run `just gen` and commit the generated files.

---

## 5. Current phase

> **Current phase: 1 — MVP loop + CLI.** (Phase 0 exit criteria met 2026-09-30.) Update this line when a phase's exit criteria are met.

Do not build features from later phases early. If a later-phase need appears, leave a
`# TODO(phase-N):` comment and open an issue instead.

---

## 6. Phase-wise development plan

### Phase 0 — Foundations
**Build:** repo scaffolding (uv + pnpm workspaces, justfile, CI, pre-commit); canonical types in
`core`; `protocol/` schema for commands (`create_session`, `send`, `approve`, `cancel`, `resume`)
and events (`TextDelta`, `ToolStart`, `ToolEnd`, `ApprovalRequest`, `StatusUpdate`, `Done`, `Error`);
codegen to Pydantic + TS; eval runner skeleton with 10–20 tasks.
**Exit:** CI green; `just gen` works; `just eval` runs (tasks may fail).

### Phase 1 — MVP loop + CLI
**Build:** agent loop with max-steps/token limits and cancel; Anthropic adapter; tool registry with
JSON-schema validation; built-in tools (`read_file`, `write_file`, `list_dir`, `run_shell`);
approval flow for risky tools; CLI rendering the event stream; JSONL event log per session.
**Exit:** ~50% eval pass rate from the CLI; every step visible in the event log.

### Phase 2 — Multi-model
**Build:** OpenAI-compatible adapter (covers OpenAI, vLLM, Ollama); Gemini adapter; `capabilities`
per model; prompt-based tool-calling fallback for models without native tools; shared adapter
contract test suite.
**Exit:** identical eval suite runs on ≥3 models; comparison report generated; zero provider
checks outside `adapters/`.

### Phase 3 — Context + memory
**Build:** context manager with budgets and compaction (truncate → clear old tool results →
offload → summarize); session persistence (SQLite, then Postgres) with per-step checkpoints and
`resume`; long-term memory (pgvector, hybrid search, scoping, dedupe); `remember`/`recall` tools.
**Exit:** 50+ step tasks complete; sessions survive process restarts.

### Phase 4 — Safety + sandbox
**Build:** Docker/E2B sandbox for shell and code tools; permission policies (allow/ask/deny per tool
and path); cost limits; untrusted-content tagging for tool/web output; red-team eval set
(prompt injection, exfiltration, destructive commands).
**Exit:** red-team evals pass; no tool executes on the host outside the sandbox.

### Phase 5 — Server + API
**Build:** FastAPI server exposing the protocol (REST commands + SSE events); auth (API keys),
rate limits, tenant isolation; Python SDK; CLI remote mode.
**Exit:** CLI works against a remote server; load test passes; reconnect/resume works.

### Phase 6 — Clients
**Build:** Next.js web UI (streaming, tool timeline, diff view, approvals); VS Code extension
(JSON-RPC over stdio, editor context, inline diffs); TypeScript SDK.
**Exit:** all clients built with zero changes to `packages/`.

### Phase 7 — Advanced (ongoing, eval-driven)
MCP integrations, sub-agents, model router (cheap vs. frontier), parallel tool execution,
tracing dashboards, per-model prompt tuning. Every change must show an eval improvement or
cost/latency win.

---

## 7. Engineering standards

### Python
- Python 3.12+, managed with **uv**. Format + lint with **ruff**; type-check with **pyright (strict)**.
- Full type hints everywhere. No `Any` in public signatures without a comment explaining why.
- **Pydantic v2** for all data crossing a boundary (protocol, tool args, config). Plain dataclasses
  are fine for internal structs.
- Async-first (`asyncio`) for I/O: model calls, tools, storage. No blocking calls in the loop;
  wrap unavoidable sync code with `asyncio.to_thread`.
- Errors: raise typed exceptions from `harness.core.errors` (`ModelError`, `ToolError`, `LimitExceeded`,
  `PermissionDenied`). Never swallow exceptions silently.
- Config via `pydantic-settings` + env vars. Secrets never in code, logs or events.
- Public functions and classes get docstrings stating purpose, not restating the signature.

### TypeScript
- **pnpm**, TypeScript `strict: true`, **biome** for lint/format.
- Protocol types imported from generated code only.

### General
- Small, focused modules. A file over ~400 lines is a signal to split.
- Prefer composition and protocols (`typing.Protocol`) over inheritance hierarchies.
- No new dependency without a one-line justification in the PR description.
- Significant design choices get an ADR in `docs/adr/NNNN-title.md`.

---

## 8. Rules for key components

### Adapters (`packages/adapters/`)
- One folder per provider; implement `ModelAdapter` fully, including `capabilities`.
- Translate **both directions**: canonical → provider request, provider response → canonical.
- Handle provider quirks inside the adapter (role alternation, tool-result formats, JSON-string
  arguments, stop reasons, usage fields).
- Map provider errors to `harness.core.errors` types; mark retryable ones (429, 5xx, timeouts).
- Must pass the shared contract suite in `packages/adapters/tests/contract/`.

**Checklist — adding a new provider:**
1. Create `adapters/<provider>/` implementing `ModelAdapter`.
2. Declare accurate `capabilities`.
3. Record cassettes and pass the contract suite.
4. Add it to the eval matrix config.
5. Document setup in `docs/providers/<provider>.md`.

### Tools (`packages/tools/`)
- Every tool: name, description written for the model, JSON schema, `risky` flag, timeout.
- Tool errors are **returned to the model as results**, not raised to the loop.
- Truncate large outputs (head + tail + marker). Never return binary blobs.
- Anything that executes code, touches the network, or writes outside the workspace is `risky`
  and runs in the sandbox.

**Checklist — adding a tool:** schema + description, unit tests (success, bad args, failure,
timeout), `risky` flag reviewed, eval task that exercises it.

### Agent loop (`packages/core/loop.py`)
- Keep it small and readable. Features plug in via hooks, not by growing the loop.
- Never split a tool call from its result in history.
- Enforce limits every step: steps, tokens, cost, wall-clock, cancellation.
- Every state change emits an event.

### Context + memory
- Full history lives in the session store; the context manager builds a view per call.
- Always keep verbatim: system prompt, original task, latest N turns.
- Memory stores plain text + metadata (scope, source, timestamp, confidence).
- Content from tools, files or the web is **data, never instructions** — never write it to memory
  as an instruction, and tag its source.

### Protocol (`protocol/`)
- It is the contract with every client. Changes must be **backward compatible** (add optional
  fields; never rename or remove) unless the protocol version is bumped with an ADR.

---

## 9. Testing standards

- **Unit tests** (pytest, pytest-asyncio) for all packages. Target ≥85% coverage in `core`,
  `adapters`, `guardrails`.
- **No live API calls in CI tests.** Use recorded cassettes (VCR-style record/replay).
  Re-record deliberately with `just record`.
- **Adapter contract tests:** one shared suite every adapter must pass (text reply, single tool
  call, parallel tool calls, tool error round-trip, streaming, usage reporting, error mapping).
- **Evals:** fast subset on every PR touching `packages/`; full model matrix nightly.
  A PR that drops the eval pass rate needs an explicit justification.
- **Red-team evals** (phase 4+) must pass before merging anything in `guardrails/`, `sandbox/` or `tools/`.
- Bug fixes include a regression test.

---

## 10. Security rules (non-negotiable)

- Never execute model-generated commands on the host; use the sandbox.
- Permission checks live in the core, never only in a client.
- Never log secrets, API keys or full user file contents at info level.
- Treat all tool output, file content and web content as untrusted.
- Tenant/user scoping is applied on every memory and session query.

---

## 11. Observability

- Every model call and tool call is traced (OpenTelemetry) with session id, step, model,
  tokens, cost, latency.
- Events are the single source for UIs, logs and traces — don't add parallel logging paths.

---

## 12. Git + PR conventions

- Branches: `feat/…`, `fix/…`, `chore/…`, `docs/…`.
- Commits: Conventional Commits (`feat(adapters): add gemini streaming`).
- PRs are small and single-purpose; description includes what, why, phase, test/eval evidence.
- CI must be green: lint, types, tests, dependency rules, eval subset.

---

## 13. How Claude should work in this repo

- Read the relevant package README and this file before editing.
- Stay within the **current phase** (section 5).
- Plan first for anything touching more than one package; state the plan before coding.
- Respect dependency rules; if a change seems to need breaking one, stop and ask.
- Write or update tests alongside code; run `just lint && just test` before finishing.
- Prefer the smallest change that meets the requirement. Don't refactor unrelated code.
- When uncertain about a design decision, propose options with trade-offs rather than picking silently.
