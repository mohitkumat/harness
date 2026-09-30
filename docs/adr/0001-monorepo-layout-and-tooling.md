# 0001 — Monorepo layout and Python packaging

- Status: accepted
- Date: 2026-09-30
- Phase: 0

## Context

CLAUDE.md fixes the directory layout and the dependency rules between packages, but not how the
Python packages are distributed or named, nor how the rules are enforced.

## Decision

1. **One uv workspace member per package** (`packages/*`, `evals/runner`). Each is its own
   distribution (`harness-core`, `harness-adapters`, …) declaring only the deps it is allowed to
   have, so an illegal dependency is visible in its `pyproject.toml`.
2. **Shared `harness.*` namespace package** (PEP 420): `harness.core`, `harness.adapters`, …
   A bare top-level `core` would collide with other libraries. So CLAUDE.md's `core.errors`
   is `harness.core.errors` in code. The eval runner is an app, not a package, so it lives
   outside the namespace as `harness_evals`.
3. **import-linter enforces the dependency rules** (`just lint-deps`, also run in pre-commit and CI):
   core imports nothing from the repo; packages are mutually independent; packages never import
   evals/apps; provider SDKs (`anthropic`, `openai`, `google`, `ollama`) may be imported directly
   only from `harness.adapters`. Contracts live in the root `pyproject.toml`.
4. The root `pyproject.toml` is a virtual project (`package = false`) holding shared tool config
   (ruff, pyright strict, pytest, codegen, import-linter) and the dev dependency group.
5. The JS side is a pnpm workspace (`sdks/typescript`, later `apps/*`) linted by biome.

## Consequences

- `uv sync --all-packages` installs everything editable into one `.venv`.
- Tests use `--import-mode=importlib` because every package has its own `tests/` directory.
- Forbidding `google` blocks every direct `google.*` import outside adapters (import-linter can't
  target subpackages of external packages). If a legitimate need appears (e.g. `google.protobuf`
  for OpenTelemetry), narrow the contract in a follow-up.
