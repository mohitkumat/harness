# Agent harness task runner. `just` with no args lists recipes.
set shell := ["bash", "-euo", "pipefail", "-c"]

generated := "packages/core/src/harness/core/protocol/_generated.py sdks/typescript/src/protocol.gen.ts"

default:
    @just --list

# Install Python + JS deps and git hooks
setup:
    uv sync --all-packages
    pnpm install
    uv run pre-commit install

# Unit + contract tests (replay mode, no network)
test:
    uv run pytest

# All linters: Python, TypeScript, dependency rules
lint: lint-py lint-ts lint-deps

lint-py:
    uv run ruff check .
    uv run ruff format --check .
    uv run pyright

lint-ts:
    pnpm biome check .
    pnpm -r typecheck

# Enforce package dependency rules (CLAUDE.md §3)
lint-deps:
    uv run lint-imports

# Auto-fix formatting and safe lint issues
fmt:
    uv run ruff format .
    uv run ruff check --fix .
    pnpm biome check --write .

# Regenerate Python + TS types from protocol/
gen:
    uv run datamodel-codegen
    pnpm -r gen
    pnpm biome format --write sdks/typescript/src/protocol.gen.ts

# Fail if generated code is stale (CI)
gen-check: gen
    @if [ -n "$(git status --porcelain -- {{generated}})" ]; then \
        echo "Generated protocol types are stale: run 'just gen' and commit." >&2; \
        git status --porcelain -- {{generated}} >&2; git diff -- {{generated}} >&2; exit 1; \
    fi

# Fast eval subset against the default agent (extra args go to harness-eval)
eval *ARGS:
    uv run harness-eval --tag fast {{ARGS}}

# Full eval suite across all configured models (nightly)
eval-matrix:
    @echo "TODO(phase-2): eval matrix needs >=2 adapters." >&2; exit 1

# Run the CLI against local core, e.g. `just dev-cli chat` or `just dev-cli run "..."`
dev-cli *ARGS:
    uv run harness {{ARGS}}

# Run the API server with hot reload
dev-server:
    @echo "TODO(phase-5): apps/server does not exist yet." >&2; exit 1

# Re-record model cassettes for a test (uses real API keys from the env or .env)
record TEST:
    set -a; if [ -f .env ]; then . ./.env; fi; set +a; HARNESS_RECORD=1 uv run pytest {{TEST}}
