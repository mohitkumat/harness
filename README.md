# agent-harness

A model-agnostic agent harness. The harness owns the loop, tools, context, memory, safety and
interfaces, and models are swappable adapters. See [CLAUDE.md](CLAUDE.md) for architecture, phases
and standards.

## Quick start

Prereqs: [uv](https://docs.astral.sh/uv/), Node 22 + [pnpm](https://pnpm.io), [just](https://just.systems).

```bash
just setup    # install deps + git hooks
just lint     # ruff, pyright, biome, tsc, dependency rules
just test     # unit + contract tests
just gen      # regenerate types from protocol/
just eval     # fast eval subset
```

Status: **Phase 1 — MVP loop + CLI** (Phase 0 complete).
