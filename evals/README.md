# evals/

- `tasks/*.toml`: one task per file (file name = task `id`). A task has a prompt, seed files
  (`[setup.files]`) and declarative `[[checks]]`: `file_exists`, `file_absent`, `file_contains`,
  `file_not_contains`, `file_equals`, `file_json_equals`, `file_unchanged`, `answer_contains`,
  `answer_matches`. Checks never execute code.
- Tag `fast` marks the subset run by `just eval` on every PR.
- `runner/`: the `harness-eval` CLI. Each task runs in a fresh temp workspace, and a JSON report
  goes to `evals/results/` (git-ignored).

Agents:

- `null` (default, used by CI): does nothing, so every task fails. It checks the eval plumbing
  without API calls.
- `harness`: the real agent, built by `harness_cli.runtime.build_harness`, the same wiring as the
  CLI. It needs credentials (`ANTHROPIC_API_KEY` in the env or `.env`) and costs money. Risky tools
  are **denied** because nobody can approve them, so `run_shell` tasks fail unless you pass
  `--allow-host-shell`. That flag runs model-written commands on your machine inside each task's temp
  workspace ([ADR 0003](../docs/adr/0003-host-shell-before-sandbox.md)). Local use only, never CI.

```bash
just eval                                        # fast subset, null agent (what CI runs)
just eval --agent harness                        # fast subset, real agent
uv run harness-eval --agent harness              # full suite
uv run harness-eval --agent harness --allow-host-shell --model claude-sonnet-5-5
uv run harness-eval --help

# Free local model via Ollama (docs/providers/openai_compat.md)
HARNESS_PROVIDER=openai_compat HARNESS_MODEL=qwen2.5:7b HARNESS_MAX_OUTPUT_TOKENS=8192 \
  uv run harness-eval --agent harness
```

Each run writes a JSON report to `evals/results/`. With the `harness` agent it also writes the
per-task session event logs to `evals/results/sessions/`.
