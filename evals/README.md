# evals/

- `tasks/*.toml`: one task per file (file name = task `id`). A task has a prompt, seed files
  (`[setup.files]`) and declarative `[[checks]]`: `file_exists`, `file_absent`, `file_contains`,
  `file_not_contains`, `file_equals`, `file_json_equals`, `file_unchanged`, `answer_contains`,
  `answer_matches`. Checks never execute code.
- Tag `fast` marks the subset run by `just eval` on every PR.
- `runner/`: the `harness-eval` CLI. Each task runs in a fresh temp workspace, and a JSON report
  goes to `evals/results/` (git-ignored).

```bash
just eval                          # fast subset
just eval --tag write_file         # add a tag filter
uv run harness-eval --help
```

Phase 0 only has the `null` agent (every task fails by design). Phase 1 wires in the real loop.
