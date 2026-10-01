# harness-cli

`harness` command (Typer + Rich). A thin client: it sends protocol commands to an in-process
`harness.core.Harness` and renders the events that come back.

```bash
just dev-cli run "Create hello.txt containing Hello" -w ./scratch   # one task, exit 0 iff completed
just dev-cli run "..." --json                                         # raw JSONL events on stdout; risky tools denied
just dev-cli chat -w .                                                # interactive; /exit or Ctrl-D quits
```

- **Approvals:** risky tools (`run_shell`) show the call and wait for `y`. Anything else is a deny.
  Until Phase 4 an approved command runs **on your machine** ([ADR 0003](../../docs/adr/0003-host-shell-before-sandbox.md)).
- **Ctrl-C** cancels the active run (`Done(reason=cancelled)`). History stays consistent.
- **Event log:** every event of every session is appended to `~/.harness/sessions/<session_id>.jsonl`.
- Untrusted text (model output, tool args/output) is printed without Rich markup and with terminal
  control characters removed.

## Configuration

`HARNESS_*` environment variables or `.env` in the current directory (`harness_cli.config.Settings`):

| Variable | Default |
|---|---|
| `ANTHROPIC_API_KEY` | (SDK credential chain) |
| `HARNESS_MODEL` | `claude-opus-5-5` |
| `HARNESS_EFFORT` | `high` |
| `HARNESS_MAX_STEPS` | 30 |
| `HARNESS_MAX_TOKENS` | unlimited (input + output across the run) |
| `HARNESS_MAX_COST_USD` | 5.0 |
| `HARNESS_TIMEOUT_S` | 1800 |
| `HARNESS_LOG_DIR` | `~/.harness/sessions` |
| `HARNESS_SYSTEM_PROMPT` | see `config.py` |

`harness_cli.runtime.build_harness` is the single place the agent is wired together. The eval
runner uses it too, so evals measure exactly what the CLI runs.
