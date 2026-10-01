# 0003 — `run_shell` on the host until the Phase 4 sandbox

- Status: accepted
- Date: 2026-10-01
- Phase: 1

## Context

Phase 1 requires a `run_shell` built-in tool. CLAUDE.md §10 says model-generated commands never run
on the host. The Docker/E2B sandbox is Phase 4 work, and building it early would break the phase
discipline in CLAUDE.md §5.

## Decision

Until Phase 4, `run_shell` is a **temporary, explicitly scoped exception** to §10:

1. `run_shell` is `risky=True`. The Phase 1 guardrail policy requires **user approval for every
   risky call**. No "always allow" option exists.
2. The command runs with `bash -c` and the session workspace as its working directory, in its own
   process group (a timeout kills the whole group). It gets a minimal environment (`PATH`, `HOME`,
   locale, `TERM`, `TMPDIR`, `USER`), so API keys and other secrets in the harness's environment
   are not inherited.
3. Evals have no human to approve, so they **deny** risky tools by default. `harness-eval
   --allow-host-shell` approves them automatically. Only use it locally: each task runs in a
   throwaway temp workspace, but the command still runs on the host. CI never passes this flag.
4. The code carries `TODO(phase-4)` markers at the execution site and in the policy.

## Consequences

- The approval prompt is the only barrier, so a user who approves a destructive command will run it.
  The workspace `cwd` does not confine a shell.
- Eval tasks tagged `run_shell` fail in default eval runs.
- Phase 4 replaces the executor with the sandbox and removes this exception. Its exit criterion
  ("no tool executes on the host outside the sandbox") closes this ADR.
