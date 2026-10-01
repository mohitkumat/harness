# harness-guardrails

Permission policies, limits and untrusted-content tagging. May import `harness.core` only
(CLAUDE.md §3).

Phase 1 policies (implement `harness.core.ApprovalPolicy`):

| Policy | Risky tool | Other tools |
|---|---|---|
| `AskForRisky()` | `ask`: the loop emits `ApprovalRequest` and waits for `approve` | `allow` |
| `Headless(allow_risky=False)` | `deny` (returned to the model as an error result) | `allow` |
| `Headless(allow_risky=True)` | `allow`, which runs on the host until Phase 4 ([ADR 0003](../../docs/adr/0003-host-shell-before-sandbox.md)) | `allow` |

Run limits (steps, tokens, cost, wall-clock) are enforced in `harness.core.loop` (`Limits`).

Coming in Phase 4: per-tool/per-path allow/ask/deny rules, cost policies, untrusted-content tagging.
