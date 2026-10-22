# harness-observability

Tracing, cost tracking and the event log. May import `harness.core` only (CLAUDE.md §3).

- `JsonlEventLog(directory)`: an event sink that appends every protocol event of a session to
  `<directory>/<session_id>.jsonl`. Pass it to `Harness(..., sinks=[log])` so every step of every
  session is recorded. The log holds tool output (possibly file contents), so treat it as session
  data, not an info-level log.

Coming later (dashboards in Phase 7): OpenTelemetry tracing per model/tool call (session id, step, model, tokens, cost,
latency), driven by the same events.
