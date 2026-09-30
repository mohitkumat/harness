# 0002 — Protocol as a single JSON Schema with generated types

- Status: accepted
- Date: 2026-09-30
- Phase: 0

## Context

`protocol/` is the source of truth for the command/event contract with every client, and both
Python and TypeScript need types generated from it.

## Decision

- **One file**, `protocol/protocol.schema.json` (JSON Schema 2020-12), with every message in
  `$defs`. `Command` and `Event` are `oneOf` unions discriminated on a `type` string
  (`discriminator` keyword with an explicit mapping). A single file keeps `$ref`s simple for both
  generators.
- **Envelope:** every event has `session_id`, a per-session monotonically increasing `seq` (used
  by `resume.after_seq`), `ts` (RFC 3339) and, where it applies, `step`. Every command has a
  client-chosen `command_id` that related events echo back.
- **Versioning:** `protocol/VERSION` (integer, currently `1`) mirrored by `PROTOCOL_VERSION` in
  Python and TS; a test keeps them equal. Changes must be additive. Receivers ignore unknown fields
  (Pydantic's default `extra="ignore"`).
- **Codegen** (`just gen`):
  - Python: `datamodel-code-generator` → `harness/core/protocol/_generated.py` (Pydantic v2,
    `Literal` enums, discriminated `RootModel` unions). `harness.core.protocol` re-exports the
    types and adds `parse_command` / `parse_event`.
  - TypeScript: `json-schema-to-typescript` → `sdks/typescript/src/protocol.gen.ts`.
  - Generated files are committed, and CI runs `just gen-check` to fail on stale output.
- **Examples:** `protocol/examples/*.json` holds one example per message type. Tests validate
  each against the schema and round-trip it through the generated models.

## Consequences

- Free-form objects (tool `arguments`) must declare `"additionalProperties": true`, or the TS
  generator emits `{}`.
- Protocol `TextDelta` (an event) and `harness.core.TextDelta` (a model streaming delta) share a
  name but live in different modules; import the protocol one via `harness.core.protocol`.
