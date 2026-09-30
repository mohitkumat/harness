# protocol/

The wire contract between the harness and every client. **Source of truth**: never hand-write
these types elsewhere.

- `protocol.schema.json`: commands (`create_session`, `send`, `approve`, `cancel`, `resume`) and
  events (`text_delta`, `tool_start`, `tool_end`, `approval_request`, `status_update`, `done`,
  `error`).
- `examples/`: one valid payload per message type. Add one when you add a message.
- `VERSION`: protocol major version.

After any change: `just gen`, then commit the regenerated files. Changes must be backward
compatible (add optional fields only); anything else needs a version bump and an ADR. See
[ADR 0002](../docs/adr/0002-protocol-schema-and-codegen.md).
