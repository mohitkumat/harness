# harness-tools

Tool registry and built-in tools. May import `harness.core` only (CLAUDE.md §3).

- `ToolRegistry(workspace, specs)` implements `harness.core.ToolExecutor`. It validates arguments
  against each tool's JSON schema (`jsonschema`, Draft 2020-12) and enforces `Tool.timeout_s`.
  Output is truncated to head + tail with a marker. Every failure comes back to the model as an
  `is_error` result: unknown tool, bad args, `ToolError`, an unexpected exception or a timeout.
- A tool is a `ToolSpec(tool: Tool, handler)`. The handler is `async (args, ToolContext) -> str`
  and raises `ToolError` for expected failures.

| Tool | Risky | Notes |
|---|---|---|
| `read_file` | no | UTF-8 text only; refuses binary files and anything outside the workspace (symlinks resolved) |
| `write_file` | no | creates parent dirs; workspace-confined |
| `list_dir` | no | optional `recursive`; capped at 1000 entries |
| `run_shell` | **yes** | `bash -c` in the workspace, minimal env, process group killed on timeout. Runs on the **host** until Phase 4: see [ADR 0003](../../docs/adr/0003-host-shell-before-sandbox.md) |

Coming later: MCP client (Phase 7) and sandboxed execution (Phase 4).
