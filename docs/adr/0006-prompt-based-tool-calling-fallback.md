# 0006 — Prompt-based tool calling as a wrapper adapter

- Status: accepted
- Date: 2026-10-23
- Phase: 2

## Context

Phase 2 needs tool calling for models without native tool support (CLAUDE.md §6). Example:
`qwen2.5-coder` under Ollama, whose chat template has no tools section. The loop must not branch
on provider or model (§1). The fallback also has to work with cassettes, which replay a request
only if its body is byte-identical to the recording (ADR 0004).

Options considered:

1. **In the loop or the context manager**: check `capabilities.native_tools` and rewrite requests
   there. This grows the loop (§8: "features plug in via hooks, not by growing the loop"), and
   parsing model output into tool calls is an adapter's job: provider response → canonical.
2. **In each adapter**: every adapter gets a "no native tools" mode. This duplicates the same
   prompt format and parser in each adapter.
3. **A wrapper adapter** (`adapters/prompt_tools/`) that takes any `ModelAdapter` and is itself
   one. Chosen.

## Decision

1. `PromptToolsAdapter(inner)` puts tool definitions in the system prompt, after the caller's
   system text so the stable prefix stays stable. It sends the inner adapter no native tools,
   renders past `ToolCall`s and `ToolResult`s as text, and parses the reply back into canonical
   `ToolCall`s. Reasoning parts pass through untouched.
2. **Format**: `<tool_call>{"name": …, "arguments": {…}}</tool_call>`, results in
   `<tool_result name="…">` blocks in a user message. This is the Qwen/Hermes format, so many
   open models already know it.
3. **Tolerant parsing, driven by recordings.** qwen2.5:7b also writes several objects in one
   block, and bare call objects with no tags. Bare JSON counts as calls only if the *whole* reply
   is call objects naming a tool from the request, so a final answer that happens to be JSON is
   not misread. Unparseable blocks become `{"_raw": …}` calls (ADR 0005). The registry answers
   them with a JSON parse error ("not valid JSON … use double quotes"), not a schema error.
   With the schema error, qwen2.5:7b resent the same broken call. The key is now
   `harness.core.RAW_ARGUMENTS_KEY`, and `ToolCall.raw_arguments` reads it.
4. **Ids are generated, never shown to the model.** Results are matched to calls by position and
   name in the prompt, so request bodies stay deterministic for cassettes.
5. **Selection is config.** `AdapterSpec.native_tools=False` (CLI: `HARNESS_NATIVE_TOOLS=false`)
   or an inner adapter reporting `capabilities.native_tools=False` makes `build_adapter` wrap
   it. The wrapper reports `native_tools=False` and `parallel_tool_calls=True`.
6. **Streaming**: prose streams live. `<tool_call>` blocks are held back (including a tag split
   across chunks) and arrive as `ToolCall`s on the final `StopDelta.message`.

## Consequences

- Any adapter, current or future (Gemini), gets the fallback for free. The loop, `core` and the
  protocol are unchanged.
- The contract suite runs the wrapper as its own provider (`prompt_tools`, over local qwen2.5:7b
  with native tools off).
- Full eval suite on qwen2.5:7b, run without `--allow-host-shell` (2026-10-23): native tools
  10/15, prompt fallback 9/15. The fallback's misses vary from run to run, which looks like
  model noise rather than parser failures.
- The prompt costs extra input tokens (tool schemas as text), and the parser's tolerance is
  heuristic. New model quirks need a recording and a parser case, not a loop change.
- `invalid_tool_call` and `_raw` are reserved: no tool may use those names.
