# 0004 — Opaque reasoning parts and HTTP-level cassettes

- Status: accepted
- Date: 2026-10-01
- Phase: 1

## Context

1. Claude Opus 5.5 (the default model) always thinks. Its thinking blocks are signed and bound to
   the conversation that produced them. They must be passed back **unchanged** in later requests,
   and dropping or editing them degrades reasoning and can cause 400s. OpenAI's encrypted reasoning
   items and Gemini's thought signatures (Phase 2) have the same shape: opaque, provider-produced
   data that must round-trip. The canonical `Message` had no place for it.
2. CLAUDE.md §9 requires VCR-style record/replay with no live API calls in CI. The `anthropic` 1.x
   SDK is built on `httpx2`, which existing VCR libraries don't patch.
3. Server-side refusal fallbacks can rewrite a response mid-stream, so incremental deltas alone
   don't determine the message that must be echoed back.

## Decision

1. **`ReasoningPart(source, data, text)`** joins the canonical content parts (assistant messages
   only). `data` is opaque JSON that only the adapter whose id equals `source` interprets. Adapters
   replay their own parts verbatim and drop everyone else's. `text` is display-only. This keeps
   provider knowledge inside `adapters/` (prime directive).
2. **`StopDelta.message`** (optional): an adapter may stream text deltas live and put the
   authoritative final message on the stop delta. `collect_stream` prefers it. The Anthropic
   adapter streams text and builds the message from the SDK's final message.
3. **`CassetteTransport`** in `harness.adapters.testing`: an `httpx2` transport that records
   method, path, query and JSON body plus the response, and **never request headers**. Replay serves
   responses in order and fails on any request that differs. Missing cassettes skip the contract test
   with a hint. `just record TEST=…` re-records using credentials from the env or `.env`. No new
   dependency.

## Consequences

- History must stay append-only and translation deterministic, or preserved thinking breaks. The
  Phase 3 context manager has to respect this (compaction drops a leading run of reasoning, never a
  middle one) or use server-side compaction.
- The protocol has no reasoning event. Thinking summaries are not shown in clients yet.
- Contract tests skip until someone with a key records cassettes. Until then, unit tests cover the
  adapter with hand-written payloads.
