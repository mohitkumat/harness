"""``ModelAdapter`` wrapper that gives tool calling to models without native tool support."""

from __future__ import annotations

from collections.abc import AsyncIterator

from harness.adapters.prompt_tools import render
from harness.core import (
    Delta,
    ModelAdapter,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    StopDelta,
    TextDelta,
    collect_stream,
)


class PromptToolsAdapter:
    """Wraps any ``ModelAdapter`` and does tool calling through the prompt (ADR 0006).

    The wrapped adapter never sees native tools: tool definitions go into the system prompt,
    past calls and results become text, and ``<tool_call>`` blocks in the reply become canonical
    ``ToolCall`` parts. To the loop it looks like any other adapter.
    """

    def __init__(self, inner: ModelAdapter, *, parallel_tool_calls: bool = True) -> None:
        """``parallel_tool_calls``: whether the model reliably writes several blocks per reply."""
        self.inner = inner
        self._capabilities = inner.capabilities.model_copy(
            update={"native_tools": False, "parallel_tool_calls": parallel_tool_calls}
        )

    @property
    def capabilities(self) -> ModelCapabilities:
        """The wrapped model's capabilities; ``native_tools`` is False."""
        return self._capabilities

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """One non-streaming call through the wrapped adapter."""
        response = await self.inner.complete(render.to_inner_request(request))
        return render.from_inner_response(response, _names(request))

    async def stream(self, request: ModelRequest) -> AsyncIterator[Delta]:
        """Stream prose live, hiding ``<tool_call>`` blocks; the ``StopDelta`` has the calls."""
        seen: list[Delta] = []
        visible = _VisibleText()
        async for delta in self.inner.stream(render.to_inner_request(request)):
            seen.append(delta)
            if isinstance(delta, TextDelta) and (text := visible.feed(delta.text)):
                yield TextDelta(text=text)
        if tail := visible.flush():
            yield TextDelta(text=tail)

        response = render.from_inner_response(await collect_stream(_replay(seen)), _names(request))
        yield StopDelta(
            stop_reason=response.stop_reason,
            usage=response.usage,
            model=response.model,
            response_id=response.response_id,
            message=response.message,
        )


def _names(request: ModelRequest) -> set[str]:
    return {t.name for t in request.tools}


async def _replay(deltas: list[Delta]) -> AsyncIterator[Delta]:
    for delta in deltas:
        yield delta


class _VisibleText:
    """Filters streamed text, dropping ``<tool_call>…</tool_call>`` blocks.

    Holds back a tail that could be the start of a tag, so a tag split across chunks is
    still recognised.
    """

    def __init__(self) -> None:
        self._buf = ""
        self._in_call = False

    def feed(self, chunk: str) -> str:
        """Add ``chunk``; return the text that is now safe to show."""
        self._buf += chunk
        out: list[str] = []
        while True:
            if self._in_call:
                end = self._buf.find(render.CLOSE)
                if end < 0:
                    self._buf = self._buf[-(len(render.CLOSE) - 1) :]
                    break
                self._buf = self._buf[end + len(render.CLOSE) :]
                self._in_call = False
            else:
                start = self._buf.find(render.OPEN)
                if start >= 0:
                    out.append(self._buf[:start])
                    self._buf = self._buf[start + len(render.OPEN) :]
                    self._in_call = True
                    continue
                keep = _partial_tag(self._buf, render.OPEN)
                out.append(self._buf[: len(self._buf) - keep])
                self._buf = self._buf[len(self._buf) - keep :]
                break
        return "".join(out)

    def flush(self) -> str:
        """Text still held back at the end of the stream (none if a block is unfinished)."""
        text = "" if self._in_call else self._buf
        self._buf = ""
        return text


def _partial_tag(text: str, tag: str) -> int:
    """Length of the longest suffix of ``text`` that is a proper prefix of ``tag``."""
    for n in range(min(len(text), len(tag) - 1), 0, -1):
        if text.endswith(tag[:n]):
            return n
    return 0
