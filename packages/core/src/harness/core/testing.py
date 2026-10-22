"""Test doubles for code built on the core. Not used at runtime."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field

from pydantic import JsonValue

from harness.core.adapter import Delta, StopDelta, TextDelta, ToolCallArgsDelta, ToolCallStartDelta
from harness.core.hooks import PolicyDecision, ToolOutcome
from harness.core.types import (
    Message,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    StopReason,
    TextPart,
    Tool,
    ToolCall,
    ToolResult,
    Usage,
)


@dataclass(frozen=True)
class Reply:
    """One scripted model turn: optional text followed by tool calls ``(name, arguments)``."""

    text: str = ""
    calls: Sequence[tuple[str, dict[str, JsonValue]]] = ()
    usage: Usage = field(default_factory=lambda: Usage(input_tokens=10, output_tokens=5))
    delay_s: float = 0.0


class ScriptedAdapter:
    """A ``ModelAdapter`` that replays ``Reply`` objects in order and records every request."""

    def __init__(self, replies: Sequence[Reply], *, streaming: bool = True) -> None:
        self._replies = list(replies)
        self.requests: list[ModelRequest] = []
        self._capabilities = ModelCapabilities(
            context_window=100_000,
            max_output_tokens=4_096,
            native_tools=True,
            parallel_tool_calls=True,
            vision=False,
            streaming=streaming,
            tokenizer="approx",
        )

    @property
    def capabilities(self) -> ModelCapabilities:
        """Fixed test capabilities."""
        return self._capabilities

    def _next(self, request: ModelRequest) -> tuple[int, Reply]:
        self.requests.append(request)
        n = len(self.requests)
        if n > len(self._replies):
            raise AssertionError(f"ScriptedAdapter: unexpected model call #{n}")
        return n, self._replies[n - 1]

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Return the next scripted reply."""
        n, reply = self._next(request)
        await asyncio.sleep(reply.delay_s)
        content: list[TextPart | ToolCall] = [TextPart(text=reply.text)] if reply.text else []
        content += [
            ToolCall(id=f"call_{n}_{i}", name=name, arguments=args)
            for i, (name, args) in enumerate(reply.calls)
        ]
        return ModelResponse(
            message=Message(role="assistant", content=list(content)),
            stop_reason=StopReason.TOOL_USE if reply.calls else StopReason.END_TURN,
            usage=reply.usage,
            model="scripted",
        )

    async def stream(self, request: ModelRequest) -> AsyncIterator[Delta]:
        """Stream the next scripted reply word by word, then its tool calls."""
        n, reply = self._next(request)
        await asyncio.sleep(reply.delay_s)
        for word in reply.text.split(" ") if reply.text else []:
            yield TextDelta(text=word + " ")
        for i, (name, args) in enumerate(reply.calls):
            yield ToolCallStartDelta(index=i, id=f"call_{n}_{i}", name=name)
            raw = json.dumps(args)
            yield ToolCallArgsDelta(index=i, json_fragment=raw[: len(raw) // 2])
            yield ToolCallArgsDelta(index=i, json_fragment=raw[len(raw) // 2 :])
        yield StopDelta(
            stop_reason=StopReason.TOOL_USE if reply.calls else StopReason.END_TURN,
            usage=reply.usage,
            model="scripted",
        )


class EchoTools:
    """A ``ToolExecutor`` with an ``echo`` tool and a risky ``danger`` tool; records calls."""

    def __init__(self, delay_s: float = 0.0) -> None:
        self.calls: list[ToolCall] = []
        self.delay_s = delay_s
        schema: dict[str, JsonValue] = {
            "type": "object",
            "properties": {"text": {"type": "string"}},
        }
        self._tools = [
            Tool(name="echo", description="Echo text.", input_schema=schema),
            Tool(name="danger", description="Risky.", input_schema=schema, risky=True),
        ]

    @property
    def tools(self) -> list[Tool]:
        """The two test tools."""
        return self._tools

    async def execute(self, call: ToolCall) -> ToolOutcome:
        """Echo the ``text`` argument back."""
        self.calls.append(call)
        await asyncio.sleep(self.delay_s)
        if call.name not in {"echo", "danger"}:
            return ToolOutcome(ToolResult(tool_call_id=call.id, content="unknown", is_error=True))
        return ToolOutcome(
            ToolResult(tool_call_id=call.id, content=str(call.arguments.get("text")))
        )


@dataclass(frozen=True)
class RiskyAsks:
    """An ``ApprovalPolicy``: risky tools need approval, everything else is allowed."""

    def check(self, tool: Tool | None, call: ToolCall) -> PolicyDecision:
        """Ask for risky tools."""
        if tool is not None and tool.risky:
            return PolicyDecision("ask", f"{tool.name} is risky")
        return PolicyDecision("allow")
