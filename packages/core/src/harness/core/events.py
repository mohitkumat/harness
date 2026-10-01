"""Per-session event bus: assigns ``seq``/``ts`` to protocol events and fans them out.

Events are the single source for UIs, logs and traces (CLAUDE.md §11). Every state change in a
session goes through ``EventBus``; subscribers render, log or trace them.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Awaitable, Callable
from datetime import UTC, datetime
from typing import Literal

from pydantic import JsonValue

from harness.core.protocol import (
    AnyEvent,
    ApprovalRequest,
    Done,
    Error,
    StatusUpdate,
    TextDelta,
    ToolEnd,
    ToolStart,
)
from harness.core.protocol import Usage as ProtocolUsage
from harness.core.types import Usage

EventSink = Callable[[AnyEvent], Awaitable[None]]
"""An async subscriber. Exceptions propagate to the emitter: a broken sink is a bug, not noise."""

Status = Literal["created", "idle", "thinking", "running_tool", "awaiting_approval", "compacting"]
DoneReason = Literal[
    "completed", "max_steps", "max_tokens", "max_cost", "timeout", "cancelled", "error"
]


def _now() -> datetime:
    return datetime.now(UTC)


class EventBus:
    """Builds envelope-stamped events for one session and delivers them to every subscriber.

    ``seq`` is assigned synchronously, so it is strictly increasing in emission order. The bus
    keeps every event in memory so ``resume`` can replay them.
    """

    # TODO(phase-3): replay from the persistent session store instead of an in-memory list.

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._seq = 0
        self._sinks: list[EventSink] = []
        self._history: list[AnyEvent] = []

    # ------------------------------------------------------------------ subscription

    def subscribe(self, sink: EventSink) -> Callable[[], None]:
        """Register ``sink`` for future events. Returns a function that unsubscribes it."""
        self._sinks.append(sink)
        return lambda: self._sinks.remove(sink)

    def stream(self) -> AsyncGenerator[AnyEvent]:
        """Subscribe now and return an iterator over every event emitted from here on.

        Subscribing happens on call, not on first iteration, so no event is missed between creating
        the iterator and starting to consume it. Close it (``contextlib.aclosing``) to unsubscribe.
        """
        queue: asyncio.Queue[AnyEvent] = asyncio.Queue()

        async def put(event: AnyEvent) -> None:
            queue.put_nowait(event)

        unsubscribe = self.subscribe(put)

        async def iterate() -> AsyncGenerator[AnyEvent]:
            try:
                while True:
                    yield await queue.get()
            finally:
                unsubscribe()

        return iterate()

    def replay(self, after_seq: int = -1) -> list[AnyEvent]:
        """Events already emitted with ``seq > after_seq``."""
        return [e for e in self._history if e.seq > after_seq]

    async def publish(self, event: AnyEvent) -> None:
        """Record ``event`` and deliver it to every subscriber, in subscription order."""
        self._history.append(event)
        for sink in list(self._sinks):
            await sink(event)

    def _next_seq(self) -> int:
        seq = self._seq
        self._seq += 1
        return seq

    # ------------------------------------------------------------------ typed emitters

    async def text_delta(self, step: int, text: str) -> None:
        """Emit a chunk of assistant text."""
        await self.publish(
            TextDelta(
                type="text_delta",
                session_id=self.session_id,
                seq=self._next_seq(),
                ts=_now(),
                step=step,
                text=text,
            )
        )

    async def tool_start(
        self, step: int, tool_call_id: str, tool_name: str, arguments: dict[str, JsonValue]
    ) -> None:
        """Emit that a tool call is about to execute."""
        await self.publish(
            ToolStart(
                type="tool_start",
                session_id=self.session_id,
                seq=self._next_seq(),
                ts=_now(),
                step=step,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                arguments=dict(arguments),
            )
        )

    async def tool_end(
        self,
        step: int,
        tool_call_id: str,
        tool_name: str,
        *,
        output: str,
        is_error: bool,
        truncated: bool,
        duration_ms: int,
    ) -> None:
        """Emit that a tool call finished (or was refused)."""
        await self.publish(
            ToolEnd(
                type="tool_end",
                session_id=self.session_id,
                seq=self._next_seq(),
                ts=_now(),
                step=step,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                is_error=is_error,
                output=output,
                truncated=truncated,
                duration_ms=duration_ms,
            )
        )

    async def approval_request(
        self,
        step: int,
        approval_id: str,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, JsonValue],
        reason: str,
    ) -> None:
        """Emit that a tool call is waiting for an ``approve`` command."""
        await self.publish(
            ApprovalRequest(
                type="approval_request",
                session_id=self.session_id,
                seq=self._next_seq(),
                ts=_now(),
                step=step,
                approval_id=approval_id,
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                arguments=dict(arguments),
                reason=reason,
            )
        )

    async def status(
        self,
        status: Status,
        *,
        step: int | None = None,
        command_id: str | None = None,
        message: str | None = None,
    ) -> None:
        """Emit a session status change."""
        await self.publish(
            StatusUpdate(
                type="status_update",
                session_id=self.session_id,
                seq=self._next_seq(),
                ts=_now(),
                status=status,
                step=step,
                command_id=command_id,
                message=message,
            )
        )

    async def done(
        self, reason: DoneReason, *, steps: int, final_text: str | None, usage: Usage
    ) -> None:
        """Emit the end of a run."""
        await self.publish(
            Done(
                type="done",
                session_id=self.session_id,
                seq=self._next_seq(),
                ts=_now(),
                reason=reason,
                steps=steps,
                final_text=final_text,
                usage=ProtocolUsage.model_validate(usage.model_dump()),
            )
        )

    async def error(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        fatal: bool = False,
        command_id: str | None = None,
    ) -> None:
        """Emit an error. Fatal errors must be followed by ``done(reason="error")``."""
        await self.publish(
            Error(
                type="error",
                session_id=self.session_id,
                seq=self._next_seq(),
                ts=_now(),
                code=code,
                message=message,
                retryable=retryable,
                fatal=fatal,
                command_id=command_id,
            )
        )
