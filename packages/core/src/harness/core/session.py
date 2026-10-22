"""In-process harness runtime: turns protocol commands into session runs and events.

Interfaces (CLI now, server later) talk to the core only through these command handlers and
the events each session's ``EventBus`` emits.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Sequence
from typing import Protocol

from harness.core.approvals import ApprovalGate
from harness.core.errors import InvalidCommand
from harness.core.events import EventBus, EventSink
from harness.core.loop import AgentLoop, RunResult
from harness.core.protocol import (
    AnyEvent,
    ApproveCommand,
    CancelCommand,
    CreateSessionCommand,
    ResumeCommand,
    SendCommand,
)
from harness.core.types import Message


class LoopFactory(Protocol):
    """Builds the agent loop for a new session (model, tools, policy) from its create command."""

    def __call__(
        self, command: CreateSessionCommand, events: EventBus, approvals: ApprovalGate
    ) -> AgentLoop:
        """Return a loop wired to ``events`` and ``approvals``."""
        ...


class Session:
    """One conversation: its history, event bus and at most one active run."""

    # TODO(phase-3): persist history and checkpoints in the session store.

    def __init__(self, session_id: str, loop: AgentLoop, approvals: ApprovalGate) -> None:
        self.id = session_id
        self.loop = loop
        self.events = loop.events
        self.approvals = approvals
        self.history: list[Message] = []
        self._task: asyncio.Task[RunResult] | None = None

    @property
    def running(self) -> bool:
        """Whether a run is in progress."""
        return self._task is not None and not self._task.done()

    async def send(self, text: str) -> None:
        """Start a run for a new user message. Raises ``InvalidCommand`` if one is active."""
        if self.running:
            raise InvalidCommand(f"session {self.id} already has an active run")
        self.history.append(Message.user(text))
        self._task = asyncio.create_task(self.loop.run(self.history))
        # Let the run enter its try-block, so an immediate cancel still ends with a Done event.
        await asyncio.sleep(0)

    async def wait(self) -> RunResult:
        """Wait for the active (or last) run to finish and return how it ended."""
        if self._task is None:
            raise InvalidCommand(f"session {self.id} has not started a run")
        return await self._task

    def cancel(self) -> None:
        """Cancel the active run, if any. It ends with ``Done(reason="cancelled")``."""
        if self._task is not None and not self._task.done():
            self._task.cancel()


class Harness:
    """Holds sessions and applies protocol commands to them."""

    def __init__(self, factory: LoopFactory, sinks: Sequence[EventSink] = ()) -> None:
        """``sinks`` are subscribed to every new session before its first event."""
        self._factory = factory
        self._sinks = list(sinks)
        self.sessions: dict[str, Session] = {}

    def _get(self, session_id: str) -> Session:
        try:
            return self.sessions[session_id]
        except KeyError:
            raise InvalidCommand(f"unknown session {session_id!r}") from None

    async def create_session(self, command: CreateSessionCommand) -> Session:
        """Create a session and emit ``StatusUpdate(created)`` carrying its id."""
        session_id = f"ses_{uuid.uuid4().hex[:16]}"
        events = EventBus(session_id)
        for sink in self._sinks:
            events.subscribe(sink)
        approvals = ApprovalGate()
        session = Session(session_id, self._factory(command, events, approvals), approvals)
        self.sessions[session_id] = session
        await events.status("created", command_id=command.command_id)
        return session

    async def send(self, command: SendCommand) -> Session:
        """Start a run in the session with the command's text."""
        session = self._get(command.session_id)
        await session.send(command.text)
        return session

    def approve(self, command: ApproveCommand) -> None:
        """Answer a pending approval."""
        session = self._get(command.session_id)
        session.approvals.resolve(command.approval_id, command.decision, command.reason)

    def cancel(self, command: CancelCommand) -> None:
        """Cancel the session's active run."""
        self._get(command.session_id).cancel()

    def resume(self, command: ResumeCommand) -> list[AnyEvent]:
        """Events the client missed (``seq > after_seq``)."""
        return self._get(command.session_id).events.replay(command.after_seq)
