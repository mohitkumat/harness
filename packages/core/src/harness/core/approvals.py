"""Pending approvals: the loop waits on them, ``approve`` commands resolve them."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from typing import Literal

from harness.core.errors import InvalidCommand

Decision = Literal["approve", "deny"]


@dataclass(frozen=True)
class ApprovalAnswer:
    """The user's answer to one ``ApprovalRequest``."""

    decision: Decision
    reason: str | None = None


class ApprovalGate:
    """Tracks approval requests that are waiting for an answer."""

    def __init__(self) -> None:
        self._pending: dict[str, asyncio.Future[ApprovalAnswer]] = {}

    def open(self) -> str:
        """Create a pending approval and return its id."""
        approval_id = f"apr_{uuid.uuid4().hex[:12]}"
        self._pending[approval_id] = asyncio.get_running_loop().create_future()
        return approval_id

    async def wait(self, approval_id: str) -> ApprovalAnswer:
        """Wait for the answer to ``approval_id``."""
        try:
            return await self._pending[approval_id]
        finally:
            self._pending.pop(approval_id, None)

    def resolve(self, approval_id: str, decision: Decision, reason: str | None = None) -> None:
        """Answer a pending approval. Raises ``InvalidCommand`` for unknown or answered ids."""
        future = self._pending.get(approval_id)
        if future is None or future.done():
            raise InvalidCommand(f"no pending approval {approval_id!r}")
        future.set_result(ApprovalAnswer(decision, reason))

    @property
    def pending(self) -> list[str]:
        """Ids still waiting for an answer."""
        return [k for k, f in self._pending.items() if not f.done()]
