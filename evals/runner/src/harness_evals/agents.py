"""The interface between the eval runner and the agent being evaluated."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from harness.core import Usage


@dataclass(frozen=True)
class AgentOutcome:
    """What an agent produced for one task."""

    final_answer: str | None
    steps: int = 0
    usage: Usage = field(default_factory=Usage)
    error: str | None = None


class AgentUnderTest(Protocol):
    """Anything that can attempt a task inside a prepared workspace."""

    name: str

    async def run(self, prompt: str, workspace: Path, *, max_steps: int) -> AgentOutcome:
        """Attempt the task. ``workspace`` is a throwaway directory seeded with the task's files."""
        ...


class NullAgent:
    """Placeholder until the agent loop exists: does nothing, so every task fails.

    It proves the runner, task files and checks work end to end.
    """

    name = "null"

    async def run(self, prompt: str, workspace: Path, *, max_steps: int) -> AgentOutcome:
        """Return an empty outcome without touching the workspace."""
        # TODO(phase-1): replace with an agent that drives harness.core's loop via the CLI config.
        return AgentOutcome(final_answer=None, error="no agent available yet (phase 0)")


AGENTS: dict[str, type[AgentUnderTest]] = {"null": NullAgent}
