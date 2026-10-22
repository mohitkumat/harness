"""The interface between the eval runner and the agent being evaluated."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from harness.core import Usage
from harness.core.protocol import CreateSessionCommand, SendCommand
from harness.guardrails import Headless
from harness.observability import JsonlEventLog
from harness_cli.config import Settings
from harness_cli.runtime import build_harness, limits_from, new_command_id


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


@dataclass(frozen=True)
class AgentOptions:
    """Command-line options passed to agent factories."""

    model: str | None = None
    allow_host_shell: bool = False
    log_dir: Path | None = None


class NullAgent:
    """Does nothing, so every task fails. Proves the runner, task files and checks work end to end."""

    name = "null"

    async def run(self, prompt: str, workspace: Path, *, max_steps: int) -> AgentOutcome:
        """Return an empty outcome without touching the workspace."""
        return AgentOutcome(final_answer=None, error="null agent")


class HarnessAgent:
    """The real agent, wired exactly like the CLI (``harness_cli.runtime.build_harness``).

    Nobody can answer approvals during an eval, so risky tools are denied unless
    ``allow_host_shell`` is set. Setting it runs model-written shell commands **on this machine**
    inside the task's temp workspace (ADR 0003). Never set it in CI.
    """

    def __init__(self, options: AgentOptions) -> None:
        overrides = {"model": options.model} if options.model else {}
        self.settings = Settings(**overrides)  # pyright: ignore[reportArgumentType]
        self.options = options
        self.name = f"harness:{self.settings.model}"

    async def run(self, prompt: str, workspace: Path, *, max_steps: int) -> AgentOutcome:
        """Run one session in ``workspace`` until the agent finishes or hits a limit."""
        sinks = [JsonlEventLog(self.options.log_dir)] if self.options.log_dir else []
        harness = build_harness(
            self.settings,
            policy=Headless(allow_risky=self.options.allow_host_shell),
            sinks=sinks,
            limits=limits_from(self.settings, max_steps=max_steps),
        )
        session = await harness.create_session(
            CreateSessionCommand(
                type="create_session", command_id=new_command_id(), workspace=str(workspace)
            )
        )
        await harness.send(
            SendCommand(
                type="send", command_id=new_command_id(), session_id=session.id, text=prompt
            )
        )
        result = await session.wait()
        error = None
        if result.reason != "completed":
            detail = f": {result.error.message}" if result.error else ""
            error = f"run ended with {result.reason}{detail} (session {session.id})"
        return AgentOutcome(
            final_answer=result.final_text, steps=result.steps, usage=result.usage, error=error
        )


AgentFactory = Callable[[AgentOptions], AgentUnderTest]

AGENTS: dict[str, AgentFactory] = {
    "null": lambda options: NullAgent(),
    "harness": HarnessAgent,
}
