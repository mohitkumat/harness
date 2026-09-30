"""Runs tasks against an agent in isolated temporary workspaces and builds a report."""

from __future__ import annotations

import asyncio
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harness_evals.agents import AgentOutcome, AgentUnderTest
from harness_evals.checks import CheckResult, run_check
from harness_evals.tasks import Task


@dataclass(frozen=True)
class TaskResult:
    """Outcome of one task: passed only if the agent finished without error and all checks pass."""

    task_id: str
    passed: bool
    checks: list[CheckResult]
    steps: int
    duration_s: float
    input_tokens: int
    output_tokens: int
    error: str | None


@dataclass(frozen=True)
class Report:
    """Results for a whole eval run."""

    agent: str
    started_at: str
    results: list[TaskResult]

    @property
    def passed(self) -> int:
        """Number of passing tasks."""
        return sum(r.passed for r in self.results)

    @property
    def pass_rate(self) -> float:
        """Fraction of tasks passed (0.0 for an empty run)."""
        return self.passed / len(self.results) if self.results else 0.0

    def to_json_dict(self) -> dict[str, Any]:  # Any: heterogeneous JSON document for json.dumps
        """Return the serialisable form written to ``evals/results/``."""
        return {
            "agent": self.agent,
            "started_at": self.started_at,
            "passed": self.passed,
            "total": len(self.results),
            "pass_rate": self.pass_rate,
            "results": [asdict(r) for r in self.results],
        }


def _seed(workspace: Path, task: Task) -> None:
    for rel, content in task.setup.files.items():
        path = workspace / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)


async def run_task(agent: AgentUnderTest, task: Task) -> TaskResult:
    """Run one task in a fresh temporary workspace and score it."""
    with tempfile.TemporaryDirectory(prefix=f"eval-{task.id}-") as tmp:
        workspace = Path(tmp)
        _seed(workspace, task)
        start = time.monotonic()
        try:
            outcome = await asyncio.wait_for(
                agent.run(task.prompt, workspace, max_steps=task.max_steps), task.timeout_s
            )
        except TimeoutError:
            outcome = AgentOutcome(final_answer=None, error=f"timed out after {task.timeout_s}s")
        except Exception as e:  # an agent crash fails the task, it must not abort the suite
            outcome = AgentOutcome(final_answer=None, error=f"{type(e).__name__}: {e}")
        duration = time.monotonic() - start
        checks = [run_check(c, task, workspace, outcome.final_answer) for c in task.checks]

    return TaskResult(
        task_id=task.id,
        passed=outcome.error is None and all(c.passed for c in checks),
        checks=checks,
        steps=outcome.steps,
        duration_s=round(duration, 3),
        input_tokens=outcome.usage.input_tokens,
        output_tokens=outcome.usage.output_tokens,
        error=outcome.error,
    )


async def run_suite(agent: AgentUnderTest, tasks: list[Task]) -> Report:
    """Run tasks sequentially (parallelism is a later-phase optimisation)."""
    started = datetime.now(UTC).isoformat(timespec="seconds")
    results = [await run_task(agent, t) for t in tasks]
    return Report(agent=agent.name, started_at=started, results=results)
