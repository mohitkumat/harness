import asyncio
import json
from pathlib import Path

import pytest

from harness_evals.agents import AgentOutcome, NullAgent
from harness_evals.cli import main
from harness_evals.runner import run_suite, run_task
from harness_evals.tasks import Task

TASK = Task.model_validate(
    {
        "id": "write-hello",
        "title": "t",
        "prompt": "write hello",
        "timeout_s": 0.5,
        "setup": {"files": {"in.txt": "seed"}},
        "checks": [
            {"type": "file_equals", "path": "out.txt", "text": "hello"},
            {"type": "answer_contains", "text": "done"},
        ],
    }
)


class SolvingAgent:
    name = "solver"

    async def run(self, prompt: str, workspace: Path, *, max_steps: int) -> AgentOutcome:
        assert (workspace / "in.txt").read_text() == "seed"
        (workspace / "out.txt").write_text("hello")
        return AgentOutcome(final_answer="done", steps=1)


class SlowAgent:
    name = "slow"

    async def run(self, prompt: str, workspace: Path, *, max_steps: int) -> AgentOutcome:
        await asyncio.sleep(5)
        return AgentOutcome(final_answer="done")


class CancelSwallowingAgent:
    """Like the real harness agent: catches the cancel and returns a "cancelled" outcome."""

    name = "swallow"

    async def run(self, prompt: str, workspace: Path, *, max_steps: int) -> AgentOutcome:
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            if (task := asyncio.current_task()) is not None:
                task.uncancel()
            return AgentOutcome(final_answer="partial", steps=3, error="run ended with cancelled")
        return AgentOutcome(final_answer="done")


class CrashingAgent:
    name = "crash"

    async def run(self, prompt: str, workspace: Path, *, max_steps: int) -> AgentOutcome:
        raise RuntimeError("kaboom")


async def test_solving_agent_passes() -> None:
    result = await run_task(SolvingAgent(), TASK)
    assert result.passed, result
    assert result.steps == 1


async def test_null_agent_fails() -> None:
    report = await run_suite(NullAgent(), [TASK])
    assert report.passed == 0
    assert report.pass_rate == 0.0
    assert report.results[0].error


@pytest.mark.parametrize(
    ("agent", "fragment"),
    [
        (SlowAgent(), "timed out"),
        (CancelSwallowingAgent(), "timed out"),
        (CrashingAgent(), "kaboom"),
    ],
)
async def test_agent_failures_fail_task_not_suite(
    agent: SlowAgent | CancelSwallowingAgent | CrashingAgent, fragment: str
) -> None:
    result = await run_task(agent, TASK)
    assert not result.passed
    assert result.error is not None
    assert fragment in result.error


async def test_timeout_is_reported_even_if_agent_swallows_cancel() -> None:
    """Regression for #5: the timeout used to be reported as the agent's "cancelled"."""
    result = await run_task(CancelSwallowingAgent(), TASK)

    assert result.error == "timed out after 0.5s"
    assert result.steps == 3  # what the agent did before the deadline is kept
    assert not result.passed


async def test_agent_error_before_deadline_is_kept() -> None:
    class FailingFast:
        name = "fail-fast"

        async def run(self, prompt: str, workspace: Path, *, max_steps: int) -> AgentOutcome:
            return AgentOutcome(final_answer=None, error="run ended with max_steps")

    result = await run_task(FailingFast(), TASK)

    assert result.error == "run ended with max_steps"


def test_cli_writes_report(tmp_path: Path) -> None:
    tasks_dir = Path(__file__).resolve().parents[2] / "tasks"
    code = main(["--tasks-dir", str(tasks_dir), "--tag", "fast", "--out-dir", str(tmp_path)])
    assert code == 0
    (report_file,) = tmp_path.glob("*.json")
    report = json.loads(report_file.read_text())
    assert report["agent"] == "null"
    assert report["total"] > 0
    assert report["passed"] == 0
    assert (
        main(["--tasks-dir", str(tasks_dir), "--out-dir", str(tmp_path), "--fail-under", "0.5"])
        == 1
    )
