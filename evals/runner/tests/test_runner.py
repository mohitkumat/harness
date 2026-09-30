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
    ("agent", "fragment"), [(SlowAgent(), "timed out"), (CrashingAgent(), "kaboom")]
)
async def test_agent_failures_fail_task_not_suite(
    agent: SlowAgent | CrashingAgent, fragment: str
) -> None:
    result = await run_task(agent, TASK)
    assert not result.passed
    assert result.error is not None
    assert fragment in result.error


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
