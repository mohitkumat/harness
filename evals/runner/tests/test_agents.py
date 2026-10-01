from pathlib import Path

import pytest

import harness_cli.runtime as runtime
from harness.core import ModelAdapter
from harness.core.testing import Reply, ScriptedAdapter
from harness_cli.config import Settings
from harness_evals.agents import AGENTS, AgentOptions
from harness_evals.runner import run_task
from harness_evals.tasks import load_task

TASKS = Path(__file__).resolve().parents[2] / "tasks"


def use_model(monkeypatch: pytest.MonkeyPatch, *replies: Reply) -> None:
    adapter = ScriptedAdapter(list(replies))

    def fake(settings: Settings, model: str | None = None) -> ModelAdapter:
        return adapter

    monkeypatch.setattr(runtime, "build_adapter", fake)


async def test_harness_agent_passes_write_hello(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_model(
        monkeypatch,
        Reply(calls=[("write_file", {"path": "hello.txt", "content": "Hello, harness!"})]),
        Reply("Created hello.txt."),
    )
    agent = AGENTS["harness"](AgentOptions(log_dir=tmp_path / "sessions"))

    result = await run_task(agent, load_task(TASKS / "write-hello.toml"))

    assert result.passed, result
    assert result.steps == 2
    assert list((tmp_path / "sessions").glob("*.jsonl"))


@pytest.mark.parametrize(("allow", "passes"), [(False, False), (True, True)])
async def test_shell_needs_explicit_opt_in(
    monkeypatch: pytest.MonkeyPatch, allow: bool, passes: bool
) -> None:
    use_model(
        monkeypatch,
        Reply(calls=[("run_shell", {"command": "mv notes_old.md notes.md"})]),
        Reply("Renamed."),
    )
    agent = AGENTS["harness"](AgentOptions(allow_host_shell=allow))

    result = await run_task(agent, load_task(TASKS / "rename-file.toml"))

    assert result.passed is passes


async def test_limit_is_reported_as_error(monkeypatch: pytest.MonkeyPatch) -> None:
    use_model(monkeypatch, *[Reply(calls=[("list_dir", {})]) for _ in range(5)])
    agent = AGENTS["harness"](AgentOptions())
    task = load_task(TASKS / "write-hello.toml").model_copy(update={"max_steps": 2})

    result = await run_task(agent, task)

    assert not result.passed
    assert result.error is not None
    assert "max_steps" in result.error
