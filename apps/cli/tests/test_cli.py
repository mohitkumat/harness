import io
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

import harness_cli.main as cli_main
import harness_cli.runtime as runtime
from harness.core import ModelAdapter
from harness.core.approvals import Decision
from harness.core.protocol import (
    AnyEvent,
    ApprovalRequest,
    CreateSessionCommand,
    Done,
    TextDelta,
    ToolEnd,
    parse_event,
)
from harness.core.testing import Reply, ScriptedAdapter
from harness.guardrails import AskForRisky
from harness_cli.config import Settings
from harness_cli.main import run_turn
from harness_cli.render import Renderer, sanitize
from harness_cli.runtime import build_harness, new_command_id


def scripted(*replies: Reply) -> Callable[[str | None], ModelAdapter]:
    adapter = ScriptedAdapter(list(replies))
    return lambda model: adapter


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(log_dir=tmp_path / "logs", anthropic_api_key=None)


async def test_run_turn_asks_and_runs_approved_shell(settings: Settings, tmp_path: Path) -> None:
    ws = tmp_path / "ws"
    ws.mkdir()
    harness = build_harness(
        settings,
        policy=AskForRisky(),
        adapter_factory=scripted(
            Reply(calls=[("write_file", {"path": "a.txt", "content": "hi"})]),
            Reply(calls=[("run_shell", {"command": "cat a.txt"})]),
            Reply("the file says hi"),
        ),
    )
    session = await harness.create_session(
        CreateSessionCommand(type="create_session", command_id=new_command_id(), workspace=str(ws))
    )
    seen: list[AnyEvent] = []
    asked: list[str] = []

    async def approve(event: ApprovalRequest) -> tuple[Decision, str | None]:
        asked.append(event.tool_name)
        return "approve", None

    result = await run_turn(harness, session, "go", seen.append, approve)

    assert result.reason == "completed"
    assert asked == ["run_shell"]  # write_file is not risky
    outputs = [e.output for e in seen if isinstance(e, ToolEnd)]
    assert outputs[1] == "exit code: 0\nhi"
    assert isinstance(seen[-1], Done)


async def test_denied_shell_never_runs(settings: Settings, tmp_path: Path) -> None:
    harness = build_harness(
        settings,
        policy=AskForRisky(),
        default_workspace=tmp_path,
        adapter_factory=scripted(
            Reply(calls=[("run_shell", {"command": "touch pwned"})]), Reply("ok")
        ),
    )
    session = await harness.create_session(
        CreateSessionCommand(type="create_session", command_id="c")
    )

    async def deny(event: ApprovalRequest) -> tuple[Decision, str | None]:
        return "deny", "no"

    await run_turn(harness, session, "go", lambda e: None, deny)

    assert not (tmp_path / "pwned").exists()


def test_renderer_strips_terminal_escapes() -> None:
    buf = io.StringIO()
    renderer = Renderer(Console(file=buf, force_terminal=True, width=100))
    renderer.render(
        TextDelta(
            type="text_delta",
            session_id="s",
            seq=0,
            ts=datetime(2026, 10, 1, tzinfo=UTC),
            step=0,
            text="\x1b]0;evil\x07hi [bold]x[/bold]\n",
        )  # pyright: ignore[reportArgumentType]
    )
    out = buf.getvalue()
    assert "\x1b" not in out  # without ESC/BEL the OSC sequence is inert text
    assert "\x07" not in out
    assert "[bold]x[/bold]" in out  # untrusted text is not interpreted as markup
    assert sanitize("a\x1bb\tc\n") == "ab\tc\n"


def test_cli_run_json(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HARNESS_LOG_DIR", str(tmp_path / "logs"))
    adapter = ScriptedAdapter([Reply(calls=[("list_dir", {})]), Reply("2 files")])

    def fake_build_adapter(settings: Settings, model: str | None = None) -> ModelAdapter:
        return adapter

    monkeypatch.setattr(runtime, "build_adapter", fake_build_adapter)
    (tmp_path / "x.txt").write_text("x")

    result = CliRunner().invoke(cli_main.app, ["run", "count files", "-w", str(tmp_path), "--json"])

    assert result.exit_code == 0, result.output
    events = [parse_event(line) for line in result.stdout.splitlines() if line.startswith("{")]
    assert isinstance(events[-1], Done)
    assert events[-1].final_text is not None
    assert "2 files" in events[-1].final_text
    log = next((tmp_path / "logs").glob("*.jsonl"))
    assert len(log.read_text().splitlines()) >= len(events)


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HARNESS_MODEL", "claude-sonnet-5-5")
    monkeypatch.setenv("HARNESS_MAX_STEPS", "7")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    s = Settings()
    assert s.model == "claude-sonnet-5-5"
    assert s.max_steps == 7
    assert s.anthropic_api_key is not None
    assert "sk-test" not in repr(s)
    assert json.loads(s.model_dump_json())["anthropic_api_key"] == "**********"
