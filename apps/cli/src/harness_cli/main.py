"""Agent harness CLI: `run` one task or `chat` interactively.

The CLI only sends protocol commands to an in-process Harness and renders the events it gets back.
Risky tools ask for approval on the terminal; Ctrl-C cancels the active run.
"""

from __future__ import annotations

import asyncio
import contextlib
import signal
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.prompt import Confirm, Prompt

from harness.core import ApprovalPolicy, Harness, RunResult, Session
from harness.core.approvals import Decision
from harness.core.protocol import (
    AnyEvent,
    ApprovalRequest,
    ApproveCommand,
    CancelCommand,
    CreateSessionCommand,
    Done,
    SendCommand,
)
from harness.guardrails import AskForRisky, Headless
from harness.observability import JsonlEventLog
from harness_cli.config import Settings
from harness_cli.render import Renderer, sanitize
from harness_cli.runtime import build_harness, new_command_id

app = typer.Typer(add_completion=False, no_args_is_help=True, help=(__doc__ or "").strip())

Approver = Callable[[ApprovalRequest], Awaitable[tuple[Decision, str | None]]]
EventHandler = Callable[[AnyEvent], None]


async def run_turn(
    harness: Harness, session: Session, text: str, on_event: EventHandler, approve: Approver
) -> RunResult:
    """Send one user message and drive the run to ``Done``, answering approvals via ``approve``."""
    events = session.events.stream()
    async with contextlib.aclosing(events):
        await harness.send(
            SendCommand(type="send", command_id=new_command_id(), session_id=session.id, text=text)
        )
        async for event in events:
            on_event(event)
            if isinstance(event, ApprovalRequest):
                decision, reason = await approve(event)
                harness.approve(
                    ApproveCommand(
                        type="approve",
                        command_id=new_command_id(),
                        session_id=session.id,
                        approval_id=event.approval_id,
                        decision=decision,
                        reason=reason,
                    )
                )
            elif isinstance(event, Done):
                break
    return await session.wait()


def terminal_approver(console: Console) -> Approver:
    """Ask on the terminal. Anything but an explicit yes is a deny."""

    async def approve(event: ApprovalRequest) -> tuple[Decision, str | None]:
        allowed = await asyncio.to_thread(
            Confirm.ask,
            f"Allow [bold]{escape(sanitize(event.tool_name))}[/bold]?",
            console=console,
            default=False,
        )
        return ("approve", None) if allowed else ("deny", "the user declined")

    return approve


async def _no_approvals(event: ApprovalRequest) -> tuple[Decision, str | None]:
    return "deny", "non-interactive run"


def _install_cancel(harness: Harness, session: Session, console: Console) -> None:
    """First Ctrl-C cancels the active run; with no run active it only prints a hint."""

    def on_sigint() -> None:
        if session.running:
            console.print("\n[yellow]cancelling…[/yellow]")
            harness.cancel(
                CancelCommand(type="cancel", command_id=new_command_id(), session_id=session.id)
            )
        else:
            console.print("\n[dim](type /exit or press Ctrl-D to quit)[/dim]")

    with contextlib.suppress(NotImplementedError):  # e.g. Windows event loops
        asyncio.get_running_loop().add_signal_handler(signal.SIGINT, on_sigint)


async def _open(
    settings: Settings, workspace: Path, policy: ApprovalPolicy, console: Console
) -> tuple[Harness, Session, JsonlEventLog]:
    log = JsonlEventLog(settings.log_dir)
    harness = build_harness(settings, policy=policy, sinks=[log])
    session = await harness.create_session(
        CreateSessionCommand(
            type="create_session", command_id=new_command_id(), workspace=str(workspace)
        )
    )
    _install_cancel(harness, session, console)
    return harness, session, log


WorkspaceOpt = Annotated[
    Path,
    typer.Option(
        "--workspace", "-w", help="Directory the tools operate on.", file_okay=False, exists=True
    ),
]
ModelOpt = Annotated[
    str | None, typer.Option("--model", "-m", help="Model id (default from HARNESS_MODEL).")
]


@app.command()
def run(
    task: Annotated[str, typer.Argument(help="What the agent should do.")],
    workspace: WorkspaceOpt = Path(),
    model: ModelOpt = None,
    max_steps: Annotated[int | None, typer.Option(help="Step limit for this run.")] = None,
    json_events: Annotated[
        bool, typer.Option("--json", help="Print raw JSONL events; risky tools are denied.")
    ] = False,
) -> None:
    """Run one task to completion. Exits 0 only if the run completed."""
    overrides = {k: v for k, v in {"model": model, "max_steps": max_steps}.items() if v is not None}
    settings = Settings(**overrides)  # pyright: ignore[reportArgumentType]
    console = Console(stderr=json_events)
    workspace = workspace.resolve()

    async def main() -> RunResult:
        policy: ApprovalPolicy = Headless() if json_events else AskForRisky()
        harness, session, log = await _open(settings, workspace, policy, console)
        if json_events:
            on_event: EventHandler = lambda e: print(e.model_dump_json(), flush=True)  # noqa: E731
            approve = _no_approvals
        else:
            on_event, approve = Renderer(console).render, terminal_approver(console)
        result = await run_turn(harness, session, task, on_event, approve)
        console.print(f"[dim]event log: {log.path_for(session.id)}[/dim]")
        return result

    result = asyncio.run(main())
    raise typer.Exit(0 if result.reason == "completed" else 1)


@app.command()
def chat(workspace: WorkspaceOpt = Path(), model: ModelOpt = None) -> None:
    """Interactive session. Ctrl-C cancels the current run; /exit or Ctrl-D quits."""
    settings = Settings(**({"model": model} if model else {}))  # pyright: ignore[reportArgumentType]
    console = Console()
    workspace = workspace.resolve()

    async def main() -> None:
        harness, session, log = await _open(settings, workspace, AskForRisky(), console)
        renderer, approve = Renderer(console), terminal_approver(console)
        console.print(f"[dim]session {session.id} · workspace {workspace} · /exit to quit[/dim]")
        while True:
            try:
                text = await asyncio.to_thread(Prompt.ask, "[bold]>[/bold]", console=console)
            except EOFError:
                break
            if text.strip() in {"/exit", "/quit"}:
                break
            if text.strip():
                await run_turn(harness, session, text, renderer.render, approve)
        console.print(f"[dim]event log: {log.path_for(session.id)}[/dim]")

    asyncio.run(main())


if __name__ == "__main__":
    sys.exit(app())
