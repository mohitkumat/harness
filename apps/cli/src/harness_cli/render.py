"""Renders protocol events to the terminal with Rich.

Model text, tool arguments and tool output are untrusted, so they are printed without Rich markup
and with terminal control characters removed (no ANSI escape injection).
"""

from __future__ import annotations

import json
import re

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from harness.core.protocol import (
    AnyEvent,
    ApprovalRequest,
    Done,
    Error,
    StatusUpdate,
    TextDelta,
    ToolEnd,
    ToolStart,
)

_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")
_PREVIEW_LINES = 6
_ARGS_CHARS = 200


def sanitize(text: str) -> str:
    """Drop terminal control characters (keeps newlines and tabs)."""
    return _CONTROL.sub("", text)


def _args(arguments: dict[str, object]) -> str:
    raw = json.dumps(arguments, ensure_ascii=False)
    return raw if len(raw) <= _ARGS_CHARS else raw[:_ARGS_CHARS] + "…"


class Renderer:
    """Turns the event stream into readable terminal output."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self._mid_line = False

    def _newline(self) -> None:
        if self._mid_line:
            self.console.out("")
            self._mid_line = False

    def render(self, event: AnyEvent) -> None:
        """Print one event."""
        match event:
            case TextDelta(text=text):
                clean = sanitize(text)
                self.console.out(clean, end="", highlight=False)
                self._mid_line = not clean.endswith("\n")
            case ToolStart(tool_name=name, arguments=arguments):
                self._newline()
                self.console.print(
                    Text.assemble(
                        ("→ ", "cyan"),
                        (sanitize(name), "bold cyan"),
                        " ",
                        (sanitize(_args(arguments)), "dim"),
                    )
                )
            case ToolEnd(
                tool_name=name,
                is_error=is_error,
                output=output,
                duration_ms=ms,
                truncated=truncated,
            ):
                mark = ("✗ ", "red") if is_error else ("✓ ", "green")
                note = " (truncated)" if truncated else ""
                self.console.print(
                    Text.assemble(mark, (sanitize(name), "bold"), (f" {ms} ms{note}", "dim"))
                )
                lines = sanitize(output).splitlines()
                preview = lines[:_PREVIEW_LINES]
                if len(lines) > _PREVIEW_LINES:
                    preview.append(f"… {len(lines) - _PREVIEW_LINES} more lines")
                for line in preview:
                    self.console.print(Text("  " + line, style="dim"))
            case ApprovalRequest():
                self._newline()
                self.console.print(self.approval_panel(event))
            case Error(code=code, message=message):
                self._newline()
                self.console.print(Text(f"error [{code}]: {sanitize(message)}", style="bold red"))
            case Done():
                self._newline()
                self.console.print(self.summary(event))
            case StatusUpdate():
                pass  # TODO(phase-6): richer status display (spinner, step counter)

    @staticmethod
    def approval_panel(event: ApprovalRequest) -> Panel:
        """Build the box shown while a risky tool call waits for the user."""
        body = Text.assemble(
            (sanitize(event.tool_name), "bold"),
            "\n",
            (sanitize(json.dumps(event.arguments, indent=2, ensure_ascii=False)), ""),
            "\n\n",
            (sanitize(event.reason), "dim"),
        )
        return Panel(body, title="[yellow]Approval needed[/yellow]", border_style="yellow")

    @staticmethod
    def summary(event: Done) -> Text:
        """Describe in one line how a run ended."""
        u = event.usage
        cost = f" · ${u.cost_usd:.4f}" if u.cost_usd is not None else ""
        style = "green" if event.reason == "completed" else "yellow"
        return Text(
            f"[{escape(event.reason)}] {event.steps} step(s) · "
            f"{u.input_tokens:,} in / {u.output_tokens:,} out tokens{cost}",
            style=style,
        )
