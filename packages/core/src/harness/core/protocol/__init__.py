"""Pydantic models for the wire protocol, generated from ``protocol/protocol.schema.json``.

Never edit ``_generated.py`` by hand: change the schema and run ``just gen``.
"""

from __future__ import annotations

from pydantic import JsonValue, TypeAdapter

from harness.core.protocol._generated import (
    ApprovalRequest,
    ApproveCommand,
    CancelCommand,
    Command,
    CreateSessionCommand,
    Done,
    Error,
    Event,
    ResumeCommand,
    SendCommand,
    StatusUpdate,
    TextDelta,
    ToolEnd,
    ToolStart,
    Usage,
)

PROTOCOL_VERSION = 1
"""Must match ``protocol/VERSION``. Bumping it requires an ADR."""

AnyCommand = CreateSessionCommand | SendCommand | ApproveCommand | CancelCommand | ResumeCommand
AnyEvent = TextDelta | ToolStart | ToolEnd | ApprovalRequest | StatusUpdate | Done | Error

_commands: TypeAdapter[Command] = TypeAdapter(Command)
_events: TypeAdapter[Event] = TypeAdapter(Event)


def parse_command(data: str | bytes | dict[str, JsonValue]) -> AnyCommand:
    """Parse a command from JSON text or a decoded object, dispatching on ``type``."""
    if isinstance(data, dict):
        return _commands.validate_python(data).root
    return _commands.validate_json(data).root


def parse_event(data: str | bytes | dict[str, JsonValue]) -> AnyEvent:
    """Parse an event from JSON text or a decoded object, dispatching on ``type``."""
    if isinstance(data, dict):
        return _events.validate_python(data).root
    return _events.validate_json(data).root


__all__ = [
    "PROTOCOL_VERSION",
    "AnyCommand",
    "AnyEvent",
    "ApprovalRequest",
    "ApproveCommand",
    "CancelCommand",
    "Command",
    "CreateSessionCommand",
    "Done",
    "Error",
    "Event",
    "ResumeCommand",
    "SendCommand",
    "StatusUpdate",
    "TextDelta",
    "ToolEnd",
    "ToolStart",
    "Usage",
    "parse_command",
    "parse_event",
]
