"""JSONL event log: every protocol event of a session, one JSON object per line.

Subscribe ``JsonlEventLog`` to a session's ``EventBus`` (or pass it as a ``Harness`` sink). Each
line parses with ``harness.core.protocol.parse_event``.

The log holds tool outputs, which may include file contents. Treat it as session data with the
same access as the workspace, not as an info-level application log (CLAUDE.md §10).
"""

# TODO(phase-3): the session store becomes the system of record; keep this as an export format.

from __future__ import annotations

import asyncio
from pathlib import Path

from harness.core.protocol import AnyEvent


class JsonlEventLog:
    """Appends events to ``<directory>/<session_id>.jsonl``. Lines are written in ``seq`` order."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._lock = asyncio.Lock()

    def path_for(self, session_id: str) -> Path:
        """Where the log of ``session_id`` lives."""
        return self.directory / f"{session_id}.jsonl"

    async def __call__(self, event: AnyEvent) -> None:
        """Append one event. Errors propagate: a session must not run with a broken log."""
        if event.session_id is None:
            raise ValueError("JsonlEventLog only records session events")
        line = event.model_dump_json() + "\n"
        path = self.path_for(event.session_id)

        def append() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as f:
                f.write(line)

        async with self._lock:  # keep lines in emission order across concurrent sessions
            await asyncio.to_thread(append)
