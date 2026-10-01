from pathlib import Path

import pytest

from harness.core import AgentLoop, EventBus, Message
from harness.core.protocol import Done, parse_event
from harness.core.testing import EchoTools, Reply, RiskyAsks, ScriptedAdapter
from harness.observability import JsonlEventLog


async def test_every_step_is_logged_in_order(tmp_path: Path) -> None:
    log = JsonlEventLog(tmp_path / "logs")
    bus = EventBus("ses_log")
    bus.subscribe(log)
    loop = AgentLoop(
        adapter=ScriptedAdapter(
            [Reply("checking", calls=[("echo", {"text": "a"})]), Reply("done")]
        ),
        tools=EchoTools(),
        policy=RiskyAsks(),
        events=bus,
    )

    await loop.run([Message.user("go")])

    lines = log.path_for("ses_log").read_text().splitlines()
    events = [parse_event(line) for line in lines]
    assert [e.seq for e in events] == list(range(len(events)))
    assert {e.type for e in events} >= {
        "status_update",
        "text_delta",
        "tool_start",
        "tool_end",
        "done",
    }
    assert isinstance(events[-2], Done)
    assert events == bus.replay()


async def test_sessionless_event_rejected(tmp_path: Path) -> None:
    bus = EventBus("ses_x")
    log = JsonlEventLog(tmp_path)
    await bus.status("idle")
    event = bus.replay()[0].model_copy(update={"session_id": None})
    with pytest.raises(ValueError, match="session"):
        await log(event)
