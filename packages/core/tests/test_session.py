import asyncio
from collections.abc import AsyncIterator

import pytest

from harness.core import (
    AgentLoop,
    ApprovalGate,
    EventBus,
    Harness,
    InvalidCommand,
    ModelError,
    collect_stream,
)
from harness.core.adapter import Delta, StopDelta, TextDelta, ToolCallArgsDelta, ToolCallStartDelta
from harness.core.protocol import (
    AnyEvent,
    ApprovalRequest,
    ApproveCommand,
    CancelCommand,
    CreateSessionCommand,
    Done,
    ResumeCommand,
    SendCommand,
    StatusUpdate,
    parse_event,
)
from harness.core.testing import EchoTools, Reply, RiskyAsks, ScriptedAdapter
from harness.core.types import Message, ReasoningPart, StopReason, TextPart, ToolCall, Usage


def harness_with(
    replies: list[Reply], sink_events: list[AnyEvent], approval_seen: asyncio.Event | None = None
) -> Harness:
    def factory(
        command: CreateSessionCommand, events: EventBus, approvals: ApprovalGate
    ) -> AgentLoop:
        return AgentLoop(
            adapter=ScriptedAdapter(replies),
            tools=EchoTools(delay_s=0.01),
            policy=RiskyAsks(),
            events=events,
            approvals=approvals,
            system=command.system_prompt,
        )

    async def sink(event: AnyEvent) -> None:
        sink_events.append(event)
        if approval_seen is not None and isinstance(event, ApprovalRequest):
            approval_seen.set()

    return Harness(factory, sinks=[sink])


async def test_create_send_approve_done() -> None:
    seen: list[AnyEvent] = []
    approval_seen = asyncio.Event()
    harness = harness_with(
        [Reply(calls=[("danger", {"text": "x"})]), Reply("ok")], seen, approval_seen
    )
    session = await harness.create_session(
        CreateSessionCommand(type="create_session", command_id="c1")
    )
    created = seen[0]
    assert isinstance(created, StatusUpdate)
    assert created.status == "created"
    assert created.command_id == "c1"
    assert created.session_id == session.id

    await harness.send(SendCommand(type="send", command_id="c2", session_id=session.id, text="go"))
    await approval_seen.wait()
    req = next(e for e in seen if isinstance(e, ApprovalRequest))
    harness.approve(
        ApproveCommand(
            type="approve",
            command_id="c3",
            session_id=session.id,
            approval_id=req.approval_id,
            decision="approve",
        )
    )
    result = await session.wait()

    assert result.reason == "completed"
    assert isinstance(seen[-2], Done)
    # Every event serialises to valid protocol JSON.
    for e in seen:
        assert parse_event(e.model_dump_json()) == e


async def test_cancel_immediately_after_send_still_emits_done() -> None:
    seen: list[AnyEvent] = []
    harness = harness_with([Reply("slow", delay_s=1)], seen)
    session = await harness.create_session(
        CreateSessionCommand(type="create_session", command_id="c1")
    )
    await harness.send(SendCommand(type="send", command_id="c2", session_id=session.id, text="go"))
    harness.cancel(CancelCommand(type="cancel", command_id="c3", session_id=session.id))

    assert (await session.wait()).reason == "cancelled"
    assert [e.reason for e in seen if isinstance(e, Done)] == ["cancelled"]


async def test_second_send_while_running_is_rejected() -> None:
    harness = harness_with([Reply("slow", delay_s=0.2)], [])
    session = await harness.create_session(
        CreateSessionCommand(type="create_session", command_id="c1")
    )
    await session.send("one")
    with pytest.raises(InvalidCommand):
        await session.send("two")
    await session.wait()


async def test_unknown_session_and_approval() -> None:
    harness = harness_with([], [])
    with pytest.raises(InvalidCommand):
        await harness.send(SendCommand(type="send", command_id="c", session_id="nope", text="x"))
    session = await harness.create_session(
        CreateSessionCommand(type="create_session", command_id="c1")
    )
    with pytest.raises(InvalidCommand):
        harness.approve(
            ApproveCommand(
                type="approve",
                command_id="c",
                session_id=session.id,
                approval_id="apr_missing",
                decision="deny",
            )
        )


async def test_resume_replays_after_seq() -> None:
    harness = harness_with([Reply("hi")], [])
    session = await harness.create_session(
        CreateSessionCommand(type="create_session", command_id="c1")
    )
    await session.send("hello")
    await session.wait()

    events = harness.resume(
        ResumeCommand(type="resume", command_id="r", session_id=session.id, after_seq=2)
    )
    assert events
    assert [e.seq for e in events] == list(range(3, 3 + len(events)))


async def test_bus_stream_yields_live_events() -> None:
    bus = EventBus("ses_x")
    received: list[AnyEvent] = []

    async def consume() -> None:
        async for event in bus.stream():
            received.append(event)
            if isinstance(event, StatusUpdate) and event.status == "idle":
                return

    task = asyncio.create_task(consume())
    await asyncio.sleep(0)
    await bus.status("thinking")
    await bus.status("idle")
    await task
    assert [e.seq for e in received] == [0, 1]


# ----------------------------------------------------------------------------- collect_stream


async def _deltas(*items: Delta) -> AsyncIterator[Delta]:
    for item in items:
        yield item


async def test_collect_stream_preserves_order_and_merges_text() -> None:
    stop = StopDelta(stop_reason=StopReason.TOOL_USE, usage=Usage(input_tokens=1), model="m")
    response = await collect_stream(
        _deltas(
            TextDelta(text="a"),
            TextDelta(text="b"),
            ToolCallStartDelta(index=0, id="t1", name="echo"),
            ToolCallArgsDelta(index=0, json_fragment='{"te'),
            ToolCallArgsDelta(index=0, json_fragment='xt": 1}'),
            ToolCallStartDelta(index=1, id="t2", name="noargs"),
            stop,
        )
    )
    assert response.message.text == "ab"
    assert [(c.name, c.arguments) for c in response.message.tool_calls] == [
        ("echo", {"text": 1}),
        ("noargs", {}),
    ]
    assert response.stop_reason == StopReason.TOOL_USE


@pytest.mark.parametrize(
    "items",
    [
        [TextDelta(text="no stop")],
        [
            ToolCallStartDelta(index=0, id="t", name="x"),
            ToolCallArgsDelta(index=0, json_fragment="{bad"),
            StopDelta(stop_reason=StopReason.TOOL_USE, usage=Usage(), model="m"),
        ],
        [
            ToolCallStartDelta(index=0, id="t", name="x"),
            ToolCallArgsDelta(index=0, json_fragment="[1]"),
            StopDelta(stop_reason=StopReason.TOOL_USE, usage=Usage(), model="m"),
        ],
    ],
)
async def test_collect_stream_errors(items: list[Delta]) -> None:
    with pytest.raises(ModelError):
        await collect_stream(_deltas(*items))


async def test_collect_stream_prefers_authoritative_message() -> None:
    final = Message(
        role="assistant",
        content=[
            ReasoningPart(source="p", data={"sig": "abc"}),
            TextPart(text="hi"),
            ToolCall(id="t1", name="echo", arguments={}),
        ],
    )
    response = await collect_stream(
        _deltas(
            TextDelta(text="hi"),
            StopDelta(stop_reason=StopReason.TOOL_USE, usage=Usage(), model="m", message=final),
        )
    )
    assert response.message == final


def test_reasoning_only_in_assistant_messages() -> None:
    part = ReasoningPart(source="p", data={})
    Message(role="assistant", content=[part])
    with pytest.raises(ValueError, match="not allowed"):
        Message(role="user", content=[part])
