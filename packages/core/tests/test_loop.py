import asyncio

import pytest

from harness.core import (
    AgentLoop,
    ApprovalGate,
    EventBus,
    Limits,
    Message,
    ModelError,
    ModelRequest,
    ModelResponse,
    PolicyDecision,
    Tool,
    ToolCall,
    Usage,
)
from harness.core.protocol import AnyEvent, ApprovalRequest, Done, Error, TextDelta, ToolEnd
from harness.core.testing import EchoTools, Reply, RiskyAsks, ScriptedAdapter


class Recorder:
    def __init__(self, bus: EventBus) -> None:
        self.events: list[AnyEvent] = []
        bus.subscribe(self.record)

    async def record(self, event: AnyEvent) -> None:
        self.events.append(event)

    def of[T](self, kind: type[T]) -> list[T]:
        return [e for e in self.events if isinstance(e, kind)]

    @property
    def done(self) -> Done:
        (done,) = self.of(Done)
        return done


def make_loop(
    replies: list[Reply],
    *,
    limits: Limits | None = None,
    streaming: bool = True,
    tools: EchoTools | None = None,
    approvals: ApprovalGate | None = None,
) -> tuple[AgentLoop, ScriptedAdapter, EchoTools, Recorder]:
    adapter = ScriptedAdapter(replies, streaming=streaming)
    tools = tools or EchoTools()
    bus = EventBus("ses_test")
    loop = AgentLoop(
        adapter=adapter,
        tools=tools,
        policy=RiskyAsks(),
        events=bus,
        approvals=approvals,
        system="be brief",
        limits=limits,
    )
    return loop, adapter, tools, Recorder(bus)


def assert_consistent(history: list[Message]) -> None:
    """Every assistant tool call is answered by the immediately following tool message."""
    for i, msg in enumerate(history):
        if msg.role == "assistant" and msg.tool_calls:
            nxt = history[i + 1]
            assert nxt.role == "tool"
            ids = {p.tool_call_id for p in nxt.content if p.type == "tool_result"}
            assert ids == {c.id for c in msg.tool_calls}


@pytest.mark.parametrize("streaming", [True, False])
async def test_plain_answer_completes(streaming: bool) -> None:
    loop, adapter, _, rec = make_loop([Reply(text="hello there")], streaming=streaming)
    history = [Message.user("hi")]

    result = await loop.run(history)

    assert result.reason == "completed"
    assert result.steps == 1
    assert result.final_text is not None
    assert result.final_text.strip() == "hello there"
    assert "".join(e.text for e in rec.of(TextDelta)).strip() == "hello there"
    assert rec.done.reason == "completed"
    assert adapter.requests[0].system == "be brief"
    assert [t.name for t in adapter.requests[0].tools] == ["echo", "danger"]
    assert [e.seq for e in rec.events] == list(range(len(rec.events)))


async def test_tool_call_round_trip() -> None:
    loop, adapter, tools, rec = make_loop(
        [Reply("checking", calls=[("echo", {"text": "a"}), ("echo", {"text": "b"})]), Reply("done")]
    )
    history = [Message.user("go")]

    result = await loop.run(history)

    assert result.reason == "completed"
    assert result.steps == 2
    assert [c.arguments for c in tools.calls] == [{"text": "a"}, {"text": "b"}]
    assert [m.role for m in history] == ["user", "assistant", "tool", "assistant"]
    assert_consistent(history)
    # The second model call saw the tool results.
    assert adapter.requests[1].messages[2].role == "tool"
    assert [e.output for e in rec.of(ToolEnd)] == ["a", "b"]
    assert result.usage == Usage(input_tokens=20, output_tokens=10)


async def test_max_steps() -> None:
    replies = [Reply(calls=[("echo", {"text": str(i)})]) for i in range(3)]
    loop, _, _, rec = make_loop(replies, limits=Limits(max_steps=2))
    history = [Message.user("loop forever")]

    result = await loop.run(history)

    assert result.reason == "max_steps"
    assert result.steps == 2
    assert rec.done.reason == "max_steps"
    assert_consistent(history)


async def test_token_limit_checked_before_each_call() -> None:
    big = Usage(input_tokens=60, output_tokens=0)
    replies = [Reply(calls=[("echo", {"text": "x"})], usage=big), Reply("never")]
    loop, adapter, _, rec = make_loop(replies, limits=Limits(max_tokens=50))

    result = await loop.run([Message.user("go")])

    assert result.reason == "max_tokens"
    assert len(adapter.requests) == 1
    assert rec.done.usage.input_tokens == 60


async def test_cost_limit() -> None:
    costly = Usage(input_tokens=1, output_tokens=1, cost_usd=0.5)
    replies = [Reply(calls=[("echo", {"text": "x"})], usage=costly), Reply("never")]
    loop, _, _, _ = make_loop(replies, limits=Limits(max_cost_usd=0.25))

    assert (await loop.run([Message.user("go")])).reason == "max_cost"


async def test_wall_clock_timeout_keeps_history_consistent() -> None:
    replies = [Reply(calls=[("echo", {"text": "slow"})])]
    loop, _, _, rec = make_loop(
        replies, limits=Limits(max_wall_clock_s=0.05), tools=EchoTools(delay_s=1)
    )
    history = [Message.user("go")]

    result = await loop.run(history)

    assert result.reason == "timeout"
    assert rec.done.reason == "timeout"
    assert_consistent(history)
    assert history[-1].content[0].type == "tool_result"


async def test_cancel_mid_tool() -> None:
    replies = [Reply(calls=[("echo", {"text": "a"}), ("echo", {"text": "b"})])]
    loop, _, _, rec = make_loop(replies, tools=EchoTools(delay_s=1))
    history = [Message.user("go")]

    task = asyncio.create_task(loop.run(history))
    await asyncio.sleep(0.05)
    task.cancel()
    result = await task

    assert result.reason == "cancelled"
    assert rec.done.reason == "cancelled"
    assert_consistent(history)


async def test_model_error_is_reported_and_ends_run() -> None:
    class Failing(ScriptedAdapter):
        async def complete(self, request: ModelRequest) -> ModelResponse:
            raise ModelError("overloaded", retryable=True, status_code=529)

    adapter = Failing([], streaming=False)
    bus = EventBus("ses_test")
    rec = Recorder(bus)
    loop = AgentLoop(
        adapter=adapter,
        tools=EchoTools(),
        policy=RiskyAsks(),
        events=bus,
        limits=Limits(max_model_retries=0),
    )

    result = await loop.run([Message.user("hi")])

    assert result.reason == "error"
    assert isinstance(result.error, ModelError)
    (err,) = rec.of(Error)
    assert err.code == "model_error"
    assert err.retryable
    assert err.fatal
    assert rec.done.reason == "error"


async def test_unexpected_exception_is_reported_then_raised() -> None:
    loop, _, _, rec = make_loop([])  # no replies: the adapter raises AssertionError

    with pytest.raises(AssertionError):
        await loop.run([Message.user("hi")])

    assert rec.of(Error)[0].code == "internal"
    assert rec.done.reason == "error"


async def test_approval_granted_runs_tool() -> None:
    gate = ApprovalGate()
    replies = [Reply(calls=[("danger", {"text": "rm"})]), Reply("ok")]
    loop, _, tools, rec = make_loop(replies, approvals=gate)

    async def approver(event: AnyEvent) -> None:
        if isinstance(event, ApprovalRequest):
            gate.resolve(event.approval_id, "approve")

    loop.events.subscribe(approver)
    result = await loop.run([Message.user("go")])

    assert result.reason == "completed"
    assert [c.name for c in tools.calls] == ["danger"]
    (req,) = rec.of(ApprovalRequest)
    assert req.reason == "danger is risky"
    assert req.arguments == {"text": "rm"}


async def test_approval_denied_returns_error_result_to_model() -> None:
    gate = ApprovalGate()
    replies = [Reply(calls=[("danger", {"text": "rm"})]), Reply("ok, I won't")]
    loop, adapter, tools, rec = make_loop(replies, approvals=gate)

    async def denier(event: AnyEvent) -> None:
        if isinstance(event, ApprovalRequest):
            gate.resolve(event.approval_id, "deny", "too dangerous")

    loop.events.subscribe(denier)
    history = [Message.user("go")]
    result = await loop.run(history)

    assert result.reason == "completed"
    assert tools.calls == []
    (end,) = rec.of(ToolEnd)
    assert end.is_error
    assert "too dangerous" in end.output
    tool_msg = adapter.requests[1].messages[2]
    assert tool_msg.content[0].type == "tool_result"
    assert tool_msg.content[0].is_error


async def test_policy_deny_never_asks() -> None:
    class DenyAll:
        def check(self, tool: Tool | None, call: ToolCall) -> PolicyDecision:
            return PolicyDecision("deny", "read-only session")

    bus = EventBus("ses_test")
    rec = Recorder(bus)
    tools = EchoTools()
    loop = AgentLoop(
        adapter=ScriptedAdapter([Reply(calls=[("echo", {"text": "x"})]), Reply("fine")]),
        tools=tools,
        policy=DenyAll(),
        events=bus,
    )

    await loop.run([Message.user("go")])

    assert tools.calls == []
    assert rec.of(ApprovalRequest) == []
    assert "read-only session" in rec.of(ToolEnd)[0].output


# ----------------------------------------------------------------------------- model retries (#6)


class Flaky(ScriptedAdapter):
    """Fails with ``errors`` (in order) before serving the scripted replies."""

    def __init__(self, errors: list[ModelError], replies: list[Reply]) -> None:
        super().__init__(replies, streaming=False)
        self.errors = list(errors)
        self.attempts = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.attempts += 1
        if self.errors:
            raise self.errors.pop(0)
        return await super().complete(request)


def flaky_loop(
    errors: list[ModelError], replies: list[Reply], **limits: int | float
) -> tuple[AgentLoop, Flaky, Recorder]:
    adapter = Flaky(errors, replies)
    bus = EventBus("ses_test")
    loop = AgentLoop(
        adapter=adapter,
        tools=EchoTools(),
        policy=RiskyAsks(),
        events=bus,
        limits=Limits(**{"model_retry_backoff_s": 0, **limits}),  # pyright: ignore[reportArgumentType]
    )
    return loop, adapter, Recorder(bus)


STALL = ModelError("connection error: Request timed out.", retryable=True)


async def test_stalled_call_is_retried_then_succeeds() -> None:
    loop, adapter, rec = flaky_loop([STALL], [Reply("hello")])

    result = await loop.run([Message.user("hi")])

    assert result.reason == "completed"
    assert result.final_text is not None
    assert result.final_text.strip() == "hello"
    assert adapter.attempts == 2
    assert result.steps == 1  # a retry is not a new step
    (err,) = rec.of(Error)
    assert not err.fatal
    assert err.retryable
    assert "retrying (1/2)" in err.message


async def test_retries_exhausted_ends_with_error() -> None:
    loop, adapter, rec = flaky_loop([STALL, STALL, STALL], [Reply("never")], max_model_retries=2)

    result = await loop.run([Message.user("hi")])

    assert result.reason == "error"
    assert result.error is STALL
    assert adapter.attempts == 3
    assert [e.fatal for e in rec.of(Error)] == [False, False, True]


async def test_non_retryable_error_is_not_retried() -> None:
    bad = ModelError("HTTP 400: bad request", status_code=400)
    loop, adapter, rec = flaky_loop([bad], [Reply("never")])

    result = await loop.run([Message.user("hi")])

    assert result.reason == "error"
    assert adapter.attempts == 1
    (err,) = rec.of(Error)
    assert err.fatal


async def test_retry_backoff_grows() -> None:
    loop, _, _ = flaky_loop([STALL, STALL], [Reply("ok")], model_retry_backoff_s=0.05)
    started = asyncio.get_running_loop().time()

    result = await loop.run([Message.user("hi")])

    assert result.reason == "completed"
    assert asyncio.get_running_loop().time() - started >= 0.05 + 0.1


async def test_hung_call_is_bounded_by_wall_clock() -> None:
    """Retries never outlive the run's own deadline."""
    loop, _, _ = flaky_loop(
        [STALL] * 5,
        [Reply("never")],
        max_model_retries=5,
        model_retry_backoff_s=10,
        max_wall_clock_s=0.2,
    )

    result = await loop.run([Message.user("hi")])

    assert result.reason == "timeout"
