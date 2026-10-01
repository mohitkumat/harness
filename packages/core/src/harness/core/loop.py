"""The agent loop: think → act → observe → repeat, until a final answer or a limit.

Kept deliberately small. Context building, tool execution and approval policy plug in through
the protocols in ``harness.core.hooks``; every state change is emitted on the ``EventBus``.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from harness.core.adapter import ModelAdapter
from harness.core.approvals import ApprovalGate
from harness.core.errors import HarnessError, LimitExceeded
from harness.core.events import DoneReason, EventBus
from harness.core.hooks import ApprovalPolicy, ContextBuilder, PassthroughContext, ToolExecutor
from harness.core.streaming import collect_stream
from harness.core.types import (
    ContentPart,
    Message,
    ModelRequest,
    ModelResponse,
    ToolCall,
    ToolResult,
    Usage,
)

_LIMIT_REASONS: dict[str, DoneReason] = {
    "steps": "max_steps",
    "tokens": "max_tokens",
    "cost": "max_cost",
    "wall_clock": "timeout",
}


@dataclass(frozen=True)
class Limits:
    """Per-run budgets. ``None`` disables a limit.

    ``max_tokens`` counts input + output tokens across all model calls in the run. ``max_cost_usd``
    is only enforced when the adapter reports cost. ``max_wall_clock_s`` includes time spent
    waiting for approvals.
    """

    max_steps: int = 25
    max_tokens: int | None = None
    max_cost_usd: float | None = None
    max_wall_clock_s: float | None = None


@dataclass(frozen=True)
class RunResult:
    """How a run ended. Mirrors the ``Done`` event."""

    reason: DoneReason
    steps: int
    final_text: str | None
    usage: Usage
    error: HarnessError | None = None


@dataclass
class _RunState:
    steps: int = 0
    usage: Usage = field(default_factory=Usage)
    final_text: str | None = None
    results: list[ToolResult] = field(default_factory=list[ToolResult])


class AgentLoop:
    """Runs one user turn to completion against a session's history."""

    def __init__(
        self,
        *,
        adapter: ModelAdapter,
        tools: ToolExecutor,
        policy: ApprovalPolicy,
        events: EventBus,
        approvals: ApprovalGate | None = None,
        system: str | None = None,
        limits: Limits | None = None,
        context: ContextBuilder | None = None,
    ) -> None:
        self.adapter = adapter
        self.tools = tools
        self.policy = policy
        self.events = events
        self.approvals = approvals or ApprovalGate()
        self.system = system
        self.limits = limits or Limits()
        self.context: ContextBuilder = context or PassthroughContext()

    async def run(self, history: list[Message]) -> RunResult:
        """Run until the model answers without tool calls or a limit/cancel/error stops it.

        ``history`` must end with the new user message; the loop appends to it in place and leaves
        it consistent (every tool call followed by its results) however the run ends. Cancelling
        the task running this coroutine ends the run with ``Done(reason="cancelled")``.
        Unexpected (non-harness) exceptions are reported as events and then re-raised.
        """
        state = _RunState()
        error: HarnessError | None = None
        try:
            async with asyncio.timeout(self.limits.max_wall_clock_s):
                reason = await self._steps(history, state)
        except TimeoutError:
            reason = "timeout"
        except asyncio.CancelledError:
            if (task := asyncio.current_task()) is not None:
                task.uncancel()
            reason = "cancelled"
        except LimitExceeded as e:
            reason = _LIMIT_REASONS[e.limit]
        except HarnessError as e:
            error, reason = e, "error"
            await self.events.error(e.code, e.message, retryable=e.retryable, fatal=True)
        except Exception as e:
            _close_dangling_calls(history, state.results)
            await self.events.error("internal", f"{type(e).__name__}: {e}", fatal=True)
            await self._finish("error", state)
            raise

        _close_dangling_calls(history, state.results)
        await self._finish(reason, state)
        return RunResult(reason, state.steps, state.final_text, state.usage, error)

    async def _finish(self, reason: DoneReason, state: _RunState) -> None:
        await self.events.done(
            reason, steps=state.steps, final_text=state.final_text, usage=state.usage
        )
        await self.events.status("idle")

    # ------------------------------------------------------------------ steps

    async def _steps(self, history: list[Message], state: _RunState) -> DoneReason:
        tools = self.tools.tools
        for step in range(self.limits.max_steps):
            self._check_budget(state.usage)
            state.steps = step + 1
            await self.events.status("thinking", step=step)
            request = self.context.build(system=self.system, history=history, tools=tools)
            response = await self._call_model(request, step)
            state.usage += response.usage
            history.append(response.message)

            calls = response.message.tool_calls
            if not calls:
                state.final_text = response.message.text
                return "completed"

            state.results = []
            for call in calls:
                state.results.append(await self._run_tool(call, step))
            history.append(Message(role="tool", content=list[ContentPart](state.results)))
            state.results = []

        raise LimitExceeded(
            "steps", limit_value=self.limits.max_steps, actual=self.limits.max_steps
        )

    def _check_budget(self, usage: Usage) -> None:
        tokens = usage.input_tokens + usage.output_tokens
        if self.limits.max_tokens is not None and tokens >= self.limits.max_tokens:
            raise LimitExceeded("tokens", limit_value=self.limits.max_tokens, actual=tokens)
        cost, max_cost = usage.cost_usd, self.limits.max_cost_usd
        if max_cost is not None and cost is not None and cost >= max_cost:
            raise LimitExceeded("cost", limit_value=max_cost, actual=cost)

    async def _call_model(self, request: ModelRequest, step: int) -> ModelResponse:
        if self.adapter.capabilities.streaming:

            async def on_text(text: str) -> None:
                await self.events.text_delta(step, text)

            return await collect_stream(self.adapter.stream(request), on_text)

        response = await self.adapter.complete(request)
        if text := response.message.text:
            await self.events.text_delta(step, text)
        return response

    # ------------------------------------------------------------------ tools

    async def _run_tool(self, call: ToolCall, step: int) -> ToolResult:
        tool = next((t for t in self.tools.tools if t.name == call.name), None)
        decision = self.policy.check(tool, call)

        if decision.action == "deny":
            return await self._refuse(call, step, f"Denied by policy: {decision.reason}")
        if decision.action == "ask":
            approval_id = self.approvals.open()
            await self.events.approval_request(
                step, approval_id, call.id, call.name, call.arguments, decision.reason
            )
            await self.events.status("awaiting_approval", step=step)
            answer = await self.approvals.wait(approval_id)
            if answer.decision == "deny":
                detail = f": {answer.reason}" if answer.reason else "."
                return await self._refuse(call, step, f"The user denied this tool call{detail}")

        await self.events.status("running_tool", step=step, message=call.name)
        await self.events.tool_start(step, call.id, call.name, call.arguments)
        started = time.monotonic()
        outcome = await self.tools.execute(call)
        await self.events.tool_end(
            step,
            call.id,
            call.name,
            output=outcome.result.content,
            is_error=outcome.result.is_error,
            truncated=outcome.truncated,
            duration_ms=round((time.monotonic() - started) * 1000),
        )
        return outcome.result

    async def _refuse(self, call: ToolCall, step: int, message: str) -> ToolResult:
        """Report a refused call. Emits only ``ToolEnd``: the tool never started."""
        await self.events.tool_end(
            step, call.id, call.name, output=message, is_error=True, truncated=False, duration_ms=0
        )
        return ToolResult(tool_call_id=call.id, content=message, is_error=True)


def _close_dangling_calls(history: list[Message], partial: list[ToolResult]) -> None:
    """If the run stopped mid-tools, append results so no tool call is left without one."""
    if not history or history[-1].role != "assistant" or not history[-1].tool_calls:
        return
    done = {r.tool_call_id for r in partial}
    missing = [
        ToolResult(tool_call_id=c.id, content="Not run: the run was stopped.", is_error=True)
        for c in history[-1].tool_calls
        if c.id not in done
    ]
    history.append(Message(role="tool", content=[*partial, *missing]))
