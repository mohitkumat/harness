"""Wires adapters, tools, guardrails and observability into a ``Harness``.

Shared by the CLI and the eval agent so both run exactly the same agent.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Sequence
from pathlib import Path

from harness.adapters import AdapterSpec
from harness.adapters import build_adapter as build_provider_adapter
from harness.core import (
    AgentLoop,
    ApprovalGate,
    ApprovalPolicy,
    EventBus,
    EventSink,
    Harness,
    Limits,
    ModelAdapter,
)
from harness.core.protocol import CreateSessionCommand
from harness.tools import BUILTINS, ToolRegistry
from harness_cli.config import Settings


def new_command_id() -> str:
    """Return a fresh client-side command id."""
    return f"cmd_{uuid.uuid4().hex[:12]}"


def build_adapter(settings: Settings, model: str | None = None) -> ModelAdapter:
    """Build the model adapter for ``model`` (default: ``settings.model``)."""
    return build_provider_adapter(
        AdapterSpec(
            provider=settings.provider,
            model=model or settings.model,
            base_url=settings.base_url,
            api_key=settings.api_key,
            anthropic_api_key=settings.anthropic_api_key,
            effort=settings.effort,
            max_output_tokens=settings.max_output_tokens,
            idle_timeout_s=settings.model_idle_timeout_s,
            native_tools=settings.native_tools,
        )
    )


def limits_from(settings: Settings, *, max_steps: int | None = None) -> Limits:
    """Build run limits from settings, optionally overriding the step budget."""
    return Limits(
        max_steps=max_steps or settings.max_steps,
        max_tokens=settings.max_tokens,
        max_cost_usd=settings.max_cost_usd,
        max_wall_clock_s=settings.timeout_s,
        max_model_retries=settings.max_model_retries,
    )


def build_harness(
    settings: Settings,
    *,
    policy: ApprovalPolicy,
    sinks: Sequence[EventSink] = (),
    default_workspace: Path | None = None,
    limits: Limits | None = None,
    adapter_factory: Callable[[str | None], ModelAdapter] | None = None,
) -> Harness:
    """Build a ``Harness`` whose sessions use the built-in tools on their workspace.

    ``adapter_factory`` maps a session's model id to an adapter; it defaults to ``build_adapter``
    and exists so tests can inject a scripted model.
    """

    def factory(
        command: CreateSessionCommand, events: EventBus, approvals: ApprovalGate
    ) -> AgentLoop:
        workspace = (
            Path(command.workspace) if command.workspace else default_workspace or Path.cwd()
        )
        if not workspace.is_dir():
            raise NotADirectoryError(f"workspace {workspace} is not a directory")
        return AgentLoop(
            adapter=(
                adapter_factory(command.model)
                if adapter_factory is not None
                else build_adapter(settings, command.model)
            ),
            tools=ToolRegistry(workspace, list(BUILTINS)),
            policy=policy,
            events=events,
            approvals=approvals,
            system=command.system_prompt or settings.system_prompt,
            limits=limits or limits_from(settings),
        )

    return Harness(factory, sinks=sinks)
