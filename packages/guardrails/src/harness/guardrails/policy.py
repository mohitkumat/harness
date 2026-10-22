"""Phase 1 approval policies, keyed only on a tool's ``risky`` flag.

They implement ``harness.core.ApprovalPolicy``.
"""

# TODO(phase-4): allow/ask/deny rules per tool and path, configured per session.

from __future__ import annotations

from dataclasses import dataclass

from harness.core import PolicyDecision, Tool, ToolCall


@dataclass(frozen=True)
class AskForRisky:
    """Interactive default: every risky tool call needs user approval; the rest run.

    Unknown tools are allowed through so the registry can return an "unknown tool" error result.
    """

    def check(self, tool: Tool | None, call: ToolCall) -> PolicyDecision:
        """Ask for risky tools, allow everything else."""
        if tool is not None and tool.risky:
            return PolicyDecision("ask", f"{tool.name} is a risky tool and needs your approval.")
        return PolicyDecision("allow")


@dataclass(frozen=True)
class Headless:
    """Non-interactive runs (evals, scripts), where nobody can answer an approval request.

    Risky calls are denied unless ``allow_risky`` is set. Setting it runs them **on the host**
    until the Phase 4 sandbox exists (ADR 0003).
    """

    allow_risky: bool = False

    def check(self, tool: Tool | None, call: ToolCall) -> PolicyDecision:
        """Allow non-risky tools. Allow or deny risky ones according to ``allow_risky``."""
        if tool is None or not tool.risky or self.allow_risky:
            return PolicyDecision("allow")
        return PolicyDecision(
            "deny", f"{tool.name} is risky and this run is non-interactive (no approvals)."
        )
