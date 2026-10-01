"""Harness core: canonical types, the ModelAdapter contract, agent loop, event bus and errors.

This package imports nothing else from the repo; everything else builds on it.
"""

from harness.core.adapter import (
    Delta,
    ModelAdapter,
    StopDelta,
    TextDelta,
    ToolCallArgsDelta,
    ToolCallStartDelta,
)
from harness.core.approvals import ApprovalAnswer, ApprovalGate
from harness.core.errors import (
    HarnessError,
    InvalidCommand,
    LimitExceeded,
    ModelError,
    PermissionDenied,
    ToolError,
)
from harness.core.events import DoneReason, EventBus, EventSink, Status
from harness.core.hooks import (
    ApprovalPolicy,
    ContextBuilder,
    PassthroughContext,
    PolicyAction,
    PolicyDecision,
    ToolExecutor,
    ToolOutcome,
)
from harness.core.loop import AgentLoop, Limits, RunResult
from harness.core.session import Harness, LoopFactory, Session
from harness.core.streaming import collect_stream
from harness.core.types import (
    ContentPart,
    ImagePart,
    Message,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    Role,
    StopReason,
    TextPart,
    Tool,
    ToolCall,
    ToolResult,
    Usage,
)

__all__ = [
    "AgentLoop",
    "ApprovalAnswer",
    "ApprovalGate",
    "ApprovalPolicy",
    "ContentPart",
    "ContextBuilder",
    "Delta",
    "DoneReason",
    "EventBus",
    "EventSink",
    "Harness",
    "HarnessError",
    "ImagePart",
    "InvalidCommand",
    "LimitExceeded",
    "Limits",
    "LoopFactory",
    "Message",
    "ModelAdapter",
    "ModelCapabilities",
    "ModelError",
    "ModelRequest",
    "ModelResponse",
    "PassthroughContext",
    "PermissionDenied",
    "PolicyAction",
    "PolicyDecision",
    "Role",
    "RunResult",
    "Session",
    "Status",
    "StopDelta",
    "StopReason",
    "TextDelta",
    "TextPart",
    "Tool",
    "ToolCall",
    "ToolCallArgsDelta",
    "ToolCallStartDelta",
    "ToolError",
    "ToolExecutor",
    "ToolOutcome",
    "ToolResult",
    "Usage",
    "collect_stream",
]
