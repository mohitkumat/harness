"""Harness core: canonical types, the ModelAdapter contract, errors and protocol models.

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
from harness.core.errors import (
    HarnessError,
    LimitExceeded,
    ModelError,
    PermissionDenied,
    ToolError,
)
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
    "ContentPart",
    "Delta",
    "HarnessError",
    "ImagePart",
    "LimitExceeded",
    "Message",
    "ModelAdapter",
    "ModelCapabilities",
    "ModelError",
    "ModelRequest",
    "ModelResponse",
    "PermissionDenied",
    "Role",
    "StopDelta",
    "StopReason",
    "TextDelta",
    "TextPart",
    "Tool",
    "ToolCall",
    "ToolCallArgsDelta",
    "ToolCallStartDelta",
    "ToolError",
    "ToolResult",
    "Usage",
]
