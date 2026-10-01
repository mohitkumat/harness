import pytest

from harness.core import ApprovalPolicy, Tool, ToolCall
from harness.guardrails import AskForRisky, Headless

SAFE = Tool(name="read_file", description="", input_schema={"type": "object"})
RISKY = Tool(name="run_shell", description="", input_schema={"type": "object"}, risky=True)
CALL = ToolCall(id="c", name="x", arguments={})


@pytest.mark.parametrize(
    ("policy", "tool", "action"),
    [
        (AskForRisky(), SAFE, "allow"),
        (AskForRisky(), RISKY, "ask"),
        (AskForRisky(), None, "allow"),
        (Headless(), SAFE, "allow"),
        (Headless(), RISKY, "deny"),
        (Headless(), None, "allow"),
        (Headless(allow_risky=True), RISKY, "allow"),
    ],
)
def test_decisions(policy: ApprovalPolicy, tool: Tool | None, action: str) -> None:
    decision = policy.check(tool, CALL)
    assert decision.action == action
    if action != "allow":
        assert "run_shell" in decision.reason
