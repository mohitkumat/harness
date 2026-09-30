import pytest
from pydantic import ValidationError

from harness.core import (
    ImagePart,
    Message,
    ModelResponse,
    StopReason,
    TextPart,
    Tool,
    ToolCall,
    ToolResult,
    Usage,
)


def test_message_helpers() -> None:
    msg = Message(
        role="assistant",
        content=[
            TextPart(text="Looking "),
            ToolCall(id="t1", name="read_file", arguments={"path": "a.txt"}),
            TextPart(text="now"),
        ],
    )
    assert msg.text == "Looking now"
    assert [c.id for c in msg.tool_calls] == ["t1"]
    assert Message.user("hi").text == "hi"


@pytest.mark.parametrize(
    ("role", "part"),
    [
        ("user", ToolCall(id="t1", name="x", arguments={})),
        ("assistant", ToolResult(tool_call_id="t1", content="ok")),
        ("tool", TextPart(text="hi")),
    ],
)
def test_message_rejects_parts_invalid_for_role(role: str, part: object) -> None:
    with pytest.raises(ValidationError):
        Message.model_validate({"role": role, "content": [part]})


def test_message_json_round_trip() -> None:
    msgs = [
        Message.user("hi"),
        Message(
            role="assistant", content=[ToolCall(id="t1", name="ls", arguments={"p": [1, "x"]})]
        ),
        Message(
            role="tool", content=[ToolResult(tool_call_id="t1", content="boom", is_error=True)]
        ),
    ]
    for m in msgs:
        assert Message.model_validate_json(m.model_dump_json()) == m


def test_image_part_needs_exactly_one_source() -> None:
    ImagePart(media_type="image/png", url="https://example.com/a.png")
    with pytest.raises(ValidationError):
        ImagePart(media_type="image/png")
    with pytest.raises(ValidationError):
        ImagePart(media_type="image/png", url="u", data="d")


def test_tool_name_and_timeout_validated() -> None:
    Tool(name="read_file", description="d", input_schema={"type": "object"})
    with pytest.raises(ValidationError):
        Tool(name="bad name!", description="d", input_schema={})
    with pytest.raises(ValidationError):
        Tool(name="t", description="d", input_schema={}, timeout_s=0)


def test_usage_addition() -> None:
    a = Usage(input_tokens=10, output_tokens=2, cache_read_tokens=5)
    b = Usage(input_tokens=1, output_tokens=1, cost_usd=0.5)
    total = a + b
    assert (total.input_tokens, total.output_tokens, total.cache_read_tokens) == (11, 3, 5)
    assert total.cost_usd == 0.5
    assert (a + a).cost_usd is None


def test_model_response_must_be_assistant() -> None:
    with pytest.raises(ValidationError):
        ModelResponse(
            message=Message.user("hi"), stop_reason=StopReason.END_TURN, usage=Usage(), model="m"
        )


def test_types_are_immutable() -> None:
    part = TextPart(text="a")
    with pytest.raises(ValidationError):
        part.text = "b"  # type: ignore[misc]
