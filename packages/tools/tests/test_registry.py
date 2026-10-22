import asyncio
from pathlib import Path

import pytest
from pydantic import JsonValue

from harness.core import Tool, ToolCall, ToolError
from harness.tools import Handler, ToolContext, ToolRegistry, ToolSpec, truncate

SCHEMA: dict[str, JsonValue] = {
    "type": "object",
    "properties": {"n": {"type": "integer"}},
    "required": ["n"],
    "additionalProperties": False,
}


def spec(name: str, handler: Handler, timeout_s: float = 1.0) -> ToolSpec:
    return ToolSpec(
        Tool(name=name, description=name, input_schema=SCHEMA, timeout_s=timeout_s), handler
    )


async def double(args: dict[str, JsonValue], ctx: ToolContext) -> str:
    n = args["n"]
    assert isinstance(n, int)
    return str(n * 2)


async def expected_failure(args: dict[str, JsonValue], ctx: ToolContext) -> str:
    raise ToolError("nope", tool_name="fail")


async def crash(args: dict[str, JsonValue], ctx: ToolContext) -> str:
    raise RuntimeError("bug")


async def slow(args: dict[str, JsonValue], ctx: ToolContext) -> str:
    await asyncio.sleep(10)
    return "late"


async def huge(args: dict[str, JsonValue], ctx: ToolContext) -> str:
    return "a" * 100 + "b" * 100


@pytest.fixture
def registry(tmp_path: Path) -> ToolRegistry:
    return ToolRegistry(
        tmp_path,
        [
            spec("double", double),
            spec("fail", expected_failure),
            spec("crash", crash),
            spec("slow", slow, timeout_s=0.05),
            spec("huge", huge),
        ],
        max_output_chars=50,
    )


def call(name: str, **args: JsonValue) -> ToolCall:
    return ToolCall(id="c1", name=name, arguments=args)


async def test_success(registry: ToolRegistry) -> None:
    outcome = await registry.execute(call("double", n=21))
    assert outcome.result.content == "42"
    assert not outcome.result.is_error
    assert outcome.result.tool_call_id == "c1"


@pytest.mark.parametrize(
    ("args", "fragment"),
    [({}, "'n' is a required property"), ({"n": "x"}, "n: 'x' is not of type 'integer'")],
)
async def test_bad_args(registry: ToolRegistry, args: dict[str, JsonValue], fragment: str) -> None:
    outcome = await registry.execute(ToolCall(id="c1", name="double", arguments=args))
    assert outcome.result.is_error
    assert fragment in outcome.result.content


async def test_unknown_tool_lists_available(registry: ToolRegistry) -> None:
    outcome = await registry.execute(call("nope"))
    assert outcome.result.is_error
    assert "double" in outcome.result.content


async def test_tool_error_becomes_result(registry: ToolRegistry) -> None:
    outcome = await registry.execute(call("fail", n=1))
    assert outcome.result.is_error
    assert outcome.result.content == "nope"


async def test_handler_crash_becomes_result(registry: ToolRegistry) -> None:
    outcome = await registry.execute(call("crash", n=1))
    assert outcome.result.is_error
    assert "RuntimeError: bug" in outcome.result.content


async def test_timeout(registry: ToolRegistry) -> None:
    outcome = await registry.execute(call("slow", n=1))
    assert outcome.result.is_error
    assert "timed out" in outcome.result.content


async def test_output_truncated_head_and_tail(registry: ToolRegistry) -> None:
    outcome = await registry.execute(call("huge", n=1))
    assert outcome.truncated
    assert outcome.result.content.startswith("a" * 25)
    assert outcome.result.content.endswith("b" * 25)
    assert "150 characters truncated" in outcome.result.content


def test_duplicate_and_invalid_schema_rejected(tmp_path: Path) -> None:
    registry = ToolRegistry(tmp_path, [spec("double", double)])
    with pytest.raises(ValueError, match="already registered"):
        registry.register(spec("double", double))
    bad = ToolSpec(Tool(name="bad", description="", input_schema={"type": 5}), double)
    with pytest.raises(Exception, match="is not valid"):
        registry.register(bad)


def test_truncate_noop() -> None:
    assert truncate("short", 10) == ("short", False)
