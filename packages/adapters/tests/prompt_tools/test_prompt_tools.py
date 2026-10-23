"""Unit tests for prompt-based tool calling: request rendering, reply parsing and streaming.

These wrap ``ScriptedAdapter``. Behaviour against a real model is covered by the contract suite.
"""

import pytest
from pydantic import JsonValue

from harness.adapters import AdapterSpec, build_adapter
from harness.adapters.openai_compat import OpenAICompatAdapter
from harness.adapters.prompt_tools import PromptToolsAdapter
from harness.adapters.prompt_tools.adapter import (
    _VisibleText,  # pyright: ignore[reportPrivateUsage]
)
from harness.adapters.prompt_tools.render import (
    INVALID_CALL_NAME,
    parse_calls,
    to_inner_request,
)
from harness.core import (
    Message,
    ModelRequest,
    ReasoningPart,
    StopReason,
    TextDelta,
    TextPart,
    Tool,
    ToolCall,
    ToolResult,
    collect_stream,
)
from harness.core.testing import Reply, ScriptedAdapter

SCHEMA: dict[str, JsonValue] = {"type": "object", "properties": {"path": {"type": "string"}}}
TOOLS = [Tool(name="read_file", description="Read a file.", input_schema=SCHEMA)]
CALL = '<tool_call>\n{"name": "read_file", "arguments": {"path": "a.txt"}}\n</tool_call>'


def req(*messages: Message, **kwargs: object) -> ModelRequest:
    return ModelRequest(system="Be brief.", messages=list(messages), tools=TOOLS, **kwargs)  # pyright: ignore[reportArgumentType]


# ----------------------------------------------------------------------------- request rendering


def test_tools_move_into_system_prompt() -> None:
    inner = to_inner_request(req(Message.user("hi")))

    assert inner.tools == []
    assert inner.system is not None
    assert inner.system.startswith("Be brief.\n\n# Tools")
    assert '"name": "read_file"' in inner.system
    assert "<tool_call>" in inner.system


def test_request_without_tools_is_unchanged() -> None:
    request = ModelRequest(system="s", messages=[Message.user("hi")])

    assert to_inner_request(request) is request


def test_tool_choice_variants() -> None:
    required = to_inner_request(req(Message.user("hi"), tool_choice="required"))
    none = to_inner_request(req(Message.user("hi"), tool_choice="none"))

    assert required.system is not None
    assert "must call at least one tool" in required.system
    assert none.system is not None
    assert "Do not call any tools" in none.system
    assert "read_file" not in none.system
    assert required.tool_choice == none.tool_choice == "auto"


def test_history_becomes_text_with_alternating_roles() -> None:
    call = ToolCall(id="call_x", name="read_file", arguments={"path": "a.txt"})
    reasoning = ReasoningPart(source="test", data={"k": "v"})
    history = [
        Message.user("read a.txt"),
        Message(role="assistant", content=[reasoning, TextPart(text="Reading."), call]),
        Message(
            role="tool",
            content=[ToolResult(tool_call_id="call_x", content="no such file", is_error=True)],
        ),
        Message.user("try again"),
    ]

    messages = to_inner_request(req(*history)).messages

    assert [m.role for m in messages] == ["user", "assistant", "user"]
    assistant = messages[1]
    assert assistant.content[0] == reasoning
    assert assistant.text == f"Reading.\n\n{CALL}"
    assert messages[2].text == (
        '<tool_result name="read_file" error="true">\nno such file\n</tool_result>\n\ntry again'
    )
    assert "call_x" not in str([m.model_dump() for m in messages])  # ids stay out of the prompt


def test_unparseable_call_is_replayed_verbatim() -> None:
    call = ToolCall(id="c", name="read_file", arguments={"_raw": '{"name": "read_file", oops'})
    messages = to_inner_request(req(Message(role="assistant", content=[call]))).messages

    assert messages[0].text == '<tool_call>\n{"name": "read_file", oops\n</tool_call>'


# ----------------------------------------------------------------------------- reply parsing


def test_parse_single_call() -> None:
    prose, calls = parse_calls(f"Let me look.\n{CALL}")

    assert prose == "Let me look."
    (call,) = calls
    assert (call.name, call.arguments) == ("read_file", {"path": "a.txt"})
    assert call.id.startswith("call_")


def test_parse_parallel_calls_get_unique_ids() -> None:
    _, calls = parse_calls(CALL + "\n" + CALL.replace("a.txt", "b.txt"))

    assert [c.arguments["path"] for c in calls] == ["a.txt", "b.txt"]
    assert len({c.id for c in calls}) == 2


@pytest.mark.parametrize(
    "body",
    [
        '```json\n{"name": "read_file", "arguments": {"path": "a.txt"}}\n```',
        '{"name": "read_file", "parameters": {"path": "a.txt"}}',
        '{"name": "read_file", "arguments": "{\\"path\\": \\"a.txt\\"}"}',
    ],
    ids=["fenced", "parameters-key", "string-arguments"],
)
def test_parse_tolerates_common_variants(body: str) -> None:
    _, (call,) = parse_calls(f"<tool_call>{body}</tool_call>")

    assert (call.name, call.arguments) == ("read_file", {"path": "a.txt"})


def test_parse_unclosed_block_at_end() -> None:
    prose, (call,) = parse_calls('ok <tool_call>{"name": "read_file", "arguments": {}}')

    assert prose == "ok"
    assert call.arguments == {}


def test_malformed_json_becomes_raw_call() -> None:
    _, (call,) = parse_calls('<tool_call>{"name": "read_file", "arguments": {"path": </tool_call>')

    assert call.name == "read_file"
    assert call.arguments == {"_raw": '{"name": "read_file", "arguments": {"path":'}


def test_block_without_name_is_invalid_call() -> None:
    _, (call,) = parse_calls('<tool_call>{"arguments": {}}</tool_call>')

    assert call.name == INVALID_CALL_NAME


def test_several_objects_in_one_block() -> None:
    body = '{"name": "read_file", "arguments": {"path": "a"}}\n,...\n{"name": "read_file", "arguments": {"path": "b"}}'
    _, calls = parse_calls(f"<tool_call>\n{body}\n</tool_call>")

    assert [c.arguments for c in calls] == [{"path": "a"}, {"path": "b"}]


def test_bare_call_without_tags_needs_known_tool() -> None:
    bare = '{"name": "read_file", "arguments": {"path": "a.txt"}}'

    prose, (call,) = parse_calls(bare, {"read_file"})
    assert (prose, call.name, call.arguments) == ("", "read_file", {"path": "a.txt"})
    assert parse_calls(bare, {"write_file"}) == (bare, [])
    assert parse_calls('{"name": "read_file", "version": 2}', {"read_file"})[1] == []
    assert parse_calls(f"{bare} and some prose", {"read_file"})[1] == []

    _, calls = parse_calls(f"{bare}\n{bare.replace('a.txt', 'b.txt')}", {"read_file"})
    assert [c.arguments["path"] for c in calls] == ["a.txt", "b.txt"]


def test_plain_text_has_no_calls() -> None:
    assert parse_calls("  just an answer \n") == ("just an answer", [])


# ----------------------------------------------------------------------------- adapter


async def test_complete_parses_calls_and_sets_stop_reason() -> None:
    inner = ScriptedAdapter([Reply(text=f"Checking. {CALL}"), Reply(text="Done.")])
    adapter = PromptToolsAdapter(inner)

    first = await adapter.complete(req(Message.user("read a.txt")))
    second = await adapter.complete(req(Message.user("thanks")))

    assert first.stop_reason == StopReason.TOOL_USE
    assert first.message.text == "Checking."
    assert [c.name for c in first.message.tool_calls] == ["read_file"]
    assert second.stop_reason == StopReason.END_TURN
    assert inner.requests[0].tools == []


async def test_stream_hides_call_blocks_and_returns_calls() -> None:
    inner = ScriptedAdapter([Reply(text=f"Checking. {CALL} ")])
    chunks: list[str] = []

    async def on_text(text: str) -> None:
        chunks.append(text)

    response = await collect_stream(
        PromptToolsAdapter(inner).stream(req(Message.user("read a.txt"))), on_text
    )

    assert "".join(chunks).strip() == "Checking."
    assert response.stop_reason == StopReason.TOOL_USE
    (call,) = response.message.tool_calls
    assert call.arguments == {"path": "a.txt"}


async def test_native_calls_from_inner_are_kept() -> None:
    inner = ScriptedAdapter([Reply(calls=[("read_file", {"path": "a.txt"})])])

    response = await collect_stream(PromptToolsAdapter(inner).stream(req(Message.user("go"))))

    assert [c.name for c in response.message.tool_calls] == ["read_file"]
    assert response.stop_reason == StopReason.TOOL_USE


def test_capabilities() -> None:
    caps = PromptToolsAdapter(ScriptedAdapter([]), parallel_tool_calls=False).capabilities

    assert not caps.native_tools
    assert not caps.parallel_tool_calls
    assert caps.context_window == 100_000


@pytest.mark.parametrize("size", [1, 2, 3, 5, 7, 11])
def test_visible_text_handles_tags_split_across_chunks(size: int) -> None:
    text = f"a <b> {CALL} c <tool d"
    visible = _VisibleText()
    out = "".join(visible.feed(text[i : i + size]) for i in range(0, len(text), size))

    assert out + visible.flush() == "a <b>  c <tool d"


def test_visible_text_drops_unfinished_block() -> None:
    visible = _VisibleText()

    assert visible.feed('hi <tool_call>{"name"') == "hi "
    assert visible.flush() == ""


async def test_stream_yields_text_live() -> None:
    inner = ScriptedAdapter([Reply(text="one two three")])
    deltas = [d async for d in PromptToolsAdapter(inner).stream(req(Message.user("go")))]

    assert len([d for d in deltas if isinstance(d, TextDelta)]) >= 3


# ----------------------------------------------------------------------------- registry


def test_build_adapter_wraps_when_native_tools_off() -> None:
    spec = AdapterSpec(provider="openai_compat", model="qwen2.5-coder:7b", native_tools=False)

    adapter = build_adapter(spec)

    assert isinstance(adapter, PromptToolsAdapter)
    assert isinstance(adapter.inner, OpenAICompatAdapter)
    assert not adapter.inner.capabilities.native_tools


def test_build_adapter_native_by_default() -> None:
    adapter = build_adapter(AdapterSpec(provider="openai_compat", model="qwen2.5:7b"))

    assert isinstance(adapter, OpenAICompatAdapter)
