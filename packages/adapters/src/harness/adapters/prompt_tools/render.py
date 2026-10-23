"""Canonical tools ↔ prompt text, for models without native tool calling.

Tools are described in the system prompt. The model calls one by writing a block in the
format Qwen and Hermes-style models are trained on::

    <tool_call>
    {"name": "read_file", "arguments": {"path": "a.txt"}}
    </tool_call>

Tool results go back in ``<tool_result>`` blocks inside a user message. Tool-call ids never
appear in the prompt (the model doesn't need them), so requests stay deterministic for cassettes.
"""

from __future__ import annotations

import contextlib
import json
import re
import uuid
from collections.abc import Collection

from pydantic import JsonValue

from harness.core import (
    RAW_ARGUMENTS_KEY,
    ContentPart,
    Message,
    ModelRequest,
    ModelResponse,
    StopReason,
    TextPart,
    Tool,
    ToolCall,
    ToolResult,
)

OPEN = "<tool_call>"
CLOSE = "</tool_call>"
INVALID_CALL_NAME = "invalid_tool_call"
"""Name given to a block without a usable tool name, so the registry returns an error result."""

_CALL = re.compile(re.escape(OPEN) + r"(.*?)(?:" + re.escape(CLOSE) + r"|\Z)", re.DOTALL)
_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)

_INSTRUCTIONS = f"""\
To call a tool, reply with one block per call, in exactly this format:
{OPEN}
{{"name": "<tool name>", "arguments": {{<arguments as a JSON object>}}}}
{CLOSE}

You may make several calls in one reply. After your calls, stop and wait: each result comes \
back to you in a <tool_result> block. Never write a <tool_result> block yourself. When you need \
no more tools, reply with your final answer in plain text, without any {OPEN} block."""


def tools_prompt(tools: list[Tool], tool_choice: str) -> str:
    """Build the system-prompt section that describes ``tools`` and how to call them."""
    if tool_choice == "none":
        return "# Tools\n\nDo not call any tools in this reply. Answer in plain text."
    specs = "\n".join(
        json.dumps(
            {"name": t.name, "description": t.description, "parameters": t.input_schema},
            ensure_ascii=False,
        )
        for t in tools
    )
    rule = "\n\nYou must call at least one tool in this reply." if tool_choice == "required" else ""
    return (
        "# Tools\n\nYou can call these tools. Arguments must match each tool's JSON schema.\n\n"
        f"<tools>\n{specs}\n</tools>\n\n{_INSTRUCTIONS}{rule}"
    )


def render_call(call: ToolCall) -> str:
    """Render a past tool call as the text block the model would have written."""
    if (
        call.raw_arguments is not None
    ):  # an unparseable call goes back exactly as the model wrote it
        return f"{OPEN}\n{call.raw_arguments}\n{CLOSE}"
    body = json.dumps({"name": call.name, "arguments": call.arguments}, ensure_ascii=False)
    return f"{OPEN}\n{body}\n{CLOSE}"


def render_result(result: ToolResult, name: str) -> str:
    """Render a tool result as a ``<tool_result>`` block."""
    error = ' error="true"' if result.is_error else ""
    return f'<tool_result name="{name}"{error}>\n{result.content}\n</tool_result>'


def to_inner_request(request: ModelRequest) -> ModelRequest:
    """Rewrite ``request`` so it carries no native tools, tool calls or tool messages."""
    if not request.tools and not any(m.tool_calls or m.role == "tool" for m in request.messages):
        return request
    section = tools_prompt(request.tools, request.tool_choice)
    system = f"{request.system}\n\n{section}" if request.system else section
    return request.model_copy(
        update={
            "system": system,
            "messages": _to_inner_messages(request.messages),
            "tools": [],
            "tool_choice": "auto",
        }
    )


def _to_inner_messages(messages: list[Message]) -> list[Message]:
    names: dict[str, str] = {}
    out: list[Message] = []
    for message in messages:
        if message.role == "assistant":
            content: list[ContentPart] = []
            for part in message.content:
                if isinstance(part, ToolCall):
                    names[part.id] = part.name
                    _append_text(content, render_call(part))
                elif isinstance(part, TextPart):
                    _append_text(content, part.text)
                else:
                    content.append(part)
            converted = Message(role="assistant", content=content)
        elif message.role == "tool":
            blocks = [
                render_result(p, names.get(p.tool_call_id, "unknown"))
                for p in message.content
                if isinstance(p, ToolResult)
            ]
            converted = Message.user("\n\n".join(blocks))
        else:
            converted = message
        if out and out[-1].role == converted.role == "user":  # keep roles alternating
            merged = list(out[-1].content)
            for part in converted.content:
                if isinstance(part, TextPart):
                    _append_text(merged, part.text)
                else:
                    merged.append(part)
            out[-1] = Message(role="user", content=merged)
        else:
            out.append(converted)
    return out


def _append_text(content: list[ContentPart], text: str) -> None:
    last = content[-1] if content else None
    if isinstance(last, TextPart):
        content[-1] = TextPart(text=f"{last.text}\n\n{text}")
    else:
        content.append(TextPart(text=text))


def from_inner_response(response: ModelResponse, tool_names: Collection[str]) -> ModelResponse:
    """Turn ``<tool_call>`` blocks in the reply's text into canonical ``ToolCall`` parts."""
    content: list[ContentPart] = []
    calls: list[ToolCall] = []
    for part in response.message.content:
        if isinstance(part, TextPart):
            text, parsed = parse_calls(part.text, tool_names)
            if text:
                content.append(TextPart(text=text))
            calls += parsed
        elif isinstance(part, ToolCall):  # a native call slipped through; keep it
            calls.append(part)
        else:
            content.append(part)
    if calls:
        stop = StopReason.TOOL_USE
    elif response.stop_reason == StopReason.TOOL_USE:
        stop = StopReason.END_TURN
    else:
        stop = response.stop_reason
    return response.model_copy(
        update={
            "message": Message(role="assistant", content=[*content, *calls]),
            "stop_reason": stop,
        }
    )


def parse_calls(text: str, tool_names: Collection[str] = ()) -> tuple[str, list[ToolCall]]:
    """Split ``text`` into the prose outside ``<tool_call>`` blocks and the calls inside them.

    Tolerates what small models actually write: a block cut off by the end of the reply,
    several call objects in one block, and (only if it names one of ``tool_names``) a whole
    reply that is a bare call object without tags. A block that isn't valid JSON becomes a call
    with ``{RAW_ARGUMENTS_KEY: <body>}`` arguments (ADR 0005), so the model gets a validation error back as
    a tool result instead of the call silently vanishing.
    """
    calls = [call for m in _CALL.finditer(text) for call in _to_calls(m.group(1))]
    if not calls:
        bare = _bare_calls(text, tool_names)
        return ("", bare) if bare else (text.strip(), [])
    return _CALL.sub("", text).strip(), calls


def _unfence(body: str) -> str:
    body = body.strip()
    fenced = _FENCE.match(body)
    return fenced.group(1) if fenced else body


def _bare_calls(text: str, tool_names: Collection[str]) -> list[ToolCall]:
    """Return the calls in a reply made only of call objects naming known tools; else none."""
    body = _unfence(text)
    decoder = json.JSONDecoder()
    calls: list[ToolCall] = []
    i = 0
    while i < len(body):
        if body[i] in " \t\r\n,":
            i += 1
            continue
        try:
            decoded: tuple[JsonValue, int] = decoder.raw_decode(body, i)
        except json.JSONDecodeError:
            return []
        value, end = decoded
        if not (
            isinstance(value, dict) and value.get("name") in tool_names and "arguments" in value
        ):
            return []
        calls.append(_to_call(value, body[i:end]))
        i = end
    return calls


def _to_calls(body: str) -> list[ToolCall]:
    body = _unfence(body)
    try:
        value: JsonValue = json.loads(body)
    except json.JSONDecodeError:
        objects = _json_objects(body)
        if objects:
            return [_to_call(o, json.dumps(o)) for o in objects]
        return [ToolCall(id=_new_id(), name=_guess_name(body), arguments={RAW_ARGUMENTS_KEY: body})]
    return [_to_call(value, body)]


def _json_objects(body: str) -> list[dict[str, JsonValue]]:
    """Every top-level JSON object with a ``name`` in ``body``, skipping junk between them."""
    decoder = json.JSONDecoder()
    found: list[dict[str, JsonValue]] = []
    i = body.find("{")
    while i >= 0:
        try:
            decoded: tuple[JsonValue, int] = decoder.raw_decode(body, i)
        except json.JSONDecodeError:
            i = body.find("{", i + 1)
            continue
        value, end = decoded
        if isinstance(value, dict) and "name" in value:
            found.append(value)
        i = body.find("{", end)
    return found


def _new_id() -> str:
    return f"call_{uuid.uuid4().hex[:12]}"


def _to_call(value: JsonValue, body: str) -> ToolCall:
    call_id = _new_id()
    if not isinstance(value, dict) or not isinstance(value.get("name"), str):
        return ToolCall(id=call_id, name=INVALID_CALL_NAME, arguments={RAW_ARGUMENTS_KEY: body})
    name = str(value["name"])
    arguments: JsonValue = value.get("arguments", value.get("parameters", {}))
    if isinstance(arguments, str):  # some models double-encode the arguments
        with contextlib.suppress(json.JSONDecodeError):
            decoded: JsonValue = json.loads(arguments)
            arguments = decoded
    if not isinstance(arguments, dict):
        return ToolCall(id=call_id, name=name, arguments={RAW_ARGUMENTS_KEY: body})
    return ToolCall(id=call_id, name=name, arguments=arguments)


def _guess_name(body: str) -> str:
    match = re.search(r'"name"\s*:\s*"([a-zA-Z0-9_-]{1,64})"', body)
    return match.group(1) if match else INVALID_CALL_NAME
