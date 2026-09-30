"""Contract tests: schema, examples and generated models must agree."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from harness.core.protocol import PROTOCOL_VERSION, Done, SendCommand, parse_command, parse_event

PROTOCOL_DIR = Path(__file__).resolve().parents[3] / "protocol"
SCHEMA = json.loads((PROTOCOL_DIR / "protocol.schema.json").read_text())
EXAMPLES = sorted((PROTOCOL_DIR / "examples").glob("*.json"))
COMMAND_TYPES = {"create_session", "send", "approve", "cancel", "resume"}


def validate(instance: object) -> None:
    validator = Draft202012Validator(SCHEMA, format_checker=FormatChecker())
    validator.validate(instance)  # pyright: ignore[reportUnknownMemberType]  (stubs are untyped here)


def test_schema_is_valid_json_schema() -> None:
    Draft202012Validator.check_schema(SCHEMA)


def test_version_matches() -> None:
    assert int((PROTOCOL_DIR / "VERSION").read_text().strip()) == PROTOCOL_VERSION


def test_every_message_type_has_an_example() -> None:
    types = {json.loads(p.read_text())["type"] for p in EXAMPLES}
    defs = SCHEMA["$defs"]
    expected = set(defs["Command"]["discriminator"]["mapping"]) | set(
        defs["Event"]["discriminator"]["mapping"]
    )
    assert types == expected


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_example_validates_and_round_trips(path: Path) -> None:
    raw = json.loads(path.read_text())
    validate(raw)

    parse = parse_command if raw["type"] in COMMAND_TYPES else parse_event
    model = parse(raw)
    assert model.type == raw["type"]
    # Generated model -> JSON must still satisfy the schema.
    validate(json.loads(model.model_dump_json()))
    assert parse(model.model_dump_json()) == model


def test_unknown_fields_are_ignored_for_forward_compat() -> None:
    cmd = parse_command(
        {"type": "send", "command_id": "c", "session_id": "s", "text": "hi", "future": 1}
    )
    assert isinstance(cmd, SendCommand)


def test_invalid_payload_rejected() -> None:
    with pytest.raises(ValueError, match="text"):
        parse_command({"type": "send", "command_id": "c", "session_id": "s"})
    with pytest.raises(ValueError, match="nope"):
        parse_event({"type": "nope"})


def test_done_usage_defaults() -> None:
    ev = parse_event(
        '{"type":"done","session_id":"s","seq":1,"ts":"2026-01-01T00:00:00Z",'
        '"reason":"cancelled","steps":0,"usage":{"input_tokens":0,"output_tokens":0}}'
    )
    assert isinstance(ev, Done)
    assert ev.usage.cache_read_tokens == 0
