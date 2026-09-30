from pathlib import Path

import pytest
from pydantic import TypeAdapter

from harness_evals.checks import run_check
from harness_evals.tasks import Check, Task

CHECK = TypeAdapter[Check](Check)
TASK = Task.model_validate(
    {
        "id": "t",
        "title": "t",
        "prompt": "p",
        "setup": {"files": {"keep.txt": "orig\n"}},
        "checks": [{"type": "answer_contains", "text": "x"}],
    }
)


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    (tmp_path / "keep.txt").write_text("orig\n")
    (tmp_path / "a.txt").write_text("hello world\n")
    (tmp_path / "c.json").write_text('{"k": [1, 2]}')
    (tmp_path / "bad.json").write_text("{nope")
    return tmp_path


@pytest.mark.parametrize(
    ("check", "answer", "expected"),
    [
        ({"type": "file_exists", "path": "a.txt"}, None, True),
        ({"type": "file_exists", "path": "zz.txt"}, None, False),
        ({"type": "file_absent", "path": "zz.txt"}, None, True),
        ({"type": "file_absent", "path": "a.txt"}, None, False),
        ({"type": "file_contains", "path": "a.txt", "text": "world"}, None, True),
        ({"type": "file_contains", "path": "zz.txt", "text": "world"}, None, False),
        ({"type": "file_not_contains", "path": "a.txt", "text": "world"}, None, False),
        ({"type": "file_not_contains", "path": "zz.txt", "text": "x"}, None, False),
        ({"type": "file_equals", "path": "a.txt", "text": "hello world"}, None, True),
        ({"type": "file_json_equals", "path": "c.json", "value": {"k": [1, 2]}}, None, True),
        ({"type": "file_json_equals", "path": "c.json", "value": {"k": [2, 1]}}, None, False),
        ({"type": "file_json_equals", "path": "bad.json", "value": {}}, None, False),
        ({"type": "file_unchanged", "path": "keep.txt"}, None, True),
        ({"type": "file_unchanged", "path": "a.txt"}, None, False),  # not seeded
        ({"type": "answer_contains", "text": "PORT 80"}, "the port 80 it is", True),
        ({"type": "answer_contains", "text": "x"}, None, False),
        ({"type": "answer_matches", "pattern": r"\b3\b"}, "there are 3 files", True),
        ({"type": "answer_matches", "pattern": r"\b3\b"}, "there are 33", False),
    ],
)
def test_check(ws: Path, check: dict[str, object], answer: str | None, expected: bool) -> None:
    result = run_check(CHECK.validate_python(check), TASK, ws, answer)
    assert result.passed is expected
    assert bool(result.detail) is not expected


def test_file_unchanged_detects_modification(ws: Path) -> None:
    (ws / "keep.txt").write_text("changed")
    check = CHECK.validate_python({"type": "file_unchanged", "path": "keep.txt"})
    assert not run_check(check, TASK, ws, None).passed
