"""Eval task definitions, loaded from ``evals/tasks/*.toml``.

A task is a prompt, a set of files seeded into a fresh workspace, and declarative checks run
against the final workspace and the agent's final answer. Checks never execute code: running
model-written code on the host is forbidden (CLAUDE.md §10).
"""

from __future__ import annotations

import tomllib
from pathlib import Path, PurePosixPath
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator


def _check_relative(path: str) -> str:
    p = PurePosixPath(path)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise ValueError(f"path must be relative and inside the workspace: {path!r}")
    return path


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class _PathCheck(_Strict):
    path: str

    @field_validator("path")
    @classmethod
    def _relative(cls, v: str) -> str:
        return _check_relative(v)


class FileExists(_PathCheck):
    """The file exists after the run."""

    type: Literal["file_exists"]


class FileAbsent(_PathCheck):
    """The file does not exist after the run."""

    type: Literal["file_absent"]


class FileContains(_PathCheck):
    """The file contains ``text`` (case-sensitive substring)."""

    type: Literal["file_contains"]
    text: str


class FileNotContains(_PathCheck):
    """The file exists and does not contain ``text``."""

    type: Literal["file_not_contains"]
    text: str


class FileEquals(_PathCheck):
    """The file's content equals ``text``, ignoring trailing whitespace."""

    type: Literal["file_equals"]
    text: str


class FileJsonEquals(_PathCheck):
    """The file parses as JSON equal to ``value``."""

    type: Literal["file_json_equals"]
    value: JsonValue


class FileUnchanged(_PathCheck):
    """The file still has exactly its seeded content."""

    type: Literal["file_unchanged"]


class AnswerContains(_Strict):
    """The final answer contains ``text`` (case-insensitive)."""

    type: Literal["answer_contains"]
    text: str


class AnswerMatches(_Strict):
    """The final answer matches the regular expression ``pattern`` (``re.search``)."""

    type: Literal["answer_matches"]
    pattern: str


Check = Annotated[
    FileExists
    | FileAbsent
    | FileContains
    | FileNotContains
    | FileEquals
    | FileJsonEquals
    | FileUnchanged
    | AnswerContains
    | AnswerMatches,
    Field(discriminator="type"),
]


class Setup(_Strict):
    """Workspace seed: relative path -> file content."""

    files: dict[str, str] = {}

    @field_validator("files")
    @classmethod
    def _relative_paths(cls, v: dict[str, str]) -> dict[str, str]:
        for path in v:
            _check_relative(path)
        return v


class Task(_Strict):
    """One eval task."""

    id: Annotated[str, Field(pattern=r"^[a-z0-9][a-z0-9-]*$")]
    title: str
    prompt: str
    tags: list[str] = []
    max_steps: Annotated[int, Field(gt=0)] = 20
    timeout_s: Annotated[float, Field(gt=0)] = 120.0
    setup: Setup = Setup()
    checks: Annotated[list[Check], Field(min_length=1)]


def load_task(path: Path) -> Task:
    """Load and validate one task file. The file stem must equal the task id."""
    with path.open("rb") as f:
        task = Task.model_validate(tomllib.load(f))
    if task.id != path.stem:
        raise ValueError(f"{path}: task id {task.id!r} must match file name {path.stem!r}")
    return task


def load_tasks(directory: Path, tags: set[str] | None = None) -> list[Task]:
    """Load every ``*.toml`` task in ``directory``, optionally keeping only tasks with any of ``tags``."""
    tasks = [load_task(p) for p in sorted(directory.glob("*.toml"))]
    if tags:
        tasks = [t for t in tasks if tags & set(t.tags)]
    return tasks
