"""Evaluation of declarative checks against a finished workspace and final answer."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from harness_evals.tasks import (
    AnswerContains,
    AnswerMatches,
    Check,
    FileAbsent,
    FileContains,
    FileEquals,
    FileExists,
    FileJsonEquals,
    FileNotContains,
    FileUnchanged,
    Task,
)


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single check."""

    check: str
    passed: bool
    detail: str = ""


def _read(workspace: Path, rel: str) -> str | None:
    path = workspace / rel
    if not path.is_file():
        return None
    return path.read_text(errors="replace")


def run_check(check: Check, task: Task, workspace: Path, answer: str | None) -> CheckResult:
    """Evaluate one check. Never raises for a failing check; returns ``passed=False`` with detail."""
    name = check.type
    match check:
        case FileExists(path=p):
            ok = (workspace / p).is_file()
            return CheckResult(name, ok, "" if ok else f"{p} missing")
        case FileAbsent(path=p):
            ok = not (workspace / p).exists()
            return CheckResult(name, ok, "" if ok else f"{p} still exists")
        case FileContains(path=p, text=t):
            content = _read(workspace, p)
            if content is None:
                return CheckResult(name, False, f"{p} missing")
            return CheckResult(name, t in content, "" if t in content else f"{t!r} not in {p}")
        case FileNotContains(path=p, text=t):
            content = _read(workspace, p)
            if content is None:
                return CheckResult(name, False, f"{p} missing")
            return CheckResult(name, t not in content, "" if t not in content else f"{t!r} in {p}")
        case FileEquals(path=p, text=t):
            content = _read(workspace, p)
            ok = content is not None and content.rstrip() == t.rstrip()
            return CheckResult(name, ok, "" if ok else f"{p} content differs")
        case FileJsonEquals(path=p, value=v):
            content = _read(workspace, p)
            if content is None:
                return CheckResult(name, False, f"{p} missing")
            try:
                ok = json.loads(content) == v
            except json.JSONDecodeError as e:
                return CheckResult(name, False, f"{p} is not valid JSON: {e}")
            return CheckResult(name, ok, "" if ok else f"{p} JSON differs")
        case FileUnchanged(path=p):
            original = task.setup.files.get(p)
            ok = original is not None and _read(workspace, p) == original
            return CheckResult(name, ok, "" if ok else f"{p} was modified or removed")
        case AnswerContains(text=t):
            ok = answer is not None and t.lower() in answer.lower()
            return CheckResult(name, ok, "" if ok else f"answer lacks {t!r}")
        case AnswerMatches(pattern=pat):
            ok = answer is not None and re.search(pat, answer) is not None
            return CheckResult(name, ok, "" if ok else f"answer does not match /{pat}/")
