from pathlib import Path

import pytest
from pydantic import ValidationError

from harness_evals.tasks import Task, load_task, load_tasks

TASKS_DIR = Path(__file__).resolve().parents[2] / "tasks"


def test_all_repo_tasks_load() -> None:
    tasks = load_tasks(TASKS_DIR)
    assert 10 <= len(tasks) <= 20
    assert len({t.id for t in tasks}) == len(tasks)
    assert load_tasks(TASKS_DIR, {"fast"}), "the fast subset must not be empty"


def test_tag_filter() -> None:
    fast = load_tasks(TASKS_DIR, {"fast"})
    assert all("fast" in t.tags for t in fast)


def _task(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "t",
        "title": "t",
        "prompt": "p",
        "checks": [{"type": "answer_contains", "text": "x"}],
    }
    return base | overrides


@pytest.mark.parametrize("bad", ["../escape.txt", "/etc/passwd", "a/../../b"])
def test_paths_must_stay_in_workspace(bad: str) -> None:
    with pytest.raises(ValidationError):
        Task.model_validate(_task(setup={"files": {bad: "x"}}))
    with pytest.raises(ValidationError):
        Task.model_validate(_task(checks=[{"type": "file_exists", "path": bad}]))


def test_task_requires_checks_and_known_fields() -> None:
    with pytest.raises(ValidationError):
        Task.model_validate(_task(checks=[]))
    with pytest.raises(ValidationError):
        Task.model_validate(_task(unknown=1))


def test_id_must_match_filename(tmp_path: Path) -> None:
    path = tmp_path / "other.toml"
    path.write_text(
        'id = "t"\ntitle = "t"\nprompt = "p"\n[[checks]]\ntype = "answer_contains"\ntext = "x"\n'
    )
    with pytest.raises(ValueError, match="must match file name"):
        load_task(path)
