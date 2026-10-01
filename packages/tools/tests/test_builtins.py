import asyncio
from pathlib import Path

import pytest
from pydantic import JsonValue

from harness.core import ToolCall
from harness.tools import BUILTINS, RUN_SHELL, ToolRegistry, ToolSpec


@pytest.fixture
def ws(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("print('a')\n")
    (tmp_path / "src" / "pkg").mkdir()
    (tmp_path / "src" / "pkg" / "b.py").write_text("")
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01\x02")
    return tmp_path


@pytest.fixture
def registry(ws: Path) -> ToolRegistry:
    return ToolRegistry(ws, list(BUILTINS))


async def run(registry: ToolRegistry, name: str, **args: JsonValue) -> tuple[str, bool]:
    outcome = await registry.execute(ToolCall(id="c", name=name, arguments=args))
    return outcome.result.content, outcome.result.is_error


def test_only_run_shell_is_risky(registry: ToolRegistry) -> None:
    assert {t.name: t.risky for t in registry.tools} == {
        "read_file": False,
        "write_file": False,
        "list_dir": False,
        "run_shell": True,
    }


# ----------------------------------------------------------------------------- read_file


async def test_read_file(registry: ToolRegistry) -> None:
    assert await run(registry, "read_file", path="src/a.py") == ("print('a')\n", False)


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("missing.txt", "No such file: missing.txt"),
        ("blob.bin", "binary file"),
        ("../outside.txt", "outside the workspace"),
        ("/etc/passwd", "outside the workspace"),
    ],
)
async def test_read_file_failures(registry: ToolRegistry, path: str, message: str) -> None:
    content, is_error = await run(registry, "read_file", path=path)
    assert is_error
    assert message in content


async def test_read_file_bad_args(registry: ToolRegistry) -> None:
    content, is_error = await run(registry, "read_file", file="x")
    assert is_error
    assert "Invalid arguments" in content


async def test_symlink_escape_refused(
    registry: ToolRegistry, ws: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    secret = tmp_path_factory.mktemp("outside") / "secret.txt"
    secret.write_text("s3cret")
    (ws / "link.txt").symlink_to(secret)
    content, is_error = await run(registry, "read_file", path="link.txt")
    assert is_error
    assert "s3cret" not in content


# ----------------------------------------------------------------------------- write_file


async def test_write_file_creates_parents(registry: ToolRegistry, ws: Path) -> None:
    content, is_error = await run(
        registry, "write_file", path="docs/guides/setup.md", content="# Setup\n"
    )
    assert not is_error
    assert (ws / "docs/guides/setup.md").read_text() == "# Setup\n"
    assert "docs/guides/setup.md" in content


async def test_write_file_failures(registry: ToolRegistry) -> None:
    assert (await run(registry, "write_file", path="../x", content="x"))[1]
    content, is_error = await run(registry, "write_file", path="src", content="x")
    assert is_error
    assert "directory" in content
    assert (await run(registry, "write_file", path="x"))[1]  # missing content


# ----------------------------------------------------------------------------- list_dir


async def test_list_dir(registry: ToolRegistry) -> None:
    content, is_error = await run(registry, "list_dir", path="src")
    assert not is_error
    assert content.splitlines() == ["a.py", "pkg/"]


async def test_list_dir_recursive_default_root(registry: ToolRegistry) -> None:
    content, _ = await run(registry, "list_dir", recursive=True)
    assert "src/pkg/b.py" in content.splitlines()


async def test_list_dir_failures(registry: ToolRegistry) -> None:
    assert (await run(registry, "list_dir", path="nope"))[1]
    assert (await run(registry, "list_dir", path=".."))[1]
    assert (await run(registry, "list_dir", recursive="yes"))[1]


# ----------------------------------------------------------------------------- run_shell


async def test_run_shell(registry: ToolRegistry) -> None:
    content, is_error = await run(registry, "run_shell", command="ls src && echo err >&2 && exit 3")
    assert not is_error  # a non-zero exit is information for the model, not a tool failure
    assert content.startswith("exit code: 3\n")
    assert "a.py" in content
    assert "err" in content


async def test_run_shell_does_not_leak_env(
    registry: ToolRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-leak")
    content, _ = await run(registry, "run_shell", command="env")
    assert "sk-should-not-leak" not in content


async def test_run_shell_timeout_kills_process(ws: Path) -> None:
    fast = ToolSpec(RUN_SHELL.tool.model_copy(update={"timeout_s": 0.2}), RUN_SHELL.handler)
    registry = ToolRegistry(ws, [fast])
    marker = ws / "marker"
    content, is_error = await run(registry, "run_shell", command=f"sleep 1 && touch {marker}")
    assert is_error
    assert "timed out" in content
    await asyncio.sleep(1.2)
    assert not marker.exists()


async def test_run_shell_bad_args(registry: ToolRegistry) -> None:
    assert (await run(registry, "run_shell", cmd="ls"))[1]
