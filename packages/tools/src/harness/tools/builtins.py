"""Built-in tools: ``read_file``, ``write_file``, ``list_dir`` and ``run_shell``.

File tools are confined to the workspace (symlinks included) and are not risky. ``run_shell``
executes model-generated commands, so it is risky and always needs approval.
"""

from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path

from pydantic import JsonValue

from harness.core import Tool, ToolError
from harness.tools.registry import ToolContext, ToolSpec

MAX_LIST_ENTRIES = 1_000
_SHELL_ENV_KEYS = ("PATH", "HOME", "LANG", "LC_ALL", "TERM", "TMPDIR", "USER")


def resolve_in_workspace(ctx: ToolContext, path: str, *, tool: str) -> Path:
    """Resolve ``path`` (relative to the workspace) and refuse anything that escapes it."""
    target = (ctx.workspace / path).resolve()
    if not target.is_relative_to(ctx.workspace):
        raise ToolError(f"Path {path!r} is outside the workspace.", tool_name=tool)
    return target


def _rel(ctx: ToolContext, path: Path) -> str:
    return path.relative_to(ctx.workspace).as_posix() or "."


def _str_arg(args: dict[str, JsonValue], key: str, default: str | None = None) -> str:
    value = args.get(key, default)
    if not isinstance(value, str):  # schema validation guarantees this; keeps pyright honest
        raise ToolError(f"{key!r} must be a string", tool_name="?")
    return value


# ----------------------------------------------------------------------------- read_file


async def _read_file(args: dict[str, JsonValue], ctx: ToolContext) -> str:
    path = resolve_in_workspace(ctx, _str_arg(args, "path"), tool="read_file")
    if not path.is_file():
        raise ToolError(f"No such file: {_rel(ctx, path)}", tool_name="read_file")
    data = await asyncio.to_thread(path.read_bytes)
    if b"\0" in data[:8192]:
        raise ToolError(f"{_rel(ctx, path)} is a binary file.", tool_name="read_file")
    try:
        return data.decode()
    except UnicodeDecodeError:
        raise ToolError(f"{_rel(ctx, path)} is not UTF-8 text.", tool_name="read_file") from None


READ_FILE = ToolSpec(
    Tool(
        name="read_file",
        description=(
            "Read a UTF-8 text file from the workspace and return its full contents. "
            "`path` is relative to the workspace root. Very large files are truncated in the middle."
        ),
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "File path, e.g. src/app.py"}},
            "required": ["path"],
            "additionalProperties": False,
        },
        timeout_s=10,
    ),
    _read_file,
)


# ----------------------------------------------------------------------------- write_file


async def _write_file(args: dict[str, JsonValue], ctx: ToolContext) -> str:
    path = resolve_in_workspace(ctx, _str_arg(args, "path"), tool="write_file")
    content = _str_arg(args, "content")
    if path.is_dir():
        raise ToolError(f"{_rel(ctx, path)} is a directory.", tool_name="write_file")

    def write() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    await asyncio.to_thread(write)
    return f"Wrote {len(content)} characters to {_rel(ctx, path)}."


WRITE_FILE = ToolSpec(
    Tool(
        name="write_file",
        description=(
            "Create or overwrite a text file in the workspace with exactly `content`. "
            "Parent directories are created as needed. To edit a file, read it first and write "
            "back the complete new contents."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to the workspace."},
                "content": {"type": "string", "description": "The complete new file contents."},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        timeout_s=10,
    ),
    _write_file,
)


# ----------------------------------------------------------------------------- list_dir


async def _list_dir(args: dict[str, JsonValue], ctx: ToolContext) -> str:
    root = resolve_in_workspace(ctx, _str_arg(args, "path", "."), tool="list_dir")
    recursive = args.get("recursive", False) is True
    if not root.is_dir():
        raise ToolError(f"No such directory: {_rel(ctx, root)}", tool_name="list_dir")

    def listing() -> list[str]:
        paths = root.rglob("*") if recursive else root.iterdir()
        entries: list[str] = []
        for p in sorted(paths):
            if len(entries) >= MAX_LIST_ENTRIES:
                entries.append(f"[... stopped after {MAX_LIST_ENTRIES} entries ...]")
                break
            entries.append(p.relative_to(root).as_posix() + ("/" if p.is_dir() else ""))
        return entries

    entries = await asyncio.to_thread(listing)
    return "\n".join(entries) if entries else "(empty directory)"


LIST_DIR = ToolSpec(
    Tool(
        name="list_dir",
        description=(
            "List the entries of a workspace directory, one per line, relative to that directory. "
            "Directories end with '/'. Set `recursive` to include all nested files."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path; defaults to '.'."},
                "recursive": {"type": "boolean", "description": "Walk subdirectories too."},
            },
            "additionalProperties": False,
        },
        timeout_s=10,
    ),
    _list_dir,
)


# ----------------------------------------------------------------------------- run_shell


async def _run_shell(args: dict[str, JsonValue], ctx: ToolContext) -> str:
    # TODO(phase-4): execute inside the sandbox; until then this runs on the host and the
    # guardrails require explicit approval for every call (see docs/adr/0003).
    command = _str_arg(args, "command")
    env = {k: v for k in _SHELL_ENV_KEYS if (v := os.environ.get(k)) is not None}
    proc = await asyncio.create_subprocess_exec(
        "bash",
        "-c",
        command,
        cwd=ctx.workspace,
        env=env,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,  # own process group, so a timeout kills its children too
    )
    try:
        stdout, _ = await proc.communicate()
    except BaseException:  # timeout/cancel: don't leave the command running
        if proc.returncode is None:
            os.killpg(proc.pid, signal.SIGKILL)
            await proc.wait()
        raise
    output = stdout.decode(errors="replace")
    return f"exit code: {proc.returncode}\n{output}" if output else f"exit code: {proc.returncode}"


RUN_SHELL = ToolSpec(
    Tool(
        name="run_shell",
        description=(
            "Run a shell command (bash) with the workspace as the working directory and return "
            "its exit code and combined stdout/stderr. Requires user approval each time. "
            "Non-interactive: stdin is empty. Environment variables are not inherited."
        ),
        input_schema={
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The command line."}},
            "required": ["command"],
            "additionalProperties": False,
        },
        risky=True,
        timeout_s=120,
    ),
    _run_shell,
)

BUILTINS: tuple[ToolSpec, ...] = (READ_FILE, WRITE_FILE, LIST_DIR, RUN_SHELL)
