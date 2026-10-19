"""Run one agent task in a workspace, and save what happened. Made for a container.

The container is the sandbox here: it sees only /workspace (the files to work on) and
/output (where the record goes), and on the compose network it can reach the model
server but not the internet. So the permission policy defaults to "auto": nobody is
there to answer a prompt, and the container is what keeps the agent in.

    python run_task.py "Add type hints to utils.py"
    python run_task.py --seed /app/seeds/sample_project --check "python test_stats.py" \\
        "The test in test_stats.py fails. Find the bug in stats.py and fix it."

The model comes from LIL_BASE_URL and LIL_MODEL (and LIL_API_KEY for hosted APIs), so
the same image works with Ollama, vLLM or a hosted model.

After the run, /output/run.json holds the task, the model, how the run ended, the
optional check's result and the whole conversation, so whatever started the
container can see what happened without reading its logs.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypeAlias

import click
from openai import APIConnectionError

from lil_harness.agent.loop import RunResult, run_agent
from lil_harness.backends.base import ChatBackend
from lil_harness.backends.openai_compat import OpenAICompatBackend
from lil_harness.permissions import PRESETS, PermissionPolicy
from lil_harness.tools.coding import CodingTools
from lil_harness.types import Message

RunRecord: TypeAlias = dict[str, Any]
"""What goes in run.json; see `make_record`."""

SYSTEM_PROMPT = """You are a coding assistant working in a project folder.
- Look before you change anything: list the files, and read the ones you need.
- Make the smallest change that does the job, with edit_file.
- Check your work, for example by running the tests.
- When you're done, say in one or two sentences what you changed."""


def seed_workspace(seed: Path, workspace: Path) -> bool:
    """Copy a starting project into the workspace, if the workspace is empty.

    Args:
        seed: The folder to copy, e.g. a project with a known bug.
        workspace: Where the agent works.

    Returns:
        True if it copied, False if the workspace already had files (they're kept, so
        a second run continues where the first left off).
    """
    workspace.mkdir(parents=True, exist_ok=True)
    if any(workspace.iterdir()):
        return False
    # Copy what's inside the seed, not the folder itself: the workspace is a mount
    # point, and its own owner and permissions belong to the host.
    for item in seed.iterdir():
        if item.is_dir():
            shutil.copytree(item, workspace / item.name)
        else:
            shutil.copy2(item, workspace / item.name)
    return True


def run_check(command: str, workspace: Path, timeout: float = 120) -> dict[str, Any]:
    """Run a command after the agent is done, to see whether the task was solved.

    Args:
        command: A shell command, e.g. "python test_stats.py". Exit code 0 means passed.
        workspace: The folder to run it in.
        timeout: Seconds before the command is stopped and counted as failed.

    Returns:
        The command, whether it passed, its exit code, and the end of its output.
    """
    try:
        completed = subprocess.run(
            command, shell=True, cwd=workspace, capture_output=True, text=True, timeout=timeout
        )
        exit_code, output = completed.returncode, completed.stdout + completed.stderr
    except subprocess.TimeoutExpired:
        exit_code, output = None, f"Timed out after {timeout} seconds"
    return {
        "command": command,
        "passed": exit_code == 0,
        "exit_code": exit_code,
        "output": output[-2000:],
    }


def make_record(
    task: str, model: str, result: RunResult, seconds: float, check: dict[str, Any] | None
) -> RunRecord:
    """Put everything about a run into one JSON-ready dict.

    Args:
        task: The task the agent was given.
        model: The model's name.
        result: What run_agent returned.
        seconds: How long the agent ran.
        check: What run_check returned, or None if there was no check.

    Returns:
        The record: task, model, stop_reason, steps, seconds, answer, check, messages.
    """
    return {
        "task": task,
        "model": model,
        "stop_reason": result.stop_reason,
        "steps": result.steps,
        "seconds": round(seconds, 1),
        "answer": result.answer,
        "check": check,
        "messages": [asdict(message) for message in result.messages],
    }


def print_transcript(result: RunResult) -> None:
    """Print every call, the start of every result, and every reply, like step 5.

    Args:
        result: What run_agent returned.
    """
    for message in result.messages[2:]:
        for tool_call in message.tool_calls:
            click.secho(f"[call] {tool_call.name}({tool_call.raw_arguments})", fg="cyan")
        if message.role == "tool":
            lines = (message.content or "").splitlines()
            shown = "\n       ".join(lines[:6]) + ("\n       ..." if len(lines) > 6 else "")
            click.secho(f"[tool] {shown}", dim=True)
        elif message.content:
            click.echo(f"[{message.role}] {message.content}")


async def run_task(
    backend: ChatBackend,
    task: str,
    workspace: Path,
    policy: PermissionPolicy,
    *,
    max_steps: int = 20,
) -> RunResult:
    """Run the agent on one task with the coding tools.

    Args:
        backend: The model.
        task: What to do.
        workspace: The folder the tools are confined to.
        policy: Which tool calls run. In the container, usually "auto".
        max_steps: The step budget.

    Returns:
        How the run ended.
    """
    registry = CodingTools(workspace).registry(policy)
    messages = [Message.system(SYSTEM_PROMPT), Message.user(task)]
    return await run_agent(backend, registry, messages, max_steps=max_steps)


@click.command()
@click.argument("task", nargs=-1, required=True)
@click.option(
    "--workspace", type=click.Path(file_okay=False, path_type=Path),
    default=Path("/workspace"), show_default=True,
)  # fmt: skip
@click.option(
    "--output", type=click.Path(file_okay=False, path_type=Path),
    default=Path("/output"), show_default=True, help="run.json is written here.",
)  # fmt: skip
@click.option(
    "--seed", type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Copy this project into the workspace first, if the workspace is empty.",
)  # fmt: skip
@click.option("--check", help='Run this after the agent, e.g. "python test_stats.py".')
@click.option(
    "--permissions", "preset", type=click.Choice(list(PRESETS)), default="auto",
    show_default=True, help="Inside a sandbox there's nobody to ask, so auto.",
)  # fmt: skip
@click.option("--max-steps", default=20, show_default=True)
def main(
    task: tuple[str, ...],
    workspace: Path,
    output: Path,
    seed: Path | None,
    check: str | None,
    preset: str,
    max_steps: int,
) -> None:
    """Run one coding task and save the record to OUTPUT/run.json.

    \f
    Args:
        task: The task, as separate words.
        workspace: The folder to work in.
        output: Where run.json goes.
        seed: A project to copy into an empty workspace first, or None.
        check: A command whose exit code says whether the task was solved, or None.
        preset: The permission preset.
        max_steps: The step budget.

    Raises:
        click.ClickException: If the model server can't be reached.
    """
    if seed and seed_workspace(seed, workspace):
        click.echo(f"Copied {seed} into {workspace}")
    backend = OpenAICompatBackend.from_env()
    text = " ".join(task)
    click.echo(f"model: {backend.model} at {backend.client.base_url}\ntask: {text}\n")

    started = time.perf_counter()
    try:
        result = asyncio.run(
            run_task(backend, text, workspace, PermissionPolicy.preset(preset), max_steps=max_steps)
        )
    except APIConnectionError:
        raise click.ClickException(
            f"Can't reach {backend.client.base_url}. Is the model server running?"
        ) from None
    seconds = time.perf_counter() - started
    print_transcript(result)

    check_result = run_check(check, workspace) if check else None
    record = make_record(text, backend.model, result, seconds, check_result)
    output.mkdir(parents=True, exist_ok=True)
    (output / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

    click.echo(f"\nstop_reason={result.stop_reason} steps={result.steps} seconds={seconds:.0f}")
    if check_result:
        verdict = "passed" if check_result["passed"] else "FAILED"
        click.secho(f"check {verdict}: {check}", fg="green" if check_result["passed"] else "red")
    click.echo(f"Record saved to {output / 'run.json'}")
    if check_result and not check_result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
