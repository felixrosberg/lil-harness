"""A local model fixing a bug with the coding tools.

By default it copies examples/sample_project (a stats module with a bug and a failing
test) to a temporary folder and asks the model to fix it, so the original stays broken
and you can run it again:
    uv run python examples/coding.py ollama --model qwen3.5:4b

Point it at your own folder with --workspace, and give it a task:
    uv run python examples/coding.py ollama --workspace C:\\code\\app "Add docstrings"

Every write and command asks you first. Change that with --permissions (readonly,
default or auto) and --rules FILE (see permissions.example.toml), for example to let
"python test_stats.py" run without asking. Use auto only in a sandbox you can throw away:
an approved command can reach anything your user account can.
"""

import asyncio
import shutil
import tempfile
from pathlib import Path

import click
from openai import APIConnectionError

from lil_harness.agent.loop import run_agent
from lil_harness.backends.ollama import OllamaBackend
from lil_harness.backends.vllm import VLLMBackend
from lil_harness.permissions import PRESETS, PermissionPolicy, terminal_approver
from lil_harness.tools.coding import CodingTools
from lil_harness.types import Message

SAMPLE = Path(__file__).parent / "sample_project"
SAMPLE_TASK = "The test in test_stats.py fails. Find the bug in stats.py and fix it."

SYSTEM_PROMPT = """You are a coding assistant working in a project folder.
- Look before you change anything: list the files, and read the ones you need.
- Make the smallest change that does the job, with edit_file.
- If you can run commands, check your work, for example by running the tests.
- When you're done, say in one or two sentences what you changed."""


@click.command()
@click.argument("engine", type=click.Choice(["ollama", "vllm"]), default="ollama")
@click.argument("task", nargs=-1)
@click.option("--model", help="Model name. Defaults to qwen3:4b for Ollama; vLLM asks the server.")
@click.option("--think", is_flag=True, help="Turn on Qwen3's thinking mode.")
@click.option(
    "--workspace", type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Folder to work in. Default: a fresh copy of examples/sample_project.",
)  # fmt: skip
@click.option(
    "--permissions", "preset", type=click.Choice(list(PRESETS)),
    help="readonly: no writes or commands. default: ask first. auto: never ask.  [default: "
    "the rules file's preset, else default]",
)  # fmt: skip
@click.option(
    "--rules", type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="TOML file with permission rules, such as commands that run without asking.",
)  # fmt: skip
@click.option("--max-steps", default=20, show_default=True)
def main(
    engine: str,
    task: tuple[str, ...],
    model: str | None,
    think: bool,
    workspace: Path | None,
    preset: str | None,
    rules: Path | None,
    max_steps: int,
) -> None:
    """Give a local model a coding task in a folder.

    \f
    Args:
        engine: "ollama" or "vllm".
        task: The task, as separate words. Defaults to fixing the sample project.
        model: The model name, or None for the engine's default.
        think: Turn on thinking mode.
        workspace: The folder to work in, or None for a fresh copy of the sample.
        preset: Overrides the rules file's preset; "default" if neither is given.
        rules: A TOML permission rules file.
        max_steps: The step budget for the agent loop.
    """
    if workspace is None:
        workspace = Path(tempfile.mkdtemp(prefix="lil_harness_")) / "sample_project"
        shutil.copytree(SAMPLE, workspace)
    if rules:
        policy = PermissionPolicy.from_file(rules, preset=preset, approver=terminal_approver)
    else:
        policy = PermissionPolicy.preset(preset or "default", approver=terminal_approver)
    # The rules file can't be changed by the model, even if it sits in the workspace.
    tools = CodingTools(workspace, protected=[rules] if rules else [])
    text = " ".join(task) or SAMPLE_TASK
    asyncio.run(run(engine, text, model, think, tools, policy, max_steps))


async def run(
    engine: str,
    task: str,
    model: str | None,
    think: bool,
    tools: CodingTools,
    policy: PermissionPolicy,
    max_steps: int,
) -> None:
    """Run the agent on the task and print every call and result.

    Args:
        engine: "ollama" or "vllm".
        task: What to do.
        model: The model name, or None for the engine's default.
        think: Turn on thinking mode.
        tools: The coding tools, already bound to the workspace.
        policy: The permission policy, with the terminal as approver.
        max_steps: The step budget for the agent loop.

    Raises:
        click.ClickException: If the server can't be reached.
    """
    backend_cls = OllamaBackend if engine == "ollama" else VLLMBackend
    backend = backend_cls(enable_thinking=think, **({"model": model} if model else {}))
    registry = tools.registry(policy)
    click.echo(f"workspace: {tools.root}")
    modes = ", ".join(f"{risk} {mode}" for risk, mode in policy.modes.items())
    click.echo(f"permissions: {modes}")
    click.echo(f"tools: {', '.join(spec.name for spec in registry.specs())}\n")
    try:
        result = await run_agent(
            backend,
            registry,
            [Message.system(SYSTEM_PROMPT), Message.user(task)],
            max_steps=max_steps,
        )
    except APIConnectionError:
        raise click.ClickException(
            f"Can't reach {backend.client.base_url}. Is the {engine} server running?"
        ) from None

    for message in result.messages[2:]:
        if message.reasoning:
            click.secho(f"[thinking] {message.reasoning.strip()}", dim=True)
        for tool_call in message.tool_calls:
            click.secho(f"[call] {tool_call.name}({tool_call.raw_arguments})", fg="cyan")
        if message.role == "tool":
            # Tool results can be long; show the first few lines.
            lines = (message.content or "").splitlines()
            shown = "\n       ".join(lines[:8]) + ("\n       ..." if len(lines) > 8 else "")
            click.secho(f"[tool] {shown}", dim=True)
        elif message.content:
            click.echo(f"[{message.role}] {message.content}")
    click.echo(f"\nstop_reason={result.stop_reason} steps={result.steps}")
    click.echo(f"The changed files are in {tools.root}")


if __name__ == "__main__":
    main()
