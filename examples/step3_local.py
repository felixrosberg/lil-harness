"""Step 3 demo: the agent loop against a model running on your own GPU.

Start Ollama or vLLM first (see the README), then run one of:
    uv run python examples/step3_local.py ollama "What is 1234 * 5678?"
    uv run python examples/step3_local.py vllm "What is 1234 * 5678?"

Add --think to turn on Qwen3's thinking mode, or --model NAME to try another model,
for example --model qwen3.5:4b with Ollama. Run with --help to see all options.
"""

import asyncio
from datetime import UTC, datetime
from typing import Literal

import click
from openai import APIConnectionError

from lil_harness.agent.loop import run_agent
from lil_harness.backends.ollama import OllamaBackend
from lil_harness.backends.vllm import VLLMBackend
from lil_harness.tools.registry import ToolRegistry
from lil_harness.types import Message

registry = ToolRegistry()


@registry.tool
def calculate(a: float, b: float, op: Literal["add", "subtract", "multiply", "divide"]):
    """Do basic arithmetic on two numbers. Use this instead of doing math in your head."""
    return {"add": a + b, "subtract": a - b, "multiply": a * b, "divide": a / b}[op]


@registry.tool
def current_time_utc() -> str:
    """Get the current date and time in UTC, in ISO 8601 format."""
    return datetime.now(UTC).isoformat(timespec="seconds")


@click.command()
@click.argument("engine", type=click.Choice(["ollama", "vllm"]), default="ollama")
@click.argument("question", nargs=-1)
@click.option("--model", help="Model name. Defaults to qwen3:4b for Ollama; vLLM asks the server.")
@click.option("--think", is_flag=True, help="Turn on Qwen3's thinking mode.")
def main(engine: str, question: tuple[str, ...], model: str | None, think: bool) -> None:
    """Ask a question to a local model that can use a calculator and a clock."""
    text = " ".join(question) or "What is 1234 * 5678, and what time is it in UTC?"
    # click is synchronous, so this is where we enter async code.
    asyncio.run(run(engine, text, model, think))


async def run(engine: str, question: str, model: str | None, think: bool) -> None:
    # The only engine-specific line: everything below is the same for both.
    backend_cls = OllamaBackend if engine == "ollama" else VLLMBackend
    backend = backend_cls(enable_thinking=think, **({"model": model} if model else {}))
    try:
        result = await run_agent(
            backend,
            registry,
            [
                Message.system("You are a helpful assistant. Use the tools when they help."),
                Message.user(question),
            ],
        )
    except APIConnectionError:
        raise click.ClickException(
            f"Can't reach {backend.client.base_url}. Is the {engine} server running?"
        ) from None
    click.echo(f"{engine}: {backend.model} (thinking {'on' if think else 'off'})\n")

    for m in result.messages[1:]:
        if m.reasoning:
            click.secho(f"[thinking] {m.reasoning.strip()}", dim=True)
        if m.tool_calls:
            for c in m.tool_calls:
                # raw_arguments is exactly what the model emitted.
                click.echo(f"[{m.role}] calls {c.name}({c.raw_arguments})")
        else:
            click.echo(f"[{m.role}] {m.content}")
    click.echo(f"\nstop_reason={result.stop_reason} steps={result.steps}")


if __name__ == "__main__":
    main()
