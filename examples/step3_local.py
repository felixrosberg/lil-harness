"""Step 3 demo: the agent loop against a model running on your own GPU.

Start Ollama or vLLM first (see the README), then run one of:
    uv run python examples/step3_local.py ollama "What is 1234 * 5678?"
    uv run python examples/step3_local.py vllm "What is 1234 * 5678?"

Add --think to turn on Qwen3's thinking mode, or --model NAME to try another model,
for example --model qwen3.5:4b with Ollama. Run with --help to see all options.
"""

import asyncio

import click
from openai import APIConnectionError

from lil_harness.agent.loop import run_agent
from lil_harness.backends.ollama import OllamaBackend
from lil_harness.backends.vllm import VLLMBackend
from lil_harness.tools.builtin import default_registry
from lil_harness.types import Message

registry = default_registry()


@click.command()
@click.argument("engine", type=click.Choice(["ollama", "vllm"]), default="ollama")
@click.argument("question", nargs=-1)
@click.option("--model", help="Model name. Defaults to qwen3:4b for Ollama; vLLM asks the server.")
@click.option("--think", is_flag=True, help="Turn on Qwen3's thinking mode.")
def main(engine: str, question: tuple[str, ...], model: str | None, think: bool) -> None:
    """Ask a question to a local model that can use a calculator and a clock.

    \f
    Args:
        engine: "ollama" or "vllm".
        question: The question, as separate words from the command line.
        model: The model name, or None for the engine's default.
        think: Turn on thinking mode.
    """
    text = " ".join(question) or "What is 1234 * 5678, and what time is it in UTC?"
    # click is synchronous, so this is where we enter async code.
    asyncio.run(run(engine, text, model, think))


async def run(engine: str, question: str, model: str | None, think: bool) -> None:
    """Run the agent once and print every message, with thinking dimmed.

    Args:
        engine: "ollama" or "vllm".
        question: The user's question.
        model: The model name, or None for the engine's default.
        think: Turn on thinking mode.

    Raises:
        click.ClickException: If the server can't be reached.
    """
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
