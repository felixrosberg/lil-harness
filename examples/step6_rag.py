"""Step 6 demo: ask questions about a folder of docs and code, answered with search.

It indexes this repository (or --root), gives the model `search_docs` plus the
read-only coding tools, and asks your question:
    ollama pull nomic-embed-text
    uv run python examples/step6_rag.py ollama --model qwen3.5:4b "How do permissions work?"

The index is saved in .lil_index/ and reused until a file changes. Try
--embedder hash to run without an embedding model: search then only matches words.
Add --reranker llm (your chat model rates each result) or --reranker cross-encoder
(needs `uv sync --extra rerank`) to rerank every search's top 20.
"""

import asyncio
from pathlib import Path

import click
from openai import APIConnectionError, NotFoundError

from lil_harness.agent.loop import run_agent
from lil_harness.backends.ollama import OllamaBackend
from lil_harness.backends.vllm import VLLMBackend
from lil_harness.permissions import PermissionPolicy
from lil_harness.rag.embeddings import Embedder
from lil_harness.rag.evaluate import make_embedder, make_reranker
from lil_harness.rag.index import ensure_index
from lil_harness.rag.rerank import Reranker
from lil_harness.rag.tools import RagTools
from lil_harness.tools.coding import CodingTools
from lil_harness.types import Message

REPO = Path(__file__).parent.parent
SAMPLE_QUESTION = "How does the harness decide whether a tool call needs my permission?"

SYSTEM_PROMPT = """You answer questions about a project, using its files.
- Start with search_docs. Search with names you expect in the answer, such as
  function, setting or file names.
- If the results don't answer the question, search again with different words, or
  read the file around a result with read_file.
- Answer only from what you found. Cite each fact as file:lines, like
  (src/app.py:10-24). If you couldn't find it, say so."""


@click.command()
@click.argument("engine", type=click.Choice(["ollama", "vllm"]), default="ollama")
@click.argument("question", nargs=-1)
@click.option("--model", help="Chat model. Defaults to qwen3:4b for Ollama; vLLM asks the server.")
@click.option("--think", is_flag=True, help="Turn on Qwen3's thinking mode.")
@click.option(
    "--embedder", "embedder_kind", type=click.Choice(["ollama", "hash"]), default="ollama",
    show_default=True, help="hash needs no model: a word-matching baseline.",
)  # fmt: skip
@click.option("--embed-model", help="Ollama embedding model. Default: nomic-embed-text.")
@click.option(
    "--reranker", "reranker_kind", type=click.Choice(["none", "llm", "cross-encoder"]),
    default="none", show_default=True, help="Rerank each search's top 20 results.",
)  # fmt: skip
@click.option(
    "--rerank-model",
    help="Default: the chat model for llm, BAAI/bge-reranker-v2-m3 for cross-encoder.",
)
@click.option(
    "--root", type=click.Path(exists=True, file_okay=False, path_type=Path), default=REPO,
    help="The folder to index and answer about. Default: this repository.",
)  # fmt: skip
@click.option(
    "--index-dir", type=click.Path(file_okay=False, path_type=Path),
    help="Where to save the index. Default: .lil_index in the root folder.",
)  # fmt: skip
@click.option("--rebuild", is_flag=True, help="Rebuild the index even if it's current.")
@click.option("--max-steps", default=10, show_default=True)
def main(
    engine: str,
    question: tuple[str, ...],
    model: str | None,
    think: bool,
    embedder_kind: str,
    embed_model: str | None,
    reranker_kind: str,
    rerank_model: str | None,
    root: Path,
    index_dir: Path | None,
    rebuild: bool,
    max_steps: int,
) -> None:
    """Ask a local model a question about a folder, answered with search.

    \f
    Args:
        engine: "ollama" or "vllm", for the chat model.
        question: The question, as separate words. Defaults to one about permissions.
        model: The chat model name, or None for the engine's default.
        think: Turn on thinking mode.
        embedder_kind: "ollama" or "hash".
        embed_model: The Ollama embedding model, or None for the default.
        reranker_kind: "none", "llm" or "cross-encoder".
        rerank_model: The reranker's model, or None for the default.
        root: The folder to index.
        index_dir: Where the index is saved, or None for root/.lil_index.
        rebuild: Rebuild the index even if it's current.
        max_steps: The step budget for the agent loop.
    """
    embedder = make_embedder(embedder_kind, embed_model)
    # The LLM reranker uses the chat model unless told otherwise.
    if reranker_kind == "llm" and not rerank_model:
        rerank_model = model
    reranker = make_reranker(reranker_kind, rerank_model)
    # One subfolder per embedder, so switching models doesn't throw away an index.
    folder = (index_dir or root / ".lil_index") / embedder.name.replace(":", "_")
    text = " ".join(question) or SAMPLE_QUESTION
    asyncio.run(
        run(engine, text, model, think, embedder, reranker, root, folder, rebuild, max_steps)
    )


async def run(
    engine: str,
    question: str,
    model: str | None,
    think: bool,
    embedder: Embedder,
    reranker: Reranker | None,
    root: Path,
    folder: Path,
    rebuild: bool,
    max_steps: int,
) -> None:
    """Build or load the index, run the agent, and print every call and result.

    Args:
        engine: "ollama" or "vllm", for the chat model.
        question: What to ask.
        model: The chat model name, or None for the engine's default.
        think: Turn on thinking mode.
        embedder: The embedding model.
        reranker: Reranks each search's results, or None.
        root: The folder to index.
        folder: Where this embedder's index is saved.
        rebuild: Rebuild the index even if it's current.
        max_steps: The step budget for the agent loop.

    Raises:
        click.ClickException: If a server can't be reached or lacks the embedding model.
    """
    try:
        index, built = await ensure_index(root, folder, embedder, rebuild=rebuild)
    except APIConnectionError:
        raise click.ClickException("Can't reach Ollama for embeddings. Is it running?") from None
    except NotFoundError:
        raise click.ClickException(
            f"Ollama doesn't have {embedder.name!r}. Run: ollama pull {embedder.name}"
        ) from None
    click.echo(f"{'Built' if built else 'Loaded'} index: {len(index.chunks)} chunks from {root}")

    # Read-only: the model can search and read, but not change anything or run commands.
    registry = CodingTools(root).registry(PermissionPolicy.preset("readonly"))
    RagTools(index, embedder, reranker=reranker).register(registry)
    click.echo(f"tools: {', '.join(spec.name for spec in registry.specs())}\n")

    backend_cls = OllamaBackend if engine == "ollama" else VLLMBackend
    backend = backend_cls(enable_thinking=think, **({"model": model} if model else {}))
    try:
        result = await run_agent(
            backend,
            registry,
            [Message.system(SYSTEM_PROMPT), Message.user(question)],
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
            # Search results are long; show the first few lines.
            lines = (message.content or "").splitlines()
            shown = "\n       ".join(lines[:8]) + ("\n       ..." if len(lines) > 8 else "")
            click.secho(f"[tool] {shown}", dim=True)
        elif message.content:
            click.echo(f"[{message.role}] {message.content}")
    click.echo(f"\nstop_reason={result.stop_reason} steps={result.steps}")


if __name__ == "__main__":
    main()
