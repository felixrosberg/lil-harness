"""Retrieval evals: for questions with a known answer location, does search find it?

This tests the index on its own, without a chat model, so it's fast and pinpoints
retrieval problems. If the right chunk isn't in the top 5, no model can answer from it.

Cases live in YAML under `evals/retrieval/`:

    - question: How many tool calls can run at once in the eval runner?
      source: src/lil_harness/evals/runner.py
      contains: Semaphore

A search result is a hit when it comes from `source` and its text contains `contains`
(case-insensitive). The numbers per search mode:

- **recall@k:** the share of questions with at least one hit in the top k (the
  results the model would see).
- **MRR** (mean reciprocal rank): 1 for a hit in first place, 1/2 in second, 1/3 in
  third, ... and 0 for none in the top k, averaged over questions. It rewards ranking
  the right chunk high, not just somewhere in the top k.
- **recall@depth** (default 20): the same for the top 20. That's the shortlist a
  reranker gets, so it's the most reranking could ever reach.
- **s/query:** seconds per search, since a reranker costs a model call per result.

`lil-rag-eval` runs every mode (hybrid, semantic, keyword) on the same cases, so you
can see what each kind of search contributes. With `--reranker llm` or
`--reranker cross-encoder` it adds a "hybrid+rerank" row.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path

import click
import yaml
from openai import APIConnectionError, NotFoundError
from pydantic import BaseModel, ConfigDict

from lil_harness.backends.ollama import OllamaBackend
from lil_harness.rag.embeddings import Embedder, HashEmbedder, OllamaEmbedder
from lil_harness.rag.index import DocIndex, SearchHit, SearchMode, ensure_index
from lil_harness.rag.rerank import CrossEncoderReranker, LLMReranker, Reranker


class RetrievalCase(BaseModel):
    """One question and where its answer is.

    Attributes:
        question: The search query.
        source: The file the answer is in, relative to the corpus folder.
        contains: Text the right chunk must contain, to tell apart chunks of the same
            file. Leave it out to accept any chunk of `source`.
    """

    model_config = ConfigDict(extra="forbid")

    question: str
    source: str
    contains: str | None = None

    def is_hit(self, hit: SearchHit) -> bool:
        """Whether a search result answers this case.

        Args:
            hit: One search result.

        Returns:
            True if it comes from `source` and contains `contains`.
        """
        if hit.chunk.source != self.source:
            return False
        return self.contains is None or self.contains.lower() in hit.chunk.text.lower()


def load_retrieval_cases(directory: Path) -> list[RetrievalCase]:
    """Load every case from the .yaml files in `directory`.

    Args:
        directory: The folder with the case files, usually evals/retrieval.

    Returns:
        All cases, in file and list order.
    """
    cases: list[RetrievalCase] = []
    for path in sorted(directory.glob("*.yaml")):
        for item in yaml.safe_load(path.read_text(encoding="utf-8")) or []:
            cases.append(RetrievalCase.model_validate(item))
    return cases


@dataclass
class RetrievalScore:
    """How well one way of searching did.

    Attributes:
        label: The row name in the report, e.g. "hybrid" or "hybrid+rerank".
        k: The cutoff for recall@k and MRR, i.e. how many results the model sees.
        depth: How many results were fetched, for recall@depth.
        hit_ranks: For each case, the rank of its first hit (1 = top) within `depth`,
            or None.
        seconds: Total search time over all cases.
    """

    label: str
    k: int
    depth: int
    hit_ranks: list[int | None] = field(default_factory=list)
    seconds: float = 0.0

    def recall_at(self, n: int) -> float:
        """The share of cases with a hit in the top n.

        Args:
            n: The cutoff, at most `depth`.

        Returns:
            A fraction from 0 to 1.
        """
        hits = sum(rank is not None and rank <= n for rank in self.hit_ranks)
        return hits / (len(self.hit_ranks) or 1)

    @property
    def recall(self) -> float:
        """Returns: recall@k, the share of cases with a hit in the top k."""
        return self.recall_at(self.k)

    @property
    def mrr(self) -> float:
        """Returns: the mean of 1/rank within the top k, counting a miss as 0."""
        total = sum(1 / rank for rank in self.hit_ranks if rank and rank <= self.k)
        return total / (len(self.hit_ranks) or 1)


async def score_retrieval(
    index: DocIndex,
    embedder: Embedder,
    cases: list[RetrievalCase],
    *,
    mode: SearchMode,
    k: int = 5,
    depth: int = 20,
    reranker: Reranker | None = None,
) -> RetrievalScore:
    """Search every case's question and record where the first hit landed.

    Args:
        index: The index to test.
        embedder: The model the index was built with.
        cases: The questions and where their answers are.
        mode: Which search to use.
        k: The cutoff for recall@k and MRR.
        depth: How many results to fetch per question. With a reranker, it reranks
            this many, so recall@depth is the same with and without one.
        reranker: Rerank the results with this, or None.

    Returns:
        The rank of the first hit for each case, and the time taken.
    """
    label = f"{mode}+rerank" if reranker else mode
    score = RetrievalScore(label, k, depth)
    started = time.perf_counter()
    for case in cases:
        hits = await index.search(
            case.question, embedder, k=depth, mode=mode, reranker=reranker, rerank_pool=depth
        )
        first = next((rank for rank, hit in enumerate(hits, 1) if case.is_hit(hit)), None)
        score.hit_ranks.append(first)
    score.seconds = time.perf_counter() - started
    return score


def make_embedder(kind: str, model: str | None) -> Embedder:
    """Pick an embedder from command-line options.

    Args:
        kind: "ollama" or "hash".
        model: The Ollama model name, or None for the default.

    Returns:
        The embedder.
    """
    if kind == "hash":
        return HashEmbedder()
    return OllamaEmbedder(**({"model": model} if model else {}))


def make_reranker(kind: str, model: str | None) -> Reranker | None:
    """Pick a reranker from command-line options.

    Args:
        kind: "none", "llm" (a chat model through Ollama) or "cross-encoder".
        model: The model name, or None for the default (qwen3:4b, or
            BAAI/bge-reranker-v2-m3).

    Returns:
        The reranker, or None for "none".

    Raises:
        click.ClickException: If the cross-encoder's library isn't installed.
    """
    if kind == "llm":
        backend = OllamaBackend(**({"model": model} if model else {}))
        return LLMReranker(backend, name=f"llm:{backend.model}")
    if kind == "cross-encoder":
        try:
            return CrossEncoderReranker(**({"model": model} if model else {}))
        except ImportError as error:
            raise click.ClickException(str(error)) from None
    return None


@click.command()
@click.option(
    "--embedder", "embedder_kind", type=click.Choice(["ollama", "hash"]), default="ollama",
    show_default=True, help="hash needs no model: a word-matching baseline.",
)  # fmt: skip
@click.option("--embed-model", help="Ollama embedding model. Default: nomic-embed-text.")
@click.option(
    "--reranker", "reranker_kind", type=click.Choice(["none", "llm", "cross-encoder"]),
    default="none", show_default=True,
    help="Also score hybrid search with this reranker. llm uses a chat model in Ollama; "
    "cross-encoder needs: uv sync --extra rerank",
)  # fmt: skip
@click.option(
    "--rerank-model",
    help="Default: qwen3:4b for llm, BAAI/bge-reranker-v2-m3 for cross-encoder.",
)
@click.option(
    "--root", type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("."), show_default=True, help="The folder to index.",
)  # fmt: skip
@click.option(
    "--index-dir", type=click.Path(file_okay=False, path_type=Path),
    default=Path(".lil_index"), show_default=True,
)  # fmt: skip
@click.option(
    "--cases-dir", type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path("evals/retrieval"), show_default=True,
)  # fmt: skip
@click.option("-k", "k", default=5, show_default=True, help="Results the model would see.")
@click.option(
    "--depth", default=20, show_default=True,
    help="Results fetched per question, and how many the reranker reorders.",
)  # fmt: skip
@click.option("--rebuild", is_flag=True, help="Rebuild the index even if it's current.")
@click.option("-v", "--verbose", is_flag=True, help="List the questions each mode missed.")
def main(
    embedder_kind: str,
    embed_model: str | None,
    reranker_kind: str,
    rerank_model: str | None,
    root: Path,
    index_dir: Path,
    cases_dir: Path,
    k: int,
    depth: int,
    rebuild: bool,
    verbose: bool,
) -> None:
    """Measure whether search finds the right chunks, for each search mode.

    \f
    Args:
        embedder_kind: "ollama" or "hash".
        embed_model: The Ollama embedding model, or None for the default.
        reranker_kind: "none", "llm" or "cross-encoder".
        rerank_model: The reranker's model, or None for the default.
        root: The folder to index.
        index_dir: Where the index is saved. Each embedder gets its own subfolder.
            It's kept apart from step 6's index, since this one leaves out the cases.
        cases_dir: Where the retrieval cases are.
        k: The cutoff for recall@k and MRR: how many results the model would see.
        depth: How many results to fetch per question, and to rerank.
        rebuild: Rebuild the index even if it's current.
        verbose: List the questions each mode missed.

    Raises:
        click.ClickException: If the embedding server can't be reached or lacks the model.
    """
    embedder = make_embedder(embedder_kind, embed_model)
    reranker = make_reranker(reranker_kind, rerank_model)
    cases = load_retrieval_cases(cases_dir)
    # The case files hold the questions word for word, so if they were indexed they'd
    # come out on top of every search. Leave them out.
    try:
        exclude = (f"{cases_dir.resolve().relative_to(root.resolve()).as_posix()}/*",)
    except ValueError:  # the cases aren't inside the corpus
        exclude = ()
    folder = index_dir / f"eval-{embedder.name.replace(':', '_')}"
    asyncio.run(run(embedder, root, folder, cases, k, depth, rebuild, verbose, exclude, reranker))


async def run(
    embedder: Embedder,
    root: Path,
    index_dir: Path,
    cases: list[RetrievalCase],
    k: int,
    depth: int,
    rebuild: bool,
    verbose: bool,
    exclude: tuple[str, ...] = (),
    reranker: Reranker | None = None,
) -> None:
    """Build or load the index, score every mode, and print a table.

    Args:
        embedder: The embedding model.
        root: The folder to index.
        index_dir: Where this embedder's index is saved.
        cases: The retrieval cases.
        k: The cutoff for recall@k and MRR.
        depth: How many results to fetch per question, and to rerank.
        rebuild: Rebuild the index even if it's current.
        verbose: List the questions each mode missed.
        exclude: Patterns for files to leave out of the index.
        reranker: If given, hybrid search is also scored with it, as "hybrid+rerank".

    Raises:
        click.ClickException: If the Ollama server can't be reached or lacks a model.
    """
    try:
        index, built = await ensure_index(
            root, index_dir, embedder, exclude=exclude, rebuild=rebuild
        )
        click.echo(
            f"{'Built' if built else 'Loaded'} index: {len(index.chunks)} chunks, {embedder.name}"
        )
        scores = [
            await score_retrieval(index, embedder, cases, mode=mode, k=k, depth=depth)
            for mode in ("hybrid", "semantic", "keyword")
        ]
        if reranker:
            click.echo(f"Reranking the top {depth} with {reranker.name}...")
            scores.append(
                await score_retrieval(
                    index, embedder, cases, mode="hybrid", k=k, depth=depth, reranker=reranker
                )
            )
    except APIConnectionError:
        raise click.ClickException("Can't reach Ollama. Is it running?") from None
    except NotFoundError as error:
        raise click.ClickException(
            f"Ollama is missing a model ({error.message}). Pull it with: ollama pull <model>"
        ) from None

    recall_k, recall_depth = f"recall@{k}", f"recall@{depth}"
    click.echo(f"\n  {len(cases)} questions\n")
    click.echo(f"  {'mode':<14} {recall_k:>9}  {'MRR':>5}  {recall_depth:>9}  {'s/query':>7}")
    for score in scores:
        per_query = score.seconds / (len(cases) or 1)
        click.echo(
            f"  {score.label:<14} {score.recall:>9.0%}  {score.mrr:>5.2f}"
            f"  {score.recall_at(depth):>9.0%}  {per_query:>7.2f}"
        )
        if verbose:
            for case, rank in zip(cases, score.hit_ranks, strict=True):
                if rank is None or rank > k:
                    where = f"rank {rank}" if rank else "not found"
                    click.secho(f"                 missed ({where}): {case.question}", dim=True)


if __name__ == "__main__":
    main()
