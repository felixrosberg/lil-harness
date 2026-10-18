"""The search index: chunks, their vectors, and hybrid search over both.

Searching runs two searches and merges them:

- **Semantic:** embed the query and rank chunks by cosine similarity (how closely
  their vectors point the same way). Finds "how do I limit concurrent requests?" in a
  chunk about `asyncio.Semaphore`, even with no words in common.
- **Keyword:** BM25 over the chunk texts. Finds `run_agent` exactly.

The two produce scores on different scales, so they're merged by rank, not by score:
reciprocal rank fusion gives each chunk 1 / (60 + rank) from each list and adds them
up. A chunk near the top of both lists wins; one that's first in one list and absent
from the other still does well.

An index is saved as a folder: `chunks.jsonl` (one chunk per line), `vectors.npy`
(the embeddings), and `meta.json` (which model made them, and a fingerprint of the
files, so a changed corpus is noticed and rebuilt).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from lil_harness.rag.chunking import DEFAULT_PATTERNS, Chunk, corpus_files, load_corpus
from lil_harness.rag.embeddings import Embedder
from lil_harness.rag.keyword import BM25
from lil_harness.rag.rerank import Reranker

SearchMode = Literal["hybrid", "semantic", "keyword"]

# The usual constant for reciprocal rank fusion. Larger values flatten the difference
# between rank 1 and rank 10.
RRF_K = 60

# How many results each search contributes to the fusion. Fixed rather than tied to
# k, so the top 5 of a k=20 search are the same as a k=5 search's.
FUSION_POOL = 50

# How many fused results a reranker looks at.
RERANK_POOL = 20


@dataclass
class SearchHit:
    """One search result.

    Attributes:
        chunk: The chunk that matched.
        score: The fused score (higher is better). Only comparable within one search.
        semantic_rank: Its place in the semantic results (1 = best), or None.
        keyword_rank: Its place in the keyword results (1 = best), or None if it shares
            no words with the query.
        rerank_score: The reranker's score, or None without a reranker. When set, the
            hits are ordered by it.
    """

    chunk: Chunk
    score: float
    semantic_rank: int | None = None
    keyword_rank: int | None = None
    rerank_score: float | None = None


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to length 1, so a dot product is the cosine similarity.

    Args:
        vectors: An array of shape (n, dimensions).

    Returns:
        The scaled array. All-zero rows stay zero.
    """
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(lengths == 0, 1, lengths)


def ranks(scores: np.ndarray, *, limit: int, positive_only: bool = False) -> dict[int, int]:
    """Rank positions by score.

    Args:
        scores: One score per chunk.
        limit: Keep only this many of the best.
        positive_only: Drop chunks scoring 0 or less (for keyword search, no match).

    Returns:
        Chunk position -> rank, starting at 1 for the best.
    """
    order = np.argsort(-scores)[:limit]
    if positive_only:
        order = [position for position in order if scores[position] > 0]
    return {int(position): rank for rank, position in enumerate(order, 1)}


def corpus_fingerprint(
    root: Path, patterns: tuple[str, ...] = DEFAULT_PATTERNS, exclude: tuple[str, ...] = ()
) -> str:
    """A short hash of the corpus's file names, sizes and modification times.

    Args:
        root: The indexed folder.
        patterns: The glob patterns used to find files.
        exclude: Patterns for files left out.

    Returns:
        A hex string that changes whenever a file is added, removed or edited.
    """
    digest = hashlib.sha256()
    for path in corpus_files(root, patterns, exclude):
        stat = path.stat()
        digest.update(
            f"{path.relative_to(root).as_posix()}|{stat.st_size}|{stat.st_mtime_ns}\n".encode()
        )
    return digest.hexdigest()[:16]


class DocIndex:
    """Chunks plus their embeddings, searchable by meaning and by keyword."""

    def __init__(
        self, chunks: list[Chunk], vectors: np.ndarray, embedder_name: str, fingerprint: str = ""
    ) -> None:
        """Wrap ready-made chunks and vectors. Use `build` or `load` to get them.

        Args:
            chunks: The chunks, in index order.
            vectors: One embedding per chunk, same order.
            embedder_name: The model that made the vectors.
            fingerprint: The corpus fingerprint at build time, to detect changes.

        Raises:
            ValueError: If there aren't as many vectors as chunks.
        """
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        self.chunks = chunks
        self.vectors = normalize_rows(vectors)
        self.embedder_name = embedder_name
        self.fingerprint = fingerprint
        self.keyword = BM25([chunk.text for chunk in chunks])

    @classmethod
    async def build(
        cls, chunks: list[Chunk], embedder: Embedder, fingerprint: str = ""
    ) -> DocIndex:
        """Embed every chunk and make an index.

        Args:
            chunks: The chunks to index, e.g. from `load_corpus`.
            embedder: The model that embeds them. Searches must use the same one.
            fingerprint: The corpus fingerprint, stored for `ensure_index`.

        Returns:
            The new index.
        """
        vectors = await embedder.embed([chunk.text for chunk in chunks])
        return cls(chunks, vectors, embedder.name, fingerprint)

    async def search(
        self,
        query: str,
        embedder: Embedder,
        *,
        k: int = 5,
        mode: SearchMode = "hybrid",
        reranker: Reranker | None = None,
        rerank_pool: int = RERANK_POOL,
    ) -> list[SearchHit]:
        """Find the chunks most relevant to a query.

        Args:
            query: What to look for.
            embedder: Embeds the query; must be the model the index was built with.
            k: How many results to return.
            mode: "hybrid" (both, fused), "semantic" (embeddings only) or "keyword"
                (BM25 only). Comparing them is what the retrieval eval is for.
            reranker: If given, the best `rerank_pool` results of the search are
                scored again by it, and the top `k` by that score are returned.
            rerank_pool: How many results the reranker looks at. More finds more,
                but each one costs a model call.

        Returns:
            Up to `k` hits, best first.

        Raises:
            ValueError: If `embedder` isn't the model that built the index.
        """
        if embedder.name != self.embedder_name:
            raise ValueError(
                f"This index was built with {self.embedder_name!r}, not {embedder.name!r}. "
                "Vectors from different models can't be compared; rebuild the index."
            )
        if not self.chunks:
            return []
        # Fuse over more candidates than we return, so a chunk ranked 8th by both
        # searches can still beat one ranked 1st by only one of them.
        candidates = max(k, FUSION_POOL)
        semantic: dict[int, int] = {}
        keyword: dict[int, int] = {}
        if mode in ("hybrid", "semantic"):
            query_vector = normalize_rows(await embedder.embed([query], is_query=True))[0]
            semantic = ranks(self.vectors @ query_vector, limit=candidates)
        if mode in ("hybrid", "keyword"):
            keyword = ranks(self.keyword.scores(query), limit=candidates, positive_only=True)

        fused: dict[int, float] = {}
        for ranking in (semantic, keyword):
            for position, rank in ranking.items():
                fused[position] = fused.get(position, 0.0) + 1 / (RRF_K + rank)
        shortlist = max(k, rerank_pool) if reranker else k
        best = sorted(fused, key=lambda position: fused[position], reverse=True)[:shortlist]
        hits = [
            SearchHit(
                self.chunks[position],
                fused[position],
                semantic.get(position),
                keyword.get(position),
            )
            for position in best
        ]
        if reranker and hits:
            scores = await reranker.score(query, [hit.chunk.text for hit in hits])
            for hit, score in zip(hits, scores, strict=True):
                hit.rerank_score = float(score)
            # sorted() is stable, so ties keep the search's order.
            hits = sorted(hits, key=lambda hit: hit.rerank_score or 0.0, reverse=True)
        return hits[:k]

    def save(self, folder: Path) -> None:
        """Write the index to `folder` (created if needed), replacing what was there.

        Args:
            folder: Where to save, e.g. ".lil_index".
        """
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "chunks.jsonl").open("w", encoding="utf-8") as file:
            for chunk in self.chunks:
                file.write(json.dumps(asdict(chunk)) + "\n")
        np.save(folder / "vectors.npy", self.vectors)
        meta = {
            "embedder": self.embedder_name,
            "fingerprint": self.fingerprint,
            "chunks": len(self.chunks),
            "dimensions": int(self.vectors.shape[1]) if len(self.vectors) else 0,
        }
        (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, folder: Path) -> DocIndex:
        """Read an index saved with `save`.

        Args:
            folder: The index folder.

        Returns:
            The index.

        Raises:
            FileNotFoundError: If the folder has no saved index.
        """
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        with (folder / "chunks.jsonl").open(encoding="utf-8") as file:
            chunks = [Chunk(**json.loads(line)) for line in file]
        vectors = np.load(folder / "vectors.npy")
        return cls(chunks, vectors, meta["embedder"], meta.get("fingerprint", ""))


async def ensure_index(
    root: Path,
    folder: Path,
    embedder: Embedder,
    *,
    patterns: tuple[str, ...] = DEFAULT_PATTERNS,
    exclude: tuple[str, ...] = (),
    rebuild: bool = False,
) -> tuple[DocIndex, bool]:
    """Load the saved index if it's still current, otherwise build and save a new one.

    It's current when it was built with the same embedding model and the corpus
    fingerprint hasn't changed (no file added, removed or edited).

    Args:
        root: The folder to index.
        folder: Where the index is saved.
        embedder: The embedding model.
        patterns: Which files to index.
        exclude: Patterns for files to leave out, e.g. the eval's own questions.
        rebuild: Build a new index even if the saved one is current.

    Returns:
        The index, and whether it was (re)built.
    """
    fingerprint = corpus_fingerprint(root, patterns, exclude)
    if not rebuild and (folder / "meta.json").exists():
        index = DocIndex.load(folder)
        if index.embedder_name == embedder.name and index.fingerprint == fingerprint:
            return index, False
    index = await DocIndex.build(load_corpus(root, patterns, exclude), embedder, fingerprint)
    index.save(folder)
    return index, True
