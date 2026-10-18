"""Rerankers: a second, more careful look at the best search results.

Search embeds the query and every chunk separately, then compares vectors. That's
fast, but the model never sees the two side by side, so it can't tell a chunk that
*answers* the question from one that merely *mentions* its words. A reranker reads
the query and one chunk together and scores how well the chunk answers it. That's
far more accurate, and far slower (one model call per chunk), so it only runs on the
shortlist search already found:

    hybrid search -> top 20 -> reranker -> top 5 -> the chat model

It can only reorder that shortlist. If the right chunk isn't in the top 20, reranking
can't bring it back, which is why `lil-rag-eval` shows recall@20 as well.

`Reranker` is the interface. Two implementations:

- `LLMReranker` asks a chat model (your qwen) to rate each chunk from 0 to 10. No new
  model to download, but slow, and small models give coarse, noisy ratings.
- `CrossEncoderReranker` runs a model trained for exactly this job, such as
  bge-reranker-v2-m3, through sentence-transformers. Install it with
  `uv sync --extra rerank`.
"""

from __future__ import annotations

import asyncio
import re
from typing import Protocol

import numpy as np

from lil_harness.backends.base import ChatBackend
from lil_harness.types import Message


class Reranker(Protocol):
    """Anything with these two members can rerank search results."""

    @property
    def name(self) -> str:
        """Identifies the reranker in reports."""
        ...

    async def score(self, query: str, texts: list[str]) -> np.ndarray:
        """Score how well each text answers the query.

        Args:
            query: The search query.
            texts: The candidate chunks' texts.

        Returns:
            One score per text, higher is better. Only the order matters.
        """
        ...


LLM_RERANK_PROMPT = """You rate search results. Given a question and a passage from a \
project's files, rate how well the passage helps answer the question:
0 = unrelated, 5 = related but doesn't answer it, 10 = answers it directly.
Reply with the number only."""

_NUMBER = re.compile(r"\d+(?:\.\d+)?")


class LLMReranker:
    """Rates each chunk with a chat model, from 0 to 10."""

    def __init__(self, backend: ChatBackend, *, name: str = "llm", concurrency: int = 4) -> None:
        """Set up the reranker.

        Args:
            backend: The chat model, e.g. OllamaBackend(model="qwen3.5:4b"). Leave
                thinking off: it's slow, and a single number doesn't need it.
            name: How the reranker is named in reports.
            concurrency: How many chunks to rate at the same time. Ollama queues
                requests beyond what it's set to run in parallel, so higher values
                only help if the server allows it (OLLAMA_NUM_PARALLEL).
        """
        self.backend = backend
        self._name = name
        self.concurrency = concurrency

    @property
    def name(self) -> str:
        """Returns: the name given at setup."""
        return self._name

    async def score(self, query: str, texts: list[str]) -> np.ndarray:
        """Ask the model to rate every text, a few at a time.

        Args:
            query: The search query.
            texts: The candidate chunks' texts.

        Returns:
            One rating per text. A reply without a number counts as 0.
        """
        semaphore = asyncio.Semaphore(self.concurrency)

        async def rate(text: str) -> float:
            messages = [
                Message.system(LLM_RERANK_PROMPT),
                Message.user(f"Question: {query}\n\nPassage:\n{text}"),
            ]
            async with semaphore:
                turn = await self.backend.complete(messages, [])
            match = _NUMBER.search(turn.content or "")
            return min(float(match.group()), 10.0) if match else 0.0

        return np.array(await asyncio.gather(*(rate(text) for text in texts)))


class CrossEncoderReranker:
    """Scores (query, chunk) pairs with a cross-encoder model from sentence-transformers."""

    def __init__(
        self,
        model: str = "BAAI/bge-reranker-v2-m3",
        *,
        device: str | None = None,
        batch_size: int = 8,
    ) -> None:
        """Load the model. The first run downloads it from Hugging Face (about 2 GB).

        Args:
            model: A cross-encoder on Hugging Face. bge-reranker-v2-m3 (0.6B
                parameters) is strong and fits easily next to a 4B chat model.
                cross-encoder/ms-marco-MiniLM-L-6-v2 is far smaller and faster, but
                English only and weaker on code.
            device: "cuda" or "cpu". None picks the GPU when PyTorch can see one.
            batch_size: How many pairs go through the model at once. Lower it if the
                GPU runs out of memory.

        Raises:
            ImportError: If sentence-transformers isn't installed.
        """
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "The cross-encoder reranker needs sentence-transformers: uv sync --extra rerank"
            ) from None
        self.model_name = model
        self.batch_size = batch_size
        self.model = CrossEncoder(model, device=device)

    @property
    def name(self) -> str:
        """Returns: the model name."""
        return self.model_name

    async def score(self, query: str, texts: list[str]) -> np.ndarray:
        """Score every (query, text) pair.

        Args:
            query: The search query.
            texts: The candidate chunks' texts.

        Returns:
            One relevance score per text.
        """
        pairs = [(query, text) for text in texts]
        # The model runs on this thread and blocks it, so run it on another one and
        # keep the event loop free (the agent may be running other tool calls).
        scores = await asyncio.to_thread(self.model.predict, pairs, batch_size=self.batch_size)
        return np.asarray(scores, dtype=np.float32)
