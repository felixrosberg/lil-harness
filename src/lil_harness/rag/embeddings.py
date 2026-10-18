"""Turn text into vectors, so texts with similar meaning end up close together.

An embedding model reads a text and returns a list of numbers (768 for
nomic-embed-text). Two texts about the same thing get vectors pointing in a similar
direction, which is what semantic search compares.

`Embedder` is the interface, shaped like `ChatBackend`: anything with `name` and an
async `embed` method works. Two implementations:

- `OpenAIEmbedder` calls any server's `/v1/embeddings` (Ollama, vLLM, hosted).
  `OllamaEmbedder` is the same with Ollama's address and a default model.
- `HashEmbedder` needs no model at all: it hashes words into a vector. It only
  matches shared words, not meaning, but it's instant and deterministic, which is
  what the tests want.
"""

from __future__ import annotations

import zlib
from typing import Protocol

import numpy as np
from openai import AsyncOpenAI

from lil_harness.rag.keyword import tokenize


class Embedder(Protocol):
    """Anything with these two members can embed text for the index."""

    @property
    def name(self) -> str:
        """Identifies the model. Vectors from different models can't be compared, so
        the index stores this and refuses to search with a different one."""
        ...

    async def embed(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        """Embed texts.

        Args:
            texts: The texts to embed.
            is_query: True for search queries, False for documents. Some models are
                trained with a different prefix for each.

        Returns:
            An array of shape (len(texts), dimensions).
        """
        ...


class OpenAIEmbedder:
    """Embeds through an OpenAI-compatible `/v1/embeddings` endpoint."""

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        *,
        query_prefix: str = "",
        document_prefix: str = "",
        batch_size: int = 32,
        client: AsyncOpenAI | None = None,
    ) -> None:
        """Set up the client. Nothing is sent until `embed` is called.

        Args:
            model: The embedding model's name on the server.
            base_url: The server's API root, ending in /v1.
            api_key: The key for hosted providers. Local servers ignore it.
            query_prefix: Text put in front of every query, for models trained that
                way (nomic-embed-text wants "search_query: ").
            document_prefix: Text put in front of every document.
            batch_size: How many texts to send per request.
            client: A ready-made client, so tests can plug in a fake transport.
        """
        self.model = model
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.batch_size = batch_size
        self.client = client or AsyncOpenAI(base_url=base_url, api_key=api_key or "EMPTY")

    @property
    def name(self) -> str:
        """Returns: the model name."""
        return self.model

    async def embed(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        """Embed texts in batches.

        Args:
            texts: The texts to embed.
            is_query: Use the query prefix instead of the document prefix.

        Returns:
            An array of shape (len(texts), dimensions), in the order given.

        Raises:
            openai.APIConnectionError: If the server can't be reached.
            openai.NotFoundError: If the server doesn't have the model.
        """
        prefix = self.query_prefix if is_query else self.document_prefix
        vectors: list[list[float]] = []
        for first in range(0, len(texts), self.batch_size):
            batch = [prefix + text for text in texts[first : first + self.batch_size]]
            response = await self.client.embeddings.create(model=self.model, input=batch)
            # The API may return items out of order; `index` says where each belongs.
            vectors += [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
        return np.array(vectors, dtype=np.float32)


# Models that were trained with task prefixes, and the prefixes they expect.
KNOWN_PREFIXES = {
    "nomic-embed-text": ("search_query: ", "search_document: "),
}


class OllamaEmbedder(OpenAIEmbedder):
    """An OpenAIEmbedder with Ollama's address and nomic-embed-text as the default."""

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434/v1",
        **kwargs: object,
    ) -> None:
        """Set up an embedder for a model served by Ollama.

        Args:
            model: An embedding model you have pulled, e.g. `ollama pull
                nomic-embed-text`. Known models get their prefixes automatically.
            base_url: Where Ollama listens.
            **kwargs: Passed on to OpenAIEmbedder, e.g. batch_size.
        """
        query_prefix, document_prefix = KNOWN_PREFIXES.get(model.split(":")[0], ("", ""))
        kwargs.setdefault("query_prefix", query_prefix)
        kwargs.setdefault("document_prefix", document_prefix)
        super().__init__(model, base_url, **kwargs)  # type: ignore[arg-type]


class HashEmbedder:
    """A model-free embedder: each word adds 1 to a bucket chosen by hashing it.

    Texts that share words get similar vectors, so it behaves like a rough keyword
    search. Useful for tests and for trying the pipeline before pulling a model.
    """

    def __init__(self, dimensions: int = 256) -> None:
        """Set up the embedder.

        Args:
            dimensions: The vector length. More buckets mean fewer words collide.
        """
        self.dimensions = dimensions

    @property
    def name(self) -> str:
        """Returns: "hash-<dimensions>"."""
        return f"hash-{self.dimensions}"

    async def embed(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        """Hash every text's words into a vector.

        Args:
            texts: The texts to embed.
            is_query: Ignored; queries and documents are embedded the same way.

        Returns:
            An array of shape (len(texts), dimensions).
        """
        vectors = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                # crc32, unlike hash(), gives the same number in every Python process.
                vectors[row, zlib.crc32(token.encode()) % self.dimensions] += 1
        return vectors
