"""The `search_docs` tool: the model searches the index itself, when it decides to.

This is "agentic" RAG. Instead of the harness retrieving chunks before the model
starts and pasting them into the prompt, the model gets a search tool. It can search,
look at what came back, and search again with better words, or read the whole file
with `read_file` when a chunk is cut off. Small models do noticeably better when they
can retry like that.

    rag = RagTools(index, embedder)              # or RagTools(index, embedder, reranker=...)
    rag.register(registry)   # add search_docs to an existing registry, e.g. coding tools
"""

from __future__ import annotations

from lil_harness.rag.embeddings import Embedder
from lil_harness.rag.index import DocIndex
from lil_harness.rag.rerank import Reranker
from lil_harness.tools.coding import clip
from lil_harness.tools.registry import ToolRegistry


class RagTools:
    """The search tool, bound to one index and its embedding model."""

    def __init__(
        self,
        index: DocIndex,
        embedder: Embedder,
        *,
        reranker: Reranker | None = None,
        max_output_chars: int = 6000,
    ) -> None:
        """Bind the tool to an index.

        Args:
            index: The index to search.
            embedder: The model the index was built with.
            reranker: Rerank every search's top results with this, or None.
            max_output_chars: The most characters one search result may have.
        """
        self.index = index
        self.embedder = embedder
        self.reranker = reranker
        self.max_output_chars = max_output_chars

    def register(self, registry: ToolRegistry) -> ToolRegistry:
        """Add `search_docs` to a registry.

        Args:
            registry: The registry to add it to. Searching only reads, so it's
                registered with risk "read".

        Returns:
            The same registry, for chaining.
        """
        registry.tool(self.search_docs, risk="read")
        return registry

    async def search_docs(self, query: str, k: int = 5) -> str:
        """Search the project's documents and code for passages about a question.

        Each result starts with its source as 'file:first-last' lines. Cite those in
        your answer. If a passage is cut off, read the file with read_file. If nothing
        fits, search again with different words.

        Args:
            query: What to look for. Use words likely to appear in the answer, such as
                names of functions, settings or files.
            k: How many passages to return, from 1 to 10.

        Returns:
            The passages, best first, each headed "[n] file:first-last".
        """
        hits = await self.index.search(
            query, self.embedder, k=min(max(k, 1), 10), reranker=self.reranker
        )
        if not hits:
            return f"No passages found for {query!r}. Try other words."
        sections: list[str] = []
        used = 0
        for number, hit in enumerate(hits, 1):
            section = f"[{number}] {hit.chunk.id}\n{hit.chunk.text}"
            # Drop whole results from the end rather than cutting one in half: the
            # best ones come first.
            if sections and used + len(section) > self.max_output_chars:
                sections.append(
                    f"[... {len(hits) - number + 1} more results cut to save space ...]"
                )
                break
            sections.append(clip(section, self.max_output_chars))
            used += len(section)
        return "\n\n".join(sections)
