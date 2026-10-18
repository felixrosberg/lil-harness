"""Step 6: chunking, keyword and semantic search, the index, and the search tool.

Everything uses HashEmbedder or a fake embeddings server, so no model is needed.
"""

import json
import sys
import types
from pathlib import Path

import httpx2 as httpx
import numpy as np
import pytest
from openai import AsyncOpenAI

from lil_harness.agent.loop import run_agent
from lil_harness.backends.fake import FakeBackend
from lil_harness.permissions import PermissionPolicy
from lil_harness.rag.chunking import chunk_text, corpus_files, load_corpus
from lil_harness.rag.embeddings import HashEmbedder, OllamaEmbedder, OpenAIEmbedder
from lil_harness.rag.evaluate import (
    RetrievalCase,
    RetrievalScore,
    load_retrieval_cases,
    score_retrieval,
)
from lil_harness.rag.index import DocIndex, ensure_index, ranks
from lil_harness.rag.keyword import BM25, tokenize
from lil_harness.rag.rerank import CrossEncoderReranker, LLMReranker
from lil_harness.rag.tools import RagTools
from lil_harness.tools.registry import ToolRegistry
from lil_harness.types import AssistantTurn, Message, ToolCall

REPO = Path(__file__).parent.parent


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "cooking.md").write_text(
        "# Pasta\nBoil water with salt. Cook the spaghetti for nine minutes.\n\n"
        "# Bread\nKnead the dough and let it rise overnight before baking.\n"
    )
    (tmp_path / "limits.py").write_text(
        "import asyncio\n\n\ndef make_limiter(concurrency):\n"
        "    return asyncio.Semaphore(concurrency)\n"
    )
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "secret.md").write_text("spaghetti secrets\n")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    return tmp_path


# Chunking


def test_chunks_cover_every_line_with_overlap():
    text = "\n".join(f"line {number} " + "x" * 40 for number in range(1, 101))
    chunks = chunk_text("a.txt", text, max_chars=500, overlap_lines=2)

    assert len(chunks) > 1
    assert chunks[0].start_line == 1
    assert chunks[-1].end_line == 100
    for previous, chunk in zip(chunks, chunks[1:], strict=False):
        assert chunk.start_line == previous.end_line - 1  # two shared lines
    for chunk in chunks:
        assert len(chunk.text) <= 500
        assert chunk.text.splitlines()[0].startswith(f"line {chunk.start_line} ")


def test_chunks_end_before_a_heading_once_they_have_substance():
    text = "# One\n" + "word " * 80 + "\n# Two\nshort\n"
    chunks = chunk_text("a.md", text, max_chars=1000, overlap_lines=0)

    assert [chunk.text.splitlines()[0] for chunk in chunks] == ["# One", "# Two"]
    assert chunks[0].id == "a.md:1-2"


def test_a_long_line_still_makes_progress():
    chunks = chunk_text("a.txt", "x" * 5000 + "\nend", max_chars=100)

    assert [chunk.start_line for chunk in chunks] == [1, 2]


def test_blank_files_give_no_chunks():
    assert chunk_text("a.txt", "\n\n   \n") == []


def test_corpus_skips_hidden_folders_and_other_file_types(corpus):
    names = [path.relative_to(corpus).as_posix() for path in corpus_files(corpus)]

    assert names == ["docs/cooking.md", "limits.py"]
    assert {chunk.source for chunk in load_corpus(corpus)} == set(names)


# Keyword search


def test_tokenize_splits_code_names_into_parts():
    assert tokenize("run_agent ChatBackend max2") == [
        "run_agent", "run", "agent", "chatbackend", "chat", "backend", "max2", "max", "2",
    ]  # fmt: skip


def test_bm25_ranks_the_document_with_rare_shared_words_first():
    bm25 = BM25(["the cat sat", "the dog sat on the semaphore", "the the the"])
    scores = bm25.scores("semaphore")

    assert scores.argmax() == 1
    assert scores[0] == 0 and scores[2] == 0


def test_ranks_can_drop_non_matches():
    scores = np.array([0.0, 2.0, 1.0])

    assert ranks(scores, limit=3) == {1: 1, 2: 2, 0: 3}
    assert ranks(scores, limit=3, positive_only=True) == {1: 1, 2: 2}


# Embeddings


def fake_embeddings_server() -> tuple[AsyncOpenAI, list[dict]]:
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        # Return the items in reverse order, to check the embedder sorts them back.
        data = [
            {"object": "embedding", "index": position, "embedding": [float(len(text)), 1.0]}
            for position, text in enumerate(body["input"])
        ][::-1]
        return httpx.Response(
            200,
            json={"object": "list", "data": data, "model": body["model"],
                  "usage": {"prompt_tokens": 1, "total_tokens": 1}},
        )  # fmt: skip

    client = AsyncOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="EMPTY",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    return client, bodies


async def test_openai_embedder_batches_and_keeps_order():
    client, bodies = fake_embeddings_server()
    embedder = OpenAIEmbedder("m", "unused", batch_size=2, client=client)

    vectors = await embedder.embed(["a", "bb", "ccc"])

    assert [len(body["input"]) for body in bodies] == [2, 1]
    assert vectors[:, 0].tolist() == [1.0, 2.0, 3.0]


async def test_ollama_embedder_adds_nomic_prefixes():
    client, bodies = fake_embeddings_server()
    embedder = OllamaEmbedder(client=client)

    await embedder.embed(["a doc"])
    await embedder.embed(["a question"], is_query=True)

    assert bodies[0]["input"] == ["search_document: a doc"]
    assert bodies[1]["input"] == ["search_query: a question"]
    assert embedder.name == "nomic-embed-text"


async def test_hash_embedder_is_deterministic_and_matches_shared_words():
    embedder = HashEmbedder()
    vectors = await embedder.embed(["boil the pasta", "boil the pasta", "knead dough"])

    assert vectors.shape == (3, 256)
    assert np.array_equal(vectors[0], vectors[1])
    assert vectors[0] @ vectors[2] == 0


# The index


async def test_hybrid_search_finds_by_word_and_by_name(corpus):
    embedder = HashEmbedder()
    index = await DocIndex.build(load_corpus(corpus), embedder)

    pasta = await index.search("how long to cook spaghetti", embedder, k=1)
    limiter = await index.search("Semaphore", embedder, k=1)

    assert pasta[0].chunk.source == "docs/cooking.md"
    assert pasta[0].chunk.text.startswith("# Pasta")
    assert limiter[0].chunk.source == "limits.py"
    assert limiter[0].keyword_rank == 1


@pytest.mark.parametrize("mode", ["hybrid", "semantic", "keyword"])
async def test_every_mode_returns_at_most_k(corpus, mode):
    embedder = HashEmbedder()
    index = await DocIndex.build(load_corpus(corpus), embedder)

    hits = await index.search("bread dough", embedder, k=2, mode=mode)

    assert 1 <= len(hits) <= 2
    assert "Bread" in hits[0].chunk.text


async def test_searching_with_another_embedder_is_refused(corpus):
    index = await DocIndex.build(load_corpus(corpus), HashEmbedder(256))

    with pytest.raises(ValueError, match="rebuild"):
        await index.search("pasta", HashEmbedder(128))


async def test_save_and_load_roundtrip(corpus, tmp_path):
    embedder = HashEmbedder()
    index = await DocIndex.build(load_corpus(corpus), embedder, fingerprint="abc")
    index.save(tmp_path / "index")

    loaded = DocIndex.load(tmp_path / "index")

    assert loaded.chunks == index.chunks
    assert np.allclose(loaded.vectors, index.vectors)
    assert (loaded.embedder_name, loaded.fingerprint) == ("hash-256", "abc")


async def test_ensure_index_rebuilds_only_when_files_change(corpus, tmp_path):
    folder = tmp_path / "index"
    embedder = HashEmbedder()

    _, built_first = await ensure_index(corpus, folder, embedder)
    _, built_again = await ensure_index(corpus, folder, embedder)
    (corpus / "new.md").write_text("# Soup\nSimmer the soup.\n")
    index, built_after_change = await ensure_index(corpus, folder, embedder)

    assert (built_first, built_again, built_after_change) == (True, False, True)
    assert "new.md" in {chunk.source for chunk in index.chunks}


# The search tool


async def test_search_docs_through_the_registry(corpus):
    embedder = HashEmbedder()
    index = await DocIndex.build(load_corpus(corpus), embedder)
    registry = RagTools(index, embedder).register(ToolRegistry(PermissionPolicy.preset("readonly")))

    assert [spec.name for spec in registry.specs()] == ["search_docs"]
    result = await registry.execute(
        ToolCall(id="1", name="search_docs", arguments={"query": "spaghetti", "k": 1})
    )

    assert not result.is_error
    assert result.content.startswith("[1] docs/cooking.md:1-")


async def test_search_docs_drops_whole_results_over_the_limit(corpus):
    embedder = HashEmbedder()
    index = await DocIndex.build(load_corpus(corpus), embedder)
    tools = RagTools(index, embedder, max_output_chars=120)

    content = await tools.search_docs("the", k=3)

    assert content.count("[1]") == 1
    assert "more results cut" in content


async def test_an_agent_can_search_then_answer(corpus):
    embedder = HashEmbedder()
    index = await DocIndex.build(load_corpus(corpus), embedder)
    registry = RagTools(index, embedder).register(ToolRegistry())
    backend = FakeBackend(
        [
            AssistantTurn(
                tool_calls=[ToolCall(id="1", name="search_docs", arguments={"query": "bread"})]
            ),
            AssistantTurn(content="Let it rise overnight (docs/cooking.md:4-5)."),
        ]
    )

    result = await run_agent(backend, registry, [Message.user("How long does bread rise?")])

    assert "Knead the dough" in result.messages[2].content
    assert result.messages[-1].content.endswith("(docs/cooking.md:4-5).")


# Retrieval evals


async def test_retrieval_scores(corpus):
    embedder = HashEmbedder()
    index = await DocIndex.build(load_corpus(corpus), embedder)
    cases = [
        RetrievalCase(question="cook spaghetti", source="docs/cooking.md", contains="Pasta"),
        RetrievalCase(question="Semaphore", source="limits.py"),
        RetrievalCase(question="quantum physics", source="nowhere.md"),
    ]

    score = await score_retrieval(index, embedder, cases, mode="hybrid", k=2)

    assert score.hit_ranks == [1, 1, None]
    assert score.recall == pytest.approx(2 / 3)
    assert score.mrr == pytest.approx(2 / 3)


def test_the_repo_retrieval_cases_load_and_point_at_real_text():
    cases = load_retrieval_cases(REPO / "evals" / "retrieval")

    assert len(cases) >= 10
    for case in cases:
        text = (REPO / case.source).read_text(encoding="utf-8")
        assert case.contains is None or case.contains.lower() in text.lower(), case.question


def test_excluded_files_are_left_out(corpus):
    files = corpus_files(corpus, exclude=("docs/*",))

    assert [path.name for path in files] == ["limits.py"]


def test_recall_at_and_mrr_use_their_cutoffs():
    score = RetrievalScore("hybrid", k=5, depth=20, hit_ranks=[1, 2, 12, None])

    assert score.recall == 0.5
    assert score.recall_at(20) == 0.75
    assert score.mrr == pytest.approx((1 + 1 / 2) / 4)  # rank 12 is past k


# Reranking


class WordReranker:
    """Scores a text by how often it contains one word, and records what it was given."""

    def __init__(self, word: str) -> None:
        self.word = word
        self.seen: list[list[str]] = []

    @property
    def name(self) -> str:
        return f"word:{self.word}"

    async def score(self, query: str, texts: list[str]) -> np.ndarray:
        self.seen.append(texts)
        return np.array([text.lower().count(self.word) for text in texts], dtype=float)


async def test_a_reranker_reorders_the_shortlist(corpus):
    embedder = HashEmbedder()
    index = await DocIndex.build(load_corpus(corpus), embedder)
    reranker = WordReranker("semaphore")

    plain = await index.search("cook spaghetti", embedder, k=1)
    reranked = await index.search("cook spaghetti", embedder, k=1, reranker=reranker)

    assert plain[0].chunk.source == "docs/cooking.md"
    assert reranked[0].chunk.source == "limits.py"
    assert reranked[0].rerank_score == 1.0
    # It saw the whole shortlist (every chunk here), not just the top k.
    assert len(reranker.seen[0]) == len(index.chunks)


async def test_rerank_pool_limits_what_the_reranker_sees(corpus):
    embedder = HashEmbedder()
    index = await DocIndex.build(load_corpus(corpus), embedder)
    reranker = WordReranker("dough")

    await index.search("spaghetti", embedder, k=1, reranker=reranker, rerank_pool=1)

    assert len(reranker.seen[0]) == 1


async def test_search_docs_uses_the_reranker(corpus):
    embedder = HashEmbedder()
    index = await DocIndex.build(load_corpus(corpus), embedder)
    tools = RagTools(index, embedder, reranker=WordReranker("semaphore"))

    content = await tools.search_docs("cook spaghetti", k=1)

    assert content.startswith("[1] limits.py:")


async def test_reranking_shows_up_in_the_retrieval_score(corpus):
    embedder = HashEmbedder()
    index = await DocIndex.build(load_corpus(corpus), embedder)
    cases = [RetrievalCase(question="cook spaghetti", source="limits.py")]

    plain = await score_retrieval(index, embedder, cases, mode="hybrid", k=1)
    reranked = await score_retrieval(
        index, embedder, cases, mode="hybrid", k=1, reranker=WordReranker("semaphore")
    )

    assert (plain.label, plain.recall, plain.recall_at(20)) == ("hybrid", 0.0, 1.0)
    assert (reranked.label, reranked.recall) == ("hybrid+rerank", 1.0)


async def test_llm_reranker_reads_the_number_from_each_reply():
    backend = FakeBackend(
        [
            AssistantTurn(content="7"),
            AssistantTurn(content="No idea"),
            AssistantTurn(content="10/10"),
        ]
    )
    reranker = LLMReranker(backend, concurrency=1)

    scores = await reranker.score("a question", ["one", "two", "three"])

    assert scores.tolist() == [7.0, 0.0, 10.0]
    assert backend.calls[0].tool_names == []
    assert "Question: a question" in backend.calls[0].messages[1].content
    assert "Passage:\none" in backend.calls[0].messages[1].content


def test_cross_encoder_says_how_to_install_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # import fails

    with pytest.raises(ImportError, match="uv sync --extra rerank"):
        CrossEncoderReranker()


async def test_cross_encoder_scores_query_text_pairs(monkeypatch):
    received = {}

    class FakeCrossEncoder:
        def __init__(self, model, device=None):
            received["model"] = model

        def predict(self, pairs, batch_size):
            received["pairs"] = pairs
            return [float(len(text)) for _, text in pairs]

    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    reranker = CrossEncoderReranker()
    scores = await reranker.score("q", ["ab", "abcd"])

    assert reranker.name == received["model"] == "BAAI/bge-reranker-v2-m3"
    assert received["pairs"] == [("q", "ab"), ("q", "abcd")]
    assert scores.tolist() == [2.0, 4.0]
