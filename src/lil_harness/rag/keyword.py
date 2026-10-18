"""Keyword search with BM25: score chunks by the words they share with the query.

BM25 is the classic search-engine formula. A chunk scores higher when it contains the
query's words often (with diminishing returns), when those words are rare across the
corpus (so "the" counts for little and "semaphore" for a lot), and when it's short
(a word in a 3-line chunk says more than the same word in a 40-line one).

Embeddings understand meaning but are fuzzy about exact names. For code and docs,
where someone asks about `run_agent` or `max_steps`, keyword search is often the one
that finds the right place, so the index combines the two.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")
_PART = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")


def tokenize(text: str) -> list[str]:
    """Split text into lowercase search tokens, splitting code names into their parts too.

    "run_agent" gives "run_agent", "run" and "agent", and "ChatBackend" gives
    "chatbackend", "chat" and "backend", so a query for either form matches.

    Args:
        text: Any text or code.

    Returns:
        The tokens, in order, with repeats.
    """
    tokens: list[str] = []
    for word in _WORD.findall(text):
        tokens.append(word.lower())
        parts = _PART.findall(word)
        if len(parts) > 1:
            tokens += [part.lower() for part in parts]
    return tokens


class BM25:
    """A BM25 scorer over a fixed list of documents (here: chunks)."""

    def __init__(self, documents: list[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        """Count the words of every document, once, so queries are fast.

        Args:
            documents: The texts to search, in index order.
            k1: How quickly repeats of a word stop adding score. 1.2 to 2 is typical.
            b: How much longer documents are penalized, from 0 (not at all) to 1.
        """
        self.k1 = k1
        self.b = b
        self.term_counts = [Counter(tokenize(document)) for document in documents]
        self.lengths = np.array([sum(counts.values()) for counts in self.term_counts])
        self.average_length = float(self.lengths.mean()) if len(documents) else 0.0
        document_frequency: Counter[str] = Counter()
        for counts in self.term_counts:
            document_frequency.update(counts.keys())
        total = len(documents)
        # Rare words weigh more. The +1 keeps every weight positive.
        self.idf = {
            term: math.log(1 + (total - count + 0.5) / (count + 0.5))
            for term, count in document_frequency.items()
        }

    def scores(self, query: str) -> np.ndarray:
        """Score every document against a query.

        Args:
            query: The search text.

        Returns:
            One score per document, in index order; 0 means no shared words.
        """
        scores = np.zeros(len(self.term_counts))
        query_terms = set(tokenize(query))
        for position, counts in enumerate(self.term_counts):
            length_factor = (
                1 - self.b + self.b * self.lengths[position] / (self.average_length or 1)
            )
            for term in query_terms & counts.keys():
                frequency = counts[term]
                scores[position] += (
                    self.idf[term]
                    * frequency
                    * (self.k1 + 1)
                    / (frequency + self.k1 * length_factor)
                )
        return scores
