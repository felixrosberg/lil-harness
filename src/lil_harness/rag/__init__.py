"""Retrieval-augmented generation: let the model search a folder of documents and code.

The pieces, in the order data flows through them:

1. `chunking.py` splits files into overlapping chunks that remember where they came
   from (`README.md:120-140`).
2. `embeddings.py` turns text into vectors through an embedding model, so similar
   meanings end up close together.
3. `keyword.py` scores chunks by the words they share with the query (BM25), which
   catches exact names like `run_agent` that embeddings often miss.
4. `index.py` holds the chunks and their vectors, combines both kinds of search, and
   saves everything to disk so it's only built once.
5. `rerank.py` optionally rescores the top results with a model that reads the query
   and each chunk together.
6. `tools.py` wraps the index as a `search_docs` tool the model can call.
7. `evaluate.py` measures whether the right chunks come back for known questions.
"""
