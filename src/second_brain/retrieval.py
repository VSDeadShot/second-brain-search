"""Retrieval: embed a query, search the index, return ranked cited chunks.

Every refusal - blank query, no index, wrong vector space, unknown project -
happens before the query is embedded, so a mistake never spends quota.

Unproven until the real index exists: how well Gemini's embeddings rank real
questions, and what score separates a relevant chunk from an irrelevant one.
There is deliberately no score cut-off yet; results are always the top `k`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .embedding import Embedder
from .store import ChunkStore

DEFAULT_K = 5
# Fetched per result wanted, so collapsing duplicate texts still leaves k.
_OVERFETCH = 2


class RetrievalError(Exception):
    """A query that can't be answered as asked - with a message saying why."""


@dataclass(frozen=True)
class RetrievedChunk:
    project: str
    rel_path: str
    heading_path: str
    chunk_index: int
    text: str
    """The stored display text - never the heading-prefixed embedding text."""
    score: float
    """Cosine similarity: 1 - cosine distance. Higher is more relevant."""
    content_hash: str


def _resolve_project(store: ChunkStore, project: str) -> str:
    names = store.projects()
    for name in names:
        if name.lower() == project.strip().lower():
            return name
    raise RetrievalError(
        f"No project named {project!r} in the index. Projects: {', '.join(names)}."
    )


def retrieve(
    query: str,
    embedder: Embedder,
    store: ChunkStore,
    *,
    k: int = DEFAULT_K,
    project: str | None = None,
) -> list[RetrievedChunk]:
    if k < 1:
        raise ValueError("k must be at least 1")
    if not query.strip():
        raise RetrievalError("The query is empty.")
    if store.count() == 0:
        raise RetrievalError("There is no index yet - run `sbs index` first.")
    if store.namespace is None:
        raise RetrievalError(
            "This index doesn't record which embedding model built it, so its results "
            "can't be trusted. Rebuild it with `sbs index`."
        )
    if store.namespace != embedder.cache_namespace:
        raise RetrievalError(
            f"The index was built with {store.namespace!r} but queries would be embedded "
            f"with {embedder.cache_namespace!r} - their vectors aren't comparable. "
            "Rebuild the index with `sbs index`."
        )
    resolved_project = _resolve_project(store, project) if project is not None else None

    vector = embedder.embed_query(query)
    rows = store.query(vector, k=k * _OVERFETCH, project=resolved_project)

    results: list[RetrievedChunk] = []
    seen: set[str] = set()
    for row in rows:  # nearest first, so the first copy of a duplicate is its best
        if row["content_hash"] in seen:
            continue
        seen.add(row["content_hash"])
        results.append(
            RetrievedChunk(
                project=row["project"],
                rel_path=row["rel_path"],
                heading_path=row["heading_path"],
                chunk_index=row["chunk_index"],
                text=row["text"],
                score=1.0 - row["distance"],
                content_hash=row["content_hash"],
            )
        )
        if len(results) == k:
            break
    return results
