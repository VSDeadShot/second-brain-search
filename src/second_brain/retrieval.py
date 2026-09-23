"""Retrieval: embed a query, search the index, return ranked cited chunks.

Every refusal - blank query, no index, wrong vector space, unknown project -
happens before the query is embedded, so a mistake never spends quota.

Unproven until the real index exists: how well Gemini's embeddings rank real
questions, and what score separates a relevant chunk from an irrelevant one.
There is deliberately no score cut-off yet; results are always the top `k`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .embedding import Embedder
from .store import ChunkStore

DEFAULT_K = 5
# Fetched per result wanted, so collapsing duplicate texts still leaves k.
_OVERFETCH = 2

# Answering reads wider than `search` shows, then caps per file and per project.
DEFAULT_ANSWER_K = 10
DEFAULT_PER_FILE = 3
DEFAULT_PER_PROJECT = 5
# Candidates ranked before the caps are applied: a good passage can sit well
# below a long document's chunks.
DEFAULT_POOL = 40


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
    mtime: float
    """The file's modification time when it was indexed - freshness for citations."""


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
                mtime=row["mtime"],
            )
        )
        if len(results) == k:
            break
    return results


def retrieve_for_answer(
    query: str,
    embedder: Embedder,
    store: ChunkStore,
    *,
    k: int = DEFAULT_ANSWER_K,
    per_file: int = DEFAULT_PER_FILE,
    per_project: int = DEFAULT_PER_PROJECT,
    pool: int = DEFAULT_POOL,
    project: str | None = None,
) -> list[RetrievedChunk]:
    """Top-k passages for answering, with no single file or project dominating.

    Measured on the real index: at k=3 one long document (Watch Tracker's
    EXPLAINER.md) took every slot on cross-project questions. A wide pool is
    ranked as usual, then walked in score order keeping a passage only while its
    file and project are under their caps.

    Capped passages are never used to top the list back up - backfilling would
    hand the dominant document the slots straight back - so fewer than `k` may
    come back. `project` narrows the search and lifts the per-project cap, since
    asking for one project on purpose is not domination.
    """
    if per_file < 1 or per_project < 1:
        raise ValueError("per_file and per_project must be at least 1")
    if pool < k:
        raise ValueError("pool must be at least k")

    candidates = retrieve(query, embedder, store, k=pool, project=project)

    per_file_seen: Counter[tuple[str, str]] = Counter()
    per_project_seen: Counter[str] = Counter()
    kept: list[RetrievedChunk] = []

    for chunk in candidates:  # already best-first
        if per_file_seen[(chunk.project, chunk.rel_path)] >= per_file:
            continue
        if project is None and per_project_seen[chunk.project] >= per_project:
            continue
        per_file_seen[(chunk.project, chunk.rel_path)] += 1
        per_project_seen[chunk.project] += 1
        kept.append(chunk)
        if len(kept) == k:
            break
    return kept
