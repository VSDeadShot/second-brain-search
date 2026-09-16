"""Full-rebuild indexing: discover -> read -> chunk -> embed -> store.

SPEC decision 5 makes every index a full rebuild, so this deliberately has no
incremental logic. Chunking is split out as `collect_chunks` so a dry run can
report exactly what indexing would embed without calling the API.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

from .chunking import DEFAULT_MAX_CHARS, DEFAULT_OVERLAP, Chunk, chunk_document
from .config import Config
from .discovery import DiscoveredDoc, discover_documents
from .embedding import BatchCallback, Embedder
from .store import ChunkStore


@dataclass(frozen=True)
class IndexReport:
    documents: int
    chunks: int
    projects: int
    skipped: list[tuple[str, str]] = field(default_factory=list)
    chunks_per_project: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ChunkCollection:
    chunks: list[Chunk]
    report: IndexReport


def collect_chunks(
    config: Config,
    *,
    docs: Sequence[DiscoveredDoc] | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
) -> ChunkCollection:
    """Discover, read and chunk - everything short of embedding.

    `docs` can be supplied to reuse an existing discovery pass; otherwise the
    scan runs here.
    """
    documents = list(docs) if docs is not None else discover_documents(config)

    chunks: list[Chunk] = []
    skipped: list[tuple[str, str]] = []
    projects: set[str] = set()
    read_count = 0

    for doc in documents:
        try:
            # Self-authored markdown should be clean UTF-8; replace rather than
            # abort so one bad byte cannot stop a whole rebuild.
            text = doc.path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            skipped.append((str(doc.path), str(exc)))
            continue

        read_count += 1
        projects.add(doc.project)
        chunks.extend(chunk_document(doc, text, max_chars=max_chars, overlap=overlap))

    report = IndexReport(
        documents=read_count,
        chunks=len(chunks),
        projects=len(projects),
        skipped=skipped,
        chunks_per_project=dict(Counter(c.project for c in chunks)),
    )
    return ChunkCollection(chunks=chunks, report=report)


def store_chunks(
    collected: ChunkCollection,
    embedder: Embedder,
    store: ChunkStore,
    *,
    rebuild: bool = False,
    on_batch: BatchCallback | None = None,
) -> IndexReport:
    """Embed an already-collected set of chunks and write them to the store.

    Upsert alone never removes anything: chunk ids are deterministic, so a
    deleted document - or the tail of a shrunk one - would stay searchable.
    `rebuild=True` makes the store match this run exactly. Embedding happens
    BEFORE the reset, so an API failure (network, quota, bad key) leaves the
    previous index intact instead of empty.
    """
    vectors = (
        embedder.embed_documents([c.text for c in collected.chunks], on_batch=on_batch)
        if collected.chunks
        else []
    )

    if rebuild:
        store.reset()
    store.upsert(collected.chunks, vectors)

    return collected.report


def index_documents(
    config: Config,
    embedder: Embedder,
    store: ChunkStore,
    *,
    docs: Sequence[DiscoveredDoc] | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap: int = DEFAULT_OVERLAP,
    rebuild: bool = False,
) -> IndexReport:
    """Collect, embed and store every discovered document in one call."""
    collected = collect_chunks(config, docs=docs, max_chars=max_chars, overlap=overlap)
    return store_chunks(collected, embedder, store, rebuild=rebuild)
